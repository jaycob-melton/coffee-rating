import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.models.utils import CoffeeDataset, TripleTrainingDataset, collate
from src.models.model import DualEncoder
from src.config import (
    TRAIN_DATA_PATH,
    VOCABS_PATH,
    QUERIES_PATH,
    SBERT_MODEL_DIR,
    TRAINED_MODEL_PATH,
    MODEL_SAVE_PATH,
    TRAIN_PARAMS
)

torch.manual_seed(189)  # for reproducibility

def load_model_train(model_path: str, numerical_dim: int, device):
    """
    Loads a trained model from a .pth file for inference, i.e. evaluation
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    vocabs = checkpoint["vocabs"]

    model = DualEncoder(vocabs, numerical_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # model.eval()
    
    print(f"Model loaded from {model_path}")
    return model, vocabs


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    # df = pd.read_csv(config["preprocessed_data_path"])
    # df["combined_text"] = df["blind assessment"].fillna() + " " + df["bottom line"].fillna("")
    df = config["preprocessed_data"]
    vocabs = config["vocabs"]

    # used by collate to look up coffee data by index
    full_coffee_dataset = CoffeeDataset(df, vocabs)

    # provides thje (query, positive_idx) pairs for training
    train_dataset = TripleTrainingDataset(config["queries_path"], df)


    train_loader = DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate(batch, full_coffee_dataset)
    )

    # model, loss, optimizer
    # if config["model_path"]:
    #     model, vocabs = load_model_train(config["model_path"], numerical_dim=len(full_coffee_dataset.numerical_cols), device=device)
    # else:
    #     model = DualEncoder(vocabs, numerical_dim=len(full_coffee_dataset.numerical_cols)).to(device)

    model, _ = build_model(
        vocabs_path=config["vocabs"],
        numerical_dim=len(full_coffee_dataset.numerical_cols),
        device=device,
        model_arch_path=config["enc_model_path"],
        model_weights_path=config["model_path"],
    )

    loss_fn = nn.TripletMarginLoss(margin=config["margin"])

    transformer_params = model.transformer.parameters()
    head_params = list(model.metadata_encoder.parameters()) + list(model.fusion_layer.parameters())

    optimizer = AdamW([
        {"params": transformer_params, "lr": config["transformer_lr"]},
        {"params": head_params, "lr": config["head_lr"]}
    ])
    
    # optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    print("Starting training...")
    loss_info = {
        "loss": [],
        "semi_hard_negatives": [],
    }

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        num_semi_hard = 0
        if config["semi_hard_mining_start_epoch"]:
            use_semi_hard_mining = epoch >= config["semi_hard_mining_start_epoch"]
        else:
            last_4_losses = loss_info["loss"][-4:] if len(loss_info["loss"]) >= 4 else []
            last_3_relative_diff = [
                abs((last_4_losses[i] - last_4_losses[i-1]) / last_4_losses[i-1]) if last_4_losses[i-1] != 0 else 0
                for i in range(1, len(last_4_losses))
            ] if len(last_4_losses) >= 4 else []
            use_semi_hard_mining = len(last_4_losses) >= 4 and all(loss < 0.02 for loss in last_3_relative_diff) 

        if use_semi_hard_mining:
            print(f"Epoch {epoch+1}/{config['epochs']} - Using Semi-Hard Negative Mining")

        else:
            print(f"Epoch {epoch+1}/{config['epochs']} - Using In-Batch Negative Mining")

        for queries, positive_indices, positive_coffee_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            for key, val in positive_coffee_batch.items():
                if isinstance(val, torch.Tensor):
                    positive_coffee_batch[key] = val.to(device)
                elif isinstance(val, dict):
                    for k, v in val.items():
                        positive_coffee_batch[key][k] = v.to(device)

            # forward pass
            query_embeddings = model.encode_queries(queries).to(device)
            # debugging
            with torch.no_grad():
                for col in ["countries_extracted", "process", "varietals"]:
                    idxs = positive_coffee_batch["categoricals"][col]
                    max_idx = int(idxs.max().item()) if idxs.numel() else -1
                    num_rows = model.metadata_encoder.__getattr__(f"{col.split('_')[0]}_embed").num_embeddings
                    if max_idx >= num_rows:
                        raise ValueError(
                            f"{col}: max idx {max_idx} >= num_rows {num_rows} "
                            f"(offsets: {positive_coffee_batch['categoricals'][col + '_offsets']})"
                        )

            positive_embeddings = model.encode_coffees(positive_coffee_batch)
            
            # all other positive embeddings are potential negatives
            negative_embeddings = positive_embeddings

            # calculate distances 
            # first calculate the distances between the queries and their positives
            pos_dists = 1 - F.cosine_similarity(query_embeddings, positive_embeddings)

            dist_matrix = 1 - F.cosine_similarity(query_embeddings.unsqueeze(1), negative_embeddings.unsqueeze(0), dim=2)

            if use_semi_hard_mining:
                final_negative_embeddings = []
                
                for i in range(len(queries)):
                    pos_dist = pos_dists[i] 
                    neg_dists = dist_matrix[i]

                    # exclude the positive from the negatives
                    neg_dists[i] = float("inf")

                    # find semi-hard negatives; negatives that are harder than the positive by violate the margin
                    semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + config["margin"])

                    if semi_hard_mask.any():
                        semi_hard_indices = torch.where(semi_hard_mask)[0]
                        rand_idx = semi_hard_indices[random.randint(0, len(semi_hard_indices)-1)]
                        num_semi_hard += 1
                    else:
                        rand_idx = torch.argmin(neg_dists)
                    
                    final_negative_embeddings.append(negative_embeddings[rand_idx])
                
                negative_embeddings = torch.stack(final_negative_embeddings)
            
            else:
                # just pick an essentially random coffee as the negative
                negative_indices = [(i+1) % len(queries) for i in range(len(queries))]
                negative_embeddings = negative_embeddings[negative_indices]

            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_info["loss"].append(avg_loss)
        loss_info["semi_hard_negatives"].append(num_semi_hard)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
        if use_semi_hard_mining:
            print(f"Number of Semi-hard Negatives Used: {num_semi_hard}")

        # checkpoint = {
        #     "model_state_dict": model.state_dict(),
        #     "vocabs": vocabs,
        #     "loss": loss_info["loss"],
        # }
        torch.save(model.state_dict(), f"{config['save_path']}{epoch+1}.pth")
    
    return loss_info


if __name__ == "__main__":
    # PREPROCESSED_PATH = "data/processed/preprocessed_data.csv"
    # TRAINING_DATA_PATH = "data/processed/training_data.jsonl"
    # MODEL_PATH = "coffee_model_simpleadam_epoch_10.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("Loading Coffee Data...")
    # df = pd.read_csv(PREPROCESSED_PATH)
    # df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    
    # print("Creating 80/20 train-test split...")
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=189)
    # train_df = train_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)
    
    # train_df.to_csv("data/processed/train_data_8_11.csv", index=False)
    # test_df.to_csv("data/processed/test_data_8_11.csv", index=False)

    # vocabs = build_all_vocabs(train_df)


    config = {
        # Data paths
        "preprocessed_data": TRAIN_DATA_PATH,
        "query_path": QUERIES_PATH,
        "vocabs": VOCABS_PATH,

        # Model paths
        "enc_model_path": SBERT_MODEL_DIR,
        "model_path": TRAINED_MODEL_PATH,
        "save_path": MODEL_SAVE_PATH,

        "batch_size": TRAIN_PARAMS["batch_size"],
        "transformer_lr": TRAIN_PARAMS["transformer_lr"],
        "head_lr": TRAIN_PARAMS["head_lr"],
        "epochs": TRAIN_PARAMS["num_epochs"],
        "margin": TRAIN_PARAMS["margin"],
        "semi_hard_mining_start_epoch": TRAIN_PARAMS["semi_hard_mining_start_epoch"],
    }

    loss_info = train(config)

    pd.DataFrame(loss_info).to_csv("data/outputs/loss-info/loss_info_8_11.csv", index=False)