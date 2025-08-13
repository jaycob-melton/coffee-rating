import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import CoffeeDataset, TripleTrainingDataset, build_all_vocabs
from model import DualEncoder

torch.manual_seed(189)  # for reproducibility

def collate(batch, coffee_dataset: CoffeeDataset):
    """
    Collate function to prepare a batch for training
    """
    queries = [item["query"] for item in batch]
    positive_indices = [item["positive_idx"] for item in batch]

    positive_coffee_batch = {
        'text': [],
        'numericals': [],
        'categoricals': {
            'roast level': [], 'test_method': [], 'price_tier': [],
            'countries_extracted': [], 'process': [], 'varietals': [],
            'countries_extracted_offsets': [0], 'process_offsets': [0], 'varietals_offsets': [0]
        }
    }
    # store all data about postive coffees
    for idx in positive_indices:
        text, numericals, categoricals = coffee_dataset[idx]
        positive_coffee_batch["text"].append(text)
        positive_coffee_batch["numericals"].append(numericals)
        for col, val in categoricals.items():
            if val.dim() == 0:
                positive_coffee_batch["categoricals"][col].append(val)
            else:
                positive_coffee_batch["categoricals"][col].append(val)
                offset_key = f"{col}_offsets"
                # need to store offset since these cats take variable length
                offsets = positive_coffee_batch["categoricals"][offset_key]
                offsets.append(offsets[-1] + len(val))
    
    # convert everything to tensors for models
    positive_coffee_batch["numericals"] = torch.stack(positive_coffee_batch["numericals"])
    for col, val in positive_coffee_batch["categoricals"].items():
        if "offsets" not in col and len(val) > 0:
            if val[0].dim() == 0:
                positive_coffee_batch["categoricals"][col] = torch.stack(val)
            else:
                positive_coffee_batch["categoricals"][col] = torch.cat(val)
        elif "offsets" in col:
            positive_coffee_batch["categoricals"][col] = torch.tensor(val[:-1], dtype=torch.long)

    return queries, positive_indices, positive_coffee_batch


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
    train_dataset = TripleTrainingDataset(config["training_data_path"], df)


    train_loader = DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate(batch, full_coffee_dataset)
    )

    # model, loss, optimizer
    if config["model_path"]:
        model = load_model_train(config["model_path"], numerical_dim=len(full_coffee_dataset.numerical_cols), device=device)
    else:
        model = DualEncoder(vocabs, numerical_dim=len(full_coffee_dataset.numerical_cols)).to(device)

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

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "vocabs": vocabs,
        }
        torch.save(checkpoint, f"data/outputs/model-weights/8-11/coffee_model_epoch_{epoch+1}.pth")
    
    return loss_info


if __name__ == "__main__":
    PREPROCESSED_PATH = "data/processed/preprocessed_data.csv"
    TRAINING_DATA_PATH = "data/processed/training_data.jsonl"
    # MODEL_PATH = "coffee_model_simpleadam_epoch_10.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_PATH)
    df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    
    print("Creating 80/20 train-test split...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=189)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_df.to_csv("data/processed/train_data_8_11.csv", index=False)
    test_df.to_csv("data/processed/test_data_8_11.csv", index=False)

    vocabs = build_all_vocabs(train_df)


    config = {
        "preprocessed_data": train_df,
        "vocabs": vocabs,
        "training_data_path": TRAINING_DATA_PATH,
        "batch_size": 32,
        "transformer_lr": 1e-5,
        "head_lr": 1e-5,
        "epochs": 25,
        "margin": 0.2,
        "semi_hard_mining_start_epoch": 0,
        "model_path": "data/outputs/model-weights/8-11/coffee_model_epoch_25.pth"
    }

    loss_info = train(config)

    pd.DataFrame(loss_info).to_csv("data/outputs/loss-info/loss_info_8_11.csv", index=False)