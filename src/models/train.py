import json
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.utils import CoffeeDataset, TripleTrainingDataset, collate
# from src.models.model import DualEncoder
from src.models.model_utils import load_vocabs, load_model
from src.config import (
    TRAIN_DATA_PATH,
    PREPROCESSED_DATA_PATH,
    VOCABS_PATH,
    QUERIES_PATH,
    SBERT_MODEL_DIR,
    TRAINED_MODEL_PATH,
    MODEL_SAVE_PATH,
    TRAIN_PARAMS,
    MODEL_PARAMS,
    today
)

#torch.manual_seed(189)  # for reproducibility

def train(device):
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    vocabs = load_vocabs(VOCABS_PATH)

    # used by collate to look up coffee data by index
    full_coffee_dataset = CoffeeDataset(df, vocabs)

    # provides thje (query, positive_idx) pairs for training
    train_dataset = TripleTrainingDataset(str(QUERIES_PATH), df)


    train_loader = DataLoader(
        train_dataset,
        batch_size = TRAIN_PARAMS["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate(batch, full_coffee_dataset)
    )

    model = load_model(
        vocabs=vocabs,
        numerical_dim=len(full_coffee_dataset.numerical_cols),
        embedding_dim=MODEL_PARAMS["embedding_dim"],
        encoder_only=MODEL_PARAMS["encoder_only"],
        device=device,
        model_arch_path=SBERT_MODEL_DIR,
        model_weights_path=TRAINED_MODEL_PATH,
    )

    loss_fn = nn.TripletMarginLoss(margin=TRAIN_PARAMS["margin"])

    transformer_params = model.transformer.parameters()
    if model.encoder_only:
        print("Training with encoder only. No head parameters will be updated.")
        head_params = []
        head_lr = 0.
    else:
        print("Training with full model. Head parameters will be updated.")
        head_params = list(model.metadata_encoder.parameters()) + list(model.fusion_layer.parameters())
        head_lr = TRAIN_PARAMS["head_lr"]

    optimizer = AdamW([
        {"params": transformer_params, "lr": TRAIN_PARAMS["transformer_lr"]},
        {"params": head_params, "lr": head_lr}
    ])

    print("Starting training...")
    loss_info = {
        "loss": [],
        "semi_hard_negatives": [],
    }

    for epoch in range(TRAIN_PARAMS["num_epochs"]):
        model.train()
        total_loss = 0
        num_semi_hard = 0
        if TRAIN_PARAMS["semi_hard_mining_start_epoch"]:
            use_semi_hard_mining = epoch >= TRAIN_PARAMS["semi_hard_mining_start_epoch"]
        else:
            last_4_losses = loss_info["loss"][-4:] if len(loss_info["loss"]) >= 4 else []
            last_3_relative_diff = [
                abs((last_4_losses[i] - last_4_losses[i-1]) / last_4_losses[i-1]) if last_4_losses[i-1] != 0 else 0
                for i in range(1, len(last_4_losses))
            ] if len(last_4_losses) >= 4 else []
            use_semi_hard_mining = len(last_4_losses) >= 4 and all(loss < 0.02 for loss in last_3_relative_diff) 

        if use_semi_hard_mining:
            print(f"Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} - Using Semi-Hard Negative Mining")

        else:
            print(f"Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} - Using In-Batch Negative Mining")
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
            if not model.encoder_only:
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
                    semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + TRAIN_PARAMS["margin"])

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

        torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}{epoch+1}.pth")

    if DEVICE.type == "cuda":
        del model
        torch.cuda.empty_cache()
    
    return loss_info


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config = {
    #     # Data paths
    #     "preprocessed_data": TRAIN_DATA_PATH,
    #     "queries_path": QUERIES_PATH,
    #     "vocabs": VOCABS_PATH,

    #     # Model paths
    #     "enc_model_path": SBERT_MODEL_DIR,
    #     "model_path": None,#TRAINED_MODEL_PATH,
    #     "save_path": MODEL_SAVE_PATH,

    #     # Training hyperparameters
    #     "batch_size": TRAIN_PARAMS["batch_size"],
    #     "transformer_lr": TRAIN_PARAMS["transformer_lr"],
    #     "head_lr": TRAIN_PARAMS["head_lr"],
    #     "epochs": TRAIN_PARAMS["num_epochs"],
    #     "margin": TRAIN_PARAMS["margin"],
    #     "semi_hard_mining_start_epoch": TRAIN_PARAMS["semi_hard_mining_start_epoch"],

    #     "device": DEVICE
    # }

    loss_info = train(DEVICE)

    pd.DataFrame(loss_info).to_csv(f"data/outputs/loss-info/loss_info_{today.month}_{today.day}.csv", index=False)

    