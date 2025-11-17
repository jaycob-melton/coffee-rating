import json
from turtle import pd
import torch
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from src.models.model import DualEncoder
from src.models.utils import CoffeeDataset
from src.config import (
    VOCABS_PATH,
    PREPROCESSED_DATA_PATH,
    TRAINED_MODEL_PATH,
    SBERT_MODEL_DIR,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    MODEL_PARAMS
)
import argparse

def load_vocabs(vocabs_path):
    with open(vocabs_path, "r") as f:
        vocabs = json.load(f)
    return vocabs


def load_model(vocabs, model_params=MODEL_PARAMS, text_encoder_dir=SBERT_MODEL_DIR, weights_path=TRAINED_MODEL_PATH, device=torch.device("cpu"), eval=False):#numerical_dim: int, embedding_dim: int, encoder_only: bool = False, device=torch.device("cpu"), model_arch_path="sentence-transformers/#all-mpnet-base-v2", model_weights_path=None, eval=False):
    """
    Loads the DualEncoder model for training or inference.
    If model_weights_path is provided, it loads trained weights, otherwise it returns the untrained base model
    """

    model = DualEncoder(
        vocabs=vocabs, 
        numerical_dim=model_params["numerical_dim"],
        embedding_dim=model_params["embedding_dim"], 
        fc_dropout=model_params["fc_dropout"],
        all_dropout=model_params["all_dropout"],
        encoder_only=model_params["encoder_only"], 
        text_model_name=text_encoder_dir,
    )
    
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {weights_path}.")
    else:
        print(f"No model weights provided. Loaded untrained model with architecture {text_encoder_dir}.")
    
    model.to(device)
    if eval:
        model.eval()
        print("Model set to evaluation mode.")
    else:
        model.train()
        print("Model set to training mode.")
        
    return model


def build_embeddings(model, coffee_df, vocabs, device, enc_only=False):
    """Encodes all coffees and saves the raw embeddings to a numpy file."""
    if enc_only:
        print("Setting inference mode to encoder only...")
        model.encoder_only = True

    print("Building embeddings for all coffees...")
    print(f"Using encoder only: {enc_only}")
    full_dataset = CoffeeDataset(coffee_df, vocabs)
    
    all_coffee_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(full_dataset)), desc="Encoding all coffees"):
            text, numericals, categoricals = full_dataset[i]
            
            coffee_batch = {
                'text': [text],
                'numericals': numericals.unsqueeze(0).to(device),
                'categoricals': {
                    'roast level': categoricals['roast level'].unsqueeze(0).to(device),
                    'test_method': categoricals['test_method'].unsqueeze(0).to(device),
                    'price_tier': categoricals['price_tier'].unsqueeze(0).to(device),
                    'countries_extracted': categoricals['countries_extracted'].to(device),
                    'countries_extracted_offsets': torch.tensor([0], dtype=torch.long).to(device),
                    'process': categoricals['process'].to(device),
                    'process_offsets': torch.tensor([0], dtype=torch.long).to(device),
                    'varietals': categoricals['varietals'].to(device),
                    'varietals_offsets': torch.tensor([0], dtype=torch.long).to(device),
                }
            }
            embedding = model.encode_coffees(coffee_batch)
            all_coffee_embeddings.append(embedding.cpu().numpy())
            
    all_coffee_embeddings = np.vstack(all_coffee_embeddings)
    return all_coffee_embeddings


def build_search_index(embeddings):
    """Encodes all coffees and builds a searchable FAISS index."""
    print("Building search index for all coffees...")
    
    index = faiss.IndexFlatIP(MODEL_PARAMS["embedding_dim"])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flag for building embeddings or search index")
    parser.add_argument("--build_embeddings", action="store_true", help="Flag to build embeddings for all coffees. Saves to a numpy file in config.EMBEDDINGS_PATH")
    parser.add_argument("--build_index", action="store_true", help="Flag to build a FAISS search index. Saves to a file in config.FAISS_INDEX_PATH. If --build_embeddings is not set, it will load embeddings from config.EMBEDDINGS_PATH")
    
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    vocabs = load_vocabs(VOCABS_PATH)

    model = load_model(
        vocabs=vocabs,
        numerical_dim=MODEL_PARAMS["numerical_dim"],  # Set to 0 since we're only encoding coffees
        embedding_dim=MODEL_PARAMS["embedding_dim"],
        encoder_only=MODEL_PARAMS["encoder_only"],
        device=DEVICE,
        model_arch_path=SBERT_MODEL_DIR,
        model_weights_path=TRAINED_MODEL_PATH,
        eval=True
    )

    print("Loading coffee data...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    embeddings = None
    if args.build_embeddings:
        embeddings = build_embeddings(model, df, vocabs, DEVICE, enc_only=MODEL_PARAMS["encoder_only"])
        try:
            np.save(EMBEDDINGS_PATH, embeddings)
            print(f"Embeddings saved to {EMBEDDINGS_PATH}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    if args.build_index:
        if embeddings is None:
            print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
            embeddings = np.load(EMBEDDINGS_PATH)
        
        index = build_search_index(embeddings)
        try:
            faiss.write_index(index, str(FAISS_INDEX_PATH))
            print(f"FAISS index saved to {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    if DEVICE.type == "cuda":
        del model
        torch.cuda.empty_cache()



