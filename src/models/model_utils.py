import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from src.models.model import DualEncoder
from src.models.utils import CoffeeDataset
import config

def load_model(vocabs_path: str, numerical_dim: int, device, model_arch_path="sentence-transformers/all-mpnet-base-v2", model_weights_path=None, eval=False):
    """
    Loads the DualEncoder model for training or inference.
    If model_weights_path is provided, it loads trained weights, otherwise it returns the untrained base model
    """
    with open(vocabs_path, "r") as f:
        vocabs = json.load(f)

    model = DualEncoder(vocabs, numerical_dim, model_arch_path)
    
    if model_weights_path:
        checkpoint = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_weights_path}.")
    else:
        print(f"No model weights provided. Loaded untrained model with architecture {model_arch_path}.")
    
    model.to(device)
    if eval:
        model.eval()
        print("Model set to evaluation mode.")
    else:
        model.train()
        print("Model set to training mode.")
        
    return model, vocabs


def build_embeddings(model, coffee_df, vocabs, device, enc_only=False):
    """Encodes all coffees and saves the raw embeddings to a numpy file."""
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
            embedding = model.encode_coffees(coffee_batch, enc_only=enc_only)
            all_coffee_embeddings.append(embedding.cpu().numpy())
            
    all_coffee_embeddings = np.vstack(all_coffee_embeddings)
    return all_coffee_embeddings


def build_search_index(embeddings):
    """Encodes all coffees and builds a searchable FAISS index."""
    print("Building search index for all coffees...")
    
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index