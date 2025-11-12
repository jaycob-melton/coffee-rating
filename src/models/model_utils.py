import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from src.models.model import DualEncoder
from src.models.utils import CoffeeDataset

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