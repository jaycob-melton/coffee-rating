import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
# from src.models.evaluate import load_model_inference
from src.models.utils import CoffeeDataset
from src.models.model_utils import load_vocabs, load_model
from src.config import (
    PREPROCESSED_DATA_PATH, 
    TRAINED_MODEL_PATH, 
    SBERT_MODEL_DIR, 
    FAISS_INDEX_PATH, 
    VOCABS_PATH,    
    MODEL_PARAMS
)
import time
import argparse


def get_recommendations(query, model, index, coffee_df, top_k=5):
    """Gets top K recommendations for a single query."""
    with torch.no_grad():
        query_embedding = model.encode_queries([query]).cpu().numpy()
        faiss.normalize_L2(query_embedding)
        
        distances, top_k_indices = index.search(query_embedding, k=top_k)
        
        # Get the indices from the search result
        result_indices = top_k_indices[0]
        
        # Return the corresponding rows from the original DataFrame
        return coffee_df.iloc[result_indices]
      
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get predictions for a given query.")
    parser.add_argument("--query", type=str, default=None, help="Query to have recommendations provided for. Type: .txt")
    parser.add_argument("--query_recs", type=str, default=None, help="Save path for recommendations for the given user query")
    parser.add_argument("--num_recommendations", type=int, default=10, help="The number of recommendations you want. Type: positive integer")
    
    args = parser.parse_args()
    
    QUERY = args.query
    QUERY_REC_PATH = args.query_recs
    TOP_K = args.num_recommendations
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    
    vocabs = load_vocabs(VOCABS_PATH)

    model = load_model(
        vocabs=vocabs,
        device=DEVICE,
        eval=True
    )

    if ".txt" in QUERY:
        with open(QUERY, "r") as f:
            query = f.read().strip()
    else:
        query = QUERY.strip()
    # load in the given faiss index
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    # acquire the top 10 recommendations and time it
    start_time = time.time()
    recommendations = get_recommendations(query, model, index, df, top_k=TOP_K)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Recommendations took {duration:.2f} seconds to run serve.")
    
    # dump recommendations to csv if requested
    display_cols = ['url', 'coffee name', 'roast level', 'process', 'test_method', 'countries_extracted', "flavor_profile", "blind assessment", 'bottom line']
    pd.set_option('display.max_columns', None)
    print(recommendations[display_cols])
    if QUERY_REC_PATH:
        recommendations[display_cols].to_csv(QUERY_REC_PATH)

    if DEVICE.type == "cuda":
        del model
        torch.cuda.empty_cache()

        
    