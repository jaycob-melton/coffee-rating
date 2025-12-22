import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
# from src.models.evaluate import load_model_inference
from src.models.utils import CoffeeDataset
from src.models.model_utils import load_vocabs, load_model, load_cross_encoder, calculate_relevance
from src.config import (
    CE_ARCHITECTURE,
    CE_WEIGHTS,
    PREPROCESSED_DATA_PATH, 
    TRAINED_MODEL_PATH, 
    SBERT_MODEL_DIR, 
    FAISS_INDEX_PATH, 
    VOCABS_PATH,    
    MODEL_PARAMS,
)
import time
import argparse


# def get_recommendations(query, dual_encoder, cross_encoder, index, coffee_df, top_k=5):
#     """Gets top K recommendations for a single query."""
#     with torch.no_grad():
#         query_embedding = model.encode_queries([query]).cpu().numpy()
#         faiss.normalize_L2(query_embedding)
        
#         distances, top_k_indices = index.search(query_embedding, k=top_k)
        
#         # Get the indices from the search result
#         result_indices = top_k_indices[0]
        
#         # Return the corresponding rows from the original DataFrame
#         return coffee_df.iloc[result_indices]
      

def get_recommendations(query, dual_encoder, faiss_index, coffee_df, cross_encoder=None, initial_k=50, final_k=10):
    device = next(dual_encoder.parameters()).device
    # Stage 1: Dual Encoder Retrieval
    with torch.no_grad():
        # Encode query with dual encoder
        query_emb = dual_encoder.encode_queries([query]).cpu().numpy()
        faiss.normalize_L2(query_emb)
        
        # Initial retrieval with dual encoder
        _, top_k_indices = faiss_index.search(
            query_emb, 
            k=initial_k if cross_encoder else final_k
        )
        candidate_indices = top_k_indices[0]
    
    candidates_df = coffee_df.iloc[candidate_indices].copy()
    
    # Stage 2: Cross Encoder Re-ranking
    if cross_encoder:
        queries = [query] * len(candidates_df)
        coffee_texts = candidates_df["combined_text"].tolist()

        with torch.no_grad():
            relevance_scores = cross_encoder.predict(queries, coffee_texts)
            relevance_scores = relevance_scores.squeeze().cpu().numpy()
        
        sorted_scores_indices = np.argsort(-relevance_scores)
        final_k_indices = sorted_scores_indices[:final_k]
        
        candidates_df = candidates_df.iloc[final_k_indices]

        candidates_df["relevance_score"] = relevance_scores[final_k_indices]

    return candidates_df

      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get predictions for a given query.")
    parser.add_argument("--query", type=str, default=None, help="Query to have recommendations provided for. Type: .txt")
    parser.add_argument("--query_recs", type=str, default=None, help="Save path for recommendations for the given user query")
    parser.add_argument("--num_recommendations", type=int, default=10, help="The number of recommendations you want. Type: positive integer")
    parser.add_argument("--cross_encoder", action="store_true", help="Whether to use cross-encoder for re-ranking")
    
    args = parser.parse_args()
    
    QUERY = args.query
    QUERY_REC_PATH = args.query_recs
    TOP_K = args.num_recommendations
    USE_CROSS_ENCODER = args.cross_encoder
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    
    vocabs = load_vocabs(VOCABS_PATH)

    dual_encoder = load_model(
        vocabs=vocabs,
        device=DEVICE,
        eval=True
    )
    
    cross_encoder = None
    if USE_CROSS_ENCODER:
        print("Loading Cross Encoder...")
        cross_encoder = load_cross_encoder(
            model_arch_path=CE_ARCHITECTURE,
            weights_path=CE_WEIGHTS,
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
    recommendations = get_recommendations(
        query=query,
        dual_encoder=dual_encoder, 
        cross_encoder=cross_encoder,
        faiss_index=index,
        coffee_df=df,
        initial_k=32,
        final_k=TOP_K   
    ) #get_recommendations(query, model, index, df, top_k=TOP_K)
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
        del dual_encoder
        del cross_encoder
        torch.cuda.empty_cache()

        
    