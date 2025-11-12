import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
# from src.models.evaluate import load_model_inference
from src.models.utils import CoffeeDataset
from src.models.model_utils import load_model, build_embeddings, build_search_index
from src.config import (
    PREPROCESSED_DATA_PATH, 
    TRAINED_MODEL_PATH, 
    SBERT_MODEL_DIR, 
    FAISS_INDEX_PATH, 
    EMBEDDINGS_PATH,
    VOCABS_PATH,    
    MODEL_PARAMS
)
import time
import argparse

# def build_embeddings(model, coffee_df, vocabs, device, enc_only=False):
#     """Encodes all coffees and saves the raw embeddings to a numpy file."""
#     print("Building embeddings for all coffees...")
#     print(f"Using encoder only: {enc_only}")
#     full_dataset = CoffeeDataset(coffee_df, vocabs)
    
#     all_coffee_embeddings = []
#     with torch.no_grad():
#         for i in tqdm(range(len(full_dataset)), desc="Encoding all coffees"):
#             text, numericals, categoricals = full_dataset[i]
            
#             coffee_batch = {
#                 'text': [text],
#                 'numericals': numericals.unsqueeze(0).to(device),
#                 'categoricals': {
#                     'roast level': categoricals['roast level'].unsqueeze(0).to(device),
#                     'test_method': categoricals['test_method'].unsqueeze(0).to(device),
#                     'price_tier': categoricals['price_tier'].unsqueeze(0).to(device),
#                     'countries_extracted': categoricals['countries_extracted'].to(device),
#                     'countries_extracted_offsets': torch.tensor([0], dtype=torch.long).to(device),
#                     'process': categoricals['process'].to(device),
#                     'process_offsets': torch.tensor([0], dtype=torch.long).to(device),
#                     'varietals': categoricals['varietals'].to(device),
#                     'varietals_offsets': torch.tensor([0], dtype=torch.long).to(device),
#                 }
#             }
#             embedding = model.encode_coffees(coffee_batch, enc_only=enc_only)
#             all_coffee_embeddings.append(embedding.cpu().numpy())
            
#     all_coffee_embeddings = np.vstack(all_coffee_embeddings)
#     return all_coffee_embeddings


# def build_search_index(model, coffee_df, vocabs, device, enc_only=False):
#     """Encodes all coffees and builds a searchable FAISS index."""
#     all_coffee_embeddings = build_embeddings(model, coffee_df, vocabs, device, enc_only=enc_only)

#     print("Building search index for all coffees...")
    
#     index = faiss.IndexFlatIP(768)
#     faiss.normalize_L2(all_coffee_embeddings)
#     index.add(all_coffee_embeddings)
#     print(f"FAISS index built with {index.ntotal} vectors.")
#     return all_coffee_embeddings, index


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
    parser = argparse.ArgumentParser(description="Run various prediction functions")
    parser.add_argument("--goal", type=str, choices=["predict", "create_index_or_embeddings", "create_embeddings"], help="'predict' means you want to get predictions for a query." \
        "This requires the additional argument 'query' and optional arguments 'query_recs' for saving the query recommendations to a csv and 'num_recommendations'." \
        "'create_index_or_embedding' is self-explanatory. Set FAISS_INDEX_PATH and EMBEDDINGS_PATH in config.py to save the index and embeddings respectively." \
        "'create_embeddings' will just create the embeddings and save them to EMBEDDINGS_PATH.")
    # this should be done in config
    # parser.add_argument("--encoder_only", default=False, action='store_true', help="Whether to only create the text embeddings without the metadata. Only applies to 'create_index_or_embeddings' and 'create_embeddings' goals.")
    # parser.add_argument("--untrained", default=False, action='store_true', help="Whether to use an untrained model for creating the index/embeddings.")
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
    
    model, vocabs = load_model(
        VOCABS_PATH,
        MODEL_PARAMS["numerical_dim"],
        DEVICE,
        SBERT_MODEL_DIR,
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
    display_cols = ['url', 'company', 'coffee name', 'roast level', 'process', 'test_method', 'countries_extracted', "flavor_profile", "blind assessment", 'bottom line']
    print(recommendations[display_cols])
    if QUERY_REC_PATH:
        recommendations[display_cols].to_csv(QUERY_REC_PATH)

    if DEVICE.type == "cuda":
        del model
        torch.cuda.empty_cache()

        
    