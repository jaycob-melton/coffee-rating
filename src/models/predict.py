import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from src.models.evaluate import load_model_inference
from src.models.utils import CoffeeDataset
from src.config import PREPROCESSED_DATA_PATH, TRAINED_MODEL_PATH, SBERT_MODEL_DIR, FAISS_INDEX_PATH, EMBEDDINGS_PATH
import time
import argparse

def build_embeddings(model, coffee_df, vocabs, device):
    """Encodes all coffees and saves the raw embeddings to a numpy file."""
    print("Building embeddings for all coffees...")
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


def build_search_index(model, coffee_df, vocabs, device):
    """Encodes all coffees and builds a searchable FAISS index."""
    all_coffee_embeddings = build_embeddings(model, coffee_df, vocabs, device)

    print("Building search index for all coffees...")
    
    index = faiss.IndexFlatIP(768)
    faiss.normalize_L2(all_coffee_embeddings)
    index.add(all_coffee_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return all_coffee_embeddings, index


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
    parser.add_argument("--query", type=str, default=None, help="Query to have recommendations provided for. Type: .txt")
    parser.add_argument("--query_recs", type=str, default=None, help="Save path for recommendations for the given user query")
    parser.add_argument("--num_recommendations", type=int, default=10, help="The number of recommendations you want. Type: positive integer")
    
    args = parser.parse_args()
    
    QUERY_PATH = args.query
    QUERY_REC_PATH = args.query_recs
    TOP_K = args.num_recommendations
    
    if args.goal == "predict":
        assert(QUERY_PATH)
    elif args.goal == "create_index_or_embeddings":
        assert(FAISS_INDEX_PATH or EMBEDDINGS_PATH)
    else:
        assert(EMBEDDINGS_PATH)

    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    
    model, vocabs = load_model_inference(
        TRAINED_MODEL_PATH, 
        numerical_dim=10, 
        device=DEVICE,           
        model_location=SBERT_MODEL_DIR
    )
    
    if args.goal == "predict":
        with open(QUERY_PATH, "r") as f:
            query = f.read().strip()
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

    elif args.goal == "create_index_or_embeddings":
        # build embeddings/search_index
        all_embeddings, search_index = build_search_index(model, df, vocabs, DEVICE)
        # save requested files
        if FAISS_INDEX_PATH:
            faiss.write_index(search_index, str(FAISS_INDEX_PATH))
            print(f"Saved FAISS index to: {FAISS_INDEX_PATH}")
        if EMBEDDINGS_PATH:
            np.save(EMBEDDINGS_PATH, all_embeddings) 
            print(f"Saved raw embeddings to: {EMBEDDINGS_PATH}")
    else: # generate just embeddings
        all_embeddings = build_embeddings(model, df, vocabs, DEVICE)
        if EMBEDDINGS_PATH:
            np.save(EMBEDDINGS_PATH, all_embeddings) 
            print(f"Saved raw embeddings to: {EMBEDDINGS_PATH}")
        
    