import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from .evaluate import load_model_inference
from .utils import CoffeeDataset
import time
import argparse

def build_search_index(model, coffee_df, vocabs, device, output_path="faiss_index.bin"):
    """Encodes all coffees and builds a searchable FAISS index."""
    print("Building search index for all coffees...")
    # Use the full dataset for the index
    full_dataset = CoffeeDataset(coffee_df, vocabs)
    
    all_coffee_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(full_dataset)), desc="Encoding all coffees"):
            text, numericals, categoricals = full_dataset[i]
            
            # Manually create a batch of size 1
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
    
    index = faiss.IndexFlatIP(768)
    faiss.normalize_L2(all_coffee_embeddings)
    index.add(all_coffee_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    faiss.write_index(index, output_path)
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
    parser.add_argument("--goal", type=str, choices=["predict", "create_index_or_embeddings"], help="'predict' means you want to get predictions for a query." \
        "This requires the additional argument 'query' and optional argument 'query_recs' for saving the query recommendations to a csv." \
        "' create_index_or_embedding' is self-explanatory. Provide" \
        " additional arguments 'faiss_output_file', 'embedding_output_file', or both")
    parser.add_argument("--data_input_file", type=str, help="Input path for the main coffee data. Type: .csv")
    parser.add_argument("--model_pth_file", type=str, help="Input path for the model weights. Type: .pth")
    parser.add_argument("--faiss_output_file", type=str, default=None, help="Output path for the faiss index. Type: .bin")
    parser.add_argument("--embedding_output_file", type=str, default=None, help="Output path for the raw numpy embeddings. Type: .npy")
    parser.add_argument("--query", type=str, default=None, help="Query to have recommendations provided for. Type: .txt")
    parser.add_argument("--query_recs", type=str, default=None, help="Save path for recommendations for the given user query")
    
    args = parser.parse_args()
    
    PREPROCESSED_PATH = args.data_input_file
    MODEL_PATH = args.model_pth_file
    INDEX_PATH = args.faiss_output_file
    EMBEDDINGS_PATH = args.embedding_output_file
    QUERY_PATH = args.query
    QUERY_REC_PATH = args.query_recs
    
    assert(PREPROCESSED_PATH and MODEL_PATH)
    if args.goal == "predict":
        assert(INDEX_PATH and QUERY_PATH)
    else:
        assert(INDEX_PATH or EMBEDDINGS_PATH)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_PATH)
    df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    
    model, vocabs = load_model_inference(MODEL_PATH, numerical_dim=10, device=DEVICE, model_location="/home/jaycob-laptop/Projects/coffee-rating/sbert_model")
    
    if args.goal == "predict":
        query = open(QUERY_PATH, "r")
        # load in the given faiss index
        index = faiss.read_index(INDEX_PATH)
        
        # acquire the top 10 recommendations and time it
        start_time = time.time()
        recommendations = get_recommendations(query, model, index, df, top_k=10)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Recommendations took {duration:.2f} seconds to run serve.")
        
        # dump recommendations to csv if requested
        display_cols = ['url', 'company', 'coffee name', 'roast level', 'process', 'test_method', 'countries_extracted', "flavor_profile", "blind assessment", 'bottom line']
        print(recommendations[display_cols])
        if QUERY_REC_PATH:
            recommendations[display_cols].to_csv(QUERY_REC_PATH)

    else:
        # build embeddings/search_index
        all_embeddings, search_index = build_search_index(model, df, vocabs, DEVICE)
        # save requested files
        if INDEX_PATH:
            faiss.write_index(search_index, INDEX_PATH)
            print(f"Saved FAISS index to: {INDEX_PATH}")
        if EMBEDDINGS_PATH:
            np.save(EMBEDDINGS_PATH, all_embeddings) 
            print(f"Saved raw embeddings to: {EMBEDDINGS_PATH}")
    