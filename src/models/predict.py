import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from .evaluate import load_model_inference
from .utils import CoffeeDataset
import time

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
    return index


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
    PREPROCESSED_PATH = rf"data\processed\preprocessed_data.csv"
    MODEL_PATH = rf"data\outputs\model-weights\coffee_model_epoch_11_semi_hard_epoch_3_3.pth"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_PATH)
    df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    
    model, vocabs = load_model_inference(MODEL_PATH, numerical_dim=10, device=DEVICE)
    
    # search_index = build_search_index(model, df, vocabs, DEVICE)
    search_index = faiss.read_index("faiss_index.bin")
    
    example_query = "Medium roast coffee from Ethipia with hints of berry."
    
    start_time = time.time()
    recommendations = get_recommendations(example_query, model, search_index, df, top_k=10)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Recommendations took {duration:.2f} seconds to run serve.")
    display_cols = ['url', 'company', 'coffee name', 'roast level', 'process', 'test_method', 'countries_extracted', "flavor_profile", "blind assessment", 'bottom line']
    print(recommendations[display_cols])
    recommendations[display_cols].to_csv("test_recommendations.csv")
