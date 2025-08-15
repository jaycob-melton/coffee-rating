import torch
import numpy as np
import pandas as pd
import faiss
import json
import ast
import re
from tqdm import tqdm
from utils import CoffeeDataset, to_list
from model import DualEncoder

def load_model_inference(model_path: str, numerical_dim: int, device):
    """
    Loads a trained model from a .pth file for inference, i.e. evaluation
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    vocabs = checkpoint["vocabs"]

    model = DualEncoder(vocabs, numerical_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path} and set to evaluation mode.")
    return model, vocabs


def calculate_relevance_old(query: str, coffee_row: pd.Series) -> int:
    """
    Assigns a relevance score to a query based on a query
    - 2: Highly relevant (multiple key terms match)
    - 1: Partially relevant (at least one key term matches)
    - 0: Not relevant (no key terms match)
    """
    query = query.lower()
    
    # --- Step 1: Extract all potential keywords from the coffee ---
    all_coffee_keywords = set()
    try:
        # Safely handle list-like columns
        origins = [o.lower() for o in ast.literal_eval(coffee_row.get('countries_extracted', '[]'))]
        processes = [p.lower() for p in ast.literal_eval(coffee_row.get('process', '[]'))]
        varietals = [v.lower() for v in ast.literal_eval(coffee_row.get('varietals', '[]'))]
        flavor_profile = ast.literal_eval(coffee_row.get('flavor_profile', '{}'))
        flavors = [f.lower() for f in flavor_profile.keys()]
        
        all_coffee_keywords.update(origins)
        all_coffee_keywords.update(processes)
        all_coffee_keywords.update(varietals)
        all_coffee_keywords.update(flavors)
        
        roast = str(coffee_row.get('roast level', '')).lower()
        if roast: all_coffee_keywords.add(roast)
        
        price = str(coffee_row.get('price_tier', '')).lower()
        if price: all_coffee_keywords.add(price)

    except:
        pass # Handle potential parsing errors gracefully

    # --- Step 2: Extract keywords from the query ---
    # This is a simple heuristic; a more advanced method could use NLP
    query_keywords = set(re.findall(r'\b\w+\b', query))
    
    # --- Step 3: Determine number of matching keywords ---
    matched_keywords = query_keywords.intersection(all_coffee_keywords)
    
    # --- Step 4: Assign a relevance score ---
    # This score is now relative to the complexity of the query
    if not query_keywords:
        return 0
        
    match_percentage = len(matched_keywords) / len(query_keywords)
    
    if match_percentage >= 0.99: return 4 # Perfect match
    if match_percentage >= 0.75: return 3 # High relevance
    if match_percentage >= 0.50: return 2 # Medium relevance
    if match_percentage > 0: return 1   # Low relevance
    return 0

ATTRIBUTE_WEIGHTS = {
    'origin': 1,
    'process': 1,
    'varietal': 1,
    'flavor': 1,
    'notes': 1,
    'roast': 1,
    'test_method': 1
}

def calculate_relevance(query: str, coffee_row: pd.Series) -> int:

    universal_origins = set(json.loads(open("data/universal/known_origins.json").read()))
    universal_flavors = dict(json.loads(open("data/universal/flavor_keywords.json").read()))
    universal_varietals = list(json.loads(open("data/universal/coffee_varietals.json").read()))
    universal_processes = dict(json.loads(open("data/universal/process_keywords.json").read()))

    universal_set = {
        "origin": universal_origins,
        "flavor": {flavor.lower() for flavor in universal_flavors.keys()},
        "notes": {note.lower() for flavor in universal_flavors.values() for note in flavor},
        "varietal": universal_varietals,
        "process": {proc.lower() for proc in universal_processes.keys()}.union({process_name for process in universal_processes.values() for process_name in process}),
        "roast": {"light", "medium-light", "medium", "medium-dark", "dark"},
        "test_method": {"hot_black", "espresso_with_milk", "espresso_black", "cold_with_milk", "hot_with_milk", "cold_black"},
    }

    query = query.lower()

    # --- 1. Parse the Query to find what the user is asking for ---
    query_attributes = {}
    total_possible_score = 0
    
    for attr_type, keywords in universal_set.items():
        found_keywords = {kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', query)}
        if found_keywords:
            query_attributes[attr_type] = found_keywords
            total_possible_score += ATTRIBUTE_WEIGHTS[attr_type] * len(found_keywords)

    if total_possible_score == 0:
        return 0 # The query is too generic to be scored

    # --- 2. Score the coffee based on how well it matches the query attributes ---
    achieved_score = 0
    
    # Safely get the coffee's attributes
    try:
        coffee_origins = {o.lower() for o in ast.literal_eval(coffee_row.get('countries_extracted', '[]'))}
        coffee_processes = {p.lower() for p in ast.literal_eval(coffee_row.get('process', '[]'))}
        coffee_varietals = {v.lower() for v in ast.literal_eval(coffee_row.get('varietals', '[]'))}
        flavor_profile = ast.literal_eval(coffee_row.get('flavor_profile', '{}'))
        coffee_flavors = {f.lower() for f in flavor_profile.keys()}
        coffee_notes = {note.lower() for notes in flavor_profile.values() for note in notes}
        coffee_roast = {str(coffee_row.get('roast level', '')).lower()}
        coffee_test_method = {str(coffee_row.get('test_method', '')).lower()}
    except:
        return 0 # Return 0 if coffee data is malformed

    coffee_attributes = {
        'origin': coffee_origins,
        'process': coffee_processes,
        'varietal': coffee_varietals,
        'flavor': coffee_flavors,
        "notes": coffee_notes,
        'roast': coffee_roast,
        'test_method': coffee_test_method
    }
    
    for attr_type, query_values in query_attributes.items():
        matches = query_values.intersection(coffee_attributes.get(attr_type, set()))
        achieved_score += ATTRIBUTE_WEIGHTS[attr_type] * len(matches)
        
    # --- 3. Calculate the final relevance score (0-4) ---
    match_percentage = achieved_score / total_possible_score
    
    if match_percentage >= 0.99: return 4 # Perfect
    if match_percentage >= 0.75: return 3 # High
    if match_percentage >= 0.50: return 2 # Medium
    if match_percentage > 0: return 1   # 
    return 0




def calculate_ndcg(relevance_scores: list, k: int) -> float:
    """
    Calculates NDCG@k for a list of relevance scores
    """
    relevance_scores = np.array(relevance_scores)[:k]

    dcg = np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))

    ideal_scores = np.sort(relevance_scores)[::-1]
    idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(model, test_df, vocabs, training_data_path, device, precomputed_index=None):
    """
    Evaluates a model on the test set using Recall@K and NDCG@K
    """
    print("\nStarting Evaluation...")

    test_dataset = CoffeeDataset(test_df, vocabs)

    print("Pre-computing coffee embeddings for the test set...")
    coffee_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Encoding Coffees"):
            text, numericals, categoricals = test_dataset[i]

            # batch of size 1
            coffee_batch = {
                "text": [text],
                "numericals": numericals.unsqueeze(0).to(device),
                "categoricals": {
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
            coffee_embeddings.append(embedding.cpu().numpy())

    coffee_embeddings = np.vstack(coffee_embeddings)

    # using FAISS index for speedy searching
    if precomputed_index is None:
        print("Building FAISS Index...")
        index = faiss.IndexFlatIP(768)
        faiss.normalize_L2(coffee_embeddings)
        index.add(coffee_embeddings)
        print(f"FAISS index built with {index.ntotal} vectors.")
    else:
        print("Using precomputed FAISS index...")
        index = precomputed_index

    # generate the test queries and evaluate
    print("Generating test queries and evaluating recall...")
    test_queries = []
    test_ground_truth = []
    test_id2idx = {cid: i for i, cid in test_df["id"].items()}

    with open(training_data_path, "r") as f:
        for line in f:
            data=json.loads(line)
            if data["coffee_id"] in test_id2idx:
                correct_idx = test_id2idx[data["coffee_id"]]
                for query in data["queries"]:
                    # if len(query.split()) > 4:
                    test_queries.append(query)
                    test_ground_truth.append(correct_idx)

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total_queries = len(test_queries)

    ndcg_5_scores = []
    ndcg_10_scores = []
    with torch.no_grad():
        for i in tqdm(range(total_queries), desc="Evaluating Queries"):
            query = test_queries[i]
            correct_idx = test_ground_truth[i]

            query_embedding = model.encode_queries([query]).cpu().numpy()
            faiss.normalize_L2(query_embedding)

            _, top_k_indices = index.search(query_embedding, k=10)

            # calculate recall
            top_k_indices = top_k_indices[0]

            if correct_idx in top_k_indices[:1]:
                hits_at_1 += 1
            if correct_idx in top_k_indices[:5]:
                hits_at_5 += 1
            if correct_idx in top_k_indices[:10]:
                hits_at_10 += 1

            # Calculate ndcg
            recommendation_relevance = [calculate_relevance(query, test_df.iloc[i]) for i in top_k_indices]

            ndcg_5_score = calculate_ndcg(recommendation_relevance, k=5)
            ndcg_5_scores.append(ndcg_5_score)

            ncdg_10_score = calculate_ndcg(recommendation_relevance, k=10)
            ndcg_10_scores.append(ncdg_10_score)

    print(f"Total Test Queries: {total_queries}")
    print(f"Recall@1:  {hits_at_1 / total_queries:.4f}")
    print(f"Recall@5:  {hits_at_5 / total_queries:.4f}")
    print(f"Recall@10: {hits_at_10 / total_queries:.4f}")
    print(f"NDCG@5:  {np.mean(ndcg_5_scores):.4f}")
    print(f"NDCG@10:  {np.mean(ndcg_10_scores):.4f}")
    return np.mean(ndcg_10_scores), hits_at_1 / total_queries, hits_at_5 / total_queries, hits_at_10 / total_queries


def build_search_index(model, coffee_df, vocabs, device):
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
    PREPROCESSED_PATH = "data/processed/test_data_8_11.csv"
    # TRAINING_DATA_PATH = "data/processed/llm-queries/synthetic_queries_np_4_1_nano.jsonl"
    TRAINING_DATA_PATH = "data/processed/training_data.jsonl"
    MODEL_PATH = "data/outputs/model-weights/8-11/coffee_model_epoch_11_semi_hard_3.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Coffee Data...")
    test_df = pd.read_csv(PREPROCESSED_PATH)
    # df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    
    model, vocabs = load_model_inference(MODEL_PATH, numerical_dim=10, device=DEVICE)
    evaluate(model, test_df, vocabs, TRAINING_DATA_PATH, DEVICE)