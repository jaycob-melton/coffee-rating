import torch
import numpy as np
import pandas as pd
import faiss
import json
import ast
import re
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from src.models.utils import CoffeeDataset
from src.models.model_utils import load_vocabs, load_model, build_embeddings, build_search_index
from src.config import (
    QUERIES_PATH, 
    SBERT_MODEL_DIR, 
    PREPROCESSED_DATA_PATH, 
    TRAINED_MODEL_PATH, 
    EMBEDDINGS_PATH, 
    FAISS_INDEX_PATH,
    VOCABS_PATH,
    MODEL_PARAMS,
    RELEVANCE_CACHE
)

universal_origins = set(json.loads(open("data/universal/known_origins.json").read()))
universal_flavors = dict(json.loads(open("data/universal/flavor_keywords.json").read()))
universal_varietals = list(json.loads(open("data/universal/coffee_varietals.json").read()))
universal_processes = dict(json.loads(open("data/universal/process_keywords.json").read()))

UNIVERSAL_SET = {
    "origin": universal_origins,
    "flavor": {flavor.lower() for flavor in universal_flavors.keys()},
    "notes": {note.lower() for flavor in universal_flavors.values() for note in flavor},
    "varietal": universal_varietals,
    "process": {proc.lower() for proc in universal_processes.keys()}.union({process_name for process in universal_processes.values() for process_name in process}),
    "roast": {"light", "medium-light", "medium", "medium-dark", "dark"},
    "test_method": {"hot_black", "espresso_with_milk", "espresso_black", "cold_with_milk", "hot_with_milk", "cold_black"},
}

ATTRIBUTE_WEIGHTS = {
    'origin': 3,
    'process': 3,
    'varietal': 2,
    'flavor': 2,
    'notes': 1,
    'roast': 2,
    'test_method': 3
}

def calculate_relevance(query: str, coffee_row: pd.Series) -> int:

    # universal_origins = set(json.loads(open("data/universal/known_origins.json").read()))
    # universal_flavors = dict(json.loads(open("data/universal/flavor_keywords.json").read()))
    # universal_varietals = list(json.loads(open("data/universal/coffee_varietals.json").read()))
    # universal_processes = dict(json.loads(open("data/universal/process_keywords.json").read()))

    # universal_set = {
    #     "origin": universal_origins,
    #     "flavor": {flavor.lower() for flavor in universal_flavors.keys()},
    #     "notes": {note.lower() for flavor in universal_flavors.values() for note in flavor},
    #     "varietal": universal_varietals,
    #     "process": {proc.lower() for proc in universal_processes.keys()}.union({process_name for process in universal_processes.values() for process_name in process}),
    #     "roast": {"light", "medium-light", "medium", "medium-dark", "dark"},
    #     "test_method": {"hot_black", "espresso_with_milk", "espresso_black", "cold_with_milk", "hot_with_milk", "cold_black"},
    # }

    query = query.lower()

    
    query_attributes = {}
    total_possible_score = 0
    
    for attr_type, keywords in UNIVERSAL_SET.items():
        found_keywords = {kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', query)}
        if found_keywords:
            query_attributes[attr_type] = found_keywords
            total_possible_score += ATTRIBUTE_WEIGHTS[attr_type] * len(found_keywords)

    if total_possible_score == 0:
        return 0 # The query is too generic to be scored

    
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
        
    
    match_percentage = achieved_score / total_possible_score
    
    if match_percentage >= 0.99: return 4 # Perfect
    if match_percentage >= 0.75: return 3 # High
    if match_percentage >= 0.50: return 2 # Medium
    if match_percentage > 0: return 1   # 
    return 0

def process_single_query(query, df, threshold=1):
    """
    Worker function to calculate total relevance for a single query 
    """
    total_relevant = 0
    for _, row in df.iterrows():
        if calculate_relevance(query, row) >= threshold:
            total_relevant += 1
    return (query, total_relevant)


def calculate_ndcg(relevance_scores: list, k: int) -> float:
    """
    Calculates NDCG@k for a list of relevance scores
    """
    relevance_scores = np.array(relevance_scores)[:k]

    dcg = np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))

    ideal_scores = np.sort(relevance_scores)[::-1]
    idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))

    return dcg / idcg if idcg > 0 else 0.0


def build_relevance_ground_truth(queries, df, relevance_thresh=1):
    """
    Calculates the total number of relevant documents for every unique query.
    Denomitor for Recall@k.
    """
    if os.path.exists(RELEVANCE_CACHE):
        print(f"Loading cache relevance map from {RELEVANCE_CACHE}...")
        with open(RELEVANCE_CACHE, "r") as f:
            return json.load(f)
    
    print("Building relevance ground truth map...")
    unique_queries = set(queries)
    relevance_map = {}
    
    num_workers = os.cpu_count() 
    print(f"Starting ProcessPoolExecutor with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(partial(process_single_query, df=df, threshold=relevance_thresh), unique_queries),
            total=len(unique_queries),
            desc="Calculating relevance ground truth"
        ))
    
    relevance_map = dict(results)
    # for query in tqdm(unique_queries, desc="Calculating relevance ground truth"):
    #     total_relevant = 0
    #     for _, row in df.iterrows():
    #         relevance = calculate_relevance(query, row)
    #         if relevance >= relevance_thresh:
    #             total_relevant += 1
    #     relevance_map[query] = total_relevant
    
    print(f"Caching relevance map to {RELEVANCE_CACHE}...")
    with open(RELEVANCE_CACHE, "w") as f:
        json.dump(relevance_map, f)
        
    return relevance_map
    

def evaluate(model, test_df, vocabs, training_data_path, device, precomputed_index=None):
    """
    Evaluates a model on the test set using Recall@K and NDCG@K
    """
    print("\nStarting Evaluation...")

    # test_dataset = CoffeeDataset(test_df, vocabs)
    
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    # generate the test queries and evaluate
    print("Getting test queries and evaluating recall and ndcg...")
    test_queries = []
    test_pair_truth = []
    test_id2idx = {cid: i for i, cid in test_df["id"].items()}

    with open(training_data_path, "r") as f:
        for line in f:
            data=json.loads(line)
            if data["coffee_id"] in test_id2idx:
                correct_idx = test_id2idx[data["coffee_id"]]
                for query in data["queries"]:
                    # if len(query.split()) > 4:
                    test_queries.append(query)
                    test_pair_truth.append(correct_idx)

    total_relevant_map = build_relevance_ground_truth(test_queries, test_df, relevance_thresh=1)
    
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total_queries = len(test_queries)
    
    ndcg_5_scores = []
    ndcg_10_scores = []
    precision_5_scores = []
    precision_10_scores = []
    recall_5_scores = []
    recall_10_scores = []
    
    with torch.no_grad():
        for i in tqdm(range(total_queries), desc="Evaluating Queries"):
            query = test_queries[i]
            correct_idx = test_pair_truth[i]

            query_embedding = model.encode_queries([query]).cpu().numpy()
            faiss.normalize_L2(query_embedding)

            _, top_k_indices = index.search(query_embedding, k=10)

            # calculate Hits@K
            top_k_indices = top_k_indices[0]

            if correct_idx in top_k_indices[:1]:
                hits_at_1 += 1
            if correct_idx in top_k_indices[:5]:
                hits_at_5 += 1
            if correct_idx in top_k_indices[:10]:
                hits_at_10 += 1

            
            recommendation_relevance = [calculate_relevance(query, test_df.iloc[j]) for j in top_k_indices]

            # Calculate NDCG@K
            ndcg_5_scores.append(calculate_ndcg(recommendation_relevance, k=5))
            ndcg_10_scores.append(calculate_ndcg(recommendation_relevance, k=10))
            
            # Calculate Precision@K
            relevant_in_top_5 = sum(1 for score in recommendation_relevance[:5] if score >= 1)
            relevant_in_top_10 = sum(1 for score in recommendation_relevance[:10] if score >= 1)
            precision_5_scores.append(relevant_in_top_5 / 5.)
            precision_10_scores.append(relevant_in_top_10 / 10.)
            
            # Calculate Recall@K
            total_relevant = total_relevant_map.get(query, 0)
            if total_relevant > 0:
                recall_5_scores.append(relevant_in_top_5 / total_relevant)
                recall_10_scores.append(relevant_in_top_10 / total_relevant)
            else:
                recall_5_scores.append(0.0)
                recall_10_scores.append(0.0)
                
    print("\n--- Evaluation Results ---")
    print(f"Total Test Queries: {total_queries}")
    
    hits_at_1 = float(hits_at_1) / total_queries
    hits_at_5 = float(hits_at_5) / total_queries
    hits_at_10 = float(hits_at_10) / total_queries
    
    precision_5 = np.mean(precision_5_scores)
    precision_10 = np.mean(precision_10_scores)
    
    recall_5 = np.mean(recall_5_scores)
    recall_10 = np.mean(recall_10_scores)
    
    ndcg_5 = np.mean(ndcg_5_scores)
    ndcg_10 = np.mean(ndcg_10_scores)
    
    print("\n--- Hits@K ---")
    print(f"Hits@1:  {hits_at_1:.4f}")
    print(f"Hits@5:  {hits_at_5:.4f}")
    print(f"Hits@10: {hits_at_10:.4f}")
    
    print("\n--- Precision@K ---")
    print(f"Precision@5:  {precision_5:.4f}")
    print(f"Precision@10:  {precision_10:.4f}")
    
    print("\n--- Recall@K ---")
    print(f"Recall@5:  {recall_5:.4f}")
    print(f"Recall@10:  {recall_10:.4f}")
    
    print("\n--- NDCG@K ---")
    print(f"NDCG@5:  {ndcg_5:.4f}")
    print(f"NDCG@10:  {ndcg_10:.4f}")
    return hits_at_1, hits_at_5, hits_at_10, precision_5, precision_10, recall_5, recall_10, ndcg_5, ndcg_10


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)
    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    
    # model, vocabs = load_model_inference(TRAINED_MODEL_PATH, numerical_dim=10, device=DEVICE, model_location=SBERT_MODEL_DIR)
    vocabs = load_vocabs(VOCABS_PATH)
    model = load_model(
        vocabs, 
        numerical_dim=MODEL_PARAMS["numerical_dim"],
        embedding_dim=MODEL_PARAMS["embedding_dim"],
        encoder_only=MODEL_PARAMS["encoder_only"], 
        device=DEVICE, 
        model_arch_path=SBERT_MODEL_DIR, 
        model_weights_path=TRAINED_MODEL_PATH, 
        eval=True
    )
    evaluate(model, df, vocabs, QUERIES_PATH, DEVICE, FAISS_INDEX_PATH)

    if DEVICE.type == "cuda":
        del model
        torch.cuda.empty_cache()