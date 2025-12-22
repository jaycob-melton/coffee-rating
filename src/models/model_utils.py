import json
import random
import torch
import pandas as pd
import re
import ast
import numpy as np
import faiss
from tqdm import tqdm
from src.models.model import DualEncoder, CrossEncoder
from src.models.utils import CoffeeDataset
from src.config import (
    CROSS_ENCODER_DATASET,
    VOCABS_PATH,
    PREPROCESSED_DATA_PATH,
    TRAINED_MODEL_PATH,
    SBERT_MODEL_DIR,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    MODEL_PARAMS,
    QUERIES_PATH,
    QUERY_EMBEDDINGS,
    ORIGINS,
    FLAVORS,
    VARIETALS,
    PROCESS,
    CE_ARCHITECTURE,
    CE_WEIGHTS
)
import argparse

def load_vocabs(vocabs_path):
    with open(vocabs_path, "r") as f:
        vocabs = json.load(f)
    return vocabs


def load_model(vocabs, model_params=MODEL_PARAMS, text_encoder_dir=SBERT_MODEL_DIR, weights_path=TRAINED_MODEL_PATH, device=torch.device("cpu"), eval=False):#numerical_dim: int, embedding_dim: int, encoder_only: bool = False, device=torch.device("cpu"), model_arch_path="sentence-transformers/#all-mpnet-base-v2", model_weights_path=None, eval=False):
    """
    Loads the DualEncoder model for training or inference.
    If model_weights_path is provided, it loads trained weights, otherwise it returns the untrained base model
    """

    model = DualEncoder(
        vocabs=vocabs, 
        numerical_dim=model_params["numerical_dim"],
        embedding_dim=model_params["embedding_dim"], 
        fc_dropout=model_params["fc_dropout"],
        all_dropout=model_params["all_dropout"],
        encoder_only=model_params["encoder_only"], 
        text_model_name=text_encoder_dir,
    )
    
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {weights_path}.")
    else:
        print(f"No model weights provided. Loaded untrained model with architecture {text_encoder_dir}.")
    
    model.to(device)
    if eval:
        model.eval()
        print("Model set to evaluation mode.")
    else:
        model.train()
        print("Model set to training mode.")
        
    return model


def build_embeddings(model, coffee_df, vocabs, device, enc_only=False):
    """Encodes all coffees and saves the raw embeddings to a numpy file."""
    if enc_only:
        print("Setting inference mode to encoder only...")
        model.encoder_only = True

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
            embedding = model.encode_coffees(coffee_batch)
            all_coffee_embeddings.append(embedding.cpu().numpy())
            
    all_coffee_embeddings = np.vstack(all_coffee_embeddings)
    return all_coffee_embeddings


def build_search_index(embeddings):
    """Encodes all coffees and builds a searchable FAISS index."""
    print("Building search index for all coffees...")
    
    index = faiss.IndexFlatIP(MODEL_PARAMS["embedding_dim"])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

universal_origins = set(json.loads(open(ORIGINS).read()))
universal_flavors = dict(json.loads(open(FLAVORS).read()))
universal_varietals = list(json.loads(open(VARIETALS).read()))
universal_processes = dict(json.loads(open(PROCESS).read()))

UNIVERSAL_SET = {
    "origin": universal_origins,
    "flavor": {flavor.lower() for flavor in universal_flavors.keys()},
    "notes": {note.lower() for flavor in universal_flavors.values() for note in flavor},
    "varietal": universal_varietals,
    "process": {proc.lower() for proc in universal_processes.keys()}.union({process_name for process in universal_processes.values() for process_name in process}),
    "roast": {"light", "medium-light", "medium", "medium-dark", "dark"},
    "test_method": {"hot_black", "espresso_with_milk", "espresso_black", "cold_with_milk", "hot_with_milk", "cold_black"},
}

# ATTRIBUTE_WEIGHTS = {
#     'origin': 3,
#     'process': 3,
#     'varietal': 2,
#     'flavor': 2,
#     'notes': 1,
#     'roast': 2,
#     'test_method': 3
# }

ATTRIBUTE_WEIGHTS = {
    'origin': 1,
    'process': 1,
    'varietal': 1,
    'flavor': 1,
    'notes': 1,
    'roast': 1,
    'test_method': 1
}

def calculate_relevance(query: str, coffee_row) -> int:

    query = query.lower()

    
    query_attributes = {}
    total_possible_score = 0
    
    for attr_type, keywords in UNIVERSAL_SET.items():
        found_keywords = {kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', query)}
        if found_keywords:
            query_attributes[attr_type] = found_keywords
            if attr_type == "notes":
                total_possible_score += ATTRIBUTE_WEIGHTS[attr_type] * len(found_keywords) * 2
            else:
                total_possible_score += ATTRIBUTE_WEIGHTS[attr_type] * len(found_keywords)

    if total_possible_score == 0:
        return 0 # The query is too generic to be scored

    
    achieved_score = 0
    
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
        return 0 

    coffee_attributes = {
        'origin': coffee_origins,
        'process': coffee_processes,
        'varietal': coffee_varietals,
        'flavor': coffee_flavors,
        "notes": coffee_notes,
        'roast': coffee_roast,
        'test_method': coffee_test_method
    }
    
    note_to_flavor_category = {note.lower(): category.lower() for category, notes in universal_flavors.items() for note in notes}
    coffee_notes_set = coffee_attributes.get('notes', set())
    coffee_notes_categories = {note_to_flavor_category.get(note) for note in coffee_notes_set if note_to_flavor_category.get(note)}

    for attr_type, query_values in query_attributes.items():
        if attr_type == "notes":
            for query_note in query_values:
                note_score = 0
                if query_note in coffee_notes_set:
                    note_score = 2  # Exact match (category match is implied)
                else:
                    query_note_category = note_to_flavor_category.get(query_note)
                    if query_note_category and query_note_category in coffee_notes_categories:
                        note_score = 1  # Category match only
                achieved_score += ATTRIBUTE_WEIGHTS[attr_type] * note_score
        else:
            matches = query_values.intersection(coffee_attributes.get(attr_type, set()))
            achieved_score += ATTRIBUTE_WEIGHTS[attr_type] * len(matches)
        
    
    match_percentage = achieved_score / total_possible_score if total_possible_score > 0 else 0
    
    if match_percentage >= 0.99: return 4 # Perfect
    if match_percentage >= 0.75: return 3 # High
    if match_percentage >= 0.50: return 2 # Medium
    if match_percentage > 0: return 1   # 
    return 0


def build_cross_encoder_dataset(num_search=50, num_positives=5, num_negatives=3):
    
    # Goal: create coffee pairs for each query, at most 10 negatives and the rest positives.
    # load in query embeddings and coffee index
    query_embeddings = np.load(QUERY_EMBEDDINGS)
    coffee_index = faiss.read_index(str(FAISS_INDEX_PATH))
    coffee_data = pd.read_csv(PREPROCESSED_DATA_PATH)
    
    
    coffee_id2idx = {cid: i for i, cid in coffee_data['id'].items()}
    # pair each query with its true positive 
    queries2coffees = []
    print("Finding true positives for each query...")
    with open(QUERIES_PATH, "r") as f:
        for line in f:
            data = json.loads(line)
            coffee_idx = coffee_id2idx.get(data["coffee_id"])
            if coffee_idx is not None:
                for query in data["queries"]:
                    queries2coffees.append({"query": query, "positive_idx": coffee_idx})
    
    _, topkmat = coffee_index.search(query_embeddings, num_search)  # (num_queries, 50)
    
    training_samples = []
    
    for i in tqdm(range(len(queries2coffees)), desc="Finding Training Examples"):
        query_text = queries2coffees[i]["query"]
        true_idx = queries2coffees[i]["positive_idx"]
        top_k_indices = topkmat[i]  # Indices of top 50 coffees for this query
        
        
        positives = []
        positives_idx = set()
        negatives = []
        for coffee_idx in top_k_indices:
            coffee_row = coffee_data.iloc[coffee_idx]
            rel_score = calculate_relevance(query_text, coffee_row) / 4.0
            if rel_score > 0:
                positives.append({
                    "query": query_text,
                    "positive_idx": coffee_idx,
                    "coffee_text": coffee_row["combined_text"],
                    "label": rel_score
                })
            elif len(negatives) < num_negatives:
                negatives.append({
                    "query": query_text,
                    "coffee_text": coffee_row["combined_text"],
                    "label": 0.0
                })
        
        final_positives = random.sample(positives, min(len(positives), num_positives))
        # Fix: final_positives is a list of dictionaries, this does not work
        if true_idx not in [p["positive_idx"] for p in final_positives]:
            training_samples.append({
                "query": query_text,
                "coffee_text": coffee_data.iloc[true_idx]["combined_text"],
                "label": 1.0
            })
        training_samples.extend(final_positives)
        training_samples.extend(negatives)
    
    print(f"Saving {len(training_samples)} training samples to {CROSS_ENCODER_DATASET}...")
    with open(CROSS_ENCODER_DATASET, "w") as f:
        for sample in training_samples:
            f.write(json.dumps(sample) + "\n")
        
        
def load_cross_encoder(model_arch_path=CE_ARCHITECTURE, weights_path=CE_WEIGHTS, device=torch.device("cpu"), eval=False):
    
    model = CrossEncoder(model_arch_path)
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Cross-Encoder model loaded from {weights_path}.")
    else:
        print(f"No model weights provided. Loaded untrained Cross-Encoder model with architecture {model_arch_path}.")
    
    model.to(device)
    if eval:
        model.eval()
        print("Cross-Encoder model set to evaluation mode.")
    else:
        model.train()
        print("Cross-Encoder model set to training mode.")
        
    return model


def predict_hybrid(query, dual_encoder, cross_encoder, faiss_index, coffee_df, initial_k=50, final_k=10):
    device = next(dual_encoder.parameters()).device
    # Stage 1: Dual Encoder Retrieval
    with torch.no_grad():
        # Encode query with dual encoder
        query_emb = dual_encoder.encode_queries([query]).cpu().numpy()
        faiss.normalize_L2(query_emb)
        
        # Initial retrieval with dual encoder
        _, top_k_indices = faiss_index.search(query_emb, k=initial_k)
        candidate_indices = top_k_indices[0]
    
    candidates_df = coffee_df.iloc[candidate_indices].copy()
    
    # Stage 2: Cross Encoder Re-ranking
    queries = [query] * len(candidates_df)
    coffee_texts = candidates_df["combined_text"].tolist()

    with torch.no_grad():
        relevance_scores = cross_encoder.predict(queries, coffee_texts)
        relevance_scores = relevance_scores.squeeze().cpu().numpy()
    
    sorted_scores_indices = np.argsort(-relevance_scores)
    final_k_indices = sorted_scores_indices[:final_k]
    
    final_candidates = candidates_df.iloc[final_k_indices]

    final_candidates["relevance_score"] = relevance_scores[final_k_indices]

    return final_candidates




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flag for building embeddings or search index")
    parser.add_argument("--build_embeddings", action="store_true", help="Flag to build embeddings for all coffees. Saves to a numpy file in config.EMBEDDINGS_PATH")
    parser.add_argument("--build_index", action="store_true", help="Flag to build a FAISS search index. Saves to a file in config.FAISS_INDEX_PATH. If --build_embeddings is not set, it will load embeddings from config.EMBEDDINGS_PATH")
    parser.add_argument("--build_cross_encoder_dataset", action="store_true", help="Flag to build a cross-encoder dataset. Saves to a file in config.CROSS_ENCODER_DATASET_PATH")
    
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    vocabs = load_vocabs(VOCABS_PATH)

    model = load_model(
        vocabs=vocabs,
        model_params=MODEL_PARAMS,
        device=DEVICE,
        eval=True
    )

    print("Loading coffee data...")
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    #df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    embeddings = None
    if args.build_embeddings:
        embeddings = build_embeddings(model, df, vocabs, DEVICE, enc_only=MODEL_PARAMS["encoder_only"])
        try:
            np.save(EMBEDDINGS_PATH, embeddings)
            print(f"Embeddings saved to {EMBEDDINGS_PATH}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    if args.build_index:
        if embeddings is None:
            print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
            embeddings = np.load(EMBEDDINGS_PATH)
        
        index = build_search_index(embeddings)
        try:
            faiss.write_index(index, str(FAISS_INDEX_PATH))
            print(f"FAISS index saved to {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            
    if args.build_cross_encoder_dataset:
        build_cross_encoder_dataset()
    
    if DEVICE.type == "cuda":
        del model
        torch.cuda.empty_cache()



