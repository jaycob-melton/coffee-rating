import os
from pathlib import Path
from datetime import date
today = date.today()
# from dotenv import load_dotenv

# load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# PATHS
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
UNI_DIR = DATA_DIR / "universal"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
FAISS_DIR = MODELS_DIR / "faiss"
MODEL_WEIGHTS_DIR = MODELS_DIR / "model-weights"

# Data
RAW_DATA_PATH = RAW_DIR / "full_data_scrape.csv"
PREPROCESSED_DATA_PATH = PROCESSED_DIR / "preprocessed_fulltext.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv" #"train_data_8_11.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv" #"test_data_8_11.csv"
QUERIES_PATH = DATA_DIR / "outputs" / "llm-queries" / "synthetic_queries_np_4_1_nano.jsonl"#"training_data.jsonl" #"synthetic_queries_np_4_1_nano.jsonl"
VOCABS_PATH = MODELS_DIR / "vocabs" / "best_model_vocabs.json"
RELEVANCE_CACHE = PROCESSED_DIR / "relevance_cache.json"
VARIETALS = UNI_DIR / "coffee_varietals.json"
FLAVORS = UNI_DIR / "flavor_keywords.json"
ORIGINS = UNI_DIR / "known_origins.json"
PROCESS = UNI_DIR / "process_keywords.json"

# Model

# Architecture 
SBERT_MODEL_DIR = MODELS_DIR / "MiniLM-L6-v2" #"sentence-transformers/all-MiniLM-L6-v2" # PROJECT_ROOT / "sbert_model" # set to "sentence-transformers/all-mpnet-base-v2" to use model from HuggingFace
# Weightss
TRAINED_MODEL_PATH = MODEL_WEIGHTS_DIR / "11-24" / "coffee_model_minilm_fulltext_epoch_finetune_15.pth"  #MODEL_WEIGHTS_DIR / "11-13" / "MiniLM_encoder_only_epoch_20.pth" # "11-24" / "coffee_model_minilm_epoch_finetune_10.pth" 
# Embeddings Save/Load Path
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "coffee_model_minilm_fulltext_epoch_finetune_15_embeddings.npy"
# Index Save/Load Path
FAISS_INDEX_PATH = FAISS_DIR / "coffee_model_minilm_fulltext_epoch_finetune_15_index.bin" #"faiss_index.bin"
# Query Embeddings Save/Load Path
QUERY_EMBEDDINGS = EMBEDDINGS_DIR / "coffee_model_minilm_fulltext_epoch_finetune_15_query_emb.npy"
# Train Save Path
MODEL_SAVE_DIR = MODEL_WEIGHTS_DIR / f"{today.month}-{today.day}"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "coffee_model_minilm_fulltext_epoch_"

# MODEL HYPERPARAMS
TRAIN_PARAMS = {
    "batch_size": 128,
    "semi_hard_mining_start_epoch": -1,
    "margin": 0.2,
    
    # stage 1: Warm-up (Freeze Transformer, train head)
    "warmup_epochs": 0,
    "warmup_head_lr": 1e-3,

    # stage 2: Unfreeze transformer, train all layers
    "transformer_lr": 1e-5,
    "head_lr": 1e-6,
    "num_epochs": 15,
}

MODEL_PARAMS = {
    "numerical_dim": 10,
    "embedding_dim": 384,#768, #384, 
    "encoder_only": True,
    "fc_dropout": 0.1,
    "all_dropout": 0.2,
}

# API KEYS
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")