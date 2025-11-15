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
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
FAISS_DIR = MODELS_DIR / "faiss"
MODEL_WEIGHTS_DIR = MODELS_DIR / "model-weights"

# Data
RAW_DATA_PATH = RAW_DIR / "full_data_scrape.csv"
PREPROCESSED_DATA_PATH = PROCESSED_DIR / "preprocessed_data.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv" #"train_data_8_11.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv" #"test_data_8_11.csv"
QUERIES_PATH = DATA_DIR / "outputs" / "llm-queries" / "synthetic_queries_np_4_1_nano.jsonl"
VOCABS_PATH = MODELS_DIR / "vocabs" / "best_model_vocabs.json"
RELEVANCE_CACHE = PROCESSED_DIR / "relevance_cache.json"

# Model

# Architecture 
SBERT_MODEL_DIR = MODELS_DIR / "sbert_model" #"sentence-transformers/all-MiniLM-L6-v2" # PROJECT_ROOT / "sbert_model" # set to "sentence-transformers/all-mpnet-base-v2" to use model from HuggingFace
# Weights
TRAINED_MODEL_PATH = MODEL_WEIGHTS_DIR / "11-12" / "encoder_only_epoch_6_shnm_epoch_2.pth" # None # MODEL_WEIGHTS_DIR / "11-12" / "encoder_only_epoch_6_shnm_epoch_2.pth" #"best_model_weights_only.pth" # switch to none for untrained model #"8-11" / "coffee_model_epoch_11_semi_hard_3.pth"
# Embeddings Save/Load Path
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "trained_encoder_shnm_embeddings.npy"
# Index Save/Load Path
FAISS_INDEX_PATH = FAISS_DIR / "trained_encoder_shnm_index.bin"#"faiss_index.bin"
# Train Save Path
MODEL_SAVE_DIR = MODEL_WEIGHTS_DIR / f"{today.month}-{today.day}"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "coffee_model_epoch_"

# MODEL HYPERPARAMS
TRAIN_PARAMS = {
    "batch_size": 32,
    "transformer_lr": 1e-5,
    "head_lr": 1e-6,
    "num_epochs": 10,
    "semi_hard_mining_start_epoch": 3,
    "margin": 0.2,
}

MODEL_PARAMS = {
    "numerical_dim": 10,
    "embedding_dim": 768, #384, 
    "encoder_only": True,
}

# API KEYS
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")