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
TRAIN_DATA_PATH = PROCESSED_DIR / "train_data_8_11.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test_data_8_11.csv"
QUERIES_PATH = DATA_DIR / "outputs" / "llm-queries" / "synthetic_queries_np_4_1_nano.jsonl"
VOCABS_PATH = MODELS_DIR / "vocabs" / "best_model_vocabs.json"

# Model

# Architecture
SBERT_MODEL_DIR = MODELS_DIR / "sbert_model" # set to "sentence-transformers/all-mpnet-base-v2" to use model from HuggingFace
# Weights
TRAINED_MODEL_PATH =  MODEL_WEIGHTS_DIR / "11-14" / "sbert_hybrid_epoch_finetune_17.pth" #MODEL_WEIGHTS_DIR / "11-13" / "MiniLM_encoder_only_epoch_20.pth" #MODEL_WEIGHTS_DIR / "11-13" / "encoder_only_epoch_20.pth" #None #/ MODEL_WEIGHTS_DIR / "11-12" / "encoder_only_epoch_6_shnm_epoch_2.pth" 
# FAISS Index and Embeddings
FAISS_INDEX_PATH = FAISS_DIR / "sbert_hybrid_index.bin" #"trained_encoder_shnm_index.bin "#"faiss_index.bin
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "sbert_hybrid_embeddings.npy" #"trained_encoder_shnm_embeddings.npy"
# Model training save path
MODEL_SAVE_DIR = MODEL_WEIGHTS_DIR / f"{today.month}-{today.day}"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "sbert_hybrid_epoch_"

# MODEL HYPERPARAMS
TRAIN_PARAMS = {
    "margin": 0.2,
    "batch_size": 32,
    "semi_hard_mining_start_epoch": -1,

    # stage 1: Warm-up (Freeze transformer, train head)
    "warmup_epochs": 3,
    "warmup_head_lr": 1e-3,

    # stage 2: Unfreeze transformer, train all layers
    "transformer_lr": 1e-5,
    "head_lr": 1e-4,
    "num_epochs": 17,
}

MODEL_PARAMS = {
    "numerical_dim": 10,
    "embedding_dim": 768,
    "encoder_only": False,
    "fc_dropout": 0.1,
    "dropout": 0.2,
}

# API KEYS
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")