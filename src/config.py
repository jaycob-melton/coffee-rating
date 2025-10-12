import os
from pathlib import Path
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
SBERT_MODEL_DIR = MODELS_DIR / "sbert_model" # set to "sentence-transformers/all-mpnet-base-v2" to use model from HuggingFace

RAW_DATA_PATH = RAW_DIR / "full_data_scrape.csv"
PREPROCESSED_DATA_PATH = PROCESSED_DIR / "preprocessed_data.csv"
# TRAIN_DATA_PATH = PROCESSED_DIR / "train_data_8_11.csv"
# TEST_DATA_PATH = PROCESSED_DIR / "test_data_8_11.csv"
FAISS_INDEX_PATH = FAISS_DIR / "faiss_index.bin"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "all_embeddings.npy"
TRAINED_MODEL_PATH = MODEL_WEIGHTS_DIR / "8-11" / "coffee_model_epoch_11_semi_hard_3.pth"
QUERIES_PATH = DATA_DIR / "outputs" / "llm-queries" / "synthetic_queries_np_4_1_nano.jsonl"

# MODEL HYPERPARAMS
# EMBEDDING_DIM = 768
# LEARNING_RATE = 1e-6
# BATCH_SIZE = 32
# EPOCHS = 10

# API KEYS
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")