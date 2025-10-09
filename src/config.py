import os
from pathlib import Path
# from dotenv import load_dotenv

# load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# PATHS
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SBERT_MODEL_DIR = MODELS_DIR / "sbert_model" # set to none to load model from HuggingFace

PREPROCESSED_DATA_PATH = DATA_DIR / "processed" / "preprocessed_data.csv"
FAISS_INDEX_PATH = MODELS_DIR / "faiss" / "faiss_index.bin"
TRAINED_MODEL_PATH = MODELS_DIR / "model-weights" / "8-11" / "coffee_model_epoch_11_semi_hard_3.pth"

# MODEL HYPERPARAMS
# EMBEDDING_DIM = 768
# LEARNING_RATE = 1e-6
# BATCH_SIZE = 32
# EPOCHS = 10

# API KEYS
# OPENAI_KEY = os.getenv("OPENAI_API_KEY")