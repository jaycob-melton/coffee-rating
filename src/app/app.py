from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
import faiss
import numpy as np
import os
import traceback

from ..models.model import DualEncoder
from ..models.utils import CoffeeDataset
from ..models.evaluate import load_model_inference
from ..models.predict import get_recommendations

app = Flask(__name__)

# global variables, loaded when server starts
MODEL = None
VOCABS = None
COFFEE_DF = None
FAISS_INDEX = None

def load_artifacts():
    """
    Loads the model, data, vocabs, and faiss index
    Called when Flask server initializes
    """
    global MODEL, VOCABS, COFFEE_DF, FAISS_INDEX
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preprocessed_path = "data/processed/preprocessed_data.csv"
    model_path = "data/outputs/model-weights/coffee_model_epoch_11_semi_hard_3.pth"
    faiss_path = "data/outputs/faiss_index.bin"
    
    # load coffee data
    print("Loading Coffee Data...")
    COFFEE_DF = pd.read_csv(preprocessed_path)
    COFFEE_DF["combined_text"] = COFFEE_DF["blind assessment"].fillna("") + " " + COFFEE_DF["bottom line"].fillna("")
    
    # load model and vocabs
    print("Loading Model...")
    MODEL, VOCABS = load_model_inference(model_path, numerical_dim=10, device=device)
    
    # load faiss index for search
    print("Loading FAISS Index")
    if os.path.exists(faiss_path):
        FAISS_INDEX = faiss.read_index(faiss_path)
        print(f"FAISS index loaded from {faiss_path}. Contains {FAISS_INDEX.ntotal} vectors.")
    else:
        print("COULD NOT FIND FAISS INDEX")
    
    print("-- Artifacts loaded sucessfully --")
    

# -- API Routes --
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Takes a user query and returns the top K recommendations.
    """
    if not MODEL or not FAISS_INDEX or COFFEE_DF is None:
        return jsonify({"error": "Model or data not loaded"}), 500
    print("All artifacts were found")
    try:
        print("Getting query from frontend")
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # # -- Perform Inference --
        # with torch.no_grad():
        #     # encode the user query
        #     query_embedding = MODEL.encode_queries([query]).cpu().numpy()
        #     faiss.normalize_L2(query_embedding)
            
        #     # search the faiss index for top k matches
        #     distances, top_k_indices = FAISS_INDEX.search(query_embedding, k=5)
            
        #     # get results from coffee df
        #     result_indices = top_k_indices[0]
        #     recommendations = COFFEE_DF.iloc[result_indices]
            
        #     results_json = recommendations.to_dict(orient="records")
        print("Getting recommendations...")
        recommendations = get_recommendations(query, MODEL, FAISS_INDEX, COFFEE_DF, top_k=5)
        print("Dumping recomendations to json...")
        results_json = recommendations.to_dict(orient="records")
        return jsonify(results_json)
    
    except Exception as e:
            # FIX: Add detailed error logging to the terminal
            print("--- AN ERROR OCCURRED ---")
            print(traceback.format_exc())
            print("-----------------------")
            return jsonify({"error": "An internal server error occurred."}), 500

# -- Main Execution --
if __name__ == "__main__":
    # load the necessary global artifacts
    load_artifacts()
    # run app
    app.run(debug=True, port=5000, use_reloader=False)    