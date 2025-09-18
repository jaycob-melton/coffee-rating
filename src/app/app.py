from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
import faiss
import numpy as np
import os
import traceback
import ast
import json


import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model import DualEncoder
from src.models.predict import get_recommendations, load_model_inference

app = Flask(__name__)

# We will attach artifacts to the app object instead of using globals
app.MODEL = None
app.VOCABS = None
app.COFFEE_DF = None
app.FAISS_INDEX = None

def load_artifacts(flask_app):
    """
    Loads the model, data, vocabs, and faiss index and attaches them
    to the Flask app object.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use relative paths from the app's location for robustness
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_path = os.path.join(base_dir, "../../data/processed/preprocessed_data.csv")
    model_path = os.path.join(base_dir, "../../data/outputs/model-weights/8-11/coffee_model_epoch_11_semi_hard_3.pth")
    faiss_path = os.path.join(base_dir, "../../data/outputs/faiss/faiss_index.bin")
    
    print("Loading Coffee Data...")
    flask_app.COFFEE_DF = pd.read_csv(preprocessed_path)
    flask_app.COFFEE_DF["combined_text"] = flask_app.COFFEE_DF["blind assessment"].fillna("") + " " + flask_app.COFFEE_DF["bottom line"].fillna("")
    
    print("Loading Model...")
    flask_app.MODEL, flask_app.VOCABS = load_model_inference(model_path, numerical_dim=10, device=device)
    
    print("Loading FAISS Index")
    if os.path.exists(faiss_path):
        flask_app.FAISS_INDEX = faiss.read_index(faiss_path)
        print(f"FAISS index loaded from {faiss_path}. Contains {flask_app.FAISS_INDEX.ntotal} vectors.")
    else:
        # Fallback to create it if not found
        from src.models.predict import build_search_index
        print("COULD NOT FIND FAISS INDEX, creating index...")
        flask_app.FAISS_INDEX = build_search_index(flask_app.MODEL, flask_app.COFFEE_DF, flask_app.VOCABS, device, output_path=faiss_path)
        print(f"FAISS index created and loaded. Contains {flask_app.FAISS_INDEX.ntotal} vectors.")

    
    print("-- Artifacts loaded successfully --")


load_artifacts(app)

# -- API Routes --
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Takes a user query and returns the top K recommendations.
    """
    # Access artifacts from the app object
    if not app.MODEL or not app.FAISS_INDEX or app.COFFEE_DF is None:
        return jsonify({"error": "Model or data not loaded"}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        recommendations = get_recommendations(query, app.MODEL, app.FAISS_INDEX, app.COFFEE_DF, top_k=5)
        
        # Use pandas' robust to_json method first to handle NaNs and dtypes
        valid_json_string = recommendations.to_json(orient="records")
        results_list = json.loads(valid_json_string)

        # Handle string-to-list conversion for multi-value columns
        cleaned_results = []
        list_cols = ['countries_extracted', 'process', 'varietals']
        for record in results_list:
            cleaned_record = record.copy()
            for col in list_cols:
                if col in cleaned_record and isinstance(cleaned_record[col], str):
                    try:
                        cleaned_record[col] = ast.literal_eval(cleaned_record[col])
                    except (ValueError, SyntaxError):
                        cleaned_record[col] = [cleaned_record[col]] 
            cleaned_results.append(cleaned_record)

        return jsonify(cleaned_results)
    
    except Exception as e:
        print("--- AN ERROR OCCURRED ---")
        print(traceback.format_exc())
        print("-----------------------")
        return jsonify({"error": "An internal server error occurred."}), 500

# -- Main Execution --
if __name__ == "__main__":
    # Load artifacts into the app context before running
    # load_artifacts(app)
    # Use reloader=False for stability with large in-memory models
    app.run(debug=True, port=5000, use_reloader=False)

