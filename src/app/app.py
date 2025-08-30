from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
import faiss
import numpy as np
import os
import traceback
import ast
import json

from ..models.model import DualEncoder
from ..models.utils import CoffeeDataset
from ..models.evaluate import load_model_inference
from ..models.predict import get_recommendations, build_search_index

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
    model_path = "data/outputs/model-weights/8-11/coffee_model_epoch_11_semi_hard_3.pth"
    faiss_path = "data/outputs/faiss/faiss_index.bin"
    
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
        print("COULD NOT FIND FAISS INDEX, creating index...")
        build_search_index(MODEL, COFFEE_DF, VOCABS, device, output_path=faiss_path)
        FAISS_INDEX = faiss.read_index(faiss_path)
        print(f"FAISS index created and loaded. Contains {FAISS_INDEX.ntotal}) vectors.")

    
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
        

        print("Getting recommendations...")
        recommendations = get_recommendations(query, MODEL, FAISS_INDEX, COFFEE_DF, top_k=5)
        recommendations.to_csv("latest_recommendations.csv", index=False)
        valid_json_string = recommendations.to_json(orient="records")
        results_list = json.loads(valid_json_string) # Load the clean string back to a Python list of dicts

        # Now, we only need to handle the string-to-list conversion for our multi-value columns.
        cleaned_results = []
        list_cols = ['countries_extracted', 'process', 'varietals']
        for record in results_list:
            cleaned_record = record.copy() # Start with the record, which is already mostly clean
            for col in list_cols:
                if col in cleaned_record and isinstance(cleaned_record[col], str):
                    try:
                        # Safely evaluate the string representation of a list
                        cleaned_record[col] = ast.literal_eval(cleaned_record[col])
                    except (ValueError, SyntaxError):
                        # If it's not a list-like string, just treat it as a single-item list
                        cleaned_record[col] = [cleaned_record[col]] 
            cleaned_results.append(cleaned_record)

        return jsonify(cleaned_results)
    
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
    app.run(debug=True, port=5000) #, use_reloader=False    