import pandas as pd
import json
from tqdm import tqdm
import ast

def safe_eval(val):
    try:
        if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
            return ast.literal_eval(val)
        elif isinstance(val, list):
            return val
    except (ValueError, SyntaxError):
        pass

    return []


def generate_programmatic_positives(row):
    queries = set()

    origins = safe_eval(row.get("countries_extracted", []))
    processes = safe_eval(row.get("process", []))
    varietals = safe_eval(row.get("varietals", []))

    roast = row.get("roast level", "")
    price = row.get("price_tier", "")
    test_method = row.get("test_method", "")

    flavor_profile = safe_eval(row.get("flavor_profile", {}))
    flavors = list(flavor_profile.keys()) if isinstance(flavor_profile, dict) else []

     # --- Generate Queries based on Patterns ---

    # Pattern 1: Single Attributes
    for origin in origins: queries.add(origin)
    for process in processes: queries.add(f"{process} process")
    for varietal in varietals: queries.add(f"{varietal} varietal")
    if roast: queries.add(f"{roast} roast")
    if price: queries.add(f"{price} coffee")
    if 'espresso' in test_method: queries.add("espresso")

    # Pattern 2: Two-Attribute Combinations
    if roast:
        for origin in origins: queries.add(f"{roast} {origin}")
        for flavor in flavors: queries.add(f"{roast} {flavor} coffee")
    
    for process in processes:
        for origin in origins: queries.add(f"{process} {origin}")

    # Pattern 3: Three-Attribute Combinations
    if roast:
        for process in processes:
            for origin in origins:
                queries.add(f"{roast} {process} {origin}")

    # Pattern 4: Quality-based queries
    rating = row.get('rating', 0)
    if rating >= 0.9: # Assuming normalized rating
        # queries.add("90+ point coffee")
        for origin in origins: queries.add(f"highly rated {origin}")
    
    return queries


if __name__ == "__main__":
    PREPROCESSED_PATH = "preprocessed_data.csv"
    LLM_QUERIES_PATH = "synthetic_queries_np_4_1_nano.jsonl"
    FINAL_TRAINING_PATH = "training_data.jsonl"

    print(f"Loading preprocessed data from {PREPROCESSED_PATH}...")
    df = pd.read_csv(PREPROCESSED_PATH)

    llm_queries = {}
    try:
        with open(LLM_QUERIES_PATH, "r") as f:
            for line in f:
                data = json.loads(line)
                llm_queries[data["coffee_id"]] = data["queries"]
        print(f"Loaded {len(llm_queries)} entries from {LLM_QUERIES_PATH}.")
    except FileNotFoundError:
        print(f"Warning: {LLM_QUERIES_PATH} not found. Proceeding without LLM queries.")

    print("Generating programmatic queries...")

    final_training_data = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Programmatic Queries"):
        coffee_id = row["id"]

        existing_queries = set(llm_queries.get(coffee_id, []))

        prog_queries = generate_programmatic_positives(row)

        all_queries = existing_queries.union(prog_queries)

        if all_queries:
            final_training_data.append({
                "coffee_id": coffee_id,
                "queries": sorted(list(all_queries))
            })

    with open(FINAL_TRAINING_PATH, "w") as f:
        for item in final_training_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nComplete training data saved to {FINAL_TRAINING_PATH}")
    print(f"Total Entries: {len(final_training_data)}")
    if final_training_data:
        print(f"Example queries for first coffee: {final_training_data[0]['queries'][:5]}")