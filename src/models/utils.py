import pandas as pd
import ast
import json
import torch
from torch.utils.data import Dataset

def to_list(x):
    if isinstance(x, list):
        return [str(e).strip() for e in x if pd.notna(e) and str(e).strip()]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            out = ast.literal_eval(x)
            if isinstance(out, list):
                return [str(e).strip() for e in out if pd.notna(e) and str(e).strip()]
            return [str(out).strip()]
        except Exception:
            return [s.strip() for s in x.split(",") if s.strip()]
    return []


def build_vocab(series: pd.Series, is_multivalue=False) -> dict:
    """Builds a vocabulary for a categorical feature"""
    if is_multivalue:
        all_items = set(item for sublist in series for item in sublist)
    else:
        all_items = set(series.unique())

    items = sorted(map(str, all_items))
    vocab = {item: i + 1 for i, item in enumerate(items)}
    vocab["<unknown>"] = 0
    return vocab


def build_all_vocabs(df: pd.DataFrame) -> dict:
    """Builds vocabularies for all categorical features for embedding layers"""
    single_value = ["roast level", "test_method", "price_tier"]
    multi_value = ["countries_extracted", "process", "varietals"]

    vocabs = {}
    for col in single_value:
        df[col] = df[col].astype(str)
        vocabs[col] = build_vocab(df[col])
    
    for col in multi_value:
        df[col] = df[col].apply(to_list)
        vocabs[col] = build_vocab(df[col], is_multivalue=True)

    return vocabs


class CoffeeDataset(Dataset):
    def __init__(self, df, vocabs, text_col="combined_text"):
        self.df = df
        self.vocabs = vocabs
        self.text_col = text_col

        self.single_value = ["roast level", "test_method", "price_tier"]
        self.multi_value = ["countries_extracted", "process", "varietals"]
        self.numerical_cols = [
            "rating", "aroma", "acidity", "body", "flavor", "aftertaste",
            "with milk", "agtron_1", "agtron_2", "price_per_oz"
        ]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row[self.text_col]

        numerical_features = torch.tensor([row[col] for col in self.numerical_cols], dtype=torch.float)

        cat_features = {}
        for col in self.single_value:
            item = row[col]
            cat_features[col] = torch.tensor(self.vocabs[col].get(item, 0), dtype=torch.long)

        for col in self.multi_value:
            # items = row[col]
            # indices = [self.vocabs[col].get(item, 0) for item in items]
            # cat_features[col] = torch.tensor(indices, dtype=torch.long)    
            items = row[col] if isinstance(row[col], (list, tuple)) else []
            idxs = [self.vocabs[col].get(str(item), 0) for item in items]
            if not idxs:  # avoid empty-bag edge cases
                idxs = [0]
            cat_features[col] = torch.tensor(idxs, dtype=torch.long)

        return text, numerical_features, cat_features
    

class TripleTrainingDataset(Dataset):
    def __init__(self, training_data_path: str, coffee_df: pd.DataFrame):
        self.coffee_df = coffee_df
        self.queries = []
        self.coffee_id2idx = {cid: i for i, cid in coffee_df['id'].items()}
        with open(training_data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                # print(data)
                # print(data["coffee_id"])
                coffee_idx = self.coffee_id2idx.get(data["coffee_id"])
                if coffee_idx is not None:
                    for query in data["queries"]:
                        self.queries.append({"query": query, "positive_idx": coffee_idx})
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx]