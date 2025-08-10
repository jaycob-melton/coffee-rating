import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import json
import ast
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import faiss

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


class MetadataEncoder(nn.Module):
    def __init__(self, vocabs, numerical_dim, embedding_dim=16):
        super().__init__()
        self.vocabs = vocabs
        emb_rows = lambda v: max(v.values()) + 1
        # create embedding layers for single-value cats
        self.roast_lvl_embed = nn.Embedding(emb_rows(vocabs["roast level"]), embedding_dim)
        self.test_method_embed = nn.Embedding(emb_rows(vocabs["test_method"]), embedding_dim)
        self.price_tier_embed = nn.Embedding(emb_rows(vocabs["price_tier"]), embedding_dim)

        # create embedding bag layers for multi-value cats
        self.countries_embed = nn.EmbeddingBag(emb_rows(vocabs["countries_extracted"]), embedding_dim, mode="mean")
        self.process_embed = nn.EmbeddingBag(emb_rows(vocabs["process"]), embedding_dim, mode="mean")
        self.varietals_embed = nn.EmbeddingBag(emb_rows(vocabs["varietals"]), embedding_dim, mode="mean")

        # small fully connected network for numerical features
        self.fc = nn.Sequential(
            nn.Linear(numerical_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

        total_embed_dim = embedding_dim * 7
        self.output_dim = total_embed_dim

    def forward(self, num_features, cat_features):
        roast_emb = self.roast_lvl_embed(cat_features["roast level"])
        test_emb = self.test_method_embed(cat_features["test_method"])
        price_emb = self.price_tier_embed(cat_features["price_tier"])

        # For EmbeddingBag, handle empty lists to avoid errors
        countries_emb = self.countries_embed(cat_features['countries_extracted'], cat_features['countries_extracted_offsets'])
        process_emb = self.process_embed(cat_features['process'], cat_features['process_offsets'])
        varietals_emb = self.varietals_embed(cat_features['varietals'], cat_features['varietals_offsets'])

        num_emb = self.fc(num_features)

        all_emb = torch.cat([
            roast_emb, test_emb, price_emb,
            countries_emb, process_emb, varietals_emb,
            num_emb
        ], dim=-1)

        return all_emb
    

# class DualEncoder(nn.Module):
#     def __init__(self, vocabs, numerical_dim, text_model_name="all-mpnet-base-v2", embedding_dim=768, device="cuda"):
#         super().__init__()
#         self.text_encoder = SentenceTransformer(text_model_name, device=device)
#         text_output_dim = self.text_encoder.get_sentence_embedding_dimension()

#         self.metadata_encoder = MetadataEncoder(vocabs, numerical_dim)
#         metadata_output_dim = self.metadata_encoder.output_dim

#         self.fusion_layer = nn.Linear(text_output_dim + metadata_output_dim, embedding_dim)

#     def encode_queries(self, queries):
#         return self.text_encoder.encode(queries, convert_to_tensor=True)
    
#     def encode_coffees(self, coffee_batch):#text, numerical_features, cat_features):
#         # text_emb = self.text_encoder.encode(text, convert_to_tensor=True)
#         # metadata_emb = self.metadata_encoder(numerical_features, cat_features)

#         # combined_emb = torch.cat([text_emb, metadata_emb], dim=-1)
#         # final_emb = self.fusion_layer(combined_emb)
#         text_emb = self.text_encoder.encode(coffee_batch["text"], convert_to_tensor=True)
#         metadata_emb = self.metadata_encoder(coffee_batch["numericals"], coffee_batch["categoricals"])
#         combined_emb = torch.cat([text_emb, metadata_emb], dim=-1)
#         final_emb = self.fusion_layer(combined_emb)
#         return final_emb

class DualEncoder(nn.Module):
    def __init__(self, vocabs, numerical_dim, text_model_name="sentence-transformers/all-mpnet-base-v2",
                 embedding_dim=768, max_length=256):
        super().__init__()
        # Hugging Face model + tokenizer (trainable)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.transformer = AutoModel.from_pretrained(text_model_name)
        self.text_hidden = self.transformer.config.hidden_size  # e.g., 768

        self.metadata_encoder = MetadataEncoder(vocabs, numerical_dim)
        self.fusion_layer = nn.Linear(self.text_hidden + self.metadata_encoder.output_dim, embedding_dim)

        self.max_length = max_length

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-6)                        # [B, 1]
        return summed / counts

    def _encode_texts(self, texts):
        device = next(self.parameters()).device
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.transformer(**toks)                   # gradients flow
        sent = self._mean_pool(out.last_hidden_state, toks["attention_mask"])
        return sent                                       # [B, H], requires_grad=True

    def encode_queries(self, queries):
        # returns trainable embeddings
        return self._encode_texts(queries)

    def encode_coffees(self, coffee_batch):
        text_emb = self._encode_texts(coffee_batch["text"])
        metadata_emb = self.metadata_encoder(coffee_batch["numericals"], coffee_batch["categoricals"])
        return self.fusion_layer(torch.cat([text_emb, metadata_emb], dim=-1))


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
    

def collate(batch, coffee_dataset: CoffeeDataset):
    """
    Collate function to prepare a batch for training
    """
    queries = [item["query"] for item in batch]
    positive_indices = [item["positive_idx"] for item in batch]

    positive_coffee_batch = {
        'text': [],
        'numericals': [],
        'categoricals': {
            'roast level': [], 'test_method': [], 'price_tier': [],
            'countries_extracted': [], 'process': [], 'varietals': [],
            'countries_extracted_offsets': [0], 'process_offsets': [0], 'varietals_offsets': [0]
        }
    }
    # store all data about postive coffees
    for idx in positive_indices:
        text, numericals, categoricals = coffee_dataset[idx]
        positive_coffee_batch["text"].append(text)
        positive_coffee_batch["numericals"].append(numericals)
        for col, val in categoricals.items():
            if val.dim() == 0:
                positive_coffee_batch["categoricals"][col].append(val)
            else:
                positive_coffee_batch["categoricals"][col].append(val)
                offset_key = f"{col}_offsets"
                # need to store offset since these cats take variable length
                offsets = positive_coffee_batch["categoricals"][offset_key]
                offsets.append(offsets[-1] + len(val))
    
    # convert everything to tensors for models
    positive_coffee_batch["numericals"] = torch.stack(positive_coffee_batch["numericals"])
    for col, val in positive_coffee_batch["categoricals"].items():
        if "offsets" not in col and len(val) > 0:
            if val[0].dim() == 0:
                positive_coffee_batch["categoricals"][col] = torch.stack(val)
            else:
                positive_coffee_batch["categoricals"][col] = torch.cat(val)
        elif "offsets" in col:
            positive_coffee_batch["categoricals"][col] = torch.tensor(val[:-1], dtype=torch.long)

    return queries, positive_indices, positive_coffee_batch


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    # df = pd.read_csv(config["preprocessed_data_path"])
    # df["combined_text"] = df["blind assessment"].fillna() + " " + df["bottom line"].fillna("")
    df = config["preprocessed_data"]
    vocabs = config["vocabs"]

    # used by collate to look up coffee data by index
    full_coffee_dataset = CoffeeDataset(df, vocabs)

    # provides thje (query, positive_idx) pairs for training
    train_dataset = TripleTrainingDataset(config["training_data_path"], df)


    train_loader = DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate(batch, full_coffee_dataset)
    )

    # model, loss, optimizer
    model = DualEncoder(vocabs, numerical_dim=len(full_coffee_dataset.numerical_cols)).to(device)
    loss_fn = nn.TripletMarginLoss(margin=config["margin"])

    # no_decay = ["bias", "LayerNorm.weight"]
    # transformer_params = list(model.transformer.named_parameters())
    # head_params = list(model.metadata_encoder.named_parameters()) + list(model.fusion_layer.named_parameters())

    # optimizer = AdamW([
    #     {"params": [p for n,p in transformer_params if not any(nd in n for nd in no_decay)], "lr": 2e-5, "weight_decay": 0.01},
    #     {"params": [p for n,p in transformer_params if any(nd in n for nd in no_decay)],     "lr": 2e-5, "weight_decay": 0.0},
    #     {"params": [p for n,p in head_params],                                               "lr": 1e-3, "weight_decay": 0.01},
    # ])

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    print("Starting training...")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        num_semi_hard = 0
        use_semi_hard_mining = epoch >= config["semi_hard_mining_start_epoch"]
        if use_semi_hard_mining:
            print(f"Epoch {epoch+1}/{config['epochs']} - Using Semi-Hard Negative Mining")
        else:
            print(f"Epoch {epoch+1}/{config['epochs']} - Using In-Batch Negative Mining")

        for queries, positive_indices, positive_coffee_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            for key, val in positive_coffee_batch.items():
                if isinstance(val, torch.Tensor):
                    positive_coffee_batch[key] = val.to(device)
                elif isinstance(val, dict):
                    for k, v in val.items():
                        positive_coffee_batch[key][k] = v.to(device)

            # forward pass
            query_embeddings = model.encode_queries(queries).to(device)
            # debugging
            with torch.no_grad():
                for col in ["countries_extracted", "process", "varietals"]:
                    idxs = positive_coffee_batch["categoricals"][col]
                    max_idx = int(idxs.max().item()) if idxs.numel() else -1
                    num_rows = model.metadata_encoder.__getattr__(f"{col.split('_')[0]}_embed").num_embeddings
                    if max_idx >= num_rows:
                        raise ValueError(
                            f"{col}: max idx {max_idx} >= num_rows {num_rows} "
                            f"(offsets: {positive_coffee_batch['categoricals'][col + '_offsets']})"
                        )

            positive_embeddings = model.encode_coffees(positive_coffee_batch)
            
            # all other positive embeddings are potential negatives
            negative_embeddings = positive_embeddings

            # calculate distances 
            # first calculate the distances between the queries and their positives
            pos_dists = 1 - F.cosine_similarity(query_embeddings, positive_embeddings)

            dist_matrix = 1 - F.cosine_similarity(query_embeddings.unsqueeze(1), negative_embeddings.unsqueeze(0), dim=2)

            if use_semi_hard_mining:
                final_negative_embeddings = []
                
                for i in range(len(queries)):
                    pos_dist = pos_dists[i] 
                    neg_dists = dist_matrix[i]

                    # exclude the positive from the negatives
                    neg_dists[i] = float("inf")

                    # find semi-hard negatives; negatives that are harder than the positive by violate the margin
                    semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + config["margin"])

                    if semi_hard_mask.any():
                        semi_hard_indices = torch.where(semi_hard_mask)[0]
                        rand_idx = semi_hard_indices[random.randint(0, len(semi_hard_indices)-1)]
                        num_semi_hard += 1
                    else:
                        rand_idx = torch.argmin(neg_dists)
                    
                    final_negative_embeddings.append(negative_embeddings[rand_idx])
                
                negative_embeddings = torch.stack(final_negative_embeddings)
            
            else:
                # just pick an essentially random coffee as the negative
                negative_indices = [(i+1) % len(queries) for i in range(len(queries))]
                negative_embeddings = negative_embeddings[negative_indices]

            loss = loss_fn(query_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
        if use_semi_hard_mining:
            print(f"Number of Semi-hard Negatives Used: {num_semi_hard}")
        torch.save(model.state_dict(), f"coffee_model_train_only_vocab_epoch_{epoch+1}.pth")


def load_model_inference(model_path, vocabs, numerical_dim, device):
    """
    Loads a trained model from a .pth file for inference, i.e. evaluation
    """
    model = DualEncoder(vocabs, numerical_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path} and set to evaluation mode.")
    return model


def evaluate(model, test_df, vocabs, training_data_path, device):
    """
    Evaluates a model on the test set using Recall@K
    """
    print("\nStarting Evaluation...")

    test_dataset = CoffeeDataset(test_df, vocabs)

    print("Pre-computing coffee embeddings for the test set...")
    coffee_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Encoding Coffees"):
            text, numericals, categoricals = test_dataset[i]

            # batch of size 1
            coffee_batch = {
                "text": [text],
                "numericals": numericals.unsqueeze(0).to(device),
                "categoricals": {
                    'roast level': categoricals['roast level'].unsqueeze(0).to(device),
                    'test_method': categoricals['test_method'].unsqueeze(0).to(device),
                    'price_tier': categoricals['price_tier'].unsqueeze(0).to(device),
                    'countries_extracted': categoricals['countries_extracted'].to(device),
                    'countries_extracted_offsets': torch.tensor([0], dtype=torch.long).to(device),
                    'process': categoricals['process'].to(device),
                    'process_offsets': torch.tensor([0], dtype=torch.long).to(device),
                    'varietals': categoricals['varietals'].to(device),
                    'varietals_offsets': torch.tensor([0], dtype=torch.long).to(device),
                }
            }
            embedding = model.encode_coffees(coffee_batch)
            coffee_embeddings.append(embedding.cpu().numpy())

    coffee_embeddings = np.vstack(coffee_embeddings)

    # using FAISS index for speedy searching
    print("Building FAISS Index...")
    index = faiss.IndexFlatIP(768)
    faiss.normalize_L2(coffee_embeddings)
    index.add(coffee_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # generate the test queries and evaluate
    print("Generating test queries and evaluating recall...")
    test_queries = []
    test_ground_truth = []
    test_id2idx = {cid: i for i, cid in test_df["id"].items()}

    with open(training_data_path, "r") as f:
        for line in f:
            data=json.loads(line)
            if data["coffee_id"] in test_id2idx:
                correct_idx = test_id2idx[data["coffee_id"]]
                for query in data["queries"]:
                    if len(query.split()) > 4:
                        test_queries.append(query)
                        test_ground_truth.append(correct_idx)

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total_queries = len(test_queries)

    with torch.no_grad():
        for i in tqdm(range(total_queries), desc="Evaluating Queries"):
            query = test_queries[i]
            correct_idx = test_ground_truth[i]

            query_embedding = model.encode_queries([query]).cpu().numpy()
            faiss.normalize_L2(query_embedding)

            distances, top_k_indices = index.search(query_embedding, k=10)

            top_k_indices = top_k_indices[0]

            if correct_idx in top_k_indices[:1]:
                hits_at_1 += 1
            if correct_idx in top_k_indices[:5]:
                hits_at_5 += 1
            if correct_idx in top_k_indices[:10]:
                hits_at_10 += 1

    print(f"Total Test Queries: {total_queries}")
    print(f"Recall@1:  {hits_at_1 / total_queries:.4f}")
    print(f"Recall@5:  {hits_at_5 / total_queries:.4f}")
    print(f"Recall@10: {hits_at_10 / total_queries:.4f}")
   

def build_search_index(model, coffee_df, vocabs, device):
    """Encodes all coffees and builds a searchable FAISS index."""
    print("Building search index for all coffees...")
    # Use the full dataset for the index
    full_dataset = CoffeeDataset(coffee_df, vocabs)
    
    all_coffee_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(len(full_dataset)), desc="Encoding all coffees"):
            text, numericals, categoricals = full_dataset[i]
            
            # Manually create a batch of size 1
            coffee_batch = {
                'text': [text],
                'numericals': numericals.unsqueeze(0).to(device),
                'categoricals': {
                    'roast level': categoricals['roast level'].unsqueeze(0).to(device),
                    'test_method': categoricals['test_method'].unsqueeze(0).to(device),
                    'price_tier': categoricals['price_tier'].unsqueeze(0).to(device),
                    'countries_extracted': categoricals['countries_extracted'].to(device),
                    'countries_extracted_offsets': torch.tensor([0], dtype=torch.long).to(device),
                    'process': categoricals['process'].to(device),
                    'process_offsets': torch.tensor([0], dtype=torch.long).to(device),
                    'varietals': categoricals['varietals'].to(device),
                    'varietals_offsets': torch.tensor([0], dtype=torch.long).to(device),
                }
            }
            embedding = model.encode_coffees(coffee_batch)
            all_coffee_embeddings.append(embedding.cpu().numpy())
            
    all_coffee_embeddings = np.vstack(all_coffee_embeddings)
    
    index = faiss.IndexFlatIP(768)
    faiss.normalize_L2(all_coffee_embeddings)
    index.add(all_coffee_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index


def get_recommendations(query, model, index, coffee_df, top_k=5):
    """Gets top K recommendations for a single query."""
    with torch.no_grad():
        query_embedding = model.encode_queries([query]).cpu().numpy()
        faiss.normalize_L2(query_embedding)
        
        distances, top_k_indices = index.search(query_embedding, k=top_k)
        
        # Get the indices from the search result
        result_indices = top_k_indices[0]
        
        # Return the corresponding rows from the original DataFrame
        return coffee_df.iloc[result_indices]


if __name__ == "__main__":
    PREPROCESSED_PATH = "preprocessed_data.csv"
    TRAINING_DATA_PATH = "training_data.jsonl"
    MODEL_PATH = "coffee_model_simpleadam_epoch_10.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Coffee Data...")
    df = pd.read_csv(PREPROCESSED_PATH)
    df["combined_text"] = df["blind assessment"].fillna("") + " " + df["bottom line"].fillna("")
    
    print("Creating 80/20 train-test split...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=189)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    vocabs = build_all_vocabs(df)

    # config = {
    #     "preprocessed_data": train_df,
    #     "vocabs": vocabs,
    #     "training_data_path": TRAINING_DATA_PATH,
    #     "batch_size": 32,
    #     "learning_rate": 1e-5,
    #     "epochs": 10,
    #     "margin": 0.2,
    #     "semi_hard_mining_start_epoch": 3
    # }

    # train(config)

    model = load_model_inference(MODEL_PATH, vocabs, numerical_dim=10, device=DEVICE)

    # evaluate(model, test_df, vocabs, TRAINING_DATA_PATH, DEVICE)
    
    search_index = build_search_index(model, df, vocabs, DEVICE)

    example_query = "Brazilian espresso with chocolate notes"

    print(f"Recommendations for query: {example_query}")

    recommendations = get_recommendations(example_query, model, search_index, df, top_k=10)

    display_cols = ['url', 'company', 'coffee name', 'roast level', 'process', 'test_method', 'countries_extracted', "flavor_profile", "blind assessment", 'bottom line']
    print(recommendations[display_cols])
    recommendations[display_cols].to_csv("test_recommendations.csv")




