import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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


class DualEncoder(nn.Module):
    def __init__(self, vocabs, numerical_dim, text_model_name="sentence-transformers/all-mpnet-base-v2",
                 embedding_dim=768, max_length=256):
        super().__init__()
        # Hugging Face model + tokenizer (trainable)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.transformer = AutoModel.from_pretrained(text_model_name)
        self.text_hidden = self.transformer.config.hidden_size  

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