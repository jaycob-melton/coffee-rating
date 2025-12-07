import pandas as pd
import torch
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from src.config import CROSS_ENCODER_DATASET, CE_ARCHITECTURE, CE_TRAIN_PARAMS
from src.models.evaluate import calculate_relevance, calculate_ndcg, build_relevance_ground_truth
from src.models.model_utils import load_vocabs, load_cross_encoder
from src.models.utils import CrossEncoderDataset, CrossEncoderCollater


def train_test_split():
    pass

def train():
    # better to perform tokenization in training script to avoid multiprocessing issues
    tokenizer = AutoTokenizer.from_pretrained(CE_ARCHITECTURE)
    print("Loaded Tokenizer Successfully")
    train_dataset = CrossEncoderDataset(data_path=str(CROSS_ENCODER_DATASET))
    print("Loaded CrossEncoderDataset Successfully")
    print(f"Example: {train_dataset[0]}")
    collator = CrossEncoderCollater(tokenizer)
    print("Loaded Collator Successfully")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CE_TRAIN_PARAMS["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )
    print("Loaded DataLoader Successfully")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_cross_encoder(device=device)
    print("Loaded Model Successfully")
    
    optimizer = AdamW(model.parameters(), lr=CE_TRAIN_PARAMS["lr"], weight_decay=0.01)
    
    total_steps = CE_TRAIN_PARAMS["num_epochs"] * len(train_loader)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    loss_fn = MSELoss()
    
    for epoch in range(CE_TRAIN_PARAMS["num_epochs"]):
        model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CE_TRAIN_PARAMS['num_epochs']}"):
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.squeeze(-1)
            
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
    
    
    
    
  
if __name__ == "__main__":
    train()