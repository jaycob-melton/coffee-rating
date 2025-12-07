import pandas as pd
import torch
import random
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from src.config import CROSS_ENCODER_DATASET, CE_ARCHITECTURE, CE_TRAIN_PARAMS, MODEL_SAVE_PATH
from src.models.evaluate import calculate_relevance, calculate_ndcg, build_relevance_ground_truth
from src.models.model_utils import load_vocabs, load_cross_encoder
from src.models.utils import CrossEncoderDataset, CrossEncoderCollater

random.seed(189)

def train_test_split(dataset, val_ratio=0.1):
    """
    Splits the dataset ensuring that all pairs belonging to a specific query 
    stay together in either Train or Val.
    """
    # Group indices by query
    query2indices = {}
    for idx, sample in enumerate(dataset.pairs):
        q = sample['query']
        if q not in query2indices:
            query2indices[q] = []
        query2indices[q].append(idx)
    
    # Split the unique queries
    unique_queries = list(query2indices.keys())
    random.shuffle(unique_queries)
    
    split_point = int(len(unique_queries) * (1 - val_ratio))
    train_queries = set(unique_queries[:split_point])
    
    train_indices = []
    val_indices = []
    
    # 3. Flatten back into indices
    for q, indices in query2indices.items():
        if q in train_queries:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)
            
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def validate(model, val_loader, device, loss_fn):
    """
    Runs evaluation on the validation set.
    """
    model.eval() 
    total_loss = 0.0
    
    with torch.no_grad(): 
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.squeeze(-1)
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train(train_dataset, val_dataset):
    # better to perform tokenization in training script to avoid multiprocessing issues
    tokenizer = AutoTokenizer.from_pretrained(CE_ARCHITECTURE)
    print("Loaded Tokenizer Successfully")
    
    collator = CrossEncoderCollater(tokenizer)
    print("Loaded Collator Successfully")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CE_TRAIN_PARAMS["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CE_TRAIN_PARAMS["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=4
    )
    
    print("Loaded DataLoaders Successfully")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
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
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, device, loss_fn)
        
        print(f"Epoch {epoch+1}/{CE_TRAIN_PARAMS['num_epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH + f"{epoch+1}.pth")
        else:
            print(f"  Validation loss did not improve (Best: {best_val_loss:.4f}).")
      
    
  
if __name__ == "__main__":
    dataset = CrossEncoderDataset(data_path=str(CROSS_ENCODER_DATASET))
    print("Loaded Dataset Successfully")
    train_dataset, val_dataset = train_test_split(dataset, val_ratio=0.1)
    print("Performed Train-Val Split Successfully")
    train(train_dataset, val_dataset)