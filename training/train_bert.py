import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set device and enable CUDA optimizations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class JailbreakDataset(Dataset):
    """Dataset class for jailbreak detection"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(data_path: str) -> tuple:
    """Load and preprocess the dataset"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract texts and create labels
    texts = []
    labels = []
    
    for entry in data:
        # Use both goal and target for training
        texts.append(entry['goal'])
        texts.append(entry['target'])
        
        # Use is_harmful flag for labels
        labels.append(1 if entry['is_harmful'] else 0)
        labels.append(1 if entry['is_harmful'] else 0)

    return texts, labels

def train_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 3,
    learning_rate: float = 2e-5
) -> None:
    """Train the BERT model"""
    # Move model to device
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if device.type == 'cuda':
        model.gradient_checkpointing_enable()
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Clear CUDA cache periodically
            if device.type == 'cuda' and progress_bar.n % 10 == 0:
                torch.cuda.empty_cache()
            
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Average validation loss: {avg_val_loss:.4f}")

        # Save best model and tokenizer
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = 'classifier/bert_jbb'
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Saved new best model and tokenizer to {output_dir}")

def main():
    # Use CPU for now
    device = torch.device('cpu')
    logging.info(f"Using device: {device}")

    # Load and preprocess data
    data_path = os.path.join('data', 'jbb', 'processed_data.json')
    texts, labels = load_and_preprocess_data(data_path)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    # Create datasets
    train_dataset = JailbreakDataset(train_texts, train_labels, tokenizer)
    val_dataset = JailbreakDataset(val_texts, val_labels, tokenizer)

    # Use smaller batch size for CPU
    batch_size = 8

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # No multiprocessing for CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0  # No multiprocessing for CPU
    )

    # Train model
    train_model(model, tokenizer, train_loader, val_loader, device)

if __name__ == "__main__":
    main() 