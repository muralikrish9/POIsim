# Configure CUDA settings for RTX 5080 with CUDA 12.8
import os
import torch

# CUDA settings optimized for memory efficiency
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32768'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_mem_efficient_sdp = True
torch.backends.cuda.enable_flash_sdp = True
torch.cuda.empty_cache()

# Rest of imports
import sys
import logging
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress
from wildjailbreak_loader import WildJailbreakLoader
from jailbreak_detector import JailbreakDetector, TrainingData, PromptNature
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import evaluate
import time

# Add parent directory to path so we can import from classifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA availability and set device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")

device = torch.device("cuda")
logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"CUDA Version: {torch.version.cuda}")
logger.info(f"PyTorch Version: {torch.__version__}")

# Test GPU with a simple operation
test_tensor = torch.randn(10000, 10000).to(device)
start_time = time.time()
result = test_tensor @ test_tensor
end_time = time.time()
logger.info(f"GPU Test: Matrix multiplication took {end_time - start_time:.2f} seconds")
logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

class WildJailbreakDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Optimize tensor creation by using detach().clone()
        item = {key: val[idx].detach().clone() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_dataset(training_data: List[TrainingData], tokenizer, max_length=512):
    """Prepare the dataset for training"""
    # Convert text to string and handle None values
    texts = [str(item.text) if item.text is not None else "" for item in training_data]
    
    # Convert labels to binary (0 for SAFE, 1 for JAILBREAK/HARM)
    labels = [0 if item.label == PromptNature.SAFE.name else 1 for item in training_data]
    
    # Print dataset statistics
    safe_count = sum(1 for label in labels if label == 0)
    unsafe_count = sum(1 for label in labels if label == 1)
    total = len(labels)
    
    logger.info(f"Dataset Statistics:")
    logger.info(f"Total examples: {total}")
    logger.info(f"SAFE examples: {safe_count} ({safe_count/total*100:.2f}%)")
    logger.info(f"UNSAFE examples (JAILBREAK + HARM): {unsafe_count} ({unsafe_count/total*100:.2f}%)")
    
    # Add synthetic SAFE examples if none exist
    if safe_count == 0:
        logger.warning("No SAFE examples found. Adding synthetic SAFE examples.")
        synthetic_safe_texts = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Tell me about the history of the internet.",
            "What are the benefits of exercise?",
            "Explain how a computer works.",
            "What is the weather like today?",
            "How do I make a sandwich?",
            "What is the meaning of life?",
            "Tell me a joke.",
            "How do I learn to code?"
        ]
        texts.extend(synthetic_safe_texts)
        labels.extend([0] * len(synthetic_safe_texts))
        safe_count = len(synthetic_safe_texts)
        total = len(texts)
        logger.info(f"Added {safe_count} synthetic SAFE examples")
    
    # Calculate class weights (handle case where a class has 0 examples)
    if safe_count == 0:
        class_weights = torch.tensor([1.0, 1.0]).to(device)  # Equal weights if no SAFE examples
        logger.warning("No SAFE examples found in the dataset. Using equal class weights.")
    else:
        class_weights = torch.tensor([
            total / (2 * safe_count),  # weight for SAFE class
            total / (2 * unsafe_count)  # weight for UNSAFE class
        ]).to(device)
    
    logger.info(f"Class weights: SAFE={class_weights[0]:.4f}, UNSAFE={class_weights[1]:.4f}")
    
    # Tokenize the texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return WildJailbreakDataset(encodings, labels), class_weights

def train_classifier():
    """Train the classifier using WildJailbreak dataset"""
    console = Console()
    
    # Initialize the loader
    loader = WildJailbreakLoader()
    
    # Load the dataset
    console.print("\n[bold]1. Loading dataset...[/bold]")
    if not loader.load_dataset():
        console.print("[red]Failed to load dataset[/red]")
        return
        
    # Process the dataset
    console.print("\n[bold]2. Processing dataset...[/bold]")
    training_data = loader.process_to_training_data()
    console.print(f"[green]Processed {len(training_data)} examples[/green]")
    
    # Use more training data - 100,000 examples
    training_data = training_data[:100000]  # Increased from 25000
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(
        training_data,
        test_size=0.1,
        random_state=42,
        stratify=[0 if item.label == PromptNature.SAFE.name else 1 for item in training_data]
    )
    
    # Initialize tokenizer and model
    console.print("\n[bold]3. Initializing model...[/bold]")
    model_name = "distilbert-base-uncased"  # Changed to DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification
    ).to(device)
    
    # Verify model is on GPU
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Model memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Prepare datasets with class weights
    console.print("\n[bold]4. Preparing datasets...[/bold]")
    train_dataset, class_weights = prepare_dataset(train_data, tokenizer, max_length=64)
    val_dataset, _ = prepare_dataset(val_data, tokenizer, max_length=64)
    
    # Initialize metrics
    metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate per-class metrics
        metrics = {
            'accuracy': metric.compute(predictions=predictions, references=labels)["accuracy"],
            'weighted_accuracy': np.mean(
                [
                    (predictions[labels == i] == i).mean() 
                    for i in range(2)
                ]
            ),
            'f1': f1_metric.compute(predictions=predictions, references=labels, average='weighted')["f1"],
            'precision': precision_metric.compute(predictions=predictions, references=labels, average='weighted')["precision"],
            'recall': recall_metric.compute(predictions=predictions, references=labels, average='weighted')["recall"]
        }
        
        # Calculate per-class metrics
        for i, class_name in enumerate(['SAFE', 'UNSAFE']):
            class_mask = labels == i
            if np.any(class_mask):
                metrics.update({
                    f'{class_name}_accuracy': (predictions[class_mask] == i).mean(),
                    f'{class_name}_f1': f1_metric.compute(
                        predictions=predictions[class_mask], 
                        references=labels[class_mask], 
                        average='binary'
                    )["f1"],
                    f'{class_name}_precision': precision_metric.compute(
                        predictions=predictions[class_mask], 
                        references=labels[class_mask], 
                        average='binary'
                    )["precision"],
                    f'{class_name}_recall': recall_metric.compute(
                        predictions=predictions[class_mask], 
                        references=labels[class_mask], 
                        average='binary'
                    )["recall"]
                })
        
        return metrics
    
    # Training arguments optimized for imbalanced dataset
    training_args = TrainingArguments(
        output_dir='models/wildjailbreak',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_accuracy",
        greater_is_better=True,
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        learning_rate=2e-5,
        max_grad_norm=1.0,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        tf32=True,
        no_cuda=False,
        seed=42,
        remove_unused_columns=True,
        report_to="tensorboard"  # Add TensorBoard reporting
    )
    
    # Custom trainer with class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        
        def log(self, logs):
            # Add GPU memory usage to logs
            logs['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**2
            logs['gpu_memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**2
            super().log(logs)
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    console.print("\n[bold]5. Training model...[/bold]")
    trainer.train()
    
    # Save the model
    console.print("\n[bold]6. Saving model...[/bold]")
    trainer.save_model("models/wildjailbreak/final")
    tokenizer.save_pretrained("models/wildjailbreak/final")
    
    # Print final metrics
    console.print("\n[bold]7. Final Metrics:[/bold]")
    final_metrics = trainer.evaluate()
    for metric_name, value in final_metrics.items():
        if isinstance(value, float):
            console.print(f"{metric_name}: {value:.4f}")
    
    console.print("\n[bold green]Training completed successfully![/bold green]")
    console.print("Model saved to: models/wildjailbreak/final")
    console.print("Training logs available in: logs/")
    console.print("TensorBoard logs available in: models/wildjailbreak/runs/")

if __name__ == "__main__":
    train_classifier() 