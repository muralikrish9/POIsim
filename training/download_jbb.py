import os
import json
from datasets import load_dataset
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_and_process_dataset() -> None:
    """Download and process the JailbreakBench dataset"""
    try:
        # Download both dataset configurations
        logging.info("Downloading JailbreakBench behaviors dataset...")
        behaviors_dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        
        # Process and save the data
        processed_data = []
        
        # Process harmful examples
        logging.info("Processing harmful examples...")
        for entry in behaviors_dataset['harmful']:
            processed_entry = {
                'goal': entry['Goal'],
                'target': entry['Target'],
                'behavior': entry['Behavior'],
                'category': entry['Category'],
                'source': entry['Source'],
                'is_harmful': True
            }
            processed_data.append(processed_entry)
            
        # Process benign examples
        logging.info("Processing benign examples...")
        for entry in behaviors_dataset['benign']:
            processed_entry = {
                'goal': entry['Goal'],
                'target': entry['Target'],
                'behavior': entry['Behavior'],
                'category': entry['Category'],
                'source': entry['Source'],
                'is_harmful': False
            }
            processed_data.append(processed_entry)
        
        # Save processed data
        output_path = os.path.join('data', 'jbb', 'processed_data.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Dataset processed and saved to {output_path}")
        logging.info(f"Total entries processed: {len(processed_data)}")
        logging.info(f"Harmful entries: {sum(1 for entry in processed_data if entry['is_harmful'])}")
        logging.info(f"Benign entries: {sum(1 for entry in processed_data if not entry['is_harmful'])}")
        
    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    download_and_process_dataset() 