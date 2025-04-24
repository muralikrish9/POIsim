import os
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

def download_model():
    print("Downloading model files...")
    
    # Create the directory if it doesn't exist
    model_dir = Path("classifier/bert_jbb")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model from Hugging Face
    try:
        # You should replace this with your actual model on Hugging Face
        model_name = "bert-base-uncased"  # This is a placeholder. Replace with your actual model
        
        # Download the model and tokenizer
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save them to the correct location
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        
        print(f"Model successfully downloaded to {model_dir}")
        print("You can now use the model for jailbreak detection!")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("Please make sure you have an internet connection and try again.")

if __name__ == "__main__":
    download_model() 