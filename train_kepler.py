import json
import torch
from transformers import BertTokenizer
from kepler_model import KeplerModel
from knowledge_graph_dataset import Recipe1MDataset
from trainer import Trainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(config_path):
    print("Loading configuration...")
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded.", config)

    print("Initializing tokenizer and model...")
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = KeplerModel(pretrained_model_name='bert-base-uncased')
    print("Tokenizer and model initialized.")

    # Load datasets
    print("Loading datasets...")
    train_dataset = Recipe1MDataset(file_path=config['train_data_path'], tokenizer=tokenizer)
    val_dataset = Recipe1MDataset(file_path=config['valid_data_path'], tokenizer=tokenizer)
    test_dataset = Recipe1MDataset(file_path=config['test_data_path'], tokenizer=tokenizer)
    print("Datasets loaded.")
    
    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    print("Trainer initialized.")

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate(test_dataset)

    # Save the trained model
    model.save_pretrained(config['save_path'])

if __name__ == "__main__":
    torch.cuda.empty_cache()
    config_path = "config.json"  # Replace with your config path if different
    main(config_path)
