# import json
# import torch
# from transformers import BertTokenizer
# from kepler_model import KeplerModel
# from knowledge_graph_dataset import Recipe1MDataset
# from trainer import Trainer

# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config = json.load(f)
#     return config

# def main(config_path):
#     # Load configuration
#     print(f"Loading configuration from {config_path}")
#     config = load_config(config_path)
#     print("Loaded configuration: ", config)

#     # Initialize tokenizer and model
#     print("Initializing tokenizer and model...")
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = KeplerModel(pretrained_model_name='bert-base-uncased')
#     print("Tokenizer and model initialized.")

#     # Load datasets
#     print("Loading datasets...")
#     train_dataset = Recipe1MDataset(file_path=config['train_data_path'], tokenizer=tokenizer)
#     valid_dataset = Recipe1MDataset(file_path=config['valid_data_path'], tokenizer=tokenizer)
#     print("Datasets loaded.")
    
#     # Initialize Trainer
#     print("Initializing Trainer...")
#     trainer = Trainer(
#         model=model,
#         dataset=train_dataset,
#         config=config
#     )
#     print("Trainer initialized.")

#     # Train the model
#     trainer.train()

#     # Evaluate the model
#     trainer.evaluate(valid_dataset)

#     # Save the trained model
#     model.save_pretrained(config['save_path'])

# if __name__ == "__main__":
#     config_path = "config.json"  # Replace with your config path if different
#     main(config_path)

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
    # Load configuration
    config = load_config(config_path)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = KeplerModel(pretrained_model_name='bert-base-uncased')

    # Load datasets
    train_dataset = Recipe1MDataset(file_path=config['train_data_path'], tokenizer=tokenizer)
    valid_dataset = Recipe1MDataset(file_path=config['valid_data_path'], tokenizer=tokenizer)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        config=config
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate(valid_dataset)

    # Save the trained model
    model.save_pretrained(config['save_path'])

if __name__ == "__main__":
    config_path = "config.json"  # Replace with your config path if different
    main(config_path)
