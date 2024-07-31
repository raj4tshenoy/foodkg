# import json
# import torch
# from torch.utils.data import Dataset

# class Recipe1MDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=512):
#         self.data = self.load_data(file_path)
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def load_data(self, file_path):
#         data = []
#         with open(file_path, 'r') as f:
#             for line in f:
#                 item = json.loads(line)
#                 data.append(item)
#         return data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         recipe = self.data[idx]
        
#         # Prepare text data
#         title = recipe['title']
#         ingredients = ' '.join([ingredient['text'] for ingredient in recipe['ingredients']])
#         instructions = ' '.join([instruction['text'] for instruction in recipe['instructions']])
#         text = f"{title} [SEP] {ingredients} [SEP] {instructions}"

#         # Tokenize
#         tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

#         # Extract relevant fields
#         input_ids = tokens['input_ids'].squeeze()
#         attention_mask = tokens['attention_mask'].squeeze()
#         token_type_ids = tokens['token_type_ids'].squeeze()

#         return input_ids, attention_mask, token_type_ids

import json
import torch
from torch.utils.data import Dataset

class Recipe1MDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        print(f"Loaded {len(data)} records from {file_path}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recipe = self.data[idx]
        
        # Prepare text data
        title = recipe['title']
        ingredients = ' '.join([ingredient['text'] for ingredient in recipe['ingredients']])
        instructions = ' '.join([instruction['text'] for instruction in recipe['instructions']])
        text = f"{title} [SEP] {ingredients} [SEP] {instructions}"

        # Tokenize
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        # Extract relevant fields
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        token_type_ids = tokens['token_type_ids'].squeeze()

        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Mask shape: {attention_mask.shape}")
        print(f"Token Type IDs shape: {token_type_ids.shape}")

        return input_ids, attention_mask, token_type_ids, input_ids  # Returning input_ids as labels for self-supervised learning
