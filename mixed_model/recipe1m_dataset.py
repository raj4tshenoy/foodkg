import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

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
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        recipe = self.data[idx]
        title = recipe['title']
        ingredients = ' '.join([ingredient['text'] for ingredient in recipe['ingredients']])
        instructions = ' '.join([instruction['text'] for instruction in recipe['instructions']])
        text = f"{title} [SEP] {ingredients} [SEP] {instructions}"

        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        token_type_ids = tokens['token_type_ids'].squeeze()

        return input_ids, attention_mask, token_type_ids, input_ids  # Using input_ids as labels for MLM
