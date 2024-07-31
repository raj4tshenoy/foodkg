# import torch
# from torch.utils.data import DataLoader

# class Trainer:
#     def __init__(self, model, dataset, config):
#         self.model = model
#         self.dataset = dataset
#         self.config = config
#         self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=self.collate_fn)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)

#     def collate_fn(self, batch):
#         input_ids = torch.stack([item[0] for item in batch])
#         attention_masks = torch.stack([item[1] for item in batch])
#         token_type_ids = torch.stack([item[2] for item in batch])
#         return input_ids, attention_masks, token_type_ids

#     def train(self):
#         print(f"Training on device: {self.device}")
#         self.model.train()
#         for epoch in range(self.config['max_epochs']):
#             total_loss = 0
#             for i, batch in enumerate(self.dataloader):
#                 input_ids, attention_mask, token_type_ids = batch
#                 input_ids = input_ids.to(self.device)
#                 attention_mask = attention_mask.to(self.device)
#                 token_type_ids = token_type_ids.to(self.device)
                
#                 self.optimizer.zero_grad()
#                 loss, logits = self.model(input_ids, attention_mask, token_type_ids, labels=input_ids)
#                 loss.backward()
#                 self.optimizer.step()
#                 total_loss += loss.item()

#                 if i % 10 == 0:  # Log every 10 batches
#                     print(f"Epoch [{epoch+1}/{self.config['max_epochs']}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

#             avg_loss = total_loss / len(self.dataloader)
#             print(f"Epoch [{epoch+1}/{self.config['max_epochs']}], Average Loss: {avg_loss:.4f}")

#         print("Training Complete!")

#     def evaluate(self, dataset):
#         dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, collate_fn=self.collate_fn)
#         self.model.eval()
#         total_loss = 0
#         with torch.no_grad():
#             for i, batch in enumerate(dataloader):
#                 input_ids, attention_mask, token_type_ids = batch
#                 input_ids = input_ids.to(self.device)
#                 attention_mask = attention_mask.to(self.device)
#                 token_type_ids = token_type_ids.to(self.device)
                
#                 loss, logits = self.model(input_ids, attention_mask, token_type_ids, labels=input_ids)
#                 total_loss += loss.item()

#         avg_loss = total_loss / len(dataloader)
#         print(f"Validation Loss: {avg_loss:.4f}")
#         return avg_loss

import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=self.collate_fn)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def collate_fn(self, batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_masks = torch.stack([item[1] for item in batch])
        token_type_ids = torch.stack([item[2] for item in batch])
        labels = torch.stack([item[3] for item in batch])
        print(f"Collated batch shapes - Input IDs: {input_ids.shape}, Attention Masks: {attention_masks.shape}, Token Type IDs: {token_type_ids.shape}, Labels: {labels.shape}")
        return input_ids, attention_masks, token_type_ids

    def train(self):
        print(f"Training on device: {self.device}")
        self.model.train()
        for epoch in range(self.config['max_epochs']):
            total_loss = 0
            for i, batch in enumerate(self.dataloader):
                input_ids, attention_mask, token_type_ids = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                
                self.optimizer.zero_grad()
                loss, logits = self.model(input_ids, attention_mask, token_type_ids, labels=input_ids)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if i % 10 == 0:  # Log every 10 batches
                    print(f"Epoch [{epoch+1}/{self.config['max_epochs']}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{self.config['max_epochs']}], Average Loss: {avg_loss:.4f}")

        print("Training Complete!")

    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, collate_fn=self.collate_fn)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask, token_type_ids = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                
                loss, logits = self.model(input_ids, attention_mask, token_type_ids, labels=input_ids)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
