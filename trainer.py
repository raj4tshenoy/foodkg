import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time
import numpy as np
import os
import json

class Trainer:
    def __init__(self, model, dataset, val_dataset, config):
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.config = config
        self.dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=self.collate_fn)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scaler = GradScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.accumulation_steps = 4
        self.training_metrics = {'loss': [], 'lr': [], 'grad_norm': []}
        self.validation_metrics = {'loss': [], 'cosine_similarity': []}

        # Create a directory for saving checkpoints and metrics
        self.checkpoint_dir = config['save_path']
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def collate_fn(self, batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_masks = torch.stack([item[1] for item in batch])
        token_type_ids = torch.stack([item[2] for item in batch])
        labels = torch.stack([item[3] for item in batch])
        return input_ids, attention_masks, token_type_ids, labels

    def train(self):
        print(f"Training on device: {self.device}")
        self.model.train()
        best_val_loss = float('inf')

        for epoch in range(self.config['max_epochs']):
            start_time = time.time()
            total_loss = 0
            self.optimizer.zero_grad()

            for i, batch in enumerate(self.dataloader):
                input_ids, attention_mask, token_type_ids, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    loss, logits = self.model(input_ids, attention_mask, token_type_ids, labels=labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                total_loss += loss.item()

                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.config['max_epochs']}], Step [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.dataloader)
            self.training_metrics['loss'].append(avg_loss)
            self.training_metrics['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.training_metrics['grad_norm'].append(self._get_grad_norm())

            # Evaluate on validation set
            val_loss = self.evaluate(self.val_dataset)

            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{self.config['max_epochs']}], Average Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")

            # Save metrics
            self.save_metrics()

            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

        print("Training Complete!")

    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, collate_fn=self.collate_fn)
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask, token_type_ids, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    loss, logits = self.model(input_ids, attention_mask, token_type_ids, labels=labels)
                
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_metrics(self):
        metrics = {
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics
        }
        with open(os.path.join(self.checkpoint_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved.")

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}_val_loss_{val_loss:.4f}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}.")

    def _get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
