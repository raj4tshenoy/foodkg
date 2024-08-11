import json
import torch
from transformers import BertTokenizer
from kepler_model import KeplerModel
from recipe1m_dataset import Recipe1MDataset
from foodon_graph import load_foodon_graph, get_foodon_triples, map_uris_to_indices
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    token_type_ids = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    return input_ids, attention_masks, token_type_ids, labels

def save_metrics(metrics, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main(config_path):
    config = load_config(config_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = KeplerModel(pretrained_model_name='bert-base-uncased')

    train_dataset = Recipe1MDataset(file_path=config['train_data_path'], tokenizer=tokenizer)
    valid_dataset = Recipe1MDataset(file_path=config['valid_data_path'], tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    foodon_graph = load_foodon_graph()
    foodon_triples = get_foodon_triples(foodon_graph)
    indexed_triples, entity2id, relation2id = map_uris_to_indices(foodon_triples)
    foodon_triples_tensor = torch.tensor(indexed_triples, dtype=torch.long)  # Convert triples to tensor

    metrics = {
        'train_loss': [],
        'val_loss': [],
        'step_train_loss': [],
    }
    
    step = 0
    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                loss_mlm, _ = model(input_ids, attention_mask, token_type_ids, labels=labels)
            scaler.scale(loss_mlm).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss_mlm.item()
            metrics['step_train_loss'].append(loss_mlm.item())

            if step % 100 == 0:
                print(f"Step [{step}/{len(train_loader)*config['max_epochs']}], Loss: {loss_mlm.item():.4f}")
                save_metrics(metrics, os.path.join(config['save_path'], 'metrics.json'))
            
            step += 1

        avg_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_loss)
        print(f"Epoch [{epoch+1}/{config['max_epochs']}], Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(valid_loader):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
                with autocast():
                    loss_mlm, _ = model(input_ids, attention_mask, token_type_ids, labels=labels)
                val_loss += loss_mlm.item()
        avg_val_loss = val_loss / len(valid_loader)
        metrics['val_loss'].append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if epoch % config['save_every'] == 0:
            model.save_pretrained(config['save_path'] + f"epoch_{epoch}")

        # Save metrics at the end of each epoch
        save_metrics(metrics, os.path.join(config['save_path'], 'metrics.json'))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    config_path = "config.json"
    main(config_path)
