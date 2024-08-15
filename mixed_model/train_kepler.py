import json
import torch
from transformers import BertTokenizer
from kepler_model import KeplerModel
from recipe1m_dataset import Recipe1MDataset
from foodon_graph import load_foodon_graph, get_foodon_triples, map_uris_to_indices
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def calculate_metrics(predictions, labels):
    predictions = predictions.argmax(dim=-1).cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    return precision, recall, f1

# def main(config_path):
#     config = load_config(config_path)
#     print("loaded config:", config)

#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = KeplerModel(pretrained_model_name='bert-base-uncased')
#     print("loaded tokenizer and pretrained model")

#     train_dataset = Recipe1MDataset(file_path=config['train_data_path'], tokenizer=tokenizer)
#     valid_dataset = Recipe1MDataset(file_path=config['valid_data_path'], tokenizer=tokenizer)

#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
#     valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

#     optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
#     scaler = GradScaler()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     foodon_graph = load_foodon_graph()
#     foodon_triples = get_foodon_triples(foodon_graph)
#     indexed_triples, entity2id, relation2id = map_uris_to_indices(foodon_triples)
#     foodon_triples_tensor = torch.tensor(indexed_triples, dtype=torch.long)  # Convert triples to tensor

#     metrics = {
#         'train_loss': [],
#         'val_loss': [],
#         'precision': [],
#         'recall': [],
#         'f1_score': [],
#         'step_train_loss': [],
#     }
    
#     print("starting training")
#     step = 0
#     for epoch in range(config['max_epochs']):
#         model.train()
#         total_loss = 0
#         for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
#             input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)

#             optimizer.zero_grad()
#             with autocast():
#                 loss_mlm, loss_kge = model(input_ids, attention_mask, token_type_ids, labels=labels, triples=foodon_triples_tensor)
#             loss = loss_mlm + loss_kge
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             total_loss += loss.item()
#             metrics['step_train_loss'].append(loss.item())

#             if step % 100 == 0:
#                 print(f"Step [{step}/{len(train_loader)*config['max_epochs']}], Loss: {loss.item():.4f}")
#                 save_metrics(metrics, os.path.join(config['save_path'], 'metrics.json'))
            
#             step += 1

#         avg_loss = total_loss / len(train_loader)
#         metrics['train_loss'].append(avg_loss)
#         print(f"Epoch [{epoch+1}/{config['max_epochs']}], Loss: {avg_loss:.4f}")

#         model.eval()
#         val_loss = 0
#         all_preds = []
#         all_labels = []
#         with torch.no_grad():
#             for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(valid_loader):
#                 input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
#                 with autocast():
#                     loss_mlm, loss_kge = model(input_ids, attention_mask, token_type_ids, labels=labels, triples=foodon_triples_tensor)
#                 val_loss += (loss_mlm + loss_kge).item()
                
#                 # Store predictions and labels for metric calculation
#                 logits = model(input_ids, attention_mask, token_type_ids)[0]
#                 all_preds.append(logits)
#                 all_labels.append(labels)
        
#         avg_val_loss = val_loss / len(valid_loader)
#         metrics['val_loss'].append(avg_val_loss)
#         print(f"Validation Loss: {avg_val_loss:.4f}")

#         # Calculate precision, recall, f1 score
#         all_preds = torch.cat(all_preds, dim=0)
#         all_labels = torch.cat(all_labels, dim=0)
#         precision, recall, f1 = calculate_metrics(all_preds, all_labels)
#         metrics['precision'].append(precision)
#         metrics['recall'].append(recall)
#         metrics['f1_score'].append(f1)
#         print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

#         if epoch % config['save_every'] == 0:
#             model.save_pretrained(config['save_path'] + f"epoch_{epoch}")
#             # Save embeddings and mappings
#             np.save(os.path.join(config['save_path'], 'entity_embeddings.npy'), foodon_triples_tensor.cpu().numpy())
#             with open(os.path.join(config['save_path'], 'entity2id.json'), 'w') as f:
#                 json.dump(entity2id, f)

#         # Save metrics at the end of each epoch
#         save_metrics(metrics, os.path.join(config['save_path'], 'metrics.json'))

#     print("training complete")
    
# if __name__ == "__main__":
#     config_path = "config.json"
#     main(config_path)

def main(config_path):
    print("loading config")
    config = load_config(config_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("loaded config and tokenizer: \n", config)

    # Load and prepare the FoodOn triples
    print("loading FoodOn graph")
    foodon_graph = load_foodon_graph()
    foodon_triples = get_foodon_triples(foodon_graph)
    indexed_triples, entity2id, relation2id = map_uris_to_indices(foodon_triples)
    foodon_triples_tensor = torch.tensor(indexed_triples, dtype=torch.long)
    print("loaded FoodOn graph and mapped URIs to indices")

    # Update model initialization with entity and relation vocab sizes
    print("initializing model")
    model = KeplerModel(
        pretrained_model_name='bert-base-uncased',
        entity_vocab_size=len(entity2id),
        relation_vocab_size=len(relation2id)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    foodon_triples_tensor = foodon_triples_tensor.to(device)  # Ensure triples tensor is on the same device as the model

    train_dataset = Recipe1MDataset(file_path=config['train_data_path'], tokenizer=tokenizer)
    valid_dataset = Recipe1MDataset(file_path=config['valid_data_path'], tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = GradScaler()
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'step_train_loss': [],
    }
    
    print("starting training")
    print("device: ", device)
    # print the timestamp
    import datetime
    print(datetime.datetime.now())
    # clear torch cache
    torch.cuda.empty_cache()
    step = 0
    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                loss_mlm, loss_kge = model(input_ids, attention_mask, token_type_ids, labels=labels, triples=foodon_triples_tensor)
            loss = loss_mlm + loss_kge
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            metrics['step_train_loss'].append(loss.item())

            if step % 100 == 0:
                # print step number and loss with timestamp
                print(datetime.datetime.now())
                print(f"Step [{step}/{len(train_loader)*config['max_epochs']}], Loss: {loss.item():.4f}")
                save_metrics(metrics, os.path.join(config['save_path'], 'metrics.json'))
            
            step += 1

        avg_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_loss)
        print(f"Epoch [{epoch+1}/{config['max_epochs']}], Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(valid_loader):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
                with autocast():
                    loss_mlm, loss_kge = model(input_ids, attention_mask, token_type_ids, labels=labels, triples=foodon_triples_tensor)
                val_loss += (loss_mlm + loss_kge).item()
                
                # Store predictions and labels for metric calculation
                logits = model(input_ids, attention_mask, token_type_ids)[0]
                all_preds.append(logits)
                all_labels.append(labels)
        
        avg_val_loss = val_loss / len(valid_loader)
        metrics['val_loss'].append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Calculate precision, recall, f1 score
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        precision, recall, f1 = calculate_metrics(all_preds, all_labels)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        if epoch % config['save_every'] == 0:
            model.save_pretrained(config['save_path'] + f"epoch_{epoch}")
            # Save embeddings and mappings
            np.save(os.path.join(config['save_path'], 'entity_embeddings.npy'), foodon_triples_tensor.cpu().numpy())
            with open(os.path.join(config['save_path'], 'entity2id.json'), 'w') as f:
                json.dump(entity2id, f)

        # Save metrics at the end of each epoch
        save_metrics(metrics, os.path.join(config['save_path'], 'metrics.json'))

    print("training complete")
    
if __name__ == "__main__":
    # clear torch cache
    torch.cuda.empty_cache()
    config_path = "config.json"
    main(config_path)
