# VERSION 2
import torch
import argparse
import json
from transformers import BertTokenizer
from kepler_model import KeplerModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_model_and_embeddings(model_path, entity2id_path, embedding_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    
    # The correct sizes
    entity_vocab_size = 182776
    relation_vocab_size = 59
    
    # Initialize the model with the correct sizes
    model = KeplerModel(entity_vocab_size=entity_vocab_size, relation_vocab_size=relation_vocab_size)
    
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # Load the entity to ID mapping
    with open(entity2id_path, 'r') as f:
        entity2id = json.load(f)

    # Load the entity embeddings
    embeddings = np.load(embedding_path)

    return model, entity2id, embeddings

def get_entity_embedding(entity, model, tokenizer, entity2id, embeddings):
    entity_id = entity2id.get(entity, None)
    if entity_id is not None:
        return embeddings[entity_id]
    else:
        print(f"Entity '{entity}' not found in the knowledge graph.")
        return None

def find_similar_entities(input_entity, model, tokenizer, entity2id, embeddings, top_k=5):
    input_embedding = get_entity_embedding(input_entity, model, tokenizer, entity2id, embeddings)
    if input_embedding is None:
        return []

    similarities = cosine_similarity([input_embedding], embeddings)[0]
    similar_indices = np.argsort(-similarities)[:top_k]

    similar_entities = []
    for idx in similar_indices:
        for entity, entity_id in entity2id.items():
            if entity_id == idx:
                similar_entities.append((entity, similarities[idx]))
                break

    return similar_entities

def main():
    parser = argparse.ArgumentParser(description="Find semantically similar entities using trained knowledge graph embeddings.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--entity2id_path", type=str, required=True, help="Path to the entity2id JSON file.")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the embeddings .npy file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer directory.")
    parser.add_argument("--entity", type=str, required=True, help="Input entity to find similar entities.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top similar entities to return.")

    args = parser.parse_args()

    # Load model, tokenizer, entity2id, and embeddings
    model, entity2id, embeddings = load_model_and_embeddings(args.model_path, args.entity2id_path, args.embedding_path)

    # Load tokenizer from the provided directory
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    # Find similar entities
    similar_entities = find_similar_entities(args.entity, model, tokenizer, entity2id, embeddings, top_k=args.top_k)

    if similar_entities:
        print(f"Top {args.top_k} similar entities to '{args.entity}':")
        for entity, score in similar_entities:
            print(f"Entity: {entity}, Similarity Score: {score:.4f}")
    else:
        print(f"No similar entities found for '{args.entity}'.")

if __name__ == "__main__":
    main()
