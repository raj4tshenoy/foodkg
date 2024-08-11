import torch
import torch.nn as nn
from transformers import BertModel

class KeplerModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', entity_vocab_size=10000, relation_vocab_size=500):
        super(KeplerModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.entity_embeddings = nn.Embedding(entity_vocab_size, self.bert.config.hidden_size)
        self.relation_embeddings = nn.Embedding(relation_vocab_size, self.bert.config.hidden_size)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss_fn_mlm = nn.CrossEntropyLoss()
        self.loss_fn_kge = nn.MSELoss()  # Simple loss function for KGE; you can replace it with more sophisticated ones

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, triples=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        logits = self.linear(sequence_output)

        if labels is not None:
            logits = logits.view(-1, self.bert.config.vocab_size)
            labels = labels.view(-1)
            loss_mlm = self.loss_fn_mlm(logits, labels)
        else:
            loss_mlm = None

        if triples is not None:
            triples = triples.to(input_ids.device)  # Ensure triples are on the same device as the input tensors
            head_embeddings = self.get_entity_embeddings(triples[:, 0])
            relation_embeddings = self.get_relation_embeddings(triples[:, 1])
            tail_embeddings = self.get_entity_embeddings(triples[:, 2])
            loss_kge = self.loss_fn_kge(head_embeddings + relation_embeddings, tail_embeddings)
        else:
            loss_kge = None

        return loss_mlm, loss_kge

    def get_entity_embeddings(self, entities):
        return self.entity_embeddings(entities)

    def get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations)

    def save_pretrained(self, save_path):
        self.bert.save_pretrained(save_path)
        torch.save(self.linear.state_dict(), save_path + '/linear.pth')
