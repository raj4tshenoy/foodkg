import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class KeplerModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(KeplerModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)  # Corrected output dimension
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]  # Use sequence output for token-level prediction
        logits = self.linear(sequence_output)

        if labels is not None:
            # Flatten the logits and labels for the loss function
            logits = logits.view(-1, self.bert.config.vocab_size)
            labels = labels.view(-1)
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def save_pretrained(self, save_path):
        self.bert.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)