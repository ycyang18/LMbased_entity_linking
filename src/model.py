
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

# model
def token_wise_CELoss(pooled_logits: torch.Tensor, labels: torch.Tensor, num_labels: int):
    loss_function = nn.CrossEntropyLoss(ignore_index=-100) # ignore -100 labels for padding tokens
    loss = loss_function(pooled_logits.view(-1, num_labels), labels.view(-1))
    return loss

class BertEntityLinking(nn.Module):
    def __init__(self, num_labels):
        super(BertEntityLinking, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(0.1)
        self.predict = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)[0]
        sequence_output = self.dropout(outputs)
        logits = self.predict(sequence_output) # (B, L, N)
        # Inference
        loss = None
        if labels is not None:
            loss = token_wise_CELoss(logits, labels, self.num_labels)
        return loss, logits
