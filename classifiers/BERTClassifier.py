from torch import nn
from transformers import BertForSequenceClassification

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels = num_classes,   
            output_attentions = False, 
            output_hidden_states = False
        )

    def forward(self, **x):
        return self.model(**x)

        