import torch
import torch.nn as nn
import numpy as np

class TFIDF_Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.classifier = nn.Linear(50000, 3)

    def forward(self, input_features):
        
        outputs = self.classifier(input_features)

        return outputs
    
