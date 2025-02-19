import torch
import torch.nn as nn
from models.seq_encoder import SeqEncoder
from models.han import HierarchicalAttentionNetworks
from models.hotn import HierarchicalOTNetworks
import json
import numpy as np

class HierarchicalNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.classifier = HierarchicalAttentionNetworks(word_hidden_size=args.word_hidden_size,
                                                        sentence_hidden_size=args.sentence_hidden_size,
                                                        embedding_size=config.hidden_size,
                                                        num_classes=3)
        
    def forward(self, input_ids, attention_masks):
        
        last_hidden_state = self.encoder(input_ids, attention_masks)
        outputs, word_attns, sentence_attns = self.classifier(last_hidden_state)

        return outputs, word_attns, sentence_attns