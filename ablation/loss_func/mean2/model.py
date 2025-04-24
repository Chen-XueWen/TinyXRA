import torch.nn as nn
from models.seq_encoder import SeqEncoder
from models.han import CLSHierarchicalMaxNetworks

class HierarchicalNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.classifier = nn.Linear(312, 1)
        
    def forward(self, input_ids, attention_masks):
        b, s, w = input_ids.shape
        last_hidden_state, word_attns = self.encoder(input_ids, attention_masks)
        last_hidden_state_flat = last_hidden_state.reshape([b, s*w, -1])
        outputs = self.classifier(last_hidden_state_flat.mean(1))
        sentence_attns = 0
        document_vector = 0

        return outputs, word_attns, sentence_attns, document_vector