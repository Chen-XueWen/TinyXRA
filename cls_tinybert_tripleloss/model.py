import torch.nn as nn
from models.seq_encoder import SeqEncoder
from models.han import CLSHierarchicalAttentionNetworks

class HierarchicalNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.classifier = CLSHierarchicalAttentionNetworks(sentence_hidden_size=args.sentence_hidden_size,
                                                           embedding_size=config.hidden_size,
                                                           num_classes=3)
        
    def forward(self, input_ids, attention_masks):
        
        last_hidden_state, word_attns = self.encoder(input_ids, attention_masks)
        outputs, sentence_attns, document_vector = self.classifier(last_hidden_state)

        return outputs, word_attns, sentence_attns, document_vector