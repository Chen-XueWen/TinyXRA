import torch
import torch.nn as nn
from models.seq_encoder import SeqEncoder
from models.han import HierarchicalAttentionNetworks

class HierarchicalNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.classifier = HierarchicalAttentionNetworks(word_hidden_size=args.word_hidden_size,
                                                            sentence_hidden_size=args.sentence_hidden_size,
                                                            embedding_size=config.hidden_size,
                                                            num_classes=1) # Only output the ranking score
        
    def forward(self, input_ids, attention_masks, tfidf_weights):
        
        last_hidden_state, word_attns = self.encoder(input_ids, attention_masks)
        tfidf_weighted_hidden_state = torch.cat([last_hidden_state, tfidf_weights.unsqueeze(-1)], dim=-1)
        outputs, sentence_attns, document_vector = self.classifier(tfidf_weighted_hidden_state)

        return outputs, word_attns, sentence_attns, document_vector