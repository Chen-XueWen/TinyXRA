import torch
import torch.nn as nn
from models.seq_encoder import SeqEncoder
from models.han import HierarchicalAttentionNetworks
import numpy as np

class HierarchicalNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Load precomputed GloVe embedding matrix
        glove_embeddings = np.load(f"./glove_embeddings.npy")
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.tensor(glove_embeddings, dtype=torch.float32), freeze=False) #Allow fine-tuning of glove embeddings
        self.classifier = HierarchicalAttentionNetworks(word_hidden_size=args.word_hidden_size,
                                                            sentence_hidden_size=args.sentence_hidden_size,
                                                            embedding_size=300,
                                                            num_classes=3)
    def forward(self, input_ids):
        
        last_hidden_state = self.embedding_layer(input_ids)
        outputs, word_attns, sentence_attns = self.classifier(last_hidden_state)

        return outputs, word_attns, sentence_attns
    
