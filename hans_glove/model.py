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
        #self.encoder = SeqEncoder(args)
        #config = self.encoder.config
        # Load precomputed GloVe embedding matrix
        glove_embeddings = np.load(f"../processed/GLOVE/{args.data_type}/glove_embeddings.npy")
        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.tensor(glove_embeddings, dtype=torch.float32), freeze=True)

        if args.model_type == "HAN":
            self.classifier = HierarchicalAttentionNetworks(word_hidden_size=args.word_hidden_size,
                                                            sentence_hidden_size=args.sentence_hidden_size,
                                                            embedding_size=50,
                                                            num_classes=args.num_classes)
        elif args.model_type == "HOTN":
            self.classifier = HierarchicalOTNetworks(word_hidden_size=args.word_hidden_size,
                                                     sentence_hidden_size=args.sentence_hidden_size,
                                                     embedding_size=50,
                                                     num_classes=args.num_classes,
                                                     n_iters=args.n_iters)

    def forward(self, input_ids):
        
        last_hidden_state = self.embedding_layer(input_ids)
        outputs, word_attns, sentence_attns = self.classifier(last_hidden_state)

        return outputs, word_attns, sentence_attns
    
