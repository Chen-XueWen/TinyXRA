import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np

class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        
        # Freeze all parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.config = self.bert.config

    def forward(self, input_ids, attention_masks):

        batch_size, num_sentences, seq_length = input_ids.shape
        input_ids_flat = input_ids.reshape(batch_size * num_sentences, seq_length)
        attention_masks_flat = attention_masks.view(batch_size * num_sentences, seq_length)

        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_masks_flat)

        sequence_output_flat = outputs.last_hidden_state
        sequence_output = sequence_output_flat.reshape(batch_size, num_sentences, seq_length, -1)

        return sequence_output
    