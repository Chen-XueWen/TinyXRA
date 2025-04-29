import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import itertools
import h5py
import os
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS


def collate_fn(batch):

    docs = np.array([ele[0] for ele in batch])
    attention_masks = np.array([ele[1] for ele in batch])
    labels = np.array([ele[2] for ele in batch])

    docs_tensor = torch.tensor(docs, dtype=torch.long)
    attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    output = {"input_ids": docs_tensor,
              "attention_mask": attention_masks_tensor,
              "targets": labels_tensor,
              }
    
    return output

def load_from_hdf5(file_path):
    """
    Loads processed data from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        np.array: Loaded documents.
        np.array: Loaded labels.
    """
    with h5py.File(file_path, "r") as hf:
        docs = hf["docs"][:]
        attn_masks = hf["attn_masks"][:]
        labels = hf["labels"][:]
    return docs, attn_masks, labels


risk_metric_map = {
    "std":0, 
    "skew":1, 
    "kurt":2, 
    "sortino":3
}