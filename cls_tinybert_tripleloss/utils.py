import numpy as np
import torch
import h5py

def collate_fn(batch):

    docs = np.array([ele[0] for ele in batch])
    attention_masks = np.array([ele[1] for ele in batch])
    labels = np.array([ele[2] for ele in batch])

    docs_tensor = torch.tensor(docs, dtype=torch.long)
    attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    output = {"input_ids": docs_tensor,
              "attention_masks": attention_masks_tensor,
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

def get_triplets(embeddings, labels):
    """
    embeddings: (B, embed_dim)
    labels: (B,)  with values in [0,1,2]
    Returns anchor, positive, negative
    Each is of shape (N, embed_dim), where N <= B
    """
    device = embeddings.device
    anchor_list = []
    positive_list = []
    negative_list = []
    
    # Convert labels to CPU for indexing
    labels_cpu = labels.cpu()
    
    for i in range(len(labels)):
        anchor_label = labels_cpu[i].item()
        
        # Find indices of positives and negatives
        positive_indices = (labels_cpu == anchor_label).nonzero(as_tuple=True)[0]
        negative_indices = (labels_cpu != anchor_label).nonzero(as_tuple=True)[0]
        
        # Remove the anchor itself from the positive candidates
        positive_indices = positive_indices[positive_indices != i]
        
        # We need at least 1 positive and 1 negative
        if len(positive_indices) < 1 or len(negative_indices) < 1:
            continue
        
        # Randomly select 1 positive and 1 negative
        pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
        neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
        
        anchor_list.append(embeddings[i].unsqueeze(0))
        positive_list.append(embeddings[pos_idx])
        negative_list.append(embeddings[neg_idx])
    
    if len(anchor_list) == 0:
        # In a worst-case scenario (e.g., batch too small or all same label),
        # fallback to returning None or zero-length.
        return None, None, None
    
    anchor_tensor = torch.cat(anchor_list, dim=0).to(device)      # (N, embed_dim)
    positive_tensor = torch.cat(positive_list, dim=0).to(device)  # (N, embed_dim)
    negative_tensor = torch.cat(negative_list, dim=0).to(device)  # (N, embed_dim)
    
    return anchor_tensor, positive_tensor, negative_tensor

risk_metric_map = {
    "std":0, 
    "skew":1, 
    "kurt":2, 
    "sortino":3
}