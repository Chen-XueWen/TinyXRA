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

def sinkhorn(cost_matrix, n_iters, epsilon):
    """
    Computes the optimal transport plan using Sinkhorn iterations.
    
    Args:
        cost_matrix: Tensor of shape (batch, num_words, num_prototypes)
    
    Returns:
        T: Transport plan of the same shape.
    """
    log_T = -cost_matrix / epsilon  # Initialize in log domain.
    for _ in range(n_iters):
        # Normalize over the words dimension.
        log_T = log_T - torch.logsumexp(log_T, dim=1, keepdim=True)
        # Normalize over the prototypes dimension.
        log_T = log_T - torch.logsumexp(log_T, dim=2, keepdim=True)
    T = torch.exp(log_T)
    return T
    
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