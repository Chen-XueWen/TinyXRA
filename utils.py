import torch
import h5py


def collate_fn(batch):

    docs = [ele[0] for ele in batch]
    labels = [ele[1] for ele in batch]

    # Convert to PyTorch tensors
    docs_tensor = torch.tensor(docs, dtype=torch.long)  # Long for embedding lookup
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    output = {"input_ids": docs_tensor,
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
        labels = hf["labels"][:]
    return docs, labels