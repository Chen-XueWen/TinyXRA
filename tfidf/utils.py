import numpy as np
import torch
import h5py
import itertools

def collate_fn(batch):

    docs = np.array([ele[0] for ele in batch])
    labels = np.array([ele[1] for ele in batch])

    docs_tensor = torch.tensor(docs, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    output = {"input_features": docs_tensor,
              "targets": labels_tensor,
              }
    
    return output

def pairwise_ranking_loss(outputs, targets):
    """
    Compute the pairwise ranking loss using the equation:

    L = - Σ (E(d_l, d_j) log P_lj + (1 - E(d_l, d_j)) log (1 - P_lj))

    where:
    P_lj = exp(f(d_l) - f(d_j)) / (1 + exp(f(d_l) - f(d_j)))

    """
    loss = 0
    num_pairs = 0
    
    for i, j in itertools.combinations(range(len(outputs)), 2):
        f_d_l, f_d_j = outputs[i], outputs[j]
        target_l, target_j = targets[i], targets[j]
        
        # Define pairwise label: 1 if i should be ranked higher than j
        E_lj = 1.0 if target_l > target_j else 0.0
        
        # Compute Pℓj
        P_lj = torch.sigmoid(f_d_l - f_d_j)
        
        # Binary cross-entropy loss
        loss += -(E_lj * torch.log(P_lj) + (1 - E_lj) * torch.log(1 - P_lj))
        num_pairs += 1

    return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)

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

risk_metric_map = {
    "std":0, 
    "skew":1, 
    "kurt":2, 
    "sortino":3
}