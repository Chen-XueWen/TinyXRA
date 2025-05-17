import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, kendalltau

def metric_crossentropy(all_preds, all_targets):
    """
    Compute accuracy, F1-macro, confusion matrix, and ranking correlation metrics.
    
    - Discretizes continuous scores into ordinal labels (0, 1, 2).
    - Computes Spearman's and Kendall's rank correlation on continuous scores.
    """
    
    true_labels = np.array(all_targets)  # Ground truth labels

    probs = torch.nn.functional.softmax(torch.tensor(all_preds), dim=1).float()
    indices = torch.arange(3).float()  # [0, 1, 2]

    pred_labels = np.array(torch.argmax(probs, dim=1))   # Discrete predicted labels
    risk_scores = np.array(torch.matmul(probs, indices)) # Continuous model scores

    # Compute classification metrics
    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, zero_division=1)

    print(report)

    # Compute Spearman's Rho and Kendall's Tau using **continuous scores**
    spearman_rho, _ = spearmanr(risk_scores, true_labels)
    kendall_tau, _ = kendalltau(risk_scores, true_labels)
    
    return acc, f1_macro, cm, spearman_rho, kendall_tau