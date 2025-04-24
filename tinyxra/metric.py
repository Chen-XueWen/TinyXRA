import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, kendalltau

def metric(all_preds, all_targets):
    """
    Compute accuracy, F1-macro, confusion matrix, and ranking correlation metrics.
    
    - Discretizes continuous scores into ordinal labels (0, 1, 2).
    - Computes Spearman's and Kendall's rank correlation on continuous scores.
    """
    
    pred_scores = np.array(all_preds)  # Continuous model scores
    true_labels = np.array(all_targets)  # Ground truth labels

    # Compute thresholds based on percentiles (adaptive binning)
    threshold_1 = np.percentile(pred_scores, 30)  # First threshold (30th percentile)
    threshold_2 = np.percentile(pred_scores, 70)  # Second threshold (70th percentile)

    # Convert continuous scores to discrete labels
    pred_labels = np.digitize(pred_scores, bins=[threshold_1, threshold_2])

    # Compute classification metrics
    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, zero_division=1)

    #print(report)

    # Compute Spearman's Rho and Kendall's Tau using **continuous scores**
    spearman_rho, _ = spearmanr(pred_scores, true_labels)
    kendall_tau, _ = kendalltau(pred_scores, true_labels)
    
    return acc, f1_macro, cm, spearman_rho, kendall_tau