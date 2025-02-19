import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, kendalltau

def metric(all_preds, all_targets):
    # Compute metrics using scikit-learn
    all_argmax_preds = list(map(np.argmax, all_preds))
    acc = accuracy_score(all_targets, all_argmax_preds)
    f1_macro = f1_score(all_targets, all_argmax_preds, average='macro')
    # Generate the confusion matrix and classification report
    cm = confusion_matrix(all_targets, all_argmax_preds)
    report = classification_report(all_targets, all_argmax_preds, zero_division=1)

    print(report)
    
    # Compute Spearman's Rho and Kendall's Tau for ordinal ranking correlation
    spearman_rho, _ = spearmanr(all_targets, all_argmax_preds)
    kendall_tau, _ = kendalltau(all_targets, all_argmax_preds)
    
    return acc, f1_macro, cm, spearman_rho, kendall_tau