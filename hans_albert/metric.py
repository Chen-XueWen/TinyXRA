import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def metric(all_preds, all_targets):
    # Compute metrics using scikit-learn
    all_argmax_preds = list(map(np.argmax, all_preds))
    acc = accuracy_score(all_targets, all_argmax_preds)
    f1_macro = f1_score(all_targets, all_argmax_preds, average='macro')
    # Generate the confusion matrix and classification report
    cm = confusion_matrix(all_targets, all_argmax_preds)
    report = classification_report(all_targets, all_argmax_preds, zero_division=1)
    print(report)
    
    return acc, f1_macro, cm