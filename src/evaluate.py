from sklearn.metrics import roc_auc_score, precision_score, recall_score


def evaluate_model(y_true, y_pred):
    return {
        "roc_auc": roc_auc_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred.round()),
        "recall": recall_score(y_true, y_pred.round()),
    }


def calibration_summary(y_true, y_prob):
    return {
        "notes": "Implement calibration analysis and threshold selection here."
    }
