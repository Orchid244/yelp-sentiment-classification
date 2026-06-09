import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import ID2LABEL



def evaluate_predictions(y_true, y_pred):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }



def plot_confusion(cm, labels, title: str, path: str) -> None:
    cm = np.array(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()