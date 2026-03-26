import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
 
 
def load_data(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    print("Loaded", len(df), "rows")
    print(df["label"].value_counts().to_string())
    return df
 
 
def plot_confusion_matrix(y_true, y_pred, labels, fig_dir, ts):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(fig_dir, f"{ts}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved confusion matrix to", path)
 
 
def plot_roc_curve(model, X_test, y_test, pos_label, fig_dir, ts):
    scores = model.decision_function(X_test)
    # binary SVC scores are positive for the last class in model.classes_ (alphabetically)
    # if pos_label is the first class (e.g. FAKE < REAL), scores must be negated
    classes = list(model.named_steps["clf"].classes_)
    if scores.ndim == 2:
        scores = scores[:, classes.index(pos_label)]
    elif classes.index(pos_label) == 0:
        scores = -scores
 
    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
 
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (positive = {pos_label})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(fig_dir, f"{ts}_roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved ROC curve to", path)
    return roc_auc
 
 
def plot_pr_curve(model, X_test, y_test, pos_label, fig_dir, ts):
    scores = model.decision_function(X_test)
    classes = list(model.named_steps["clf"].classes_)
    if scores.ndim == 2:
        scores = scores[:, classes.index(pos_label)]
    elif classes.index(pos_label) == 0:
        scores = -scores
 
    precision, recall, _ = precision_recall_curve(y_test, scores, pos_label=pos_label)
    ap = average_precision_score(y_test, scores, pos_label=pos_label)
 
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (positive = {pos_label})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(fig_dir, f"{ts}_pr_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved PR curve to", path)
    return ap
 
 
def save_metrics_log(metrics_dict, output_dir, ts):
    path = os.path.join(output_dir, f"{ts}_metrics.txt")
    with open(path, "w") as f:
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v}\n")
    print("Saved metrics log to", path)


def plot_cv_scores(cv_scores, fig_dir, ts):
    fold_labels = [f"{i+1}{'st' if i==0 else 'nd' if i==1 else 'rd' if i==2 else 'th'} fold" for i in range(len(cv_scores))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fold_labels, cv_scores, color="steelblue", edgecolor="black")
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"mean = {cv_scores.mean():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("CV Accuracy per Fold")
    ax.set_ylim(0.85, 1.0)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(fig_dir, f"{ts}_cv_scores.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved CV scores plot to", path)