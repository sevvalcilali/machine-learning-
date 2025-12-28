import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
)

DATA_PATH = "data/predictions.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    return df


def plot_prediction_distribution(df):
    plt.figure()
    plt.hist(df["proba"], bins=50)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.savefig("reports/prediction_distribution.png")
    plt.close()


def plot_roc_curve(df):
    fpr, tpr, _ = roc_curve(df["y_true"], df["proba"])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("reports/roc_curve.png")
    plt.close()


def plot_precision_recall(df):
    precision, recall, _ = precision_recall_curve(df["y_true"], df["proba"])

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("reports/precision_recall_curve.png")
    plt.close()


def plot_threshold_analysis(df):
    if "y_true" not in df.columns or df["y_true"].isna().all():
        return

    thresholds = [i / 100 for i in range(1, 100)]
    precisions = []
    recalls = []
    f1s = []

    y_true = df["y_true"].astype(int).to_numpy()
    for t in thresholds:
        preds = (df["proba"] >= t).astype(int).to_numpy()
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precisions,
            "recall": recalls,
            "f1": f1s,
        }
    ).to_csv("reports/threshold_metrics.csv", index=False)

    plt.figure()
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision/Recall/F1")
    plt.legend()
    plt.savefig("reports/threshold_analysis.png")
    plt.close()


def run_all_plots():
    df = load_data()
    plot_prediction_distribution(df)
    plot_roc_curve(df)
    plot_precision_recall(df)
    plot_threshold_analysis(df)
