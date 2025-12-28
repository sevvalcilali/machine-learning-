import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc
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
    thresholds = [i / 100 for i in range(1, 100)]
    accuracies = []

    for t in thresholds:
        preds = (df["proba"] >= t).astype(int)
        acc = (preds == df["y_true"]).mean()
        accuracies.append(acc)

    plt.figure()
    plt.plot(thresholds, accuracies)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Threshold vs Accuracy")
    plt.savefig("reports/threshold_analysis.png")
    plt.close()


def run_all_plots():
    df = load_data()
    plot_prediction_distribution(df)
    plot_roc_curve(df)
    plot_precision_recall(df)
    plot_threshold_analysis(df)
