# monitoring/advanced_monitoring.py
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)

PRED_PATH = Path("data/predictions.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

OPERATING_THRESHOLD = float(os.getenv("OPERATING_THRESHOLD", "0.5"))

# Alert eşikleri (isterseniz değiştirin)
MIN_PRECISION = 0.20          # precision SLA
MAX_PSI = 0.20                # drift sinyali
MAX_AUC_DROP_RATIO = 0.90     # bugün AUC, baseline'ın %90 altına düşerse alert


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (PSI) - proba dağılım drift ölçümü."""
    expected = np.clip(expected, 1e-12, 1 - 1e-12)
    actual = np.clip(actual, 1e-12, 1 - 1e-12)

    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0] = 0.0
    quantiles[-1] = 1.0

    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, 1e-12, None)
    act_perc = np.clip(act_perc, 1e-12, None)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing {PRED_PATH}. Run: python src/predict.py")

    df = pd.read_csv(PRED_PATH)

    # kolonları garantiye al
    df = df.dropna(subset=["timestamp", "proba"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["proba"] = pd.to_numeric(df["proba"], errors="coerce")
    df = df.dropna(subset=["proba"])

    # y_true yoksa sadece drift dağılımı yapılabilir (ama sizde var)
    if "y_true" in df.columns:
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    return df


def compute_daily_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["day"] = df["timestamp"].dt.date

    rows = []
    for day, g in df.groupby("day"):
        row = {"day": str(day), "n": int(len(g))}

        if "y_true" in g.columns and g["y_true"].notna().sum() > 0 and g["y_true"].nunique() > 0:
            y = g["y_true"].dropna().astype(int).to_numpy()
            p = g.loc[g["y_true"].notna(), "proba"].to_numpy()

            # AUC için iki sınıf da olmalı
            if len(np.unique(y)) == 2:
                row["roc_auc"] = float(roc_auc_score(y, p))
            else:
                row["roc_auc"] = np.nan

            row["pr_auc"] = float(average_precision_score(y, p))

            preds = (p >= threshold).astype(int)
            row["precision_at_threshold"] = float(precision_score(y, preds, zero_division=0))
            row["recall_at_threshold"] = float(recall_score(y, preds, zero_division=0))
        else:
            row["roc_auc"] = np.nan
            row["pr_auc"] = np.nan
            row["precision_at_threshold"] = np.nan
            row["recall_at_threshold"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("day")


def plot_label_aware_distribution(df: pd.DataFrame) -> None:
    plt.figure()
    if "y_true" in df.columns and df["y_true"].notna().any():
        df2 = df.dropna(subset=["y_true"]).copy()
        df2["y_true"] = df2["y_true"].astype(int)

        plt.hist(df2[df2["y_true"] == 0]["proba"], bins=50, alpha=0.6, label="y=0")
        plt.hist(df2[df2["y_true"] == 1]["proba"], bins=50, alpha=0.6, label="y=1")
        plt.legend()
        plt.title("Prediction Probability Distribution (by label)")
    else:
        plt.hist(df["proba"], bins=50)
        plt.title("Prediction Probability Distribution")

    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.savefig(REPORTS_DIR / "proba_distribution_by_label.png")
    plt.close()


def plot_metrics_over_time(daily: pd.DataFrame, threshold: float) -> None:
    # AUC over time
    if "roc_auc" in daily.columns:
        plt.figure()
        plt.plot(pd.to_datetime(daily["day"]), daily["roc_auc"])
        plt.title("ROC-AUC Over Time")
        plt.xlabel("Day")
        plt.ylabel("ROC-AUC")
        plt.savefig(REPORTS_DIR / "roc_auc_over_time.png")
        plt.close()

    # Precision/Recall over time
    plt.figure()
    plt.plot(pd.to_datetime(daily["day"]), daily["precision_at_threshold"], label=f"Precision@{threshold:.2f}")
    plt.plot(pd.to_datetime(daily["day"]), daily["recall_at_threshold"], label=f"Recall@{threshold:.2f}")
    plt.title(f"Precision/Recall Over Time (threshold={threshold:.2f})")
    plt.xlabel("Day")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(REPORTS_DIR / "precision_recall_over_time.png")
    plt.close()


def compute_drift_and_alerts(df: pd.DataFrame, daily: pd.DataFrame, threshold: float) -> dict:
    # PSI için baseline vs current (basit: en eski %50 vs en yeni %50)
    df_sorted = df.sort_values("timestamp")
    mid = len(df_sorted) // 2
    baseline = df_sorted.iloc[:mid]["proba"].to_numpy()
    current = df_sorted.iloc[mid:]["proba"].to_numpy()
    psi_val = _psi(baseline, current) if len(baseline) > 20 and len(current) > 20 else np.nan

    # AUC drop alert (son gün vs önceki günler ortalaması)
    alerts = []
    if not daily.empty and daily["roc_auc"].notna().sum() >= 2:
        last_auc = daily["roc_auc"].dropna().iloc[-1]
        base_auc = daily["roc_auc"].dropna().iloc[:-1].mean()
        if base_auc > 0 and last_auc < base_auc * MAX_AUC_DROP_RATIO:
            alerts.append(f"ROC-AUC drop: last={last_auc:.3f} baseline={base_auc:.3f}")

    # Precision SLA alert (son gün)
    if not daily.empty and daily["precision_at_threshold"].notna().any():
        last_prec = daily["precision_at_threshold"].dropna().iloc[-1]
        if last_prec < MIN_PRECISION:
            alerts.append(f"Precision SLA violated at {threshold:.2f}: {last_prec:.3f} < {MIN_PRECISION}")

    # PSI alert
    if not np.isnan(psi_val) and psi_val > MAX_PSI:
        alerts.append(f"Prediction drift (PSI) high: {psi_val:.3f} > {MAX_PSI}")

    return {
        "psi": None if np.isnan(psi_val) else float(psi_val),
        "alerts": alerts,
        "operating_threshold": float(threshold),
        "min_precision": MIN_PRECISION,
        "max_psi": MAX_PSI,
    }


def compute_threshold_recommendation(df: pd.DataFrame) -> Optional[dict[str, float]]:
    if "y_true" not in df.columns or df["y_true"].isna().all():
        return None

    y = df["y_true"].dropna().astype(int).to_numpy()
    p = df.loc[df["y_true"].notna(), "proba"].to_numpy()
    if len(np.unique(y)) < 2:
        return None

    thresholds = np.linspace(0.01, 0.99, 99)
    best = None
    for t in thresholds:
        preds = (p >= t).astype(int)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        candidate = {
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if precision >= MIN_PRECISION:
            if best is None or candidate["f1"] > best["f1"]:
                best = candidate
        elif best is None:
            best = candidate
    return best


def main():
    df = load_predictions()
    rec = compute_threshold_recommendation(df)
    threshold = rec["threshold"] if rec is not None else OPERATING_THRESHOLD
    daily = compute_daily_metrics(df, threshold=threshold)

    # çıktıları kaydet
    daily.to_csv(REPORTS_DIR / "daily_metrics.csv", index=False)

    plot_label_aware_distribution(df)
    plot_metrics_over_time(daily, threshold=threshold)

    summary = compute_drift_and_alerts(df, daily, threshold=threshold)
    if rec is not None:
        summary["recommended_threshold"] = rec["threshold"]
        summary["recommended_precision"] = rec["precision"]
        summary["recommended_recall"] = rec["recall"]
        summary["recommended_f1"] = rec["f1"]
        (REPORTS_DIR / "threshold_recommendation.json").write_text(
            json.dumps(rec, indent=2),
            encoding="utf-8",
        )
    (REPORTS_DIR / "monitoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("✅ Advanced monitoring generated:")
    print(" - reports/proba_distribution_by_label.png")
    print(" - reports/roc_auc_over_time.png")
    print(" - reports/precision_recall_over_time.png")
    print(" - reports/daily_metrics.csv")
    print(" - reports/monitoring_summary.json")
    if summary["alerts"]:
        print("⚠️ ALERTS:")
        for a in summary["alerts"]:
            print(" -", a)


if __name__ == "__main__":
    main()
