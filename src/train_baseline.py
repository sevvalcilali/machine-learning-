import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from feature_utils import to_feature_dict

import joblib

try:
    import mlflow
except Exception:  # pragma: no cover - MLflow optional
    mlflow = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def _parse_cross_list(raw: Optional[str]) -> list[tuple[str, str]]:
    if not raw:
        return [
            ("site_id", "app_id"),
            ("site_domain", "app_domain"),
            ("device_type", "device_conn_type"),
        ]
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid cross pair: {item}. Expected format colA:colB")
        pairs.append((parts[0], parts[1]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=os.getenv("AVAZU_TRAIN_GZ", "data/train.gz"))
    parser.add_argument("--train-rows", type=int, default=_env_int("BASELINE_TRAIN_ROWS", 3_000_000))
    parser.add_argument("--val-rows", type=int, default=_env_int("BASELINE_VAL_ROWS", 300_000))
    parser.add_argument("--hash-n-features", type=int, default=_env_int("HASH_N_FEATURES", 2**20))
    parser.add_argument("--use-feature-cross", action="store_true", default=_env_bool("USE_FEATURE_CROSS", True))
    parser.add_argument("--disable-feature-cross", action="store_true")
    parser.add_argument("--cross-list", default=os.getenv("CROSS_LIST", ""))
    parser.add_argument("--seed", type=int, default=_env_int("SEED", 42))
    args = parser.parse_args()

    use_feature_cross = args.use_feature_cross and not args.disable_feature_cross
    cross_list = _parse_cross_list(args.cross_list)

    np.random.seed(args.seed)
    start = time.time()

    total_rows = args.train_rows + args.val_rows
    df = pd.read_csv(args.data_path, compression="gzip", nrows=total_rows)

    train_df = df.iloc[: args.train_rows].copy()
    val_df = df.iloc[args.train_rows : args.train_rows + args.val_rows].copy()

    y_train = train_df["click"].astype(int).to_numpy()
    X_train_raw = train_df.drop(columns=["click"])
    y_val = val_df["click"].astype(int).to_numpy()
    X_val_raw = val_df.drop(columns=["click"])

    hasher = FeatureHasher(n_features=args.hash_n_features, input_type="dict")
    X_train = hasher.transform(
        to_feature_dict(X_train_raw, add_feature_cross=use_feature_cross, cross_pairs=cross_list)
    )
    X_val = hasher.transform(
        to_feature_dict(X_val_raw, add_feature_cross=use_feature_cross, cross_pairs=cross_list)
    )

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-6,
        max_iter=5,
        tol=None,
        class_weight="balanced",
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    ll = log_loss(y_val, val_proba)
    pr = average_precision_score(y_val, val_proba)

    elapsed = time.time() - start

    model_path = "models/ctr_baseline_hashing.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "hasher": hasher,
            "hash_n_features": args.hash_n_features,
            "use_feature_cross": use_feature_cross,
            "cross_pairs": cross_list,
        },
        model_path,
    )

    metrics = {
        "val_auc": float(auc),
        "val_logloss": float(ll),
        "val_pr_auc": float(pr),
        "train_rows": int(args.train_rows),
        "val_rows": int(args.val_rows),
        "val_rows_config": int(args.val_rows),
        "elapsed_seconds": float(elapsed),
        "hash_n_features": int(args.hash_n_features),
        "use_feature_cross": bool(use_feature_cross),
        "cross_pairs": cross_list,
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if mlflow is not None:
        mlflow.set_experiment("avazu_ctr_baseline")
        with mlflow.start_run(run_name="ctr_baseline") as run:
            mlflow.log_params(
                {
                    "hash_n_features": args.hash_n_features,
                    "use_feature_cross": use_feature_cross,
                    "cross_list": cross_list,
                    "train_rows": args.train_rows,
                    "val_rows": args.val_rows,
                    "seed": args.seed,
                    "model_type": "sgd_classifier",
                    "rebalancing": "class_weight_balanced",
                }
            )
            mlflow.log_metrics({"val_auc": float(auc), "val_logloss": float(ll), "val_pr_auc": float(pr)})
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact("metrics/metrics_baseline.json", artifact_path="metrics")
            metrics["mlflow_run_id"] = run.info.run_id
            with open("metrics/metrics_baseline.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    print("\n=== BASELINE ===")
    print(f"train_rows: {args.train_rows}")
    print(f"val_rows: {args.val_rows}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"VAL AUC: {auc:.5f}")
    print(f"VAL LogLoss: {ll:.5f}")
    print(f"VAL PR-AUC: {pr:.5f}")
    print(f"Model saved: {model_path}")
    print("Metrics saved: metrics/metrics_baseline.json")


if __name__ == "__main__":
    main()
