import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from feature_utils import to_feature_dict

import joblib

try:
    import mlflow
    from mlflow import MlflowClient
except Exception:  # pragma: no cover - MLflow optional
    mlflow = None
    MlflowClient = None


@dataclass
class Config:
    data_path: str
    chunk_size: int
    max_train_chunks: int
    val_rows: int
    val_chunks: int
    checkpoint_every: int
    seed: int
    hash_n_features: int
    use_feature_cross: bool
    cross_list: list[tuple[str, str]]
    ensemble_type: str
    n_estimators: int
    rebalancing: str
    metric_gate_pr_auc: float
    resume_from: Optional[str]
    register_name: str
    register_stage: str


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


def _build_models(cfg: Config, class_weight_param: Optional[dict[int, float]]) -> list[SGDClassifier]:
    if cfg.ensemble_type == "single":
        seeds = [cfg.seed]
    else:
        seeds = [cfg.seed + i for i in range(cfg.n_estimators)]

    models = []
    for s in seeds:
        models.append(
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-6,
                class_weight=class_weight_param,
                random_state=s,
            )
        )
    return models


def _train_on_chunk(
    models: list[SGDClassifier],
    X_h,
    y: np.ndarray,
    classes: np.ndarray,
):
    for m in models:
        if not hasattr(m, "classes_"):
            m.partial_fit(X_h, y, classes=classes)
        else:
            m.partial_fit(X_h, y)


def _predict_proba(models: list[SGDClassifier], X_h) -> np.ndarray:
    if len(models) == 1:
        return models[0].predict_proba(X_h)[:, 1]
    probs = [m.predict_proba(X_h)[:, 1] for m in models]
    return np.mean(probs, axis=0)


def _save_checkpoint(
    path: str,
    models: list[SGDClassifier],
    hasher: FeatureHasher,
    cfg: Config,
    chunks_trained: int,
    trained_rows: int,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "hasher": hasher,
            "chunks_trained": chunks_trained,
            "trained_rows": trained_rows,
            "config": cfg.__dict__,
        },
        path,
    )


def _load_checkpoint(path: str) -> tuple[list[SGDClassifier], FeatureHasher, int, int, Config]:
    state = joblib.load(path)
    cfg = Config(**state["config"])
    return state["models"], state["hasher"], state["chunks_trained"], state["trained_rows"], cfg


def _setup_mlflow(cfg: Config):
    if mlflow is None:
        return None
    mlflow.set_experiment("avazu_ctr_streaming")
    run = mlflow.start_run(run_name="ctr_streaming_training")
    mlflow.set_tag("mlflow.source.name", "https://github.com/sevvalcilali/machine-learning-")
    mlflow.log_params(
        {
            "hash_n_features": cfg.hash_n_features,
            "use_feature_cross": cfg.use_feature_cross,
            "cross_list": cfg.cross_list,
            "rebalancing": cfg.rebalancing,
            "chunk_size": cfg.chunk_size,
            "max_train_chunks": cfg.max_train_chunks,
            "val_rows": cfg.val_rows,
            "val_chunks": cfg.val_chunks,
            "checkpoint_every": cfg.checkpoint_every,
            "seed": cfg.seed,
            "model_type": "sgd_classifier",
            "ensemble_type": cfg.ensemble_type,
            "n_estimators": cfg.n_estimators if cfg.ensemble_type != "single" else 1,
        }
    )
    return run


def _maybe_register_model(
    run_id: Optional[str],
    model_path: str,
    cfg: Config,
    val_pr_auc: float,
):
    if mlflow is None or MlflowClient is None or run_id is None:
        return
    if val_pr_auc < cfg.metric_gate_pr_auc:
        return
    try:
        client = MlflowClient()
        name = cfg.register_name
        existing = [m.name for m in client.list_registered_models()]
        if name not in existing:
            client.create_registered_model(name)
        model_uri = f"runs:/{run_id}/model/{os.path.basename(model_path)}"
        mv = client.create_model_version(name=name, source=model_uri, run_id=run_id)
        for _ in range(30):
            v = client.get_model_version(name, mv.version)
            if v.status == "READY":
                break
            time.sleep(1)
        client.transition_model_version_stage(name, mv.version, cfg.register_stage)
    except Exception:
        # Registry may be unavailable; ignore to keep training usable
        return


def _parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=os.getenv("AVAZU_TRAIN_GZ", "data/train.gz"))
    parser.add_argument("--chunk-size", type=int, default=_env_int("CHUNK_SIZE", 200_000))
    parser.add_argument("--max-train-chunks", type=int, default=_env_int("MAX_TRAIN_CHUNKS", 25))
    parser.add_argument("--val-rows", type=int, default=_env_int("VAL_ROWS", 400_000))
    parser.add_argument("--val-chunks", type=int, default=_env_int("VAL_CHUNKS", 0))
    parser.add_argument("--checkpoint-every", type=int, default=_env_int("CHECKPOINT_EVERY", 5))
    parser.add_argument("--seed", type=int, default=_env_int("SEED", 42))
    parser.add_argument("--hash-n-features", type=int, default=_env_int("HASH_N_FEATURES", 2**20))
    parser.add_argument("--use-feature-cross", action="store_true", default=_env_bool("USE_FEATURE_CROSS", True))
    parser.add_argument("--disable-feature-cross", action="store_true")
    parser.add_argument("--cross-list", default=os.getenv("CROSS_LIST", ""))
    parser.add_argument("--ensemble", choices=["bagging_sgd", "single"], default=os.getenv("ENSEMBLE", "bagging_sgd"))
    parser.add_argument("--n-estimators", type=int, default=_env_int("N_ESTIMATORS", 5))
    parser.add_argument("--rebalancing", default=os.getenv("REBALANCING", "class_weight_balanced"))
    parser.add_argument("--metric-gate-pr-auc", type=float, default=float(os.getenv("METRIC_GATE_PR_AUC", "0.20")))
    parser.add_argument("--resume-from", default=os.getenv("RESUME_FROM", ""))
    parser.add_argument("--register-name", default=os.getenv("MODEL_REGISTER_NAME", "avazu_ctr"))
    parser.add_argument("--register-stage", default=os.getenv("MODEL_REGISTER_STAGE", "Staging"))
    args = parser.parse_args()

    use_feature_cross = args.use_feature_cross and not args.disable_feature_cross
    cross_list = _parse_cross_list(args.cross_list)

    if args.val_chunks <= 0:
        val_chunks = max(1, math.ceil(args.val_rows / args.chunk_size))
    else:
        val_chunks = args.val_chunks

    return Config(
        data_path=args.data_path,
        chunk_size=args.chunk_size,
        max_train_chunks=args.max_train_chunks,
        val_rows=args.val_rows,
        val_chunks=val_chunks,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        hash_n_features=args.hash_n_features,
        use_feature_cross=use_feature_cross,
        cross_list=cross_list,
        ensemble_type=args.ensemble,
        n_estimators=args.n_estimators,
        rebalancing=args.rebalancing,
        metric_gate_pr_auc=args.metric_gate_pr_auc,
        resume_from=args.resume_from or None,
        register_name=args.register_name,
        register_stage=args.register_stage,
    )


def main():
    cfg = _parse_args()
    np.random.seed(cfg.seed)

    start = time.time()
    classes = np.array([0, 1], dtype=int)

    hasher = FeatureHasher(n_features=cfg.hash_n_features, input_type="dict")
    trained_rows = 0
    chunks_trained = 0
    models: list[SGDClassifier]

    reader = pd.read_csv(
        cfg.data_path,
        compression="gzip",
        chunksize=cfg.chunk_size,
    )

    if cfg.resume_from:
        models, hasher, chunks_trained, trained_rows, resume_cfg = _load_checkpoint(cfg.resume_from)
        print(f"Resuming from chunk={chunks_trained} rows={trained_rows}")
        # Keep model/feature config from checkpoint to avoid mismatches
        cfg.hash_n_features = resume_cfg.hash_n_features
        cfg.use_feature_cross = resume_cfg.use_feature_cross
        cfg.cross_list = resume_cfg.cross_list
        cfg.ensemble_type = resume_cfg.ensemble_type
        cfg.n_estimators = resume_cfg.n_estimators
        cfg.rebalancing = resume_cfg.rebalancing
    else:
        first_chunk = next(reader, None)
        if first_chunk is None:
            raise ValueError("No data available in training file")
        y_first = first_chunk["click"].astype(int).to_numpy()
        if cfg.rebalancing == "class_weight_balanced":
            cw = compute_class_weight("balanced", classes=classes, y=y_first)
            class_weight_param = {int(c): float(w) for c, w in zip(classes, cw)}
        else:
            class_weight_param = None
        models = _build_models(cfg, class_weight_param)

        X_first_raw = first_chunk.drop(columns=["click"])
        X_first = hasher.transform(
            to_feature_dict(
                X_first_raw,
                add_feature_cross=cfg.use_feature_cross,
                cross_pairs=cfg.cross_list,
            )
        )
        _train_on_chunk(models, X_first, y_first, classes)
        trained_rows += len(first_chunk)
        chunks_trained += 1

    run = _setup_mlflow(cfg)

    # Train chunks
    current_chunk = 0
    last_checkpoint_path = None
    for chunk in reader:
        current_chunk += 1
        if current_chunk <= chunks_trained:
            continue
        if chunks_trained >= cfg.max_train_chunks:
            break

        y = chunk["click"].astype(int).to_numpy()
        X_raw = chunk.drop(columns=["click"])
        X_h = hasher.transform(
            to_feature_dict(
                X_raw,
                add_feature_cross=cfg.use_feature_cross,
                cross_pairs=cfg.cross_list,
            )
        )

        _train_on_chunk(models, X_h, y, classes)

        trained_rows += len(chunk)
        chunks_trained += 1

        if cfg.checkpoint_every > 0 and chunks_trained % cfg.checkpoint_every == 0:
            ckpt_path = f"models/checkpoints/ckpt_chunk_{chunks_trained}.joblib"
            _save_checkpoint(ckpt_path, models, hasher, cfg, chunks_trained, trained_rows)
            last_checkpoint_path = ckpt_path
            if mlflow is not None:
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        if chunks_trained >= cfg.max_train_chunks:
            break

    # Validation chunks (immediately after training chunks)
    val_y: list[int] = []
    val_proba: list[float] = []
    val_rows_used = 0
    val_chunks_used = 0

    for chunk in reader:
        if val_chunks_used >= cfg.val_chunks:
            break
        val_chunks_used += 1
        y = chunk["click"].astype(int).to_numpy()
        X_raw = chunk.drop(columns=["click"])
        X_h = hasher.transform(
            to_feature_dict(
                X_raw,
                add_feature_cross=cfg.use_feature_cross,
                cross_pairs=cfg.cross_list,
            )
        )
        proba = _predict_proba(models, X_h)
        val_y.extend(y.tolist())
        val_proba.extend(proba.tolist())
        val_rows_used += len(chunk)

    val_auc = roc_auc_score(val_y, val_proba) if val_y else float("nan")
    val_ll = log_loss(val_y, val_proba) if val_y else float("nan")
    val_pr = average_precision_score(val_y, val_proba) if val_y else float("nan")

    elapsed = time.time() - start
    run_type = "SMOKE/DEBUG" if trained_rows < 100000 or chunks_trained < 5 else "FINAL"

    model_path = "models/ctr_model_hashing.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "hasher": hasher,
            "hash_n_features": cfg.hash_n_features,
            "use_feature_cross": cfg.use_feature_cross,
            "cross_pairs": cfg.cross_list,
            "ensemble_type": cfg.ensemble_type,
            "n_estimators": cfg.n_estimators if cfg.ensemble_type != "single" else 1,
            "rebalancing": cfg.rebalancing,
        },
        model_path,
    )

    metrics = {
        "val_auc": float(val_auc),
        "val_logloss": float(val_ll),
        "val_pr_auc": float(val_pr),
        "train_rows": int(trained_rows),
        "trained_rows": int(trained_rows),
        "chunks_trained": int(chunks_trained),
        "val_rows": int(val_rows_used),
        "val_rows_used": int(val_rows_used),
        "val_chunks_used": int(val_chunks_used),
        "elapsed_seconds": float(elapsed),
        "hash_n_features": int(cfg.hash_n_features),
        "chunk_size": int(cfg.chunk_size),
        "max_train_chunks": int(cfg.max_train_chunks),
        "val_rows_config": int(cfg.val_rows),
        "val_chunks": int(cfg.val_chunks),
        "use_feature_cross": bool(cfg.use_feature_cross),
        "cross_pairs": cfg.cross_list,
        "rebalancing": cfg.rebalancing,
        "ensemble_type": cfg.ensemble_type,
        "n_estimators": int(cfg.n_estimators if cfg.ensemble_type != "single" else 1),
        "run_type": run_type,
    }

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if last_checkpoint_path is None:
        ckpt_path = f"models/checkpoints/ckpt_chunk_{chunks_trained}.joblib"
        _save_checkpoint(ckpt_path, models, hasher, cfg, chunks_trained, trained_rows)
        last_checkpoint_path = ckpt_path

    if mlflow is not None and run is not None:
        mlflow.log_metrics(
            {
                "val_auc": float(val_auc),
                "val_logloss": float(val_ll),
                "val_pr_auc": float(val_pr),
            }
        )
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact("metrics/metrics.json", artifact_path="metrics")
        if last_checkpoint_path:
            mlflow.log_artifact(last_checkpoint_path, artifact_path="checkpoints")

        run_id = mlflow.active_run().info.run_id
        metrics["mlflow_run_id"] = run_id
        with open("metrics/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        os.makedirs("models", exist_ok=True)
        with open("models/last_run_id.txt", "w", encoding="utf-8") as f:
            f.write(run_id)

        _maybe_register_model(run_id, model_path, cfg, val_pr_auc=val_pr)
        mlflow.end_run()

    print(f"\n=== {run_type} ===")
    print(f"trained_rows: {trained_rows}")
    print(f"chunks_trained: {chunks_trained}")
    print(f"val_rows_config: {cfg.val_rows}")
    print(f"val_rows_used: {val_rows_used}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"VAL AUC: {val_auc:.5f}")
    print(f"VAL LogLoss: {val_ll:.5f}")
    print(f"VAL PR-AUC: {val_pr:.5f}")
    print(f"Model saved: {model_path}")
    print("Metrics saved: metrics/metrics.json")


if __name__ == "__main__":
    main()
