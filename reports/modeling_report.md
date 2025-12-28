# Modeling Report (Avazu CTR)

## Overview
This report summarizes the modeling work for CTR prediction on the Avazu dataset, including feature engineering, baseline and streaming training results, and generated monitoring artifacts.

## Dataset
- Source: Avazu CTR `train.gz`
- Target: `click` (0/1)
- High-cardinality features: `device_ip`, `device_id`, `device_model`, `site_id`, `site_domain`, `app_id`

## Feature Engineering
- Hashing trick via `FeatureHasher`
- Hash size: 2^20 (1,048,576)
- Representation: `col=value` -> 1 sparse feature
- Feature crosses enabled:
  - `site_id` x `app_id`
  - `site_domain` x `app_domain`
  - `device_type` x `device_conn_type`

## Baseline Training (Batch)
- Method: Hashing + `SGDClassifier` (log_loss)
- Train rows: 1,000,000
- Validation rows: 100,000
- Metrics:
  - VAL AUC: 0.73997
  - VAL LogLoss: 1.22473
  - VAL PR-AUC: 0.31291
- Artifact: `models/ctr_baseline_hashing.joblib`
- Metrics file: `metrics/metrics_baseline.json`

## Streaming Training (Online)
- Method: chunked read + `partial_fit` (SGDClassifier)
- Train rows: 2,000,000 (20 chunks x 100,000)
- Validation rows: 100,000
- Metrics:
  - VAL AUC: 0.72299
  - VAL LogLoss: 0.61760
  - VAL PR-AUC: 0.34181
- Artifact: `models/ctr_model_hashing.joblib`
- Metrics file: `metrics/metrics.json`

## Baseline vs Streaming (Latest)
| Metric | Baseline | Streaming |
|---|---:|---:|
| Val AUC | 0.73997 | 0.72299 |
| Val PR-AUC | 0.31291 | 0.34181 |
| Val LogLoss | 1.22473 | 0.61760 |
| Train rows | 1,000,000 | 2,000,000 |
| Val rows | 100,000 | 100,000 |

## Monitoring Artifacts
Generated monitoring outputs (from `data/predictions.csv`) are stored under `reports/`:
- `prediction_distribution.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `threshold_analysis.png`
- `threshold_metrics.csv`
- `threshold_recommendation.json`
- `proba_distribution_by_label.png`
- `roc_auc_over_time.png`
- `precision_recall_over_time.png`
- `daily_metrics.csv`
- `monitoring_summary.json`

## Threshold Optimization (Latest)
- Recommended threshold: 0.01
- Precision: 0.4224
- Recall: 0.2500
- F1: 0.3141

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.01 | 0.42243 | 0.24997 | 0.31408 |
| 0.05 | 0.42387 | 0.24861 | 0.31340 |
| 0.10 | 0.42423 | 0.24747 | 0.31259 |
| 0.20 | 0.42499 | 0.24622 | 0.31180 |
| 0.30 | 0.42503 | 0.24577 | 0.31145 |
| 0.50 | 0.42561 | 0.24509 | 0.31105 |
| 0.99 | 0.42750 | 0.24054 | 0.30786 |

## Monitoring Metrics (Latest)
- Day: 2025-12-28
- N: 50,200
- ROC-AUC: 0.72920
- PR-AUC: 0.34162
- Precision@0.01: 0.42243
- Recall@0.01: 0.24997

## Drift & Alerts (Latest)
- PSI: 0.000095
- Alerts: none
- Operating threshold: 0.01

## Threshold Metrics Summary (Latest)
- Best F1: threshold=0.01, precision=0.42243, recall=0.24997, f1=0.31408
- Max precision: threshold=0.99, precision=0.42750, recall=0.24054, f1=0.30786

## Evidence & Code References
- Feature pipeline: `src/feature_utils.py`
- Baseline training: `src/train_baseline.py`
- Streaming training: `src/train_streaming.py`
- Monitoring pipeline: `monitoring/`
