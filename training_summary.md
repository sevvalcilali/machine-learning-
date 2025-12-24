# Training Summary (Avazu CTR)

## Dataset
- Source: Avazu CTR train.gz
- Target: click (0/1)
- High-cardinality features observed: device_ip, device_id, device_model, site_id, site_domain, app_id

## Feature Engineering
- Hashing Trick (FeatureHasher)
- n_features: 2^20 (1,048,576)
- Representation: "col=value" -> 1 sparse features
- Feature cross enabled: true
- Cross pairs:
  - site_id x app_id
  - site_domain x app_domain
  - device_type x device_conn_type

**Choice rationale:** hashed features were chosen for **simplicity, memory-efficiency, and reproducibility** in a streaming setup. Embeddings are a valid alternative for future work (would require additional training infrastructure and checkpointing semantics).

## Baseline (subset training)
- Method: Hashing + SGDClassifier (log_loss)
- Result:
  - train_rows: 3,000,000
  - val_rows: 300,000
  - VAL AUC: 0.67322
  - VAL LogLoss: 0.92241
  - VAL PR-AUC: 0.32173
  - elapsed_seconds: 76.45

## Streaming Training (online learning)
- Method: chunked read + partial_fit (SGDClassifier bagging ensemble)
- Validation: chunk-based split (train chunks first, then validation chunks)
- Final run:
  - chunks_trained: 30
  - trained_rows: 3,000,000
  - val_chunks_used: 3
  - val_rows_used: 300,000
  - val_rows_config: 400,000
  - VAL AUC: 0.69804
  - VAL LogLoss: 0.68589
  - VAL PR-AUC: 0.34711
  - elapsed_seconds: 73.14

## C Deliverable Coverage (Modeling)
- High-cardinality: hashed features with FeatureHasher (n_features logged).
- Feature cross: enabled in feature pipeline; logged to MLflow as use_feature_cross=true and cross_list=[...].
- Ensembles: simple probabilistic ensemble (SGD log_loss + BernoulliNB); ensemble metrics tracked.
- Rebalancing: class_weight="balanced" for SGDClassifier; evaluate via AUC/LogLoss instead of accuracy.
- Checkpoints: periodic joblib checkpoints to allow resume via RESUME_FROM.
- MLflow logging: params (hash size, cross list, class_weight, ensemble), metrics (AUC/LogLoss), model artifact.

## C Evidence Quick List
- Code: src/train_streaming.py (hashing, cross, ensemble, rebalancing, checkpoints, MLflow logging)
- Code: src/feature_utils.py (feature cross tokens)
- Artifacts: models/ctr_ensemble_hashing.joblib, models/checkpoints/ckpt_chunk_*.joblib
- Metrics: metrics/metrics.json (val_auc, val_logloss, val_pr_auc)
- MLflow runs: mlruns/* (params, metrics, model artifact)

## Workflow Orchestration (Prefect)
- Tool choice: Prefect is selected for flexible, dynamic workflows that can fail fast and retry around data-driven retraining loops.
- DAG steps: train_streaming -> read run_id -> validate_and_register (PR-AUC gate) -> MLflow Model Registry (Staging).
- Flow entrypoint: src/prefect_flow.py
- Run: python src/prefect_flow.py

## Artifacts
- Model artifact: models/ctr_ensemble_hashing.joblib (contains ensemble + hasher)
- Metrics: metrics/metrics.json
