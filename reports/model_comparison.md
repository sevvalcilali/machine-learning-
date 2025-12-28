# Model Comparison (Baseline vs Streaming)

Source metrics:
- Baseline: `metrics/metrics_baseline.json`
- Streaming: `metrics/metrics.json`

| Metric | Baseline | Streaming | Better |
|---|---:|---:|---|
| Train rows | 1,000,000 | 1,000,000 | tie |
| Val rows | 100,000 | 100,000 | tie |
| Val AUC | 0.73997 | 0.70144 | Baseline |
| Val PR-AUC | 0.31291 | 0.27560 | Baseline |
| Val LogLoss | 1.22473 | 3.63279 | Baseline (lower) |
| Elapsed seconds | 24.67 | 23.23 | Streaming |

Notes:
- Both runs used feature crosses and the same hash size (1,048,576).
- Streaming run used 10 chunks of 100,000 rows; baseline trained on a single in-memory split.
