import gzip
import csv
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


def test_val_pr_auc_written(tmp_path, monkeypatch):
    # Create small synthetic train.gz with required columns
    header = [
        "id",
        "click",
        "site_id",
        "app_id",
        "site_domain",
        "app_domain",
        "device_type",
        "device_conn_type",
    ]

    rows = []
    for i in range(100):
        click = 1 if i % 10 == 0 else 0
        rows.append(
            [
                i,
                click,
                f"s{i % 5}",
                f"a{i % 7}",
                f"sd{i % 11}",
                f"ad{i % 13}",
                f"d{i % 3}",
                f"dc{i % 2}",
            ]
        )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    gzpath = data_dir / "train.gz"
    with gzip.open(gzpath, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Ensure repo src is on sys.path so local imports like `feature_utils` resolve
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    # Load training module by file
    train = SourceFileLoader("train_streaming", str(repo_root / "src" / "train_streaming.py")).load_module()

    # Run in temporary directory so outputs (models/ and metrics/) go to tmp_path
    monkeypatch.chdir(tmp_path)

    # Override argv for CLI-based training
    argv_backup = sys.argv[:]
    sys.argv = [
        "train_streaming.py",
        "--data-path",
        str(gzpath),
        "--chunk-size",
        "30",
        "--max-train-chunks",
        "2",
        "--val-chunks",
        "1",
        "--checkpoint-every",
        "1",
        "--hash-n-features",
        str(2 ** 10),
    ]

    try:
        train.main()
    finally:
        sys.argv = argv_backup

    metrics_file = tmp_path / "metrics" / "metrics.json"
    assert metrics_file.exists(), "metrics/metrics.json should exist after training"

    metrics = json.loads(metrics_file.read_text())
    assert "val_pr_auc" in metrics, "val_pr_auc should be present in metrics.json"
    assert isinstance(metrics["val_pr_auc"], float)
