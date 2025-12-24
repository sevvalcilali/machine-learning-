import joblib
import gzip
import csv
from pathlib import Path
from importlib.machinery import SourceFileLoader

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import FeatureHasher


def load_predict_module():
    return SourceFileLoader("predict", "src/predict.py").load_module()


def test_predict_ensemble(tmp_path, capsys):
    # Create tiny dataset
    data_path = tmp_path / "train.csv.gz"
    rows = [
        [1, 0, "s1", "a1", "sd1", "ad1", "d1", "dc1"],
        [2, 1, "s2", "a2", "sd2", "ad2", "d2", "dc2"],
    ]
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
    # write gz csv
    with gzip.open(data_path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Build hasher and dummy models
    df = pd.DataFrame(rows, columns=header)
    X_raw = df.drop(columns=["click"])
    y = df["click"].astype(int)

    hasher = FeatureHasher(n_features=2**10, input_type="dict")
    from importlib.machinery import SourceFileLoader as _L
    feature_utils = _L("feature_utils", "src/feature_utils.py").load_module()
    dicts = feature_utils.to_feature_dict(X_raw, add_feature_cross=False)
    X_h = hasher.transform(dicts)

    sgd = DummyClassifier(strategy="uniform")
    nb = DummyClassifier(strategy="prior")
    sgd.fit(X_h, y)
    nb.fit(X_h, y)

    # Save artifact
    artifact_path = tmp_path / "artifact.joblib"
    joblib.dump({"sgd": sgd, "nb": nb, "hasher": hasher}, artifact_path)

    # Load predict module and override constants
    predict = load_predict_module()
    predict.ARTIFACT_PATH = str(artifact_path)
    predict.DATA_PATH = str(data_path)

    # Run main and capture stdout
    predict.main()
    captured = capsys.readouterr()
    assert "id | true_click | predicted_proba" in captured.out
    # ensure two data lines printed
    lines = [l for l in captured.out.splitlines() if "|" in l]
    assert len(lines) >= 3
