import joblib
import pandas as pd
from feature_utils import to_feature_dict


ARTIFACT_PATH = "models/ctr_model_hashing.joblib"
DATA_PATH = "data/train.gz"


def main():
    artifact = joblib.load(ARTIFACT_PATH)
    # Support artifact shapes:
    # 1) legacy single-model: {"model", "hasher"}
    # 2) legacy ensemble: {"sgd", "nb", "hasher"}
    # 3) bagging ensemble: {"models": [..], "hasher"}
    hasher = artifact.get("hasher")
    if hasher is None:
        raise ValueError("artifact missing 'hasher' key")

    use_feature_cross = bool(artifact.get("use_feature_cross", False))
    cross_pairs = artifact.get("cross_pairs")

    if "model" in artifact:
        model = artifact["model"]
        def _predict_proba(X_h):
            return model.predict_proba(X_h)[:, 1]
    elif "sgd" in artifact and "nb" in artifact:
        sgd = artifact["sgd"]
        nb = artifact["nb"]
        def _predict_proba(X_h):
            sgd_proba = sgd.predict_proba(X_h)[:, 1]
            nb_proba = nb.predict_proba(X_h)[:, 1]
            return 0.5 * (sgd_proba + nb_proba)
    elif "models" in artifact:
        models = artifact["models"]
        def _predict_proba(X_h):
            probs = [m.predict_proba(X_h)[:, 1] for m in models]
            return sum(probs) / len(probs)
    else:
        raise ValueError("Unsupported artifact format: expected 'model' or ('sgd' and 'nb')")

    # Örnek veri (ilk 20 satır)
    df = pd.read_csv(DATA_PATH, compression="gzip", nrows=20)

    y_true = df["click"].astype(int).tolist()
    ids = df["id"].tolist()
    X_raw = df.drop(columns=["click"])

    X_h = hasher.transform(
        to_feature_dict(
            X_raw,
            add_feature_cross=use_feature_cross,
            cross_pairs=cross_pairs,
        )
    )
    proba = _predict_proba(X_h).tolist()

    print("id | true_click | predicted_proba")
    for i in range(len(ids)):
        print(f"{ids[i]} | {y_true[i]} | {proba[i]:.6f}")

if __name__ == "__main__":
    main()
