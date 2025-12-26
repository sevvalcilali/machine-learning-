from typing import Dict, Any
import pandas as pd

from src.feature_utils import to_feature_dict


def build_predictor(artifact):
    hasher = artifact.get("hasher")
    if hasher is None:
        raise ValueError("artifact missing 'hasher'")

    use_feature_cross = bool(artifact.get("use_feature_cross", False))
    cross_pairs = artifact.get("cross_pairs")

    if "model" in artifact:
        model = artifact["model"]

        def predict_proba(X_h):
            return model.predict_proba(X_h)[:, 1]

    elif "sgd" in artifact and "nb" in artifact:
        sgd = artifact["sgd"]
        nb = artifact["nb"]

        def predict_proba(X_h):
            return 0.5 * (
                sgd.predict_proba(X_h)[:, 1]
                + nb.predict_proba(X_h)[:, 1]
            )

    elif "models" in artifact:
        models = artifact["models"]

        def predict_proba(X_h):
            probs = [m.predict_proba(X_h)[:, 1] for m in models]
            return sum(probs) / len(probs)

    else:
        raise ValueError("Unsupported artifact format")

    def predict_one(features: Dict[str, Any]) -> float:
        df = pd.DataFrame([features])

        tokens = to_feature_dict(
            df,
            add_feature_cross=use_feature_cross,
            cross_pairs=cross_pairs,
        )

        X_h = hasher.transform(tokens)

        return float(predict_proba(X_h)[0])

    return predict_one
