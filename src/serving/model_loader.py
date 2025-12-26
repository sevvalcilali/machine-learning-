import os
import joblib

DEFAULT_MODEL_PATH = "models/ctr_baseline_hashing.joblib"

def load_artifact():
    """
    Loads the trained model artifact from disk.

    The model path can be overridden with the MODEL_PATH environment variable.
    """
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    return joblib.load(model_path)
