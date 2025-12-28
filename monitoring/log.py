import csv
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("data/predictions.csv")

def ensure_file():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "prediction", "proba", "y_true"])

def log_prediction(prediction, proba=None, y_true=None):
    ensure_file()
    ts = datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, prediction, proba, y_true])
