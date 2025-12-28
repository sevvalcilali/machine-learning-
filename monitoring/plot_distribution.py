import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

LOG_PATH = Path("data/predictions.csv")
OUT_PATH = Path("reports/prediction_distribution.png")

df = pd.read_csv(LOG_PATH)

counts = df["prediction"].value_counts()

plt.bar(counts.index.astype(str), counts.values)
plt.title("Prediction Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

OUT_PATH.parent.mkdir(exist_ok=True)
plt.savefig(OUT_PATH)
plt.show()
