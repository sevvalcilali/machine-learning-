from pathlib import Path
import pandas as pd

LOG_PATH = Path("data/predictions.csv")

def main():
    df = pd.read_csv(LOG_PATH)

    # y_true hiç yoksa (boşsa) normal: accuracy hesaplanamaz
    if "y_true" not in df.columns or df["y_true"].isna().all() or (df["y_true"].astype(str).str.strip() == "").all():
        print("y_true yok -> accuracy şu an hesaplanamaz. (Gerçek etiket gelince hesaplanacak)")
        return

    df2 = df[df["y_true"].astype(str).str.strip() != ""].copy()
    df2["prediction"] = df2["prediction"].astype(int)
    df2 = df2.dropna(subset=["y_true"])
    df2["y_true"] = df2["y_true"].astype(int)



    acc = (df2["prediction"] == df2["y_true"]).mean()
    print("Accuracy:", round(acc, 4), f"({len(df2)} örnek)")

if __name__ == "__main__":
    main()
