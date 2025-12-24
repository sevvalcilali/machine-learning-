import pandas as pd

PATH = "data/train.gz"

def main():
    df = pd.read_csv(PATH, compression="gzip", nrows=200_000)

    print("shape:", df.shape)
    print("columns:", list(df.columns))

    if "click" not in df.columns:
        raise ValueError("Dataset'te 'click' kolonu bulunamadı. Yanlış dosya olabilir.")

    print("\nclick distribution:")
    print(df["click"].value_counts(dropna=False))

    feature_cols = [c for c in df.columns if c not in ("id", "click")]
    nunique = df[feature_cols].nunique().sort_values(ascending=False)

    print("\nTop unique columns (first 15):")
    print(nunique.head(15))

if __name__ == "__main__":
    main()

