import os
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/dataset_ProjetML_2026.csv"
PROCESSED_PATH = "data/processed/dataset_clean.csv"


def remove_outliers_iqr(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

    return df


def prepare_data():
    df = pd.read_csv(RAW_PATH)

    numeric_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    if "Source" in df.columns:
        df["Source"] = df["Source"].fillna(df["Source"].mode()[0])

    if "Categorie" in df.columns:
        df = df.dropna(subset=["Categorie"])

    df = remove_outliers_iqr(df, numeric_cols)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Données nettoyées sauvegardées dans :", PROCESSED_PATH)


if __name__ == "__main__":
    prepare_data()