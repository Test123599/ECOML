import os
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/dataset_ProjetML_2026.csv"
PROCESSED_PATH = "data/processed/dataset_clean.csv"

NUMERIC_COLS = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Prix_Revente",
]

REQUIRED_COLS = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Prix_Revente",
    "Source",
    "Rapport_Collecte",
    "Categorie",
]


def cap_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing_cols}")

    df = df.drop_duplicates()

    df = df.dropna(subset=["Categorie"])
    df = df.dropna(subset=["Prix_Revente"])

    df["Source"] = df["Source"].fillna(df["Source"].mode()[0])
    df["Rapport_Collecte"] = df["Rapport_Collecte"].fillna("texte non disponible")

    group_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]

    for col in group_cols:
        df[col] = df.groupby("Categorie")[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())

    df = cap_outliers_iqr(df, ["Conductivite", "Prix_Revente"])

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Dataset nettoyé sauvegardé :", PROCESSED_PATH)
    print("Shape :", df.shape)

    return df


if __name__ == "__main__":
    prepare_data()