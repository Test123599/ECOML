import os
import pandas as pd

DATA_PATH = "data/processed/dataset_clean.csv"

REQUIRED_COLUMNS = [
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


def test_dataset_clean_exists():
    assert os.path.exists(DATA_PATH), "dataset_clean.csv n'existe pas. Lance python src/prepare_data.py"


def test_columns_exist():
    df = pd.read_csv(DATA_PATH)

    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Colonne manquante : {col}"


def test_no_nan_after_cleaning():
    df = pd.read_csv(DATA_PATH)

    assert df["Categorie"].isna().sum() == 0
    assert df["Prix_Revente"].isna().sum() == 0
    assert df["Source"].isna().sum() == 0
    assert df["Rapport_Collecte"].isna().sum() == 0


def test_numeric_columns_no_nan():
    df = pd.read_csv(DATA_PATH)

    numeric_cols = [
        "Poids",
        "Volume",
        "Conductivite",
        "Opacite",
        "Rigidite",
        "Prix_Revente",
    ]

    for col in numeric_cols:
        assert df[col].isna().sum() == 0, f"NaN trouvé dans {col}"


def test_dataset_not_empty():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0