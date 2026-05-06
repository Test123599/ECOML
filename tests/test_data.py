import pandas as pd

DATA_PATH = "data/processed/dataset_clean.csv"


def test_columns_exist():
    df = pd.read_csv(DATA_PATH)

    expected_columns = [
        "Poids",
        "Volume",
        "Conductivite",
        "Opacite",
        "Rigidite",
        "Source",
        "Categorie",
    ]

    for col in expected_columns:
        assert col in df.columns


def test_no_nan_after_cleaning():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0