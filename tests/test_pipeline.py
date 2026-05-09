import os
import pandas as pd

from src.prepare_data import prepare_data
from src.monitor import generate_drift_report


def test_prepare_data():
    df = prepare_data()

    assert df is not None
    assert len(df) > 0
    assert "Categorie" in df.columns


def test_processed_dataset_exists():
    assert os.path.exists(
        "data/processed/dataset_clean.csv"
    )


def test_monitor_generation():
    report_path = generate_drift_report()

    assert os.path.exists(report_path)


def test_dataset_not_empty():
    df = pd.read_csv(
        "data/processed/dataset_clean.csv"
    )

    assert len(df) > 0