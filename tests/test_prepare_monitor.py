import os

from src.prepare_data import prepare_data
from src.monitor import generate_drift_report


def test_prepare_data_function():
    df = prepare_data()

    assert df is not None
    assert len(df) > 0
    assert os.path.exists("data/processed/dataset_clean.csv")


def test_monitor_function():
    report_path = generate_drift_report()

    assert report_path == "drift_report.html"
    assert os.path.exists("drift_report.html")