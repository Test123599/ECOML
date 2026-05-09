import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

DATA_PATH = "data/processed/dataset_clean.csv"
REPORT_PATH = "drift_report.html"

NUMERIC_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]


def load_evidently():
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        return Report, DataDriftPreset
    except Exception:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        return Report, DataDriftPreset


def create_current_data(reference_data: pd.DataFrame) -> pd.DataFrame:
    current_data = reference_data.copy()

    if "Poids" in current_data.columns:
        current_data["Poids"] = current_data["Poids"] * 1.5

    if "Conductivite" in current_data.columns:
        current_data["Conductivite"] = current_data["Conductivite"] + 5

    if "Volume" in current_data.columns:
        current_data["Volume"] = current_data["Volume"] + 10

    return current_data


def calculate_text_drift(reference_text: pd.Series, current_text: pd.Series) -> float:
    ref_tokens = " ".join(reference_text.fillna("").astype(str)).lower().split()
    cur_tokens = " ".join(current_text.fillna("").astype(str)).lower().split()

    ref_dist = pd.Series(ref_tokens).value_counts(normalize=True)
    cur_dist = pd.Series(cur_tokens).value_counts(normalize=True)

    vocab = sorted(set(ref_dist.index).union(set(cur_dist.index)))

    ref_vector = np.array([ref_dist.get(word, 0) for word in vocab])
    cur_vector = np.array([cur_dist.get(word, 0) for word in vocab])

    if ref_vector.sum() == 0 or cur_vector.sum() == 0:
        return 0.0

    return float(jensenshannon(ref_vector, cur_vector))


def generate_drift_report():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Lance d'abord : python src/prepare_data.py")

    reference_data = pd.read_csv(DATA_PATH)
    current_data = create_current_data(reference_data)

    Report, DataDriftPreset = load_evidently()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(REPORT_PATH)

    print("Rapport drift généré :", REPORT_PATH)

    if "Rapport_Collecte" in reference_data.columns:
        text_score = calculate_text_drift(
            reference_data["Rapport_Collecte"],
            current_data["Rapport_Collecte"],
        )

        print("Jensen-Shannon text drift :", text_score)

        if text_score > 0.3:
            print("Alerte : text drift détecté")
        else:
            print("Text drift non critique")

    return REPORT_PATH


if __name__ == "__main__":
    generate_drift_report()