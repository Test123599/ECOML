import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# =====================================================
# Dataset de référence
# =====================================================

reference_data = pd.read_csv("data/raw/dataset_ProjetML_2026.csv")

# =====================================================
# Création automatique d'un dataset drifté
# =====================================================

current_data = reference_data.copy()

# Simulation drift
current_data["Poids"] *= 3
current_data["Conductivite"] += 5
current_data["Volume"] += 10

# =====================================================
# Génération du rapport
# =====================================================

report = Report(metrics=[
    DataDriftPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# =====================================================
# Sauvegarde HTML
# =====================================================

report.save_html("drift_report.html")

print("Rapport drift généré avec succès")