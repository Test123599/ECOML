import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/dataset_clean.csv"

CLASS_MODEL_PATH = "models/modele_classification.pkl"
REG_MODEL_PATH = "models/modele_regression.pkl"
NLP_MODEL_PATH = "models/modele_nlp_tfidf_linearsvc1.pkl"

CLASS_FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]

REG_FEATURES = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Source",
    "Categorie",
]


def evaluate_classification(df: pd.DataFrame):
    model = joblib.load(CLASS_MODEL_PATH)

    X = df[CLASS_FEATURES]
    y = df["Categorie"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== Évaluation Classification =====")
    print("Accuracy :", acc)
    print("F1-score :", f1)
    print(classification_report(y_test, y_pred, zero_division=0))

    if acc < 0.70:
        raise ValueError("Accuracy classification inférieure à 0.70")

    return acc, f1


def evaluate_regression(df: pd.DataFrame):
    model = joblib.load(REG_MODEL_PATH)

    X = df[REG_FEATURES]
    y = df["Prix_Revente"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n===== Évaluation Régression =====")
    print("R2   :", r2)
    print("RMSE :", rmse)

    return r2, rmse


def evaluate_nlp(df: pd.DataFrame):
    model = joblib.load(NLP_MODEL_PATH)

    X = df["Rapport_Collecte"].fillna("texte non disponible")
    y = df["Categorie"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n===== Évaluation NLP =====")
    print("Accuracy :", acc)
    print("F1-score :", f1)

    return acc, f1


def evaluate_all():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Lance d'abord : python src/prepare_data.py")

    for path in [CLASS_MODEL_PATH, REG_MODEL_PATH, NLP_MODEL_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError("Lance d'abord : python src/train.py")

    df = pd.read_csv(DATA_PATH)

    evaluate_classification(df)
    evaluate_regression(df)
    evaluate_nlp(df)

    print("\nÉvaluation terminée avec succès.")


if __name__ == "__main__":
    evaluate_all()