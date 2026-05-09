import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/dataset_clean.csv"

CLASS_MODEL_PATH = "models/modele_classification.pkl"
REG_MODEL_PATH = "models/modele_regression.pkl"

CLASS_FEATURES = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Source",
]

REG_FEATURES = [
    "Poids",
    "Volume",
    "Conductivite",
    "Opacite",
    "Rigidite",
    "Source",
    "Categorie",
]


def test_classification_model_exists():
    assert os.path.exists(CLASS_MODEL_PATH), "Modèle classification manquant. Lance python src/train.py"


def test_regression_model_exists():
    assert os.path.exists(REG_MODEL_PATH), "Modèle regression manquant. Lance python src/train.py"


def test_classification_prediction():
    model = joblib.load(CLASS_MODEL_PATH)

    sample = pd.DataFrame(
        [
            {
                "Poids": 10.5,
                "Volume": 3.2,
                "Conductivite": 0.6,
                "Opacite": 0.5,
                "Rigidite": 0.8,
                "Source": "Maison",
            }
        ]
    )

    prediction = model.predict(sample)

    assert prediction is not None
    assert len(prediction) == 1


def test_regression_prediction():
    model = joblib.load(REG_MODEL_PATH)

    sample = pd.DataFrame(
        [
            {
                "Poids": 10.5,
                "Volume": 3.2,
                "Conductivite": 0.6,
                "Opacite": 0.5,
                "Rigidite": 0.8,
                "Source": "Maison",
                "Categorie": "Plastique",
            }
        ]
    )

    prediction = model.predict(sample)

    assert prediction is not None
    assert len(prediction) == 1
    assert float(prediction[0]) >= 0


def test_accuracy_threshold():
    df = pd.read_csv(DATA_PATH)

    X = df[CLASS_FEATURES]
    y = df["Categorie"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(CLASS_MODEL_PATH)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.70