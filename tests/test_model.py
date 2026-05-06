import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "data/processed/dataset_clean.csv"
MODEL_PATH = "models/best_model.pkl"


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        [
            {
                "Poids": 12.5,
                "Volume": 4.2,
                "Conductivite": 0.7,
                "Opacite": 0.4,
                "Rigidite": 0.8,
                "Source": "Maison",
            }
        ]
    )

    prediction = model.predict(sample)

    assert prediction is not None
    assert len(prediction) == 1


def test_accuracy_threshold():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Categorie", axis=1)
    y = df["Categorie"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    assert acc >= 0.70