import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/processed/dataset_clean.csv"
MODEL_PATH = "models/best_model.pkl"


def evaluate():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Categorie", axis=1)
    y = df["Categorie"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy :", acc)
    print(classification_report(y_test, y_pred))

    if acc < 0.70:
        raise ValueError("Accuracy inférieure à 0.70")

    print("Test performance validé : accuracy >= 0.70")


if __name__ == "__main__":
    evaluate()