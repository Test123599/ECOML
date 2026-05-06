import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/processed/dataset_clean.csv"
MODEL_PATH = "models/best_model.pkl"


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Categorie", axis=1)
    y = df["Categorie"]

    numeric_features = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
    categorical_features = ["Source"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("EcoSmart_Classification")

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(pipeline, name)

            print(name)
            print("Accuracy :", acc)
            print("F1-score :", f1)
            print("-" * 40)

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = pipeline
                best_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("Meilleur modèle :", best_name)
    print("Accuracy :", best_accuracy)
    print("Modèle sauvegardé dans :", MODEL_PATH)


if __name__ == "__main__":
    train()