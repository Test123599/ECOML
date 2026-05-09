import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

DATA_PATH = "data/processed/dataset_clean.csv"
MODELS_DIR = "models"

CLASS_MODEL_PATH = os.path.join(MODELS_DIR, "modele_classification.pkl")
REG_MODEL_PATH = os.path.join(MODELS_DIR, "modele_regression.pkl")
NLP_MODEL_PATH = os.path.join(MODELS_DIR, "modele_nlp_tfidf_linearsvc1.pkl")

NUMERIC_FEATURES = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
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


def train_classification(df: pd.DataFrame):
    X = df[CLASS_FEATURES]
    y = df["Categorie"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["Source"],
            ),
        ]
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment("EcoSmart_Classification")

    best_model = None
    best_f1 = -1
    best_name = None

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
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(pipeline, name)

            print(f"\n===== {name} =====")
            print("Accuracy :", acc)
            print("Precision:", precision)
            print("Recall   :", recall)
            print("F1-score :", f1)

            if f1 > best_f1:
                best_f1 = f1
                best_model = pipeline
                best_name = name

    joblib.dump(best_model, CLASS_MODEL_PATH)

    print("\nMeilleur modèle classification :", best_name)
    print("F1-score :", best_f1)
    print("Sauvegardé :", CLASS_MODEL_PATH)

    return best_model


def train_regression(df: pd.DataFrame):
    X = df[REG_FEATURES]
    y = df["Prix_Revente"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["Source", "Categorie"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=42,
                    verbose=0,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    mlflow.set_experiment("EcoSmart_Regression")

    with mlflow.start_run(run_name="CatBoostRegressor"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mlflow.log_param("model_name", "CatBoostRegressor")
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "CatBoostRegressor")

        print("\n===== Régression CatBoost =====")
        print("R2   :", r2)
        print("RMSE :", rmse)

    joblib.dump(model, REG_MODEL_PATH)
    print("Sauvegardé :", REG_MODEL_PATH)

    return model


def train_nlp(df: pd.DataFrame):
    X = df["Rapport_Collecte"].fillna("texte non disponible")
    y = df["Categorie"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words=None,
                    ngram_range=(1, 2),
                    max_features=5000,
                ),
            ),
            ("model", LinearSVC()),
        ]
    )

    mlflow.set_experiment("EcoSmart_NLP")

    with mlflow.start_run(run_name="TFIDF_LinearSVC"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_param("model_name", "TFIDF_LinearSVC")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "TFIDF_LinearSVC")

        print("\n===== NLP TF-IDF + LinearSVC =====")
        print("Accuracy :", acc)
        print("F1-score :", f1)

    joblib.dump(model, NLP_MODEL_PATH)
    print("Sauvegardé :", NLP_MODEL_PATH)

    return model


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    train_classification(df)
    train_regression(df)
    train_nlp(df)

    print("\nTous les modèles sont sauvegardés dans le dossier models.")


if __name__ == "__main__":
    train_all()