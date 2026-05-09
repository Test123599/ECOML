import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/processed/dataset_clean.csv"
NLP_MODEL_PATH = "models/modele_nlp_tfidf_linearsvc1.pkl"


def test_nlp_model_exists():
    assert os.path.exists(NLP_MODEL_PATH), "Modèle NLP manquant. Lance python src/train.py"


def test_rapport_collecte_column_exists():
    df = pd.read_csv(DATA_PATH)
    assert "Rapport_Collecte" in df.columns


def test_tfidf_vectorization():
    texts = [
        "bouteille plastique recyclable",
        "papier carton recyclé",
        "déchet métallique aluminium",
        "verre transparent collecté",
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    assert X.shape[0] == 4
    assert X.shape[1] > 0


def test_nlp_prediction():
    model = joblib.load(NLP_MODEL_PATH)

    prediction = model.predict(["bouteille transparente recyclable en plastique"])

    assert prediction is not None
    assert len(prediction) == 1