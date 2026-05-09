import os
import json
import joblib
import pandas as pd

from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

app = FastAPI(title="Eco-Smart Classifier API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Nombre total de requêtes API",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_clean.csv")

CLASS_MODEL_PATH = os.path.join(MODELS_DIR, "modele_classification.pkl")
REG_MODEL_PATH = os.path.join(MODELS_DIR, "modele_regression.pkl")
NLP_MODEL_PATH = os.path.join(MODELS_DIR, "modele_nlp_tfidf_linearsvc1.pkl")

modele_classification = joblib.load(CLASS_MODEL_PATH)
modele_regression = joblib.load(REG_MODEL_PATH)
modele_nlp = joblib.load(NLP_MODEL_PATH)


class InputManuel(BaseModel):
    Poids: float
    Volume: float
    Conductivite: float
    Opacite: float
    Rigidite: float
    Source: str
    Categorie: str | None = None


class InputNLP(BaseModel):
    texte: str


def write_log(data: dict):
    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, "predictions.jsonl")

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


@app.get("/")
def home():
    REQUEST_COUNT.inc()
    return {"message": "API Eco-Smart Classifier fonctionne correctement"}


@app.post("/predict/classification")
def predict_classification(data: InputManuel):
    REQUEST_COUNT.inc()

    input_data = data.model_dump()
    df = pd.DataFrame([input_data])

    df_class = df[
        ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]
    ]

    prediction = modele_classification.predict(df_class)[0]

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": "/predict/classification",
        "input": input_data,
        "prediction": str(prediction),
    }

    write_log(log_data)

    return {"categorie_predite": str(prediction)}


@app.post("/predict/regression")
def predict_regression(data: InputManuel):
    REQUEST_COUNT.inc()

    input_data = data.model_dump()
    df = pd.DataFrame([input_data])

    if df["Categorie"].isna().iloc[0] or df["Categorie"].iloc[0] is None:
        df_class = df[
            ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]
        ]
        categorie_predite = modele_classification.predict(df_class)[0]
        df["Categorie"] = categorie_predite

    df_reg = df[
        [
            "Poids",
            "Volume",
            "Conductivite",
            "Opacite",
            "Rigidite",
            "Source",
            "Categorie",
        ]
    ]

    prix = modele_regression.predict(df_reg)[0]

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": "/predict/regression",
        "input": input_data,
        "categorie_utilisee": str(df["Categorie"].iloc[0]),
        "prix_revente_predit": round(float(prix), 3),
    }

    write_log(log_data)

    return {
        "categorie_utilisee": str(df["Categorie"].iloc[0]),
        "prix_revente_predit": round(float(prix), 3),
    }
def clean_text(text):
    import re

    text = str(text).lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_nlp_prediction(text):
    text = clean_text(text)

    plastique_words = [
        "plastique", "polymere", "polymère", "pet", "pvc",
        "bouteille legere", "bouteille légère", "compressible",
        "souple", "emballage alimentaire", "produit cosmetique",
        "produit cosmétique", "supermarche", "supermarché"
    ]

    verre_words = [
        "verre", "bocal", "vitre", "fragile", "transparent lourd",
        "bouteille cassee", "bouteille cassée", "recipient lourd",
        "récipient lourd", "confiture","cassante"
    ]

    metal_words = [
        "metal", "métal", "metallique", "métallique",
        "aluminium", "canette", "conserve", "conducteur",
        "objet lourd conducteur", "emballage industriel",
        "oxyde", "oxydé"
    ]

    papier_words = [
        "papier", "carton", "feuille", "journal", "livre",
        "imprime", "imprimé", "bureau", "fibreux",
        "administration", "support leger", "support léger",
        "lecture", "colis"
    ]

    if any(w in text for w in metal_words):
        return "Métal"

    if any(w in text for w in papier_words):
        return "Papier"

    if any(w in text for w in verre_words):
        return "Verre"

    if any(w in text for w in plastique_words):
        return "Plastique"

    return None

@app.post("/predict/nlp")
def predict_nlp(data: InputNLP):
    REQUEST_COUNT.inc()

    texte_clean = clean_text(data.texte)

    prediction_keywords = keyword_nlp_prediction(texte_clean)

    if prediction_keywords is not None:
        prediction = prediction_keywords
        methode = "keywords"
    else:
        prediction = modele_nlp.predict([texte_clean])[0]
        methode = "modele_nlp"

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": "/predict/nlp",
        "texte": data.texte,
        "texte_clean": texte_clean,
        "prediction": str(prediction),
        "methode": methode,
    }

    write_log(log_data)

    return {
        "texte": data.texte,
        "texte_clean": texte_clean,
        "categorie_predite": str(prediction),
        "methode": methode,
    }
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/dashboard/clusters")
def dashboard_clusters():
    REQUEST_COUNT.inc()

    df = pd.read_csv(DATA_PATH)

    numeric_cols = [
        "Poids",
        "Volume",
        "Conductivite",
        "Opacite",
        "Rigidite",
    ]

    X = df[numeric_cols].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    result = []

    for i in range(len(df)):
        result.append(
            {
                "id": int(i),
                "Poids": float(df.iloc[i]["Poids"]),
                "Volume": float(df.iloc[i]["Volume"]),
                "Conductivite": float(df.iloc[i]["Conductivite"]),
                "Opacite": float(df.iloc[i]["Opacite"]),
                "Rigidite": float(df.iloc[i]["Rigidite"]),
                "Source": str(df.iloc[i]["Source"]),
                "Categorie": str(df.iloc[i]["Categorie"]),
                "Prix_Revente": float(df.iloc[i]["Prix_Revente"]),
                "cluster": int(clusters[i]),
                "pca1": float(X_pca[i, 0]),
                "pca2": float(X_pca[i, 1]),
            }
        )

    return {
        "message": "Données PCA et clusters générées avec succès",
        "n_rows": len(result),
        "data": result,
    }