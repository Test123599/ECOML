from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

from prometheus_client import Counter, generate_latest
from fastapi.responses import Response

app = FastAPI(title="Eco-Smart Classifier API")

# ==============================
# Prometheus Metrics
# ==============================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Nombre total de requêtes API"
)

# ==============================
# Chargement des modèles
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

modele_classification = joblib.load(
    os.path.join(MODELS_DIR, "modele_classification.pkl")
)

modele_regression = joblib.load(
    os.path.join(MODELS_DIR, "modele_regression.pkl")
)

modele_nlp = joblib.load(
    os.path.join(MODELS_DIR, "modele_nlp_tfidf_linearsvc1.pkl")
)


# ==============================
# Schémas d'entrée
# ==============================

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


# ==============================
# Routes API
# ==============================

@app.get("/")
def home():
    REQUEST_COUNT.inc()
    return {"message": "API Eco-Smart Classifier fonctionne correctement"}


@app.post("/predict/classification")
def predict_classification(data: InputManuel):
    REQUEST_COUNT.inc()

    df = pd.DataFrame([data.model_dump()])

    df_class = df[
        ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]
    ]

    prediction = modele_classification.predict(df_class)[0]

    return {"categorie_predite": str(prediction)}


@app.post("/predict/regression")
def predict_regression(data: InputManuel):
    REQUEST_COUNT.inc()

    df = pd.DataFrame([data.model_dump()])

    if df["Categorie"].isna().iloc[0]:
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

    return {
        "categorie_utilisee": str(df["Categorie"].iloc[0]),
        "prix_revente_predit": round(float(prix), 3),
    }


@app.post("/predict/nlp")
def predict_nlp(data: InputNLP):
    REQUEST_COUNT.inc()

    prediction = modele_nlp.predict([data.texte])[0]

    return {
        "texte": data.texte,
        "categorie_predite": str(prediction),
    }


# ==============================
# Endpoint Prometheus
# ==============================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")