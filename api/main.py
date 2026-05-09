import os
import json
import joblib
import pandas as pd

from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest

app = FastAPI(title="Eco-Smart Classifier API")

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Nombre total de requêtes API",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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


@app.post("/predict/nlp")
def predict_nlp(data: InputNLP):
    REQUEST_COUNT.inc()

    prediction = modele_nlp.predict([data.texte])[0]

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": "/predict/nlp",
        "texte": data.texte,
        "prediction": str(prediction),
    }

    write_log(log_data)

    return {
        "texte": data.texte,
        "categorie_predite": str(prediction),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")