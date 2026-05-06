from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Eco-Smart Classifier API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

modele_classification = joblib.load(os.path.join(MODELS_DIR, "modele_classification.pkl"))
modele_regression = joblib.load(os.path.join(MODELS_DIR, "modele_regression.pkl"))
modele_nlp = joblib.load(os.path.join(MODELS_DIR, "modele_nlp_tfidf_linearsvc1.pkl"))


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


@app.get("/")
def home():
    return {"message": "API Eco-Smart Classifier fonctionne correctement"}


@app.post("/predict/classification")
def predict_classification(data: InputManuel):
    df = pd.DataFrame([data.dict()])
    df_class = df[["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]]

    prediction = modele_classification.predict(df_class)[0]

    return {"categorie_predite": str(prediction)}


@app.post("/predict/regression")
def predict_regression(data: InputManuel):
    df = pd.DataFrame([data.dict()])

    if df["Categorie"].isna().iloc[0]:
        df_class = df[["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source"]]
        categorie_predite = modele_classification.predict(df_class)[0]
        df["Categorie"] = categorie_predite

    df_reg = df[["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source", "Categorie"]]

    prix = modele_regression.predict(df_reg)[0]

    return {
        "categorie_utilisee": str(df["Categorie"].iloc[0]),
        "prix_revente_predit": round(float(prix), 3)
    }


@app.post("/predict/nlp")
def predict_nlp(data: InputNLP):
    prediction = modele_nlp.predict([data.texte])[0]

    return {
        "texte": data.texte,
        "categorie_predite": str(prediction)
    }