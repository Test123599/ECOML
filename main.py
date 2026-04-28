from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model_nlp.pkl")

class InputText(BaseModel):
    texte: str

@app.get("/")
def home():
    return {"message": "API NLP fonctionne"}

@app.post("/predict")
def predict(data: InputText):
    prediction = model.predict([data.texte])[0]

    return {
        "texte": data.texte,
        "prediction": prediction
    }

@app.get("/health")
def health():
    return {"status": "ok"}

from pydantic import BaseModel

class NLPInput(BaseModel):
    rapport_collecte: str

@app.post("/predict")
def predict_nlp(data: NLPInput):
    prediction = model_nlp.predict([data.rapport_collecte])[0]
    proba = model_nlp.predict_proba([data.rapport_collecte])[0]

    return {
        "categorie": prediction,
        "confidence": float(max(proba)),
        "tokens": data.rapport_collecte.lower().split(),
        "tfidf_keywords": []
    }