import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_classification():
    payload = {
        "Poids": 10.5,
        "Volume": 3.2,
        "Conductivite": 0.6,
        "Opacite": 0.5,
        "Rigidite": 0.8,
        "Source": "Maison",
    }

    response = client.post("/predict/classification", json=payload)

    assert response.status_code == 200
    assert "categorie_predite" in response.json()


def test_predict_regression_without_category():
    payload = {
        "Poids": 10.5,
        "Volume": 3.2,
        "Conductivite": 0.6,
        "Opacite": 0.5,
        "Rigidite": 0.8,
        "Source": "Maison",
    }

    response = client.post("/predict/regression", json=payload)

    assert response.status_code == 200
    assert "categorie_utilisee" in response.json()
    assert "prix_revente_predit" in response.json()


def test_predict_regression_with_category():
    payload = {
        "Poids": 10.5,
        "Volume": 3.2,
        "Conductivite": 0.6,
        "Opacite": 0.5,
        "Rigidite": 0.8,
        "Source": "Maison",
        "Categorie": "Plastique",
    }

    response = client.post("/predict/regression", json=payload)

    assert response.status_code == 200
    assert "categorie_utilisee" in response.json()
    assert "prix_revente_predit" in response.json()


def test_predict_nlp():
    payload = {
        "texte": "bouteille transparente recyclable en plastique"
    }

    response = client.post("/predict/nlp", json=payload)

    assert response.status_code == 200
    assert "categorie_predite" in response.json()


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "api_requests_total" in response.text