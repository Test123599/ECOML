import os
import pandas as pd

from src.train import train_classification, train_regression, train_nlp


def test_train_classification_function():
    df = pd.read_csv("data/processed/dataset_clean.csv")
    model = train_classification(df)

    assert model is not None
    assert os.path.exists("models/modele_classification.pkl")


def test_train_regression_function():
    df = pd.read_csv("data/processed/dataset_clean.csv")
    model = train_regression(df)

    assert model is not None
    assert os.path.exists("models/modele_regression.pkl")


def test_train_nlp_function():
    df = pd.read_csv("data/processed/dataset_clean.csv")
    model = train_nlp(df)

    assert model is not None
    assert os.path.exists("models/modele_nlp_tfidf_linearsvc1.pkl")