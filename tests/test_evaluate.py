from src.evaluate import (
    evaluate_classification,
    evaluate_regression,
    evaluate_nlp,
)

import pandas as pd


def test_evaluate_functions():
    df = pd.read_csv(
        "data/processed/dataset_clean.csv"
    )

    acc, f1 = evaluate_classification(df)
    r2, rmse = evaluate_regression(df)
    nlp_acc, nlp_f1 = evaluate_nlp(df)

    assert acc >= 0.70
    assert r2 > 0
    assert nlp_acc >= 0.70