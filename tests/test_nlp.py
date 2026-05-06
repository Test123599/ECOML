from sklearn.feature_extraction.text import TfidfVectorizer


def test_tfidf_vectorization():
    texts = [
        "bouteille plastique recyclable",
        "papier carton recyclé",
        "déchet métallique aluminium",
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    assert X.shape[0] == 3
    assert X.shape[1] > 0