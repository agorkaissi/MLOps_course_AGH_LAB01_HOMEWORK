def test_model_loading():
    import joblib
    from sentence_transformers import SentenceTransformer

    classifier = joblib.load("models/classifier.joblib")
    transformer = SentenceTransformer("models/sentence_transformer.model")

    from sklearn.base import BaseEstimator

    assert isinstance(classifier, BaseEstimator)
    assert isinstance(transformer, SentenceTransformer)
