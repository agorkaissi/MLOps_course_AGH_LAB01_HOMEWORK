from fastapi.testclient import TestClient
from unittest.mock import patch
from app import app

client = TestClient(app)


def test_inference_positive():
    with (
        patch("app.transformer_model") as mock_transformer,
        patch("app.classifier_model") as mock_classifier,
    ):
        mock_transformer.encode.return_value = [[0.1, 0.2]]
        mock_classifier.predict.return_value = ["positive"]

        response = client.post("/predict", json={"text": "I love this"})

        assert response.status_code == 200
        assert response.json()["prediction"] == "positive"


def test_inference_negative():
    with (
        patch("app.transformer_model") as mock_transformer,
        patch("app.classifier_model") as mock_classifier,
    ):
        mock_transformer.encode.return_value = [[0.1, 0.2]]
        mock_classifier.predict.return_value = ["negative"]

        response = client.post("/predict", json={"text": "I hate this"})
        assert response.status_code == 200
        assert response.json()["prediction"] == "negative"


def test_inference_neutral():
    with (
        patch("app.transformer_model") as mock_transformer,
        patch("app.classifier_model") as mock_classifier,
    ):
        mock_transformer.encode.return_value = [[0.1, 0.2]]
        mock_classifier.predict.return_value = ["neutral"]

        response = client.post("/predict", json={"text": "This is okay"})
        assert response.status_code == 200
        assert response.json()["prediction"] == "neutral"
