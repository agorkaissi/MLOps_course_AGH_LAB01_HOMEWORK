from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_empty_input():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_missing_input():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_valid_input():
    response = client.post("/predict", json={"text": "Welcome"})
    assert response.status_code == 200
