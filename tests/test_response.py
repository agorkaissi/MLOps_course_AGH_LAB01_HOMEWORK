from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_response_validation_success():
    response = client.post("/predict", json={"text": "This is great"})

    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]

    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], str)
    assert len(data["prediction"]) > 0


def test_invalid_input_returns_pretty_json():
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422
    assert "application/json" in response.headers["content-type"]

    data = response.json()
    assert "detail" in data
    assert isinstance(data["detail"], list)
    assert any(err["loc"][-1] == "text" for err in data["detail"])
