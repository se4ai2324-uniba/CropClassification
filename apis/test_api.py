from fastapi.testclient import TestClient
from fastapi import FastAPI
from .main import app
import pytest

client = TestClient(app)

def read_root():
    response = client.get("/")
    assert response.status_code == 400
    assert response.json() == {"message": "Welcome to the Crop Classification API!",
                               "name": "Crop Classification API",
                               "version": "0.1.0",
                               "github": "https://github.com/se4ai2324-uniba/CropClassification.git"}

# Test predict endpoint
def test_predict():
    payload = {
        "Nitrogen": 90,
        "Phosphorus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82,
        "pH_Value": 6.5,
        "Rainfall": 202.93
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    assert "predicted" in response.json()
    assert response.json()["input_data"] == payload

# Test Model Performance endpoint
def test_performance():
    response = client.get("/performance/")
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert "precision" in response.json()
    assert "recall" in response.json()
    assert "f1" in response.json()

# Test Summary endpoint
def test_summary():
    response = client.get("/summary/")
    assert response.status_code == 200
    assert "Nitrogen" in response.json()
    assert "Phosphorus" in response.json()
    assert "Potassium" in response.json()
    assert "Temperature" in response.json()
    assert "Humidity" in response.json()
    assert "pH_Value" in response.json()
    assert "Rainfall" in response.json()


if __name__ == "__main__":
    pytest.main()