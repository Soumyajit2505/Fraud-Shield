# tests/test_api.py

from fastapi.testclient import TestClient
from api.app import app

# Create a test client for FastAPI
client = TestClient(app)

def test_root_endpoint():
    """
    Test the root endpoint "/" returns correct status and message
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Detection API is running"}

def test_predict_endpoint_valid():
    """
    Test the /predict endpoint with valid transaction data
    """
    sample_data = {"amount": 1000, "type": "transfer"}
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    # prediction should be 0 or 1
    assert response.json()["prediction"] in [0, 1]

def test_predict_endpoint_fraud():
    """
    Test the /predict endpoint with fraudulent transaction
    """
    sample_data = {"amount": 10000, "type": "transfer"}
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert response.json()["prediction"] == 1