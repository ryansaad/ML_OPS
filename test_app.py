import pytest
from fastapi.testclient import TestClient
from main import app, find_model_path

def test_prediction_logic():
    """Send a dummy request to the API and check the response format."""
    with TestClient(app) as client:
        # Testing with a phrase the 20-newsgroups model should know
        response = client.post("/predict", json={"text": "The rocket was launched into orbit around the moon."})
        assert response.status_code == 200
        json_res = response.json()
        assert "sentiment" in json_res
        assert "confidence" in json_res

def test_api_health():
    """Check if the root endpoint is online and shows the correct version."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        # Updated to match your new 2.0 versioning
        assert response.json() == {"status": "Online", "version": "2.0-Refined"}

def test_model_discovery():
    """Verify that the logic can actually locate a model folder."""
    path = find_model_path()
    assert path is not None
