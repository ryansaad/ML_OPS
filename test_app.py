import pytest
from fastapi.testclient import TestClient
from main import app, find_model_path

# We use a context manager to ensure the 'lifespan' (model loading) triggers
def test_prediction_logic():
    """Send a dummy request to the API and check the response format."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "This is a test comment"})
        assert response.status_code == 200
        assert "sentiment" in response.json()

def test_api_health():
    """Check if the root endpoint is online."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "Online"}

def test_model_discovery():
    """Verify that the logic can actually locate a model folder."""
    path = find_model_path()
    assert path is not None
    assert "MLmodel" in str(path) or "artifacts" in str(path) or "models" in str(path)






