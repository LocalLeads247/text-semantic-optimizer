from fastapi.testclient import TestClient
from app.main import app

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200

def test_optimize_text_endpoint(client, sample_text):
    response = client.post(
        "/api/optimize",
        json={
            "content": sample_text,
            "optimization_level": "medium",
            "preserve_keywords": []
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "original" in data
    assert "optimized" in data
    assert "metrics" in data
    assert "suggestions" in data

def test_invalid_optimization_level(client, sample_text):
    response = client.post(
        "/api/optimize",
        json={
            "content": sample_text,
            "optimization_level": "invalid",
            "preserve_keywords": []
        }
    )
    assert response.status_code == 422  # Validation error

def test_text_too_long(client, long_text):
    response = client.post(
        "/api/optimize",
        json={
            "content": long_text,
            "optimization_level": "medium",
            "preserve_keywords": []
        }
    )
    assert response.status_code == 422  # Validation error