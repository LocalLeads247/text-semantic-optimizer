from fastapi.testclient import TestClient
from app.main import app
from app.models import TextInput

client = TestClient(app)

def test_optimize_text_valid_input():
    text_input = TextInput(
        content="This is a sample text for testing.",
        optimization_level="medium",
        preserve_keywords=[]
    )
    response = client.post("/api/optimize", json=text_input.dict())
    assert response.status_code == 200
    assert "original" in response.json()
    assert "optimized" in response.json()
    assert "metrics" in response.json()
    assert "suggestions" in response.json()

def test_optimize_text_too_long_input():
    text_input = TextInput(
        content="a" * 10001,
        optimization_level="medium",
        preserve_keywords=[]
    )
    response = client.post("/api/optimize", json=text_input.dict())
    assert response.status_code == 400
    assert "error" in response.json()
    assert "Input text exceeds maximum length of 10000 characters" in response.json()["error"]

def test_optimize_text_too_short_input():
    text_input = TextInput(
        content="",
        optimization_level="medium",
        preserve_keywords=[]
    )
    response = client.post("/api/optimize", json=text_input.dict())
    assert response.status_code == 400
    assert "error" in response.json()
    assert "Input text cannot be empty" in response.json()["error"]

def test_optimize_text_invalid_optimization_level():
    text_input = TextInput(
        content="This is a sample text for testing.",
        optimization_level="invalid",
        preserve_keywords=[]
    )
    response = client.post("/api/optimize", json=text_input.dict())
    assert response.status_code == 400
    assert "error" in response.json()
    assert "Invalid optimization level: invalid" in response.json()["error"]
