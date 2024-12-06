import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.processors.style_guide import StyleGuideType

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_grammar_enhancement(client):
    test_text = "The cats runs fast and I saw an cat."
    response = client.post(
        "/enhance/grammar",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == test_text
    assert "issues" in data
    assert isinstance(data["improvement_score"], float)

def test_sentiment_analysis(client):
    test_text = "I am very happy with this product!"
    response = client.post(
        "/analyze/sentiment",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == test_text
    assert isinstance(data["polarity"], float)
    assert isinstance(data["subjectivity"], float)
    assert "emotional_tone" in data

def test_style_analysis(client):
    test_text = "I think this will obviously work better. It's clearly the best solution."
    response = client.post(
        "/analyze/style",
        json={
            "text": test_text,
            "style_guide": StyleGuideType.ACADEMIC.value
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original_text"] == test_text
    assert "issues" in data
    assert isinstance(data["compliance_score"], float)
    assert len(data["issues"]) > 0

def test_style_analysis_missing_guide(client):
    test_text = "Some sample text."
    response = client.post(
        "/analyze/style",
        json={"text": test_text}
    )
    assert response.status_code == 400

def test_combined_analysis(client):
    test_text = "The product is great but the service were terrible."
    response = client.post(
        "/analyze",
        json={
            "text": test_text,
            "style_guide": StyleGuideType.BUSINESS.value
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "grammar" in data
    assert "style" in data
    assert "sentiment" in data
    assert data["style"]["style_guide_type"] == StyleGuideType.BUSINESS.value

def test_combined_analysis_no_style(client):
    test_text = "The product is great but the service were terrible."
    response = client.post(
        "/analyze",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "grammar" in data
    assert data["style"] is None
    assert "sentiment" in data

def test_invalid_input(client):
    response = client.post(
        "/analyze",
        json={"invalid_field": "test"}
    )
    assert response.status_code == 422

def test_empty_text(client):
    response = client.post(
        "/analyze",
        json={"text": ""}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grammar"]["improvement_score"] == 1.0
    assert data["sentiment"]["polarity"] == 0