import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_text():
    return "The quick brown fox jumps over the lazy dog."

@pytest.fixture
def long_text():
    return "" * 10001  # Text longer than max allowed

@pytest.fixture
def complex_text():
    return """The implementation of artificial intelligence algorithms requires
    careful consideration of computational complexity and resource optimization.
    Moreover, the utilization of sophisticated mathematical frameworks enables
    the development of more efficient solutions."""