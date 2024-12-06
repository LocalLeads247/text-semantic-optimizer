import pytest
from app.text_processor import TextOptimizer
from app.exceptions import TextTooShortError, TextTooLongError

def test_text_optimization_basic(sample_text):
    optimizer = TextOptimizer()
    result, metrics, suggestions = optimizer.optimize_text(sample_text)
    assert isinstance(result, str)
    assert len(result) > 0
    assert isinstance(metrics, dict)
    assert isinstance(suggestions, list)

def test_empty_text():
    optimizer = TextOptimizer()
    with pytest.raises(TextTooShortError):
        optimizer.optimize_text("")

def test_text_too_long(long_text):
    optimizer = TextOptimizer()
    with pytest.raises(TextTooLongError):
        optimizer.optimize_text(long_text)

def test_preserve_keywords():
    optimizer = TextOptimizer()
    text = "The quick brown fox jumps over the lazy dog."
    keywords = ["quick", "fox"]
    result, _, _ = optimizer.optimize_text(text, preserve_keywords=keywords)
    assert "quick" in result
    assert "fox" in result