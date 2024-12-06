
from app.text_processor import TextOptimizer
from app.exceptions import TextTooLongError, TextTooShortError, InvalidOptimizationLevelError

def test_optimize_text_valid_input():
    optimizer = TextOptimizer()
    text = "This is a sample text for testing."
    optimization_level = "medium"
    preserve_keywords = []

    optimized_text, metrics, suggestions = optimizer.optimize_text(
        text, optimization_level, preserve_keywords
    )

    assert optimized_text is not None
    assert metrics is not None
    assert suggestions is not None

def test_optimize_text_too_long_input():
    optimizer = TextOptimizer()
    text = "a" * 10001
    optimization_level = "medium"
    preserve_keywords = []

    try:
        optimizer.optimize_text(text, optimization_level, preserve_keywords)
        assert False
    except TextTooLongError as e:
        assert str(e) == "Input text exceeds maximum length of 10000 characters"

def test_optimize_text_too_short_input():
    optimizer = TextOptimizer()
    text = ""
    optimization_level = "medium"
    preserve_keywords = []

    try:
        optimizer.optimize_text(text, optimization_level, preserve_keywords)
        assert False
    except TextTooShortError as e:
        assert str(e) == "Input text cannot be empty"

def test_optimize_text_invalid_optimization_level():
    optimizer = TextOptimizer()
    text = "This is a sample text for testing."
    optimization_level = "invalid"
    preserve_keywords = []

    try:
        optimizer.optimize_text(text, optimization_level, preserve_keywords)
        assert False
    except InvalidOptimizationLevelError as e:
        assert str(e) == "Invalid optimization level: invalid"
