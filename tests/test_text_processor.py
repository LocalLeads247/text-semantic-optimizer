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

def test_get_synonyms():
    optimizer = TextOptimizer()
    synonyms = optimizer.get_synonyms("quick")
    assert len(synonyms) == 3
    assert "rapid" in synonyms
    assert "fast" in synonyms
    assert "speedy" in synonyms

def test_optimize_sentence_structure():
    optimizer = TextOptimizer()
    sentence = "This is a very long sentence that should be split into two sentences."
    optimized_sentence = optimizer.optimize_sentence_structure(sentence)
    assert len(optimized_sentence.split(".")) == 2

def test_analyze_readability():
    optimizer = TextOptimizer()
    text = "This is a sample text for testing. It has multiple sentences and words."
    metrics = optimizer.analyze_readability(text)
    assert metrics["sentence_count"] == 2
    assert metrics["word_count"] == 12
    assert metrics["avg_sentence_length"] == 6.0
    assert metrics["avg_word_length"] == 5.5

def test_generate_suggestions():
    optimizer = TextOptimizer()
    text = "This is a long sentence that should be broken down. Another long sentence that could also be improved."
    suggestions = optimizer.generate_suggestions(optimizer.nlp(text))
    assert "Consider breaking down 2 long sentences for better readability" in suggestions
    assert "Consider simplifying complex words: sentence" in suggestions