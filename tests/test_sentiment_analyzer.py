import pytest
from app.processors.sentiment_analyzer import SentimentAnalyzer, SentimentScore

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

def test_polarity_calculation(sentiment_analyzer):
    # Test positive text
    text = "This is a good and excellent product."
    score = sentiment_analyzer.analyze_sentiment(text)
    assert score.polarity > 0
    
    # Test negative text
    text = "This is a bad and terrible product."
    score = sentiment_analyzer.analyze_sentiment(text)
    assert score.polarity < 0
    
    # Test neutral text
    text = "This is a product."
    score = sentiment_analyzer.analyze_sentiment(text)
    assert score.polarity == 0

def test_subjectivity_calculation(sentiment_analyzer):
    # Test subjective text
    text = "This is an amazing and wonderful product!"
    score = sentiment_analyzer.analyze_sentiment(text)
    assert score.subjectivity > 0.5
    
    # Test objective text
    text = "The product weighs 5 pounds and measures 10 inches."
    score = sentiment_analyzer.analyze_sentiment(text)
    assert score.subjectivity < 0.5

def test_emotional_tone_analysis(sentiment_analyzer):
    # Test happy text
    text = "I am so happy and delighted with this!"
    score = sentiment_analyzer.analyze_sentiment(text)
    assert score.emotional_tone['joy'] > score.emotional_tone['sadness']
    
    # Test angry text
    text = "I am furious and angry about this situation!"
    score = sentiment_analyzer.analyze_sentiment(text)
    assert score.emotional_tone['anger'] > score.emotional_tone['joy']

def test_sentiment_summary(sentiment_analyzer):
    text = "This is a good product that makes me happy."
    score = sentiment_analyzer.analyze_sentiment(text)
    summary = sentiment_analyzer.get_sentiment_summary(score)
    assert isinstance(summary, str)
    assert 'positive' in summary.lower()