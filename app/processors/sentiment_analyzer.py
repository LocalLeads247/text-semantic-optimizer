from typing import Dict, List, Union
from dataclasses import dataclass
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class SentimentScore:
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    emotional_tone: Dict[str, float]  # emotions and their scores
    objectivity: float  # 0 to 1

class SentimentAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.initialize_lexicons()
        self.vectorizer = TfidfVectorizer()
    
    def initialize_lexicons(self):
        """Initialize sentiment lexicons and emotion dictionaries."""
        # Example emotion categories
        self.emotion_categories = {
            'joy': ['happy', 'delighted', 'pleased'],
            'sadness': ['sad', 'disappointed', 'unhappy'],
            'anger': ['angry', 'furious', 'irritated'],
            'fear': ['scared', 'afraid', 'worried'],
            'surprise': ['surprised', 'amazed', 'astonished']
        }
        
        # Initialize polarity words
        self.positive_words = set(['good', 'great', 'excellent'])
        self.negative_words = set(['bad', 'poor', 'terrible'])
    
    def analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze the sentiment of the given text."""
        doc = self.nlp(text)
        
        # Calculate polarity
        polarity = self._calculate_polarity(doc)
        
        # Calculate subjectivity
        subjectivity = self._calculate_subjectivity(doc)
        
        # Analyze emotional tone
        emotional_tone = self._analyze_emotional_tone(doc)
        
        # Calculate objectivity
        objectivity = 1 - subjectivity
        
        return SentimentScore(
            polarity=polarity,
            subjectivity=subjectivity,
            emotional_tone=emotional_tone,
            objectivity=objectivity
        )
    
    def _calculate_polarity(self, doc: spacy.tokens.Doc) -> float:
        """Calculate the polarity score of the text."""
        positive_count = sum(1 for token in doc if token.text.lower() in self.positive_words)
        negative_count = sum(1 for token in doc if token.text.lower() in self.negative_words)
        
        total_words = len([token for token in doc if not token.is_punct])
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words
    
    def _calculate_subjectivity(self, doc: spacy.tokens.Doc) -> float:
        """Calculate the subjectivity score of the text."""
        subjective_words = self.positive_words.union(self.negative_words)
        subjective_count = sum(1 for token in doc if token.text.lower() in subjective_words)
        
        total_words = len([token for token in doc if not token.is_punct])
        if total_words == 0:
            return 0.0
        
        return subjective_count / total_words
    
    def _analyze_emotional_tone(self, doc: spacy.tokens.Doc) -> Dict[str, float]:
        """Analyze the emotional tone of the text."""
        text_words = set(token.text.lower() for token in doc)
        emotion_scores = {}
        
        for emotion, words in self.emotion_categories.items():
            emotion_words = set(words)
            intersection = text_words.intersection(emotion_words)
            score = len(intersection) / len(text_words) if text_words else 0.0
            emotion_scores[emotion] = score
        
        return emotion_scores
    
    def get_sentiment_summary(self, score: SentimentScore) -> str:
        """Generate a human-readable summary of the sentiment analysis."""
        polarity_desc = 'positive' if score.polarity > 0 else 'negative' if score.polarity < 0 else 'neutral'
        dominant_emotion = max(score.emotional_tone.items(), key=lambda x: x[1])[0]
        
        return f"The text has a {polarity_desc} tone (polarity: {score.polarity:.2f}) with {score.subjectivity:.1%} subjectivity. The dominant emotion is {dominant_emotion}."