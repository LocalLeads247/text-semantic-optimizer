import spacy
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def initialize_nlp():
    """Initialize NLP models and download required resources."""
    try:
        nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt', quiet=True)
        return nlp
    except OSError:
        raise RuntimeError("Required language model 'en_core_web_sm' not found. Please install it using 'python -m spacy download en_core_web_sm'")

def calculate_text_metrics(text: str) -> Dict[str, Any]:
    """Calculate various metrics for the input text."""
    sentences = sent_tokenize(text)
    words = text.split()
    
    return {
        "sentence_count": len(sentences),
        "word_count": len(words),
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
