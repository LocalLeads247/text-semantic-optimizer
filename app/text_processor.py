import spacy
import nltk
from typing import List, Dict, Any, Tuple
from .exceptions import *
from .utils import initialize_nlp, calculate_text_metrics
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from collections import defaultdict

class TextOptimizer:
    def __init__(self):
        self.nlp = initialize_nlp()
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        return list(synonyms)[:3]  # Return top 3 synonyms
    
    def replace_with_synonyms(self, sentence: str) -> str:
        """Replace words in a sentence with their synonyms."""
        words = sentence.split()
        optimized_words = []
        for word in words:
            synonyms = self.get_synonyms(word)
            if synonyms:
                optimized_words.append(synonyms[0])
            else:
                optimized_words.append(word)
        return ' '.join(optimized_words)
    
    def optimize_sentence_structure(self, sentence: str) -> str:
        """Optimize sentence structure using spaCy."""
        doc = self.nlp(sentence)
        
        # Basic sentence improvements
        if len(doc) > 20:  # Long sentence
            # Try to split at conjunctions
            conj_indices = [i for i, token in enumerate(doc) if token.dep_ == 'cc']
            if conj_indices:
                mid = conj_indices[len(conj_indices)//2]
                first_half = doc[:mid].text
                second_half = doc[mid+1:].text
                return f"{first_half}. {second_half}"
        
        return sentence
    
    def analyze_readability(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics."""
        sentences = sent_tokenize(text)
        words = text.split()
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        metrics = {
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'sentence_count': len(sentences),
            'word_count': len(words)
        }
        
        return metrics
    
    def generate_suggestions(self, doc) -> List[str]:
        """Generate improvement suggestions based on text analysis."""
        suggestions = []
        
        # Check sentence length
        long_sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 20]
        if long_sentences:
            suggestions.append(f"Consider breaking down {len(long_sentences)} long sentences for better readability")
        
        # Check word complexity
        complex_words = [token.text for token in doc if len(token.text) > 12]
        if complex_words:
            suggestions.append(f"Consider simplifying complex words: {', '.join(complex_words[:3])}")
        
        # Check passive voice
        passive_constructs = [sent.text for sent in doc.sents 
                            if any(token.dep_ == 'auxpass' for token in sent)]
        if passive_constructs:
            suggestions.append("Consider using active voice in some sentences")
        
        return suggestions
    
    def optimize_text(self, 
                      text: str, 
                      optimization_level: str = 'medium',
                      preserve_keywords: List[str] = None) -> Tuple[str, Dict[str, Any], List[str]]:
        """Main text optimization function."""
        if not text:
            raise TextTooShortError("Input text cannot be empty")
            
        if len(text) > 10000:
            raise TextTooLongError("Input text exceeds maximum length of 10000 characters")
            
        if optimization_level not in ['light', 'medium', 'aggressive']:
            raise InvalidOptimizationLevelError(f"Invalid optimization level: {optimization_level}")
        
        preserve_keywords = set(k.lower() for k in (preserve_keywords or []))
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Optimize sentence by sentence
            optimized_sentences = []
            for sent in doc.sents:
                optimized = self.replace_with_synonyms(sent.text)
                optimized = self.optimize_sentence_structure(optimized)
                optimized_sentences.append(optimized)
            
            # Join sentences
            optimized_text = ' '.join(optimized_sentences)
            
            # Calculate metrics
            metrics = self.analyze_readability(optimized_text)
            
            # Generate suggestions
            suggestions = self.generate_suggestions(doc)
            
            return optimized_text, metrics, suggestions
            
        except Exception as e:
            raise ProcessingError(f"Error during text optimization: {str(e)}")