import spacy
import logging
from typing import Dict, Any, Optional

def initialize_nlp(model_name: str = 'en_core_web_sm') -> spacy.Language:
    """Initialize spaCy NLP model."""
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        logging.warning(f"Downloading language model {model_name}")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def calculate_text_metrics(text: str, nlp: Optional[spacy.Language] = None) -> Dict[str, Any]:
    """Calculate various text metrics."""
    if nlp is None:
        nlp = initialize_nlp()
    
    doc = nlp(text)
    
    # Basic metrics
    metrics = {
        'word_count': len([token for token in doc if not token.is_punct]),
        'sentence_count': len(list(doc.sents)),
        'avg_word_length': sum(len(token.text) for token in doc if not token.is_punct) / 
                          len([token for token in doc if not token.is_punct]) if len(doc) > 0 else 0,
    }
    
    # Entity metrics
    entity_counts = {}
    for ent in doc.ents:
        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
    metrics['named_entities'] = entity_counts
    
    # Part of speech metrics
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    metrics['pos_distribution'] = pos_counts
    
    # Dependency metrics
    dep_counts = {}
    for token in doc:
        dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
    metrics['dependency_distribution'] = dep_counts
    
    return metrics

def extract_sentences(text: str, nlp: Optional[spacy.Language] = None) -> list:
    """Extract sentences from text with additional metadata."""
    if nlp is None:
        nlp = initialize_nlp()
    
    doc = nlp(text)
    sentences = []
    
    for sent in doc.sents:
        sentences.append({
            'text': sent.text,
            'start': sent.start_char,
            'end': sent.end_char,
            'length': len(sent),
            'root': sent.root.text,
            'entities': [(ent.text, ent.label_) for ent in sent.ents]
        })
    
    return sentences

def get_sentence_complexity(sent: spacy.tokens.Span) -> float:
    """Calculate sentence complexity score based on various factors."""
    complexity_score = 1.0
    
    # Add complexity for length
    complexity_score += len(sent) / 10
    
    # Add complexity for nested clauses
    clause_count = len([token for token in sent if token.dep_ in {'ccomp', 'xcomp', 'advcl'}])
    complexity_score += clause_count * 0.5
    
    # Add complexity for rare words
    rare_words = len([token for token in sent 
                     if not token.is_stop and not token.is_punct and len(token.text) > 7])
    complexity_score += rare_words * 0.3
    
    return complexity_score
