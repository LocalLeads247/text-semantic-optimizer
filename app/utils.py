import spacy
import logging
from typing import Dict, Any

def initialize_nlp(model_name: str = 'en_core_web_sm') -> spacy.Language:
    """Initialize spaCy NLP model."""
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        logging.warning(f"Downloading language model {model_name}")
        spacy.cli.download(model_name)
        return spacy.load(model_name)

def calculate_text_metrics(text: str) -> Dict[str, Any]:
    """Calculate various text metrics."""
    nlp = initialize_nlp()
    doc = nlp(text)
    
    # Basic metrics
    metrics = {
        'word_count': len([token for token in doc if not token.is_punct]),
        'sentence_count': len(list(doc.sents)),
        'avg_word_length': sum(len(token.text) for token in doc if not token.is_punct) / len([token for token in doc if not token.is_punct]) if len(doc) > 0 else 0,
    }
    
    # Entity metrics
    entity_counts = {}
    for ent in doc.ents:
        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
    metrics['named_entities'] = entity_counts
    
    return metrics