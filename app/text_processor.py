import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
import textstat

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

nlp = spacy.load("en_core_web_sm")

class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        return entities
    
    def cluster_topics(self, text: str, num_topics: int = 3) -> List[Dict]:
        sentences = sent_tokenize(text)
        if len(sentences) < num_topics:
            num_topics = max(1, len(sentences))
        
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            kmeans = KMeans(n_clusters=num_topics, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            topics = []
            feature_names = vectorizer.get_feature_names_out()
            
            for i in range(num_topics):
                cluster_sentences = [sent for j, sent in enumerate(sentences) if clusters[j] == i]
                
                if cluster_sentences:
                    centroid = kmeans.cluster_centers_[i]
                    top_term_indices = centroid.argsort()[-5:][::-1]
                    key_terms = [feature_names[idx] for idx in top_term_indices]
                    
                    topics.append({
                        'topic_id': i,
                        'key_terms': key_terms,
                        'sample_sentences': cluster_sentences[:2]
                    })
            
            return topics
        except ValueError:
            return [{
                'topic_id': 0,
                'key_terms': [],
                'sample_sentences': sentences
            }]
    
    def analyze_readability(self, text: str) -> Dict:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text)
        }
    
    def get_text_statistics(self, text: str) -> Dict:
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        return {
            'num_sentences': len(sentences),
            'num_words': len([token for token in doc if not token.is_punct]),
            'num_chars': len(text),
            'avg_sentence_length': np.mean([len([token for token in sent if not token.is_punct]) 
                                         for sent in sentences]) if sentences else 0,
            'avg_word_length': np.mean([len(token.text) for token in doc if not token.is_punct]) if doc else 0
        }