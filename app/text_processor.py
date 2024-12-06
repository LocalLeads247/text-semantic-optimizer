import spacy
import nltk
import textstat
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from .exceptions import *
from .utils import initialize_nlp, calculate_text_metrics, get_sentence_complexity
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class TextOptimizer:
    def __init__(self):
        self.nlp = initialize_nlp()
        # Download required NLTK data
        for resource in ['wordnet', 'averaged_perceptron_tagger', 'stopwords', 'punkt']:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {resource}: {str(e)}")
        self.stopwords = set(stopwords.words('english'))

    def get_synonyms(self, word: str, context: Optional[str] = None) -> List[str]:
        """Get contextually appropriate synonyms for a word using WordNet."""
        synonyms = []
        # Get word POS tag from context if available
        pos_tag = None
        if context:
            doc = self.nlp(context)
            for token in doc:
                if token.text.lower() == word.lower():
                    pos_tag = self._convert_spacy_pos_to_wordnet(token.pos_)
                    break

        for synset in wordnet.synsets(word):
            # Filter by POS tag if available
            if pos_tag and synset.pos() != pos_tag:
                continue
            for lemma in synset.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    # Score synonym based on similarity and frequency
                    score = synset.wup_similarity(synset) if synset else 0
                    synonyms.append((lemma.name(), score or 0))

        # Sort by score and return top synonyms
        return [syn for syn, _ in sorted(set(synonyms), key=lambda x: x[1], reverse=True)[:3]]

    def identify_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Identify named entities in the text."""
        doc = self.nlp(text)
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_
            })
        return dict(entities)

    def extract_key_phrases(self, text: str, num_phrases: int = 5) -> List[str]:
        """Extract key phrases using TF-IDF and noun phrases."""
        doc = self.nlp(text)

        # Extract noun phrases
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks
                       if not all(token.is_stop for token in chunk)]

        if not noun_phrases:
            return []

        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(noun_phrases)
            feature_names = vectorizer.get_feature_names_out()

            # Calculate phrase importance scores
            phrase_scores = []
            for i, phrase in enumerate(noun_phrases):
                score = sum(tfidf_matrix[i, vectorizer.vocabulary_[word]]
                           for word in phrase.split()
                           if word in vectorizer.vocabulary_)
                phrase_scores.append((phrase, score))

            # Return top phrases
            return [phrase for phrase, _ in sorted(phrase_scores,
                                                 key=lambda x: x[1],
                                                 reverse=True)[:num_phrases]]
        except Exception:
            # Fallback to basic frequency-based extraction
            phrase_freq = defaultdict(int)
            for phrase in noun_phrases:
                phrase_freq[phrase] += 1
            return [phrase for phrase, _ in sorted(phrase_freq.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)[:num_phrases]]

    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and coherence."""
        doc = self.nlp(text)

        # Analyze sentence structure
        sentence_types = defaultdict(int)
        transition_words = 0
        discourse_markers = set(['however', 'therefore', 'furthermore', 'moreover',
                               'consequently', 'nevertheless', 'thus', 'meanwhile',
                               'afterward', 'finally'])

        for sent in doc.sents:
            # Classify sentence type
            if any(token.dep_ == 'mark' for token in sent):
                sentence_types['complex'] += 1
            elif any(token.dep_ == 'cc' for token in sent):
                sentence_types['compound'] += 1
            else:
                sentence_types['simple'] += 1

            # Count transition words
            sent_text = sent.text.lower()
            transition_words += sum(1 for word in discourse_markers if word in sent_text)

        return {
            'sentence_types': dict(sentence_types),
            'transition_words': transition_words,
            'avg_sentence_length': sum(len(sent.text.split()) for sent in doc.sents) / len(list(doc.sents)),
            'coherence_score': self._calculate_coherence_score(doc)
        }

    def optimize_sentence_structure(self, sentence: str) -> str:
        """Optimize sentence structure using advanced NLP analysis."""
        doc = self.nlp(sentence)

        # Handle long sentences
        if len(doc) > 20:
            # Try to split at discourse boundaries
            splits = []
            current_split = []

            for token in doc:
                current_split.append(token.text)

                # Check for natural breaking points
                if (token.dep_ in ['cc', 'mark'] and len(current_split) > 5) or \
                   (token.is_punct and token.text == ',' and len(current_split) > 10):
                    splits.append(' '.join(current_split))
                    current_split = []

            if current_split:
                splits.append(' '.join(current_split))

            return '. '.join(splits)

        return sentence

    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate comprehensive readability metrics."""
        metrics = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'linsear_write_formula': textstat.linsear_write_formula(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text)
        }

        # Add sentence structure metrics
        doc = self.nlp(text)
        metrics.update({
            'avg_sentence_length': sum(len(sent.text.split()) for sent in doc.sents) / len(list(doc.sents)),
            'avg_word_length': sum(len(token.text) for token in doc if not token.is_punct) / 
                             len([token for token in doc if not token.is_punct]),
            'complex_word_ratio': len([token for token in doc if len(token.text) > 6]) / 
                                len([token for token in doc if not token.is_punct])
        })

        return metrics

    def _calculate_coherence_score(self, doc) -> float:
        """Calculate text coherence score based on semantic similarity."""
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 1.0

        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            similarity = sentences[i].similarity(sentences[i + 1])
            similarities.append(similarity)

        return sum(similarities) / len(similarities)

    def _convert_spacy_pos_to_wordnet(self, spacy_pos: str) -> Optional[str]:
        """Convert spaCy POS tags to WordNet POS tags."""
        pos_map = {
            'NOUN': 'n',
            'VERB': 'v',
            'ADJ': 'a',
            'ADV': 'r'
        }
        return pos_map.get(spacy_pos)

    def optimize_text(self,
                     text: str,
                     optimization_level: str = 'medium',
                     preserve_keywords: List[str] = None) -> Tuple[str, Dict[str, Any], List[str]]:
        """Main text optimization function with enhanced capabilities."""
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

            # Extract key phrases to preserve
            if not preserve_keywords:
                preserve_keywords.update(self.extract_key_phrases(text))

            # Optimize sentence by sentence
            optimized_sentences = []
            for sent in doc.sents:
                # Skip optimization for sentences containing preserved keywords
                if any(keyword in sent.text.lower() for keyword in preserve_keywords):
                    optimized_sentences.append(sent.text)
                    continue

                # Apply optimizations based on level
                optimized = sent.text
                if optimization_level in ['medium', 'aggressive']:
                    optimized = self.optimize_sentence_structure(optimized)

                if optimization_level == 'aggressive':
                    optimized = self.replace_with_synonyms(optimized)

                optimized_sentences.append(optimized)

            # Join sentences
            optimized_text = ' '.join(optimized_sentences)

            # Calculate comprehensive metrics
            metrics = {
                'readability': self.calculate_readability_metrics(optimized_text),
                'structure': self.analyze_text_structure(optimized_text),
                'entities': self.identify_entities(optimized_text),
                'key_phrases': self.extract_key_phrases(optimized_text)
            }

            # Generate detailed suggestions
            suggestions = self.generate_suggestions(self.nlp(optimized_text))

            return optimized_text, metrics, suggestions

        except Exception as e:
            raise ProcessingError(f"Error during text optimization: {str(e)}")

    def generate_suggestions(self, doc) -> List[Dict[str, Any]]:
        """Generate detailed improvement suggestions."""
        suggestions = []

        # Analyze sentence length
        long_sentences = [(i, sent.text) for i, sent in enumerate(doc.sents)
                         if len(sent.text.split()) > 20]
        if long_sentences:
            suggestions.append({
                'type': 'readability',
                'severity': 'medium',
                'message': f"Consider breaking down {len(long_sentences)} long sentences",
                'examples': [text for _, text in long_sentences[:2]]
            })

        # Analyze word complexity
        complex_words = [(token.i, token.text) for token in doc
                        if len(token.text) > 12 and not token.is_stop]
        if complex_words:
            suggestions.append({
                'type': 'vocabulary',
                'severity': 'low',
                'message': "Consider simplifying these complex words",
                'examples': [text for _, text in complex_words[:3]]
            })

        # Analyze passive voice
        passive_constructs = [(i, sent.text) for i, sent in enumerate(doc.sents)
                            if any(token.dep_ == 'auxpass' for token in sent)]
        if passive_constructs:
            suggestions.append({
                'type': 'style',
                'severity': 'low',
                'message': "Consider using active voice in some sentences",
                'examples': [text for _, text in passive_constructs[:2]]
            })

        # Analyze text coherence
        coherence_score = self._calculate_coherence_score(doc)
        if coherence_score < 0.5:
            suggestions.append({
                'type': 'coherence',
                'severity': 'high',
                'message': "Consider improving text flow with better transitions between sentences",
                'score': coherence_score
            })

        return suggestions
