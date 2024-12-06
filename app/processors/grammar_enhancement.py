from typing import List, Dict, Tuple
import spacy
from spacy.tokens import Doc, Token

class GrammarEnhancer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.initialize_rules()
    
    def initialize_rules(self):
        """Initialize grammar rules and patterns."""
        self.subject_verb_patterns = [
            {'POS': 'NOUN', 'DEP': 'nsubj'},
            {'POS': 'VERB', 'DEP': 'ROOT'}
        ]
        self.article_patterns = [
            {'POS': 'DET', 'DEP': 'det'},
            {'POS': 'NOUN'}
        ]
    
    def check_subject_verb_agreement(self, doc: Doc) -> List[Dict]:
        """Check for subject-verb agreement issues."""
        issues = []
        for sent in doc.sents:
            subject = None
            verb = None
            
            for token in sent:
                if token.dep_ == 'nsubj':
                    subject = token
                elif token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                    verb = token
                
                if subject and verb:
                    if not self._check_agreement(subject, verb):
                        issues.append({
                            'type': 'subject_verb_agreement',
                            'text': sent.text,
                            'subject': subject.text,
                            'verb': verb.text,
                            'start': sent.start_char,
                            'end': sent.end_char
                        })
        return issues
    
    def check_article_usage(self, doc: Doc) -> List[Dict]:
        """Check for incorrect article usage."""
        issues = []
        for token in doc:
            if token.pos_ == 'DET' and token.dep_ == 'det':
                if not self._is_correct_article(token):
                    issues.append({
                        'type': 'article_usage',
                        'text': token.sent.text,
                        'article': token.text,
                        'noun': token.head.text,
                        'start': token.sent.start_char,
                        'end': token.sent.end_char
                    })
        return issues
    
    def _check_agreement(self, subject: Token, verb: Token) -> bool:
        """Check if subject and verb agree in number."""
        subject_number = 'singular' if subject.tag_ in ['NN', 'NNP'] else 'plural'
        verb_number = self._get_verb_number(verb)
        return subject_number == verb_number
    
    def _get_verb_number(self, verb: Token) -> str:
        """Determine if verb is singular or plural."""
        if verb.tag_ in ['VBZ']:
            return 'singular'
        elif verb.tag_ in ['VBP']:
            return 'plural'
        return 'unknown'
    
    def _is_correct_article(self, article: Token) -> bool:
        """Check if article usage is correct."""
        if article.text.lower() in ['a', 'an']:
            next_word = article.head
            if next_word.text[0].lower() in 'aeiou':
                return article.text.lower() == 'an'
            return article.text.lower() == 'a'
        return True
    
    def enhance_text(self, text: str) -> Tuple[str, List[Dict]]:
        """Enhance text by fixing grammar issues."""
        doc = self.nlp(text)
        issues = []
        
        # Collect all issues
        issues.extend(self.check_subject_verb_agreement(doc))
        issues.extend(self.check_article_usage(doc))
        
        # Apply fixes
        enhanced_text = text
        for issue in sorted(issues, key=lambda x: x['start'], reverse=True):
            if issue['type'] == 'subject_verb_agreement':
                # Apply fix based on subject number
                pass
            elif issue['type'] == 'article_usage':
                # Apply article fix
                pass
        
        return enhanced_text, issues