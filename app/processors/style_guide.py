from typing import Dict, List, Optional, Tuple
import re
from enum import Enum
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc, Span, Token

class StyleGuideType(Enum):
    ACADEMIC = "academic"
    BUSINESS = "business"
    TECHNICAL = "technical"
    CREATIVE = "creative"

@dataclass
class StyleRule:
    name: str
    description: str
    pattern: str
    suggestion: str
    severity: int  # 1 (suggestion) to 3 (critical)

@dataclass
class StyleViolation:
    rule_name: str
    description: str
    text: str
    suggestion: str
    start: int
    end: int
    severity: int

class StyleGuideProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.initialize_style_guides()
    
    def initialize_style_guides(self):
        """Initialize style guides for different writing types."""
        self.style_guides: Dict[StyleGuideType, List[StyleRule]] = {
            StyleGuideType.ACADEMIC: [
                StyleRule(
                    name="first_person",
                    description="Avoid first-person pronouns in academic writing",
                    pattern=r"\b(I|me|my|mine|we|us|our|ours)\b",
                    suggestion="Use third-person or passive voice",
                    severity=2
                ),
                StyleRule(
                    name="informal_contractions",
                    description="Avoid contractions in academic writing",
                    pattern=r"\b\w+'\w+\b",
                    suggestion="Use the full form",
                    severity=1
                ),
                StyleRule(
                    name="citation_needed",
                    description="Claims may need citation",
                    pattern=r"\b(clearly|obviously|everyone knows|naturally|of course)\b",
                    suggestion="Add citation or remove claim of certainty",
                    severity=3
                )
            ],
            StyleGuideType.BUSINESS: [
                StyleRule(
                    name="passive_voice",
                    description="Prefer active voice in business writing",
                    pattern=r"\b(am|is|are|was|were|being|been|be)\s+\w+ed\b",
                    suggestion="Use active voice for clarity",
                    severity=1
                ),
                StyleRule(
                    name="jargon",
                    description="Minimize business jargon",
                    pattern=r"\b(synergy|paradigm|leverage|utilize|optimize)\b",
                    suggestion="Use simpler, clearer terms",
                    severity=2
                )
            ],
            StyleGuideType.TECHNICAL: [
                StyleRule(
                    name="ambiguous_pronouns",
                    description="Avoid ambiguous pronouns in technical writing",
                    pattern=r"\b(it|this|that|these|those)\b",
                    suggestion="Be specific about what is being referenced",
                    severity=2
                ),
                StyleRule(
                    name="future_tense",
                    description="Use present tense for technical documentation",
                    pattern=r"\b(will|shall)\s+\w+\b",
                    suggestion="Use present tense for clarity",
                    severity=1
                )
            ]
        }
    
    def check_style(self, text: str, style_type: StyleGuideType) -> List[StyleViolation]:
        """Check text against specified style guide rules."""
        doc = self.nlp(text)
        violations = []
        rules = self.style_guides.get(style_type, [])
        
        for rule in rules:
            matches = list(re.finditer(rule.pattern, text, re.IGNORECASE))
            for match in matches:
                violations.append(
                    StyleViolation(
                        rule_name=rule.name,
                        description=rule.description,
                        text=match.group(),
                        suggestion=rule.suggestion,
                        start=match.start(),
                        end=match.end(),
                        severity=rule.severity
                    )
                )
        
        # Add specific checks based on style type
        if style_type == StyleGuideType.ACADEMIC:
            violations.extend(self._check_sentence_complexity(doc))
        elif style_type == StyleGuideType.TECHNICAL:
            violations.extend(self._check_terminology_consistency(doc))
        
        return violations
    
    def _check_sentence_complexity(self, doc: Doc) -> List[StyleViolation]:
        """Check for overly complex sentences in academic writing."""
        violations = []
        
        for sent in doc.sents:
            words = len([token for token in sent if not token.is_punct])
            if words > 40:  # Arbitrary threshold for demonstration
                violations.append(
                    StyleViolation(
                        rule_name="sentence_length",
                        description="Sentence may be too complex",
                        text=sent.text,
                        suggestion="Consider breaking into multiple sentences",
                        start=sent.start_char,
                        end=sent.end_char,
                        severity=1
                    )
                )
        
        return violations
    
    def _check_terminology_consistency(self, doc: Doc) -> List[StyleViolation]:
        """Check for consistent terminology use in technical writing."""
        violations = []
        term_variants = {}
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                lowercase = token.text.lower()
                if lowercase in term_variants:
                    if token.text != term_variants[lowercase]:
                        violations.append(
                            StyleViolation(
                                rule_name="inconsistent_terminology",
                                description="Inconsistent term usage",
                                text=token.text,
                                suggestion=f"Use '{term_variants[lowercase]}' consistently",
                                start=token.idx,
                                end=token.idx + len(token.text),
                                severity=2
                            )
                        )
                else:
                    term_variants[lowercase] = token.text
        
        return violations