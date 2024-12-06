import pytest
from app.processors.grammar_enhancement import GrammarEnhancer

@pytest.fixture
def grammar_enhancer():
    return GrammarEnhancer()

def test_subject_verb_agreement(grammar_enhancer):
    # Test singular subject with singular verb
    text = "The cat runs fast."
    doc = grammar_enhancer.nlp(text)
    issues = grammar_enhancer.check_subject_verb_agreement(doc)
    assert len(issues) == 0
    
    # Test singular subject with plural verb
    text = "The cat run fast."
    doc = grammar_enhancer.nlp(text)
    issues = grammar_enhancer.check_subject_verb_agreement(doc)
    assert len(issues) == 1
    assert issues[0]['type'] == 'subject_verb_agreement'

def test_article_usage(grammar_enhancer):
    # Test correct article usage
    text = "I saw a cat and an elephant."
    doc = grammar_enhancer.nlp(text)
    issues = grammar_enhancer.check_article_usage(doc)
    assert len(issues) == 0
    
    # Test incorrect article usage
    text = "I saw an cat and a elephant."
    doc = grammar_enhancer.nlp(text)
    issues = grammar_enhancer.check_article_usage(doc)
    assert len(issues) == 2
    assert all(issue['type'] == 'article_usage' for issue in issues)

def test_enhance_text(grammar_enhancer):
    text = "The cats runs fast and I saw an cat."
    enhanced_text, issues = grammar_enhancer.enhance_text(text)
    assert len(issues) > 0
    assert isinstance(enhanced_text, str)