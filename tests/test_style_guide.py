import pytest
from app.processors.style_guide import StyleGuideProcessor, StyleGuideType, StyleViolation

@pytest.fixture
def style_processor():
    return StyleGuideProcessor()

def test_academic_style(style_processor):
    # Test first person detection
    text = "I believe this research shows important results."
    violations = style_processor.check_style(text, StyleGuideType.ACADEMIC)
    assert any(v.rule_name == "first_person" for v in violations)
    
    # Test contractions
    text = "It's important to note that the results aren't conclusive."
    violations = style_processor.check_style(text, StyleGuideType.ACADEMIC)
    assert any(v.rule_name == "informal_contractions" for v in violations)
    
    # Test citation needed
    text = "Obviously, this method is superior to others."
    violations = style_processor.check_style(text, StyleGuideType.ACADEMIC)
    assert any(v.rule_name == "citation_needed" for v in violations)

def test_business_style(style_processor):
    # Test passive voice detection
    text = "The report was written by the team."
    violations = style_processor.check_style(text, StyleGuideType.BUSINESS)
    assert any(v.rule_name == "passive_voice" for v in violations)
    
    # Test jargon detection
    text = "We need to leverage our synergies to optimize growth."
    violations = style_processor.check_style(text, StyleGuideType.BUSINESS)
    jargon_violations = [v for v in violations if v.rule_name == "jargon"]
    assert len(jargon_violations) >= 2

def test_technical_style(style_processor):
    # Test ambiguous pronouns
    text = "This can be configured in the settings. It should work after that."
    violations = style_processor.check_style(text, StyleGuideType.TECHNICAL)
    pronoun_violations = [v for v in violations if v.rule_name == "ambiguous_pronouns"]
    assert len(pronoun_violations) >= 2
    
    # Test future tense
    text = "The system will process the request and shall return a response."
    violations = style_processor.check_style(text, StyleGuideType.TECHNICAL)
    assert any(v.rule_name == "future_tense" for v in violations)

def test_sentence_complexity(style_processor):
    # Test long sentence detection
    text = "This extremely long and complex sentence contains numerous clauses and phrases " \
           "that make it difficult to read and understand because it continues for far too long " \
           "without any proper breaks or punctuation to help readers follow the meaning easily."
    violations = style_processor.check_style(text, StyleGuideType.ACADEMIC)
    assert any(v.rule_name == "sentence_length" for v in violations)

def test_terminology_consistency(style_processor):
    # Test inconsistent terminology
    text = "The API endpoint accepts JSON data. The api Endpoint returns XML."
    violations = style_processor.check_style(text, StyleGuideType.TECHNICAL)
    assert any(v.rule_name == "inconsistent_terminology" for v in violations)

def test_multiple_violations(style_processor):
    text = "I think this will obviously work better. It's clearly the best solution."
    violations = style_processor.check_style(text, StyleGuideType.ACADEMIC)
    # Should detect first person, contraction, citation needed, and future tense
    assert len(violations) >= 3

def test_violation_attributes(style_processor):
    text = "I believe this."
    violations = style_processor.check_style(text, StyleGuideType.ACADEMIC)
    violation = next(v for v in violations if v.rule_name == "first_person")
    
    assert hasattr(violation, 'rule_name')
    assert hasattr(violation, 'description')
    assert hasattr(violation, 'text')
    assert hasattr(violation, 'suggestion')
    assert hasattr(violation, 'start')
    assert hasattr(violation, 'end')
    assert hasattr(violation, 'severity')
    assert isinstance(violation.severity, int)