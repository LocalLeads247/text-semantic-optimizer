from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    TextInput, GrammarResponse, SentimentResponse, 
    TextAnalysisResponse, StyleResponse, StyleIssue
)
from .processors.grammar_enhancement import GrammarEnhancer
from .processors.sentiment_analyzer import SentimentAnalyzer
from .processors.style_guide import StyleGuideProcessor, StyleViolation

app = FastAPI(
    title="Text Semantic Optimizer",
    description="Advanced text analysis and optimization API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
grammar_enhancer = GrammarEnhancer()
sentiment_analyzer = SentimentAnalyzer()
style_processor = StyleGuideProcessor()

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(input_data: TextInput):
    """Analyze text for grammar, style, and sentiment."""
    try:
        # Grammar analysis
        enhanced_text, grammar_issues = grammar_enhancer.enhance_text(input_data.text)
        improvement_score = len(grammar_issues) / len(input_data.text.split())
        grammar_response = GrammarResponse(
            original_text=input_data.text,
            enhanced_text=enhanced_text,
            issues=grammar_issues,
            improvement_score=1 - improvement_score
        )

        # Style analysis (if style guide specified)
        style_response = None
        if input_data.style_guide:
            style_violations = style_processor.check_style(
                input_data.text, 
                input_data.style_guide
            )
            violation_weight = sum(v.severity for v in style_violations)
            max_possible_weight = len(style_violations) * 3  # max severity is 3
            compliance_score = 1 - (violation_weight / max_possible_weight if max_possible_weight > 0 else 0)
            
            style_response = StyleResponse(
                original_text=input_data.text,
                issues=[StyleIssue(**v.__dict__) for v in style_violations],
                style_guide_type=input_data.style_guide,
                compliance_score=compliance_score
            )

        # Sentiment analysis
        sentiment_score = sentiment_analyzer.analyze_sentiment(input_data.text)
        sentiment_response = SentimentResponse(
            text=input_data.text,
            polarity=sentiment_score.polarity,
            subjectivity=sentiment_score.subjectivity,
            objectivity=sentiment_score.objectivity,
            emotional_tone=sentiment_score.emotional_tone,
            summary=sentiment_analyzer.get_sentiment_summary(sentiment_score)
        )

        return TextAnalysisResponse(
            grammar=grammar_response,
            style=style_response,
            sentiment=sentiment_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhance/grammar", response_model=GrammarResponse)
async def enhance_grammar(input_data: TextInput):
    """Enhance text grammar and return detailed analysis."""
    try:
        enhanced_text, issues = grammar_enhancer.enhance_text(input_data.text)
        improvement_score = len(issues) / len(input_data.text.split())
        return GrammarResponse(
            original_text=input_data.text,
            enhanced_text=enhanced_text,
            issues=issues,
            improvement_score=1 - improvement_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/style", response_model=StyleResponse)
async def analyze_style(input_data: TextInput):
    """Analyze text style against specified style guide."""
    try:
        if not input_data.style_guide:
            raise HTTPException(
                status_code=400,
                detail="Style guide type must be specified"
            )

        violations = style_processor.check_style(
            input_data.text,
            input_data.style_guide
        )
        
        violation_weight = sum(v.severity for v in violations)
        max_possible_weight = len(violations) * 3  # max severity is 3
        compliance_score = 1 - (violation_weight / max_possible_weight if max_possible_weight > 0 else 0)
        
        return StyleResponse(
            original_text=input_data.text,
            issues=[StyleIssue(**v.__dict__) for v in violations],
            style_guide_type=input_data.style_guide,
            compliance_score=compliance_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """Analyze text sentiment and emotional tone."""
    try:
        sentiment_score = sentiment_analyzer.analyze_sentiment(input_data.text)
        return SentimentResponse(
            text=input_data.text,
            polarity=sentiment_score.polarity,
            subjectivity=sentiment_score.subjectivity,
            objectivity=sentiment_score.objectivity,
            emotional_tone=sentiment_score.emotional_tone,
            summary=sentiment_analyzer.get_sentiment_summary(sentiment_score)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}