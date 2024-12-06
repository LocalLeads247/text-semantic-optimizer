from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class TextInput(BaseModel):
    text: str = Field(..., description="Input text to analyze or enhance")
    preserve_phrases: Optional[List[str]] = Field(default=None, description="Phrases to preserve during enhancement")
    optimization_level: str = Field(default="medium", description="Optimization level: light, medium, or aggressive")

class GrammarIssue(BaseModel):
    type: str
    text: str
    start: int
    end: int
    subject: Optional[str] = None
    verb: Optional[str] = None
    article: Optional[str] = None
    noun: Optional[str] = None

class GrammarResponse(BaseModel):
    original_text: str
    enhanced_text: str
    issues: List[GrammarIssue]
    improvement_score: float

class SentimentResponse(BaseModel):
    text: str
    polarity: float
    subjectivity: float
    objectivity: float
    emotional_tone: Dict[str, float]
    summary: str

class TextAnalysisResponse(BaseModel):
    grammar: GrammarResponse
    sentiment: SentimentResponse