from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TextInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    optimization_level: str = Field(default="medium", regex="^(light|medium|aggressive)$")
    preserve_keywords: Optional[List[str]] = Field(default=[])
    language: str = Field(default="en")

class TextMetrics(BaseModel):
    word_count: int
    sentence_count: int
    avg_word_length: float
    named_entities: Dict[str, int]
    readability_scores: Dict[str, float]

class OptimizationSuggestion(BaseModel):
    type: str
    message: str
    severity: str
    span: Optional[tuple[int, int]] = None

class TextResponse(BaseModel):
    original: str
    optimized: str
    metrics: TextMetrics
    suggestions: List[OptimizationSuggestion]