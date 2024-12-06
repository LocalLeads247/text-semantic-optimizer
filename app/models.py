from pydantic import BaseModel, Field
from typing import List, Optional

class TextInput(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    optimization_level: Optional[str] = Field(default="medium", regex="^(light|medium|aggressive)$")
    preserve_keywords: Optional[List[str]] = Field(default_factory=list)
    
class TextResponse(BaseModel):
    original: str
    optimized: str
    metrics: dict
    suggestions: List[str]
