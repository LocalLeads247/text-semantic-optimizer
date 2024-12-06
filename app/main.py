from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from text_processor import TextProcessor

app = FastAPI(
    title="Semantic Text Optimizer",
    description="AI-powered text optimization with NLP features"
)

class TextInput(BaseModel):
    text: str
    num_topics: Optional[int] = 3

class TextAnalysisResponse(BaseModel):
    entities: Dict[str, List[str]]
    topics: List[Dict]
    readability: Dict
    statistics: Dict

text_processor = TextProcessor()

@app.post("/analyze")
async def analyze_text(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        return {
            "entities": text_processor.extract_entities(input_data.text),
            "topics": text_processor.cluster_topics(input_data.text, input_data.num_topics),
            "readability": text_processor.analyze_readability(input_data.text),
            "statistics": text_processor.get_text_statistics(input_data.text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")