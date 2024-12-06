from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from typing import List
from .text_processor import TextOptimizer
from .exceptions import *
from .config import get_settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Text Semantic Optimizer",
    description="API for optimizing text semantics and readability",
    version="0.1.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize text optimizer
text_optimizer = TextOptimizer()

class TextInput(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    optimization_level: str = Field(default="medium", regex="^(light|medium|aggressive)$")
    preserve_keywords: List[str] = []

    @validator('preserve_keywords', each_item=True)
    def preserve_keyword_length(cls, v):
        if len(v) > 50:
            raise ValueError("Preserve keyword must be less than 50 characters")
        return v

@app.exception_handler(TextOptimizationError)
async def optimization_exception_handler(request: Request, exc: TextOptimizationError):
    logger.error(f"Text optimization error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={"error": exc.message}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An unexpected error occurred"}
    )

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/api/optimize", response_model=TextResponse)
async def optimize_text(text_input: TextInput):
    try:
        logger.info(f"Processing text optimization request. Length: {len(text_input.content)}")
        
        optimized_text, metrics, suggestions = text_optimizer.optimize_text(
            text_input.content,
            text_input.optimization_level,
            text_input.preserve_keywords
        )
        
        logger.info("Text optimization completed successfully")
        
        return TextResponse(
            original=text_input.content,
            optimized=optimized_text,
            metrics=metrics,
            suggestions=suggestions
        )
        
    except TextOptimizationError as e:
        logger.error(f"Text optimization error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Unexpected error during optimization: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Test example
if __name__ == "__main__":
    text_input = TextInput(
        content="The quick brown fox jumps over the lazy dog.",
        optimization_level="medium",
        preserve_keywords=[]
    )
    
    optimized_text, metrics, suggestions = text_optimizer.optimize_text(
        text_input.content,
        text_input.optimization_level,
        text_input.preserve_keywords
    )
    
    print("Original text:")
    print(text_input.content)
    print("\nOptimized text:")
    print(optimized_text)
    print("\nMetrics:")
    print(metrics)
    print("\nSuggestions:")
    print(suggestions)
