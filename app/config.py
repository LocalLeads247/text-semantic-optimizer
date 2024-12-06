from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Text Semantic Optimizer"
    debug: bool = False
    model_name: str = "en_core_web_sm"
    max_text_length: int = 10000
    min_text_length: int = 1
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
