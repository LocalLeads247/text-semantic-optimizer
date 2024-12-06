from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Text Semantic Optimizer"
    version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api"
    nlp_model: str = "en_core_web_sm"
    max_text_length: int = 10000
    min_text_length: int = 1
    
    # Cache settings
    cache_ttl: int = 3600  # 1 hour
    cache_size: int = 1000
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()