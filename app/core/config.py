from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "YouTube AI Tutor"
    VERSION: str = "1.0.0"
    MONGODB_URL: str
    DATABASE_NAME: str
    
    # AI Keys
    HUGGINGFACE_API_KEY: Optional[str] = None
    QWEN_API_KEY: Optional[str] = None
    # Inside your Settings class:
    GROQ_API_KEY: Optional[str] = None

    #Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "youtube-ai-tutor"

    # Inside your Settings class
    SECRET_KEY: str = "default_secret"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080
    
    class Config:
        env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

settings = Settings()
