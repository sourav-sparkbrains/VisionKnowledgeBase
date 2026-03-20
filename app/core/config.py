"""
Application configuration.
Loads all settings from .env via pydantic-settings.
Import `settings` singleton everywhere — never instantiate Settings directly.
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #Groq
    GROQ_API_KEY: str
    OLLAMA_API_KEY: str

    #MongoDB
    MONGO_URI: str
    MONGO_DB_NAME: str
    MONGO_COLLECTION: str = "image_records"
    MONGO_USER_COLLECTION: str = "users"

    #Gdant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int =  6333
    QDRANT_COLLECTION_NAME: str = "images"
    QDRANT_API_KEY: str = ""

    #HuggingFace
    HF_API_KEY: str
    EMBEDDING_MODEL_NAME: str ="clip-ViT-B-32"
    VISION_MODEL: str = "HuggingFaceTB/SmolVLM-500M-Instruct"

    #Transformer
    LOCAL_LLM: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


    #App
    TOP_K: int = 5
    MAX_IMAGE_SIZE_MB: int = 10
    APP_ENV: str = "development"
    APP_NAME: str = "VisualKnowledgeBase"
    STORAGE_BASE_PATH: str = "storage/images"
    BASE_URL: str = "http://localhost:8000"
    EXPIRY_TIME: int = 1
    REFRESH_TIME: int = 7

    #Secret & Refresh keys
    SECRET_KEY: str
    REFRESH_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
