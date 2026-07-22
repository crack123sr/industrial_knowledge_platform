import os
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Industrial Knowledge Intelligence Platform"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # LLM & Embeddings (Updated for Gemini)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  
    LLM_MODEL: str = "gemini-3.5-flash"
    TEMPERATURE: float = 0.2
    
    # Storage Paths
    FAISS_INDEX_PATH: str = str(PROJECT_ROOT / "data" / "faiss_index")
    FAISS_DIMENSION: int = 384  
    KG_STORAGE_PATH: str = str(PROJECT_ROOT / "data" / "knowledge_graph.pkl")
    DOCUMENTS_PATH: str = str(PROJECT_ROOT / "data" / "documents")
    
    # RAG Parameters
    TOP_K_DOCUMENTS: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
Path(settings.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(settings.DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)

