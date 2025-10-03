from pathlib import Path
from pydantic import BaseModel

class Config:
    # Пути
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    OCR_DIR = DATA_DIR / "ocr"
    STRUCTURED_DIR = DATA_DIR / "structured"
    DB_DIR = DATA_DIR / "db"
    
    # Создаём директории
    for dir_path in [RAW_DIR, OCR_DIR, STRUCTURED_DIR, DB_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # OCR настройки
    OCR_LANG = 'ru'
    OCR_USE_GPU = False
    
    # Embeddings
    EMBEDDING_MODEL = 'ai-forever/ru-en-RoSBERTa'
    
    # ChromaDB
    CHROMA_COLLECTION_PREFIX = "textbook"
    
    # Ollama
    OLLAMA_MODEL = "qwen3:8b"
    
    # Chunking параметры
    MAX_CHUNK_SIZE = 2000  # токенов

class TextbookMetadata(BaseModel):
    title: str
    author: str
    year: int
    grade: int
    subject: str  # "математика" или "история"
    isbn: str | None = None
    part: int | None = None