"""Configuration management for the RAG Customer Support system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the RAG system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CHUNKS_DATA_DIR = DATA_DIR / "chunks"
    LOGS_DIR = PROJECT_ROOT / "logs"
    VECTOR_STORE_PATH = PROJECT_ROOT / os.getenv("VECTOR_STORE_PATH", "chroma_db")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chromadb")
    COLLECTION_NAME = "telecom_policies"
    
    # Chunking Configuration (as per project requirements)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # 500 tokens
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # 150 tokens
    
    # Retrieval Configuration
    TOP_K = int(os.getenv("TOP_K", "5"))  # Number of chunks to retrieve
    
    # Document files
    DOCUMENT_FILES = [
        "billing_policy.txt",
        "fup_policy.txt",
        "plan_activation.txt",
        "roaming_tariff.txt",
        "faqs.txt"
    ]
    
    # Logging Configuration
    LOG_FILE = LOGS_DIR / "interactions.log"
    LOG_LEVEL = "INFO"
    
    @classmethod
    def validate(cls):
        """Validate configuration and create necessary directories."""
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        cls.CHUNKS_DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        
        # Validate OpenAI API key
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in the .env file."
            )
        
        # Validate document files exist
        missing_files = []
        for doc_file in cls.DOCUMENT_FILES:
            if not (cls.RAW_DATA_DIR / doc_file).exists():
                missing_files.append(doc_file)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing document files in {cls.RAW_DATA_DIR}: {missing_files}"
            )
        
        return True


# Validate configuration on import (optional, can be called explicitly)
# Config.validate()
