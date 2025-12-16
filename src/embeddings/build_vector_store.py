"""Simplified vector store builder using LangChain's Chroma."""

import json
import logging
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_vector_store():
    """Build the vector store from processed chunks."""
    
    logger.info("=" * 60)
    logger.info("Building Vector Store with LangChain Chroma")
    logger.info("=" * 60)
    
    # Load chunks with embeddings
    chunks_file = Config.CHUNKS_DATA_DIR / "chunks_with_embeddings.json"
    logger.info(f"\nLoading chunks from {chunks_file}...")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize embeddings
    logger.info("\nInitializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Convert chunks to LangChain Documents
    logger.info("Converting chunks to documents...")
    documents = []
    for chunk in chunks:
        # Convert metadata values to strings
        metadata = {}
        for key, value in chunk['metadata'].items():
            metadata[key] = str(value)
        
        doc = Document(
            page_content=chunk['content'],
            metadata=metadata
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} documents")
    
    # Create Chroma vector store
    logger.info(f"\nCreating Chroma vector store at {Config.VECTOR_STORE_PATH}...")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=Config.COLLECTION_NAME,
        persist_directory=str(Config.VECTOR_STORE_PATH)
    )
    
    logger.info(f"[OK] Vector store created successfully!")
    logger.info(f"Collection: {Config.COLLECTION_NAME}")
    logger.info(f"Location: {Config.VECTOR_STORE_PATH}")
    logger.info(f"Total documents: {len(documents)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("[SUCCESS] Vector store build complete!")
    logger.info("=" * 60)
    
    return vectorstore


if __name__ == "__main__":
    try:
        build_vector_store()
    except Exception as e:
        logger.error(f"[ERROR] Failed to build vector store: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
