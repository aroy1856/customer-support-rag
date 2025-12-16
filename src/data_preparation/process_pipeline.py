"""Data processing pipeline to load, clean, and chunk documents."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.data_preparation.document_loader import DocumentLoader
from src.data_preparation.text_cleaner import TextCleaner
from src.data_preparation.chunker import DocumentChunker
from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_documents() -> List[Dict[str, Any]]:
    """Process all documents: load, clean, and chunk.
    
    Returns:
        List of all chunks from all documents
    """
    logger.info("=" * 60)
    logger.info("Starting document processing pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load documents
    logger.info("\n[Step 1/3] Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    stats = loader.get_document_stats(documents)
    logger.info(f"Loaded {stats['total_documents']} documents")
    logger.info(f"Total words: {stats['total_words']:,}")
    
    # Step 2: Clean documents
    logger.info("\n[Step 2/3] Cleaning documents...")
    cleaner = TextCleaner()
    cleaned_documents = []
    for doc in documents:
        cleaned_doc = cleaner.clean_document(doc)
        cleaned_documents.append(cleaned_doc)
        logger.info(
            f"  {doc['filename']}: "
            f"{cleaned_doc['original_length']:,} → {cleaned_doc['cleaned_length']:,} chars"
        )
    
    # Step 3: Chunk documents
    logger.info("\n[Step 3/3] Chunking documents...")
    chunker = DocumentChunker()
    all_chunks = chunker.chunk_documents(cleaned_documents)
    
    # Validate chunks
    validation_stats = chunker.validate_chunks(all_chunks)
    logger.info(f"\nChunk validation results:")
    logger.info(f"  Total chunks: {validation_stats['total_chunks']}")
    logger.info(f"  Valid chunks: {validation_stats['valid_chunks']}")
    logger.info(f"  Min tokens: {validation_stats['min_tokens']}")
    logger.info(f"  Max tokens: {validation_stats['max_tokens']}")
    logger.info(f"  Avg tokens: {validation_stats['avg_tokens']:.1f}")
    
    # Save chunks to JSON file
    chunks_file = Config.CHUNKS_DATA_DIR / "processed_chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(all_chunks)} chunks to {chunks_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Document processing pipeline completed successfully!")
    logger.info("=" * 60)
    
    return all_chunks


if __name__ == "__main__":
    chunks = process_documents()
    print(f"\n✓ Successfully processed {len(chunks)} chunks")
