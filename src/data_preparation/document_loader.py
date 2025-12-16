"""Document loader for reading and loading raw telecom policy documents."""

from pathlib import Path
from typing import List, Dict
import logging

from src.utils.config import Config

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads raw documents from the data directory."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize the document loader.
        
        Args:
            data_dir: Path to the directory containing raw documents.
                     Defaults to Config.RAW_DATA_DIR.
        """
        self.data_dir = data_dir or Config.RAW_DATA_DIR
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_document(self, filename: str) -> Dict[str, str]:
        """Load a single document.
        
        Args:
            filename: Name of the document file to load
            
        Returns:
            Dictionary with 'filename', 'content', and 'path' keys
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        logger.info(f"Loading document: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'filename': filename,
                'content': content,
                'path': str(file_path)
            }
        except Exception as e:
            logger.error(f"Error loading document {filename}: {e}")
            raise
    
    def load_all_documents(self) -> List[Dict[str, str]]:
        """Load all documents specified in the configuration.
        
        Returns:
            List of dictionaries, each containing document data
        """
        documents = []
        
        for filename in Config.DOCUMENT_FILES:
            try:
                doc = self.load_document(filename)
                documents.append(doc)
                logger.info(f"Successfully loaded: {filename} ({len(doc['content'])} characters)")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                # Continue loading other documents
                continue
        
        logger.info(f"Loaded {len(documents)} documents in total")
        return documents
    
    def get_document_stats(self, documents: List[Dict[str, str]]) -> Dict[str, int]:
        """Get statistics about loaded documents.
        
        Args:
            documents: List of loaded documents
            
        Returns:
            Dictionary with statistics
        """
        total_chars = sum(len(doc['content']) for doc in documents)
        total_words = sum(len(doc['content'].split()) for doc in documents)
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_chars_per_doc': total_chars // len(documents) if documents else 0,
            'avg_words_per_doc': total_words // len(documents) if documents else 0
        }
