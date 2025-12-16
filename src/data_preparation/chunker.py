"""Text chunking utilities for splitting documents into smaller segments."""

import logging
import tiktoken
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import Config

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunks documents into smaller segments with specified token size and overlap."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        """Initialize the document chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens (default from Config)
            chunk_overlap: Number of tokens to overlap between chunks (default from Config)
            encoding_name: Name of the tiktoken encoding to use
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.encoding_name = encoding_name
        
        # Initialize tiktoken encoder
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Could not load encoding {encoding_name}, using default: {e}")
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Initialize text splitter
        # Note: RecursiveCharacterTextSplitter uses characters, not tokens
        # We'll approximate: 1 token ≈ 4 characters (rough estimate)
        # For more accurate token-based splitting, we'll use a custom approach
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # Approximate character count
            chunk_overlap=self.chunk_overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(
            f"Initialized chunker: chunk_size={self.chunk_size} tokens, "
            f"overlap={self.chunk_overlap} tokens"
        )
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoder.encode(text))
    
    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """Chunk a text string into smaller segments.
        
        Args:
            text: Text to chunk
            source: Source identifier for the text (e.g., filename)
            
        Returns:
            List of chunk dictionaries with content, metadata, and token count
        """
        # Use LangChain's text splitter for initial splitting
        initial_chunks = self.text_splitter.split_text(text)
        
        # Refine chunks to meet token requirements
        refined_chunks = []
        for i, chunk_text in enumerate(initial_chunks):
            token_count = self.count_tokens(chunk_text)
            
            # If chunk is too large, split it further
            if token_count > self.chunk_size * 1.2:  # 20% tolerance
                # Split by sentences
                sentences = chunk_text.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    test_chunk = current_chunk + sentence + ". "
                    if self.count_tokens(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            refined_chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    refined_chunks.append(current_chunk.strip())
            else:
                refined_chunks.append(chunk_text)
        
        # Create chunk dictionaries with metadata
        chunks = []
        for i, chunk_text in enumerate(refined_chunks):
            chunk_dict = {
                'content': chunk_text,
                'metadata': {
                    'source': source,
                    'chunk_id': i,
                    'token_count': self.count_tokens(chunk_text),
                    'char_count': len(chunk_text)
                }
            }
            chunks.append(chunk_dict)
        
        logger.info(
            f"Created {len(chunks)} chunks from {source} "
            f"(original: {self.count_tokens(text)} tokens)"
        )
        
        return chunks
    
    def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, Any]]:
        """Chunk a document dictionary.
        
        Args:
            document: Dictionary with 'filename', 'content', and 'path' keys
            
        Returns:
            List of chunk dictionaries
        """
        return self.chunk_text(
            text=document['content'],
            source=document['filename']
        )
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate chunks meet the requirements.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with validation statistics
        """
        token_counts = [chunk['metadata']['token_count'] for chunk in chunks]
        
        # Check if chunks are within acceptable range (±10 tokens tolerance)
        tolerance = 10
        valid_chunks = [
            count for count in token_counts
            if count <= self.chunk_size + tolerance
        ]
        
        stats = {
            'total_chunks': len(chunks),
            'valid_chunks': len(valid_chunks),
            'invalid_chunks': len(chunks) - len(valid_chunks),
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
            'target_chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        logger.info(f"Chunk validation: {stats}")
        return stats
