"""Embedding generation module using OpenAI embeddings."""

import logging
import ssl
import certifi
import httpx
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings

from src.utils.config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text chunks using OpenAI."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the OpenAI embedding model (default from Config)
            api_key: OpenAI API key (default from Config)
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")
        
        # Create custom HTTP client with SSL verification disabled (for development)
        # Note: In production, you should fix SSL certificate issues properly
        http_client = httpx.Client(verify=False)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key,
            http_client=http_client
        )
        
        logger.info(f"Initialized embedding generator with model: {self.model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_embeddings_for_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'content' and 'metadata' keys
            
        Returns:
            List of chunks with added 'embedding' key
        """
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            chunks_with_embeddings.append(chunk_copy)
        
        logger.info(f"Added embeddings to {len(chunks_with_embeddings)} chunks")
        return chunks_with_embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of embedding vectors
        """
        # Generate a test embedding to get dimension
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
