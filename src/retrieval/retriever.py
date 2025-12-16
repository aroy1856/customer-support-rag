"""Simplified document retriever using LangChain's Chroma."""

import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from src.utils.config import Config

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Retrieves relevant document chunks using LangChain Chroma."""
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        top_k: int = None
    ):
        """Initialize the document retriever.
        
        Args:
            persist_directory: Directory where ChromaDB is persisted
            collection_name: Name of the collection
            top_k: Number of documents to retrieve
        """
        self.persist_directory = persist_directory or str(Config.VECTOR_STORE_PATH)
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.top_k = top_k or Config.TOP_K
        
        try:
            # Initialize embeddings with SSL bypass
            import httpx
            http_client = httpx.Client(verify=False)
            
            self.embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY,
                http_client=http_client
            )
            
            # Initialize Chroma vector store
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            
            logger.info(
                f"Initialized retriever with collection '{self.collection_name}' "
                f"from {self.persist_directory}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise RuntimeError(
                f"Failed to initialize document retriever. "
                f"Make sure the vector store has been built. Error: {e}"
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of retrieved document chunks with metadata and scores
        """
        k = top_k or self.top_k
        
        logger.info(f"Retrieving top {k} documents for query: '{query[:50]}...'")
        
        # Use similarity_search_with_score for scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        retrieved_chunks = []
        for doc, score in results:
            chunk = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': 1 - score,  # Convert distance to similarity
            }
            retrieved_chunks.append(chunk)
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        return retrieved_chunks
    
    def format_retrieved_chunks(
        self,
        chunks: List[Dict[str, Any]],
        include_scores: bool = False
    ) -> str:
        """Format retrieved chunks into a context string.
        
        Args:
            chunks: List of retrieved chunks
            include_scores: Whether to include relevance scores
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            content = chunk['content']
            
            if include_scores:
                score = chunk.get('score', 0)
                header = f"[Document {i} - {source} (Relevance: {score:.2%})]"
            else:
                header = f"[Document {i} - {source}]"
            
            context_parts.append(f"{header}\n{content}\n")
        
        return "\n".join(context_parts)
