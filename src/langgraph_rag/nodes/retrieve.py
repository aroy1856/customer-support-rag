"""Retrieve node - fetches documents from vector store."""

import logging
from typing import Any, Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.utils.config import Config
from src.langgraph_rag.state import GraphState, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Retrieve documents from vector store.
    
    Args:
        state: Current graph state containing the question
        
    Returns:
        Updated state with retrieved_documents and step info
    """
    question = state["question"]
    top_k = DEFAULT_CONFIG["top_k_retrieval"]
    
    logger.info(f"[RETRIEVE] Retrieving top {top_k} documents for: '{question[:50]}...'")
    
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    vectorstore = Chroma(
        collection_name=Config.COLLECTION_NAME,
        persist_directory=str(Config.VECTOR_STORE_PATH),
        embedding_function=embeddings
    )
    
    # Retrieve documents
    documents = vectorstore.similarity_search(question, k=top_k)
    
    logger.info(f"[RETRIEVE] Retrieved {len(documents)} documents")
    
    # Create step info for debugging
    step_info = {
        "node": "retrieve",
        "status": "completed",
        "documents_retrieved": len(documents),
        "sources": list(set([doc.metadata.get("source", "unknown") for doc in documents]))
    }
    
    # Initialize steps list if not present
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "retrieved_documents": documents,
        "steps": steps
    }
