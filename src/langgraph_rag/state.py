"""State schema for LangGraph RAG."""

from typing import TypedDict, List, Optional, Annotated
from langchain_core.documents import Document


class GraphState(TypedDict):
    """State that flows through the LangGraph RAG pipeline.
    
    Attributes:
        question: The user's input question
        retrieved_documents: Documents retrieved from vector store
        relevant_documents: Documents that passed relevance grading
        is_sufficient: Whether enough relevant documents exist
        generation: The generated answer
        is_grounded: Whether the answer is grounded in documents
        retry_count: Current number of regeneration attempts
        max_retries: Maximum allowed regeneration attempts
        final_answer: The final answer to return
        status: Status of the pipeline execution
        sources: List of source document names
        steps: List of execution steps for debugging
    """
    
    # Input
    question: str
    
    # Retrieval
    retrieved_documents: List[Document]
    
    # Grading
    relevant_documents: List[Document]
    
    # Sufficiency
    is_sufficient: bool
    
    # Generation
    generation: str
    confidence_score: float  # Confidence score from 0.0 to 1.0
    
    # Validation
    is_grounded: bool
    retry_count: int
    max_retries: int
    
    # Output
    final_answer: str
    status: str  # "success", "insufficient_data", "validation_failed"
    sources: List[str]
    
    # Debug
    steps: List[dict]


# Default configuration
DEFAULT_CONFIG = {
    "max_retries": 3,
    "top_k_retrieval": 10,
    "min_relevant_docs": 1,
}
