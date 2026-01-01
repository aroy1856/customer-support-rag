"""Validate answer node - checks if answer is grounded in documents using structured output."""

import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI

from src.utils.config import Config
from src.langgraph_rag.state import GraphState
from src.langgraph_rag.models import GradeAnswer

logger = logging.getLogger(__name__)

# Validation prompt for structured output
VALIDATE_SYSTEM_PROMPT = """You are a grader assessing whether an answer is grounded in / supported by a set of documents.

Check if EVERY claim in the answer is supported by the documents.
The answer should not contain information not present in the documents.
Minor paraphrasing is acceptable as long as the meaning is preserved."""


def validate_answer_node(state: GraphState) -> Dict[str, Any]:
    """Validate that the generated answer is grounded in documents using structured output.
    
    Args:
        state: Current graph state with generation and relevant_documents
        
    Returns:
        Updated state with is_grounded and step info
    """
    generation = state["generation"]
    documents = state["relevant_documents"]
    retry_count = state.get("retry_count", 0)
    
    logger.info(f"[VALIDATE] Validating answer grounding (attempt {retry_count + 1})")
    
    # Format documents for validation
    doc_texts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        doc_texts.append(f"[Document {i} - {source}]\n{doc.page_content}\n")
    
    documents_text = "\n".join(doc_texts)
    
    # Initialize LLM with structured output
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create structured LLM
    structured_llm = llm.with_structured_output(GradeAnswer)
    
    try:
        # Validate answer with structured output
        messages = [
            ("system", VALIDATE_SYSTEM_PROMPT),
            ("human", f"""Documents:
{documents_text}

Answer to Validate:
{generation}

Is this answer fully grounded in the documents?""")
        ]
        
        result: GradeAnswer = structured_llm.invoke(messages)
        is_grounded = result.grounded
        reasoning = result.reasoning
        
    except Exception as e:
        logger.warning(f"[VALIDATE] Error validating: {e}")
        # If validation fails, assume grounded to avoid infinite loops
        is_grounded = True
        reasoning = f"Assumed grounded due to error: {str(e)}"
    
    logger.info(f"[VALIDATE] Answer grounded: {is_grounded}")
    if reasoning:
        logger.info(f"[VALIDATE] Reasoning: {reasoning}")
    
    # Create step info
    step_info = {
        "node": "validate_answer",
        "status": "completed",
        "is_grounded": is_grounded,
        "reasoning": reasoning,
        "retry_count": retry_count
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "is_grounded": is_grounded,
        "steps": steps
    }
