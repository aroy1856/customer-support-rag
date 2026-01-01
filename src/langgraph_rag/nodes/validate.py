"""Validate answer node - checks if answer is grounded in documents."""

import json
import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI

from src.utils.config import Config
from src.langgraph_rag.state import GraphState
from src.langgraph_rag.prompts import VALIDATE_ANSWER_PROMPT

logger = logging.getLogger(__name__)


def validate_answer_node(state: GraphState) -> Dict[str, Any]:
    """Validate that the generated answer is grounded in documents.
    
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
    
    # Initialize LLM for validation
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Validate answer
    prompt = VALIDATE_ANSWER_PROMPT.format(
        documents=documents_text,
        answer=generation
    )
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Parse JSON response
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        result = json.loads(response_text)
        is_grounded = result.get("grounded", "no").lower() == "yes"
        
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"[VALIDATE] Error parsing validation: {e}")
        # If parsing fails, assume grounded to avoid infinite loops
        is_grounded = True
    
    logger.info(f"[VALIDATE] Answer grounded: {is_grounded}")
    
    # Create step info
    step_info = {
        "node": "validate_answer",
        "status": "completed",
        "is_grounded": is_grounded,
        "retry_count": retry_count
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "is_grounded": is_grounded,
        "steps": steps
    }
