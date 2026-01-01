"""Regenerate answer node - retries with stricter prompt."""

import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI

from src.utils.config import Config
from src.langgraph_rag.state import GraphState, DEFAULT_CONFIG
from src.langgraph_rag.prompts import REGENERATE_ANSWER_PROMPT

logger = logging.getLogger(__name__)


def regenerate_answer_node(state: GraphState) -> Dict[str, Any]:
    """Regenerate answer with stricter grounding instructions.
    
    Args:
        state: Current graph state with relevant_documents
        
    Returns:
        Updated state with new generation and incremented retry_count
    """
    question = state["question"]
    documents = state["relevant_documents"]
    retry_count = state.get("retry_count", 0) + 1
    max_retries = state.get("max_retries", DEFAULT_CONFIG["max_retries"])
    
    logger.info(f"[REGENERATE] Regenerating answer (attempt {retry_count}/{max_retries})")
    
    # Format context from documents
    context_parts = []
    sources = set()
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        sources.add(source)
        context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}\n")
    
    context = "\n".join(context_parts)
    
    # Initialize LLM with lower temperature for more factual output
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0.1,  # Lower temperature for stricter grounding
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Regenerate with stricter prompt
    prompt = REGENERATE_ANSWER_PROMPT.format(
        context=context,
        question=question,
        retry_count=retry_count,
        max_retries=max_retries
    )
    
    response = llm.invoke(prompt)
    generation = response.content.strip()
    
    logger.info(f"[REGENERATE] Regenerated answer ({len(generation)} chars)")
    
    # Create step info
    step_info = {
        "node": "regenerate_answer",
        "status": "completed",
        "retry_count": retry_count,
        "answer_length": len(generation)
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "generation": generation,
        "retry_count": retry_count,
        "sources": list(sources),
        "steps": steps
    }
