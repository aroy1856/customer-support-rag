"""Generate answer node - creates answer from relevant documents."""

import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI

from src.utils.config import Config
from src.langgraph_rag.state import GraphState
from src.langgraph_rag.prompts import GENERATE_ANSWER_PROMPT

logger = logging.getLogger(__name__)


def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """Generate answer using relevant documents.
    
    Args:
        state: Current graph state with relevant_documents
        
    Returns:
        Updated state with generation and step info
    """
    question = state["question"]
    documents = state["relevant_documents"]
    
    logger.info(f"[GENERATE] Generating answer using {len(documents)} relevant documents")
    
    # Format context from documents
    context_parts = []
    sources = set()
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        sources.add(source)
        context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}\n")
    
    context = "\n".join(context_parts)
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0.3,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Generate answer
    prompt = GENERATE_ANSWER_PROMPT.format(
        context=context,
        question=question
    )
    
    response = llm.invoke(prompt)
    generation = response.content.strip()
    
    logger.info(f"[GENERATE] Generated answer ({len(generation)} chars)")
    
    # Create step info
    step_info = {
        "node": "generate_answer",
        "status": "completed",
        "documents_used": len(documents),
        "sources": list(sources),
        "answer_length": len(generation)
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "generation": generation,
        "sources": list(sources),
        "steps": steps
    }
