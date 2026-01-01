"""Generate answer node - creates answer from relevant documents with confidence."""

import logging
from typing import Any, Dict
from langchain_openai import ChatOpenAI

from src.utils.config import Config
from src.langgraph_rag.state import GraphState
from src.langgraph_rag.models import GeneratedAnswer

logger = logging.getLogger(__name__)

# Generation system prompt
GENERATE_SYSTEM_PROMPT = """You are a helpful telecom customer support assistant. 
Answer the user's question based ONLY on the provided context documents.

Instructions:
1. Answer the question using ONLY information from the context documents
2. Be concise but comprehensive
3. If the context doesn't contain enough information, say so
4. Do not make up or infer information not present in the documents
5. Rate your confidence based on how well the documents support your answer"""


def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """Generate answer using relevant documents with structured output.
    
    Args:
        state: Current graph state with relevant_documents
        
    Returns:
        Updated state with generation, confidence_score and step info
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
    
    # Initialize LLM with structured output
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0.3,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create structured LLM
    structured_llm = llm.with_structured_output(GeneratedAnswer)
    
    try:
        # Generate answer with structured output
        messages = [
            ("system", GENERATE_SYSTEM_PROMPT),
            ("human", f"""Context Documents:
{context}

Question: {question}

Provide your answer and confidence score.""")
        ]
        
        result: GeneratedAnswer = structured_llm.invoke(messages)
        generation = result.answer
        confidence = result.confidence
        
    except Exception as e:
        logger.warning(f"[GENERATE] Structured output failed, falling back: {e}")
        # Fallback to regular generation
        llm_fallback = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.3,
            openai_api_key=Config.OPENAI_API_KEY
        )
        response = llm_fallback.invoke(f"""Based on these documents, answer the question.

Documents:
{context}

Question: {question}

Answer:""")
        generation = response.content.strip()
        confidence = 0.7  # Default confidence for fallback
    
    logger.info(f"[GENERATE] Generated answer ({len(generation)} chars, confidence: {confidence})")
    
    # Create step info
    step_info = {
        "node": "generate_answer",
        "status": "completed",
        "documents_used": len(documents),
        "sources": list(sources),
        "answer_length": len(generation),
        "confidence_score": confidence
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "generation": generation,
        "confidence_score": confidence,
        "sources": list(sources),
        "steps": steps
    }
