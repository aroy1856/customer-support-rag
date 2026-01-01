"""Grade documents node - filters documents by relevance."""

import json
import logging
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.utils.config import Config
from src.langgraph_rag.state import GraphState
from src.langgraph_rag.prompts import GRADE_DOCUMENT_PROMPT

logger = logging.getLogger(__name__)


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """Grade each retrieved document for relevance.
    
    Args:
        state: Current graph state with retrieved_documents
        
    Returns:
        Updated state with relevant_documents and step info
    """
    question = state["question"]
    documents = state["retrieved_documents"]
    
    logger.info(f"[GRADE] Grading {len(documents)} documents for relevance")
    
    # Initialize LLM for grading
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    relevant_documents: List[Document] = []
    grading_results = []
    
    for i, doc in enumerate(documents):
        # Format the grading prompt
        prompt = GRADE_DOCUMENT_PROMPT.format(
            document=doc.page_content[:1000],  # Limit document size
            question=question
        )
        
        try:
            # Get grading response
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Parse JSON response
            # Handle potential markdown code blocks
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            result = json.loads(response_text)
            is_relevant = result.get("relevant", "no").lower() == "yes"
            
            if is_relevant:
                relevant_documents.append(doc)
                grading_results.append({
                    "doc_index": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevant": True
                })
            else:
                grading_results.append({
                    "doc_index": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevant": False
                })
                
        except (json.JSONDecodeError, Exception) as e:
            # If parsing fails, include the document (err on side of inclusion)
            logger.warning(f"[GRADE] Error parsing grade for doc {i}: {e}")
            relevant_documents.append(doc)
            grading_results.append({
                "doc_index": i,
                "source": doc.metadata.get("source", "unknown"),
                "relevant": True,
                "note": "included due to parsing error"
            })
    
    logger.info(f"[GRADE] {len(relevant_documents)}/{len(documents)} documents are relevant")
    
    # Create step info
    step_info = {
        "node": "grade_documents",
        "status": "completed",
        "total_documents": len(documents),
        "relevant_documents": len(relevant_documents),
        "grading_results": grading_results
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "relevant_documents": relevant_documents,
        "steps": steps
    }
