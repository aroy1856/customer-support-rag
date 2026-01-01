"""Grade documents node - filters documents by relevance using structured output."""

import logging
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.utils.config import Config
from src.langgraph_rag.state import GraphState
from src.langgraph_rag.models import GradeDocument

logger = logging.getLogger(__name__)

# Grading prompt for structured output
GRADE_SYSTEM_PROMPT = """You are a grader assessing the relevance of a retrieved document to a user question.

If the document contains keywords, concepts, or information related to the user question, grade it as relevant.
The document does not need to fully answer the question, just be topically relevant."""


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """Grade each retrieved document for relevance using structured output.
    
    Args:
        state: Current graph state with retrieved_documents
        
    Returns:
        Updated state with relevant_documents and step info
    """
    question = state["question"]
    documents = state["retrieved_documents"]
    
    logger.info(f"[GRADE] Grading {len(documents)} documents for relevance")
    
    # Initialize LLM with structured output
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=0,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create structured LLM
    structured_llm = llm.with_structured_output(GradeDocument)
    
    relevant_documents: List[Document] = []
    grading_results = []
    
    for i, doc in enumerate(documents):
        # Format the grading prompt
        messages = [
            ("system", GRADE_SYSTEM_PROMPT),
            ("human", f"""Document:
{doc.page_content[:1000]}

Question: {question}

Is this document relevant to the question?""")
        ]
        
        try:
            # Get structured grading response
            result: GradeDocument = structured_llm.invoke(messages)
            
            if result.relevant:
                relevant_documents.append(doc)
                grading_results.append({
                    "doc_index": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevant": True,
                    "reasoning": result.reasoning
                })
            else:
                grading_results.append({
                    "doc_index": i,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevant": False,
                    "reasoning": result.reasoning
                })
                
        except Exception as e:
            # If structured output fails, include the document (err on side of inclusion)
            logger.warning(f"[GRADE] Error grading doc {i}: {e}")
            relevant_documents.append(doc)
            grading_results.append({
                "doc_index": i,
                "source": doc.metadata.get("source", "unknown"),
                "relevant": True,
                "note": f"included due to error: {str(e)}"
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
