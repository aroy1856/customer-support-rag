"""LangGraph RAG Graph Construction."""

import logging
from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, END

from src.langgraph_rag.state import GraphState, DEFAULT_CONFIG
from src.langgraph_rag.prompts import SUFFICIENCY_MESSAGE
from src.langgraph_rag.nodes import (
    retrieve_node,
    grade_documents_node,
    generate_answer_node,
    validate_answer_node,
    regenerate_answer_node,
)

logger = logging.getLogger(__name__)


# === Conditional Edge Functions ===

def check_sufficiency(state: GraphState) -> Literal["generate", "insufficient"]:
    """Check if there are sufficient relevant documents.
    
    Args:
        state: Current graph state with relevant_documents
        
    Returns:
        Next node to route to
    """
    relevant_docs = state.get("relevant_documents", [])
    min_docs = DEFAULT_CONFIG["min_relevant_docs"]
    
    if len(relevant_docs) >= min_docs:
        logger.info(f"[ROUTE] Sufficient docs ({len(relevant_docs)} >= {min_docs}) -> generate")
        return "generate"
    else:
        logger.info(f"[ROUTE] Insufficient docs ({len(relevant_docs)} < {min_docs}) -> insufficient")
        return "insufficient"


def check_validation(state: GraphState) -> Literal["end_success", "regenerate", "end_failed"]:
    """Check validation result and decide next step.
    
    Args:
        state: Current graph state with is_grounded and retry_count
        
    Returns:
        Next node to route to
    """
    is_grounded = state.get("is_grounded", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", DEFAULT_CONFIG["max_retries"])
    
    if is_grounded:
        logger.info("[ROUTE] Answer grounded -> end_success")
        return "end_success"
    elif retry_count < max_retries:
        logger.info(f"[ROUTE] Not grounded, retries left ({retry_count}/{max_retries}) -> regenerate")
        return "regenerate"
    else:
        logger.info(f"[ROUTE] Not grounded, max retries reached -> end_failed")
        return "end_failed"


# === End Node Functions ===

def end_insufficient(state: GraphState) -> Dict[str, Any]:
    """Handle insufficient data case."""
    question = state["question"]
    
    final_answer = SUFFICIENCY_MESSAGE.format(question=question)
    
    step_info = {
        "node": "end_insufficient",
        "status": "completed",
        "reason": "No relevant documents found"
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "final_answer": final_answer,
        "status": "insufficient_data",
        "is_sufficient": False,
        "steps": steps
    }


def end_success(state: GraphState) -> Dict[str, Any]:
    """Handle successful answer generation."""
    generation = state.get("generation", "")
    sources = state.get("sources", [])
    
    # Format final answer with sources
    if sources:
        source_text = "\n\n**Sources:** " + ", ".join(sources)
        final_answer = generation + source_text
    else:
        final_answer = generation
    
    step_info = {
        "node": "end_success",
        "status": "completed",
        "sources": sources
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "final_answer": final_answer,
        "status": "success",
        "is_sufficient": True,
        "steps": steps
    }


def end_failed(state: GraphState) -> Dict[str, Any]:
    """Handle validation failure after max retries."""
    generation = state.get("generation", "")
    sources = state.get("sources", [])
    retry_count = state.get("retry_count", 0)
    
    # Return the last generation with a warning
    warning = "\n\n⚠️ *Note: This answer may contain information not fully verified against the source documents.*"
    
    if sources:
        source_text = "\n\n**Sources:** " + ", ".join(sources)
        final_answer = generation + source_text + warning
    else:
        final_answer = generation + warning
    
    step_info = {
        "node": "end_failed",
        "status": "completed",
        "reason": f"Max retries ({retry_count}) reached without validation",
        "sources": sources
    }
    
    steps = state.get("steps", [])
    steps.append(step_info)
    
    return {
        "final_answer": final_answer,
        "status": "validation_failed",
        "is_sufficient": True,
        "steps": steps
    }


def create_rag_graph() -> StateGraph:
    """Create the LangGraph RAG workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating LangGraph RAG workflow...")
    
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # === Add Nodes ===
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("validate_answer", validate_answer_node)
    workflow.add_node("regenerate_answer", regenerate_answer_node)
    workflow.add_node("end_insufficient", end_insufficient)
    workflow.add_node("end_success", end_success)
    workflow.add_node("end_failed", end_failed)
    
    # === Define Edges ===
    
    # Start -> Retrieve
    workflow.set_entry_point("retrieve")
    
    # Retrieve -> Grade Documents
    workflow.add_edge("retrieve", "grade_documents")
    
    # Grade Documents -> Check Sufficiency (conditional)
    workflow.add_conditional_edges(
        "grade_documents",
        check_sufficiency,
        {
            "generate": "generate_answer",
            "insufficient": "end_insufficient"
        }
    )
    
    # Generate Answer -> Validate Answer
    workflow.add_edge("generate_answer", "validate_answer")
    
    # Validate Answer -> Check Validation (conditional)
    workflow.add_conditional_edges(
        "validate_answer",
        check_validation,
        {
            "end_success": "end_success",
            "regenerate": "regenerate_answer",
            "end_failed": "end_failed"
        }
    )
    
    # Regenerate -> Validate (loop back)
    workflow.add_edge("regenerate_answer", "validate_answer")
    
    # End nodes -> END
    workflow.add_edge("end_insufficient", END)
    workflow.add_edge("end_success", END)
    workflow.add_edge("end_failed", END)
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("LangGraph RAG workflow created successfully")
    
    return app


def run_rag_graph(question: str, max_retries: int = 3) -> Dict[str, Any]:
    """Run the RAG graph with a question.
    
    Args:
        question: The user's question
        max_retries: Maximum regeneration attempts (default: 3)
        
    Returns:
        Final state with answer and execution details
    """
    logger.info(f"Running LangGraph RAG for: '{question[:50]}...'")
    
    # Create the graph
    app = create_rag_graph()
    
    # Initialize state
    initial_state = {
        "question": question,
        "retrieved_documents": [],
        "relevant_documents": [],
        "is_sufficient": False,
        "generation": "",
        "is_grounded": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "final_answer": "",
        "status": "",
        "sources": [],
        "steps": []
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    logger.info(f"LangGraph RAG completed with status: {result.get('status')}")
    
    return result


# For direct testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_question = "What payment methods do you accept?"
    result = run_rag_graph(test_question)
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Sources: {result['sources']}")
    print(f"\nAnswer:\n{result['final_answer']}")
    print(f"\nSteps: {len(result['steps'])}")
    for step in result['steps']:
        print(f"  - {step['node']}: {step['status']}")
