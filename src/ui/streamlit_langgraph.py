"""LangGraph Enhanced Streamlit Application."""

import streamlit as st
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.langgraph_rag import run_rag_graph, GraphState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Telecom Support Assistant (LangGraph)",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .step-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .step-success {
            border-left: 4px solid #4CAF50;
        }
        .step-warning {
            border-left: 4px solid #FF9800;
        }
        .step-error {
            border-left: 4px solid #f44336;
        }
        .status-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        .status-success {
            background-color: #4CAF50;
            color: white;
        }
        .status-insufficient {
            background-color: #FF9800;
            color: white;
        }
        .status-failed {
            background-color: #f44336;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">📱 Telecom Customer Support Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enhanced with LangGraph Self-Corrective RAG</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            value=Config.OPENAI_API_KEY if Config.OPENAI_API_KEY else "",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            import os
            os.environ["OPENAI_API_KEY"] = api_key
            Config.OPENAI_API_KEY = api_key
        
        st.divider()
        
        # LangGraph settings
        st.subheader("🔧 LangGraph Settings")
        max_retries = st.slider("Max Regeneration Retries", 1, 5, 3)
        show_steps = st.checkbox("Show Execution Steps", value=True)
        
        st.divider()
        
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "What payment methods do you accept?",
            "How do I activate international roaming?",
            "What is the Fair Usage Policy?",
            "Can I change my plan anytime?",
            "What are the roaming charges for USA?",
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q[:20]}", use_container_width=True):
                st.session_state.question = q
        
        st.divider()
        
        # Info
        st.subheader("📊 LangGraph Flow")
        st.markdown("""
        1. 📚 **Retrieve** documents
        2. 🔍 **Grade** for relevance
        3. 📋 **Check** sufficiency
        4. 💡 **Generate** answer
        5. ✅ **Validate** grounding
        6. 🔄 **Regenerate** if needed
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input
        question = st.text_area(
            "Ask your question:",
            value=st.session_state.get("question", ""),
            height=100,
            placeholder="e.g., What payment methods do you accept?"
        )
        
        # Submit button
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not Config.OPENAI_API_KEY:
                st.error("Please provide your OpenAI API key in the sidebar.")
            else:
                with st.spinner("🔄 Running LangGraph RAG pipeline..."):
                    try:
                        # Run the LangGraph RAG
                        result = run_rag_graph(question, max_retries=max_retries)
                        
                        # Store result in session state
                        st.session_state.result = result
                        st.session_state.show_result = True
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.exception("Error running LangGraph RAG")
    
    with col2:
        # Status display
        if st.session_state.get("show_result") and st.session_state.get("result"):
            result = st.session_state.result
            status = result.get("status", "unknown")
            
            st.markdown("### Status")
            if status == "success":
                st.markdown('<span class="status-badge status-success">✅ Success</span>', unsafe_allow_html=True)
            elif status == "insufficient_data":
                st.markdown('<span class="status-badge status-insufficient">⚠️ Insufficient Data</span>', unsafe_allow_html=True)
            elif status == "validation_failed":
                st.markdown('<span class="status-badge status-failed">❌ Validation Failed</span>', unsafe_allow_html=True)
            
            # Quick stats
            steps = result.get("steps", [])
            st.metric("Execution Steps", len(steps))
            st.metric("Sources Used", len(result.get("sources", [])))
    
    # Display result
    if st.session_state.get("show_result") and st.session_state.get("result"):
        result = st.session_state.result
        
        st.divider()
        
        # Answer section
        st.markdown("### 💬 Answer")
        st.markdown(result.get("final_answer", "No answer generated."))
        
        # Sources
        if result.get("sources"):
            st.markdown("### 📄 Sources")
            for source in result["sources"]:
                st.markdown(f"- `{source}`")
        
        # Execution steps (debug mode)
        if show_steps and result.get("steps"):
            st.divider()
            st.markdown("### 🔍 Execution Steps")
            
            steps = result["steps"]
            
            for i, step in enumerate(steps, 1):
                node = step.get("node", "unknown")
                status = step.get("status", "unknown")
                
                # Determine step style
                if "end_success" in node:
                    style = "step-success"
                    icon = "✅"
                elif "end_insufficient" in node or "end_failed" in node:
                    style = "step-error"
                    icon = "❌"
                else:
                    style = "step-success"
                    icon = "✓"
                
                with st.expander(f"{icon} Step {i}: {node.replace('_', ' ').title()}", expanded=(i <= 3)):
                    st.markdown(f'<div class="step-box {style}">', unsafe_allow_html=True)
                    
                    # Display step details
                    for key, value in step.items():
                        if key not in ["node", "status"]:
                            if isinstance(value, list):
                                if key == "grading_results":
                                    st.markdown(f"**{key.replace('_', ' ').title()}:**")
                                    for gr in value:
                                        relevance = "✅ Relevant" if gr.get("relevant") else "❌ Not Relevant"
                                        st.markdown(f"  - {gr.get('source', 'unknown')}: {relevance}")
                                else:
                                    st.markdown(f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}")
                            else:
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Graph visualization (text-based)
            st.divider()
            st.markdown("### 📊 Execution Path")
            
            path_nodes = [step["node"] for step in steps]
            path_str = " → ".join([n.replace("_", " ").title() for n in path_nodes])
            st.code(path_str, language=None)


if __name__ == "__main__":
    main()
