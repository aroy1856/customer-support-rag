"""Streamlit web interface for the Telecom Customer Support RAG Assistant."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generation.answer_generator import AnswerGenerator
from src.utils.config import Config

# Page configuration
st.set_page_config(
    page_title="Telecom Customer Support Assistant",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-top: 1rem;
    }
    .chunk-box {
        background-color: #FEF3C7;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #F59E0B;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_answer_generator():
    """Initialize the answer generator (cached for performance)."""
    try:
        return AnswerGenerator()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.info("Please make sure you have:")
        st.info("1. Set OPENAI_API_KEY in the .env file")
        st.info("2. Built the vector store by running: python -m src.embeddings.build_vector_store")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📞 Telecom Customer Support Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered support for billing, plans, roaming, and policies</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=Config.OPENAI_API_KEY if Config.OPENAI_API_KEY else "",
            type="password",
            help="Enter your OpenAI API key. Leave empty to use .env file."
        )
        
        # Update API key in session state if provided
        if api_key_input:
            import os
            os.environ["OPENAI_API_KEY"] = api_key_input
            Config.OPENAI_API_KEY = api_key_input
        
        # Vector store rebuild button
        st.markdown("---")
        st.subheader("🔄 Vector Store Management")
        
        if st.button("🔨 Rebuild Vector Store", help="Rebuild the vector store from scratch"):
            if not Config.OPENAI_API_KEY:
                st.error("⚠️ Please provide an OpenAI API key first!")
            else:
                with st.spinner("Rebuilding vector store... This may take a minute."):
                    try:
                        from src.embeddings import build_vector_store
                        build_vector_store()
                        st.success("✅ Vector store rebuilt successfully!")
                        st.info("Please refresh the page to use the new vector store.")
                        # Clear cache to force reload
                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(f"❌ Error rebuilding vector store: {e}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info(
            "This AI assistant helps you find answers to telecom-related questions "
            "using our policy documents. Ask about:\n\n"
            "• Billing and payments\n"
            "• Data plans and FUP\n"
            "• Roaming charges\n"
            "• Plan activation/deactivation\n"
            "• General policies"
        )
        
        st.header("⚙️ Settings")
        top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=Config.TOP_K,
            help="More documents provide more context but may slow down responses"
        )
        
        show_sources = st.checkbox("Show source documents", value=True)
        show_retrieved_chunks = st.checkbox("Show retrieved chunks (debug)", value=False)
        
        st.header("📊 System Info")
        try:
            answer_gen = initialize_answer_generator()
            if answer_gen:
                # Get count from LangChain Chroma vectorstore
                total_chunks = answer_gen.retriever.vectorstore._collection.count()
                st.success("Vector store loaded")
                st.metric("Total document chunks", total_chunks)
                st.metric("LLM Model", Config.LLM_MODEL)
                st.metric("Embedding Model", Config.EMBEDDING_MODEL)
        except Exception as e:
            st.error(f"System not initialized: {e}")
    
    # Initialize answer generator
    answer_gen = initialize_answer_generator()
    
    if answer_gen is None:
        st.error("⚠️ System initialization failed. Please check the sidebar for instructions.")
        return
    
    # Main content area
    st.markdown("---")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Billing & Payments:**
            - What payment methods do you accept?
            - When will I receive my monthly bill?
            - How do I dispute a billing error?
            
            **Data & Internet:**
            - What is Fair Usage Policy?
            - How do I check my data usage?
            - Can I buy additional high-speed data?
            """)
        with col2:
            st.markdown("""
            **Roaming:**
            - What are the international roaming charges?
            - How do I activate international roaming?
            - Do I need to pay for domestic roaming?
            
            **Plans:**
            - How do I change my plan?
            - How long does plan activation take?
            - Can I port my number to your network?
            """)
    
    # Query input
    st.markdown("### 💬 Ask Your Question")
    query = st.text_area(
        "Type your question here:",
        height=100,
        placeholder="E.g., What are the roaming charges for international travel?"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit_button = st.button("🔍 Get Answer", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    # Process query
    if submit_button and query.strip():
        with st.spinner("🤔 Searching our policy documents and generating answer..."):
            try:
                # Generate answer
                result = answer_gen.generate_answer(
                    query=query,
                    top_k=top_k,
                    include_sources=show_sources
                )
                
                # Display answer
                st.markdown("---")
                st.markdown("### ✅ Answer")
                st.markdown(result['answer'])
                
                # Display retrieved chunks (debug mode)
                if show_retrieved_chunks and result['retrieved_chunks']:
                    st.markdown("---")
                    st.markdown("### 📄 Retrieved Document Chunks")
                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        with st.expander(
                            f"Chunk {i}: {chunk['metadata']['source']} "
                            f"(Distance: {chunk['distance']:.4f})"
                        ):
                            st.markdown(f"**Source:** {chunk['metadata']['source']}")
                            st.markdown(f"**Distance:** {chunk['distance']:.4f} (lower is better)")
                            st.markdown(f"**Token Count:** {chunk['metadata']['token_count']}")
                            st.markdown("**Content:**")
                            st.markdown(f'<div class="chunk-box">{chunk["content"]}</div>', 
                                      unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
                st.info("Please try rephrasing your question or contact support.")
    
    elif submit_button:
        st.warning("⚠️ Please enter a question before submitting.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #64748B; font-size: 0.9rem;">'
        'Powered by RAG (Retrieval-Augmented Generation) | '
        'OpenAI GPT-4o-mini | ChromaDB'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
