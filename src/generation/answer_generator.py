"""RAG answer generation module using OpenAI LLM."""

import logging
import httpx
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.retrieval.retriever import DocumentRetriever
from src.generation.prompt_templates import PromptTemplates
from src.utils.config import Config
from src.utils.logger import interaction_logger

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers using RAG (Retrieval-Augmented Generation)."""
    
    def __init__(
        self,
        retriever: DocumentRetriever = None,
        llm_model: str = None,
        api_key: str = None,
        temperature: float = 0.3
    ):
        """Initialize the answer generator.
        
        Args:
            retriever: Document retriever instance (creates new if None)
            llm_model: Name of the OpenAI model (default from Config)
            api_key: OpenAI API key (default from Config)
            temperature: LLM temperature for response generation
        """
        self.retriever = retriever or DocumentRetriever()
        self.llm_model = llm_model or Config.LLM_MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")
        
        # Create custom HTTP client with SSL verification disabled (for development)
        http_client = httpx.Client(verify=False)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=temperature,
            openai_api_key=self.api_key,
            http_client=http_client
        )
        
        logger.info(f"Initialized answer generator with model: {self.llm_model}")
    
    def generate_answer(
        self,
        query: str,
        top_k: int = None,
        include_sources: bool = True,
        log_interaction: bool = True
    ) -> Dict[str, Any]:
        """Generate an answer for a user query using RAG.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve (default from Config)
            include_sources: Whether to include source references in response
            log_interaction: Whether to log this interaction
            
        Returns:
            Dictionary with 'answer', 'retrieved_chunks', and 'sources' keys
        """
        logger.info(f"Generating answer for query: '{query[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_chunks = self.retriever.retrieve(query, top_k)
            
            if not retrieved_chunks:
                logger.warning("No relevant documents found for query")
                return {
                    'answer': "I apologize, but I couldn't find relevant information in our policy documents to answer your question. Please contact our customer care at 1800-XXX-XXXX for assistance.",
                    'retrieved_chunks': [],
                    'sources': []
                }
            
            # Step 2: Format context from retrieved chunks
            context = self.retriever.format_retrieved_chunks(
                retrieved_chunks,
                include_scores=False
            )
            
            # Step 3: Create prompt
            prompt = PromptTemplates.format_rag_prompt(
                query=query,
                context=context,
                include_system=False
            )
            
            # Step 4: Generate answer using LLM
            messages = [
                SystemMessage(content=PromptTemplates.SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Step 5: Format complete response with sources
            if include_sources:
                complete_answer = PromptTemplates.format_complete_response(
                    answer=answer,
                    retrieved_chunks=retrieved_chunks,
                    include_sources=True
                )
            else:
                complete_answer = answer
            
            # Extract unique sources
            sources = list(set([
                chunk['metadata']['source']
                for chunk in retrieved_chunks
            ]))
            
            result = {
                'answer': complete_answer,
                'retrieved_chunks': retrieved_chunks,
                'sources': sources,
                'query': query
            }
            
            # Log interaction
            if log_interaction:
                interaction_logger.log_interaction(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_response=complete_answer,
                    metadata={'model': self.llm_model, 'top_k': top_k or Config.TOP_K}
                )
            
            logger.info(f"Successfully generated answer ({len(answer)} characters)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            interaction_logger.log_error(str(e), query)
            raise
    
    def generate_answer_simple(self, query: str) -> str:
        """Generate a simple answer (just the text) for a query.
        
        Args:
            query: User's question
            
        Returns:
            Generated answer as string
        """
        result = self.generate_answer(query)
        return result['answer']
