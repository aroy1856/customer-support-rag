"""Prompt templates for the RAG answer generation system."""

from typing import List, Dict, Any


class PromptTemplates:
    """Collection of prompt templates for the RAG system."""
    
    # System prompt for the customer support assistant
    SYSTEM_PROMPT = """You are a helpful and professional telecom customer support assistant. Your role is to answer customer questions about telecom policies, billing, plans, roaming, and related topics.

IMPORTANT GUIDELINES:
1. Answer ONLY based on the provided context from our policy documents
2. If the answer is not in the context, politely say you don't have that information
3. Be concise, clear, and customer-friendly in your responses
4. Use simple language that customers can easily understand
5. When mentioning specific charges, dates, or procedures, cite them accurately from the context
6. If the question requires multiple pieces of information, organize your answer with bullet points
7. Always maintain a helpful and empathetic tone

Remember: You represent our telecom company's customer support. Be accurate, helpful, and professional."""

    # Template for injecting context and query
    RAG_PROMPT_TEMPLATE = """Based on the following information from our policy documents, please answer the customer's question.

CONTEXT FROM POLICY DOCUMENTS:
{context}

CUSTOMER QUESTION:
{query}

ANSWER:
Please provide a clear and helpful answer based solely on the information above. If the context doesn't contain enough information to answer the question, politely inform the customer and suggest they contact customer care for more specific assistance."""

    # Template for formatting source references
    SOURCE_REFERENCE_TEMPLATE = """

---
Source Documents:
{sources}"""

    @classmethod
    def format_rag_prompt(
        cls,
        query: str,
        context: str,
        include_system: bool = False
    ) -> str:
        """Format the RAG prompt with query and context.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            include_system: Whether to include system prompt
            
        Returns:
            Formatted prompt string
        """
        prompt = cls.RAG_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )
        
        if include_system:
            prompt = f"{cls.SYSTEM_PROMPT}\n\n{prompt}"
        
        return prompt
    
    @classmethod
    def format_source_references(
        cls,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """Format source references from retrieved chunks.
        
        Args:
            retrieved_chunks: List of retrieved document chunks
            
        Returns:
            Formatted source references string
        """
        sources = []
        seen_sources = set()
        
        for chunk in retrieved_chunks:
            source = chunk['metadata']['source']
            if source not in seen_sources:
                sources.append(f"- {source}")
                seen_sources.add(source)
        
        sources_text = "\n".join(sources)
        return cls.SOURCE_REFERENCE_TEMPLATE.format(sources=sources_text)
    
    @classmethod
    def format_complete_response(
        cls,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> str:
        """Format the complete response with answer and sources.
        
        Args:
            answer: Generated answer from LLM
            retrieved_chunks: Retrieved document chunks
            include_sources: Whether to include source references
            
        Returns:
            Complete formatted response
        """
        if include_sources:
            sources = cls.format_source_references(retrieved_chunks)
            return f"{answer}{sources}"
        return answer
