"""Pydantic models for structured LLM outputs."""

from pydantic import BaseModel, Field


class GradeDocument(BaseModel):
    """Model for document relevance grading."""
    
    relevant: bool = Field(
        description="Whether the document is relevant to the question. True if relevant, False otherwise."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation for the relevance decision."
    )


class GradeAnswer(BaseModel):
    """Model for answer grounding validation."""
    
    grounded: bool = Field(
        description="Whether the answer is fully grounded in/supported by the documents. True if grounded, False otherwise."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation for the grounding decision."
    )


class GeneratedAnswer(BaseModel):
    """Model for structured answer generation."""
    
    answer: str = Field(
        description="The generated answer based on the provided documents."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0 indicating how well the documents support this answer."
    )
