"""Logging utilities for the RAG Customer Support system."""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from src.utils.config import Config


class InteractionLogger:
    """Logger for user interactions with the RAG system."""
    
    def __init__(self, log_file: Path = None):
        """Initialize the interaction logger.
        
        Args:
            log_file: Path to the log file. Defaults to Config.LOG_FILE.
        """
        self.log_file = log_file or Config.LOG_FILE
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("RAGInteractionLogger")
        self.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # File handler for JSON logs
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_interaction(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        generated_response: str,
        metadata: Dict[str, Any] = None
    ):
        """Log a complete user interaction.
        
        Args:
            query: The user's query
            retrieved_chunks: List of retrieved document chunks with metadata
            generated_response: The generated response from the LLM
            metadata: Additional metadata about the interaction
        """
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_chunks": [
                {
                    "content": chunk.get("content", ""),
                    "source": chunk.get("source", ""),
                    "score": chunk.get("score", 0.0)
                }
                for chunk in retrieved_chunks
            ],
            "generated_response": generated_response,
            "metadata": metadata or {}
        }
        
        # Log as JSON
        self.logger.info(json.dumps(interaction_data, ensure_ascii=False))
        
        # Also append to a JSON lines file for easy parsing
        json_log_file = self.log_file.parent / "interactions.jsonl"
        with open(json_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction_data, ensure_ascii=False) + '\n')
    
    def log_error(self, error_message: str, query: str = None):
        """Log an error during interaction.
        
        Args:
            error_message: The error message
            query: The user's query that caused the error (if applicable)
        """
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "query": query,
            "error": error_message
        }
        self.logger.error(json.dumps(error_data, ensure_ascii=False))


# Create a global logger instance
interaction_logger = InteractionLogger()
