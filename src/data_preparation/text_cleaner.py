"""Text cleaning utilities for preprocessing telecom policy documents."""

import re
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and preprocesses raw text documents."""
    
    def __init__(self):
        """Initialize the text cleaner."""
        pass
    
    def remove_page_numbers(self, text: str) -> str:
        """Remove page numbers from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with page numbers removed
        """
        # Remove standalone numbers at start/end of lines (common page number pattern)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove "Page X" or "Page X of Y" patterns
        text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)
        return text
    
    def remove_headers_footers(self, text: str) -> str:
        """Remove repeated headers and footers.
        
        Args:
            text: Input text
            
        Returns:
            Text with headers/footers removed
        """
        # Remove common header/footer patterns
        # This is a simple implementation; more sophisticated logic can be added
        text = re.sub(r'={3,}', '', text)  # Remove separator lines
        text = re.sub(r'-{3,}', '', text)  # Remove dashed lines
        return text
    
    def remove_unwanted_symbols(self, text: str) -> str:
        """Remove unwanted symbols and special characters.
        
        Args:
            text: Input text
            
        Returns:
            Text with unwanted symbols removed
        """
        # Remove excessive special characters but keep basic punctuation
        # Keep: . , ! ? : ; - ( ) / @ # $ % & *
        # Remove: excessive underscores, pipes, etc.
        text = re.sub(r'_{3,}', ' ', text)  # Replace multiple underscores with space
        text = re.sub(r'\|{2,}', ' ', text)  # Replace multiple pipes with space
        text = re.sub(r'[^\w\s.,!?:;()/\-@#$%&*+=\[\]{}\'\"₹]', '', text)  # Keep common chars
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing/leading whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        # Remove leading/trailing whitespace from entire text
        text = text.strip()
        return text
    
    def remove_empty_lines(self, text: str) -> str:
        """Remove empty lines from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with empty lines removed
        """
        lines = [line for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """Apply all cleaning operations to text.
        
        Args:
            text: Input text to clean
            preserve_structure: If True, preserve paragraph structure;
                              if False, remove all empty lines
            
        Returns:
            Cleaned text
        """
        logger.info(f"Cleaning text ({len(text)} characters)")
        
        # Apply cleaning operations in sequence
        text = self.remove_page_numbers(text)
        text = self.remove_headers_footers(text)
        text = self.remove_unwanted_symbols(text)
        text = self.normalize_whitespace(text)
        
        if not preserve_structure:
            text = self.remove_empty_lines(text)
        
        logger.info(f"Cleaned text ({len(text)} characters)")
        return text
    
    def clean_document(self, document: dict) -> dict:
        """Clean a document dictionary.
        
        Args:
            document: Dictionary with 'filename', 'content', and 'path' keys
            
        Returns:
            Document dictionary with cleaned content
        """
        cleaned_content = self.clean_text(document['content'])
        
        return {
            'filename': document['filename'],
            'content': cleaned_content,
            'path': document['path'],
            'original_length': len(document['content']),
            'cleaned_length': len(cleaned_content)
        }
