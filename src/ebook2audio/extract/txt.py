"""
Text file extraction for simple text files.

This module provides simple text file extraction for testing purposes
and to support plain text inputs.
"""

from pathlib import Path
from typing import List, Optional, Union
from loguru import logger


def extract_text(file_path: Union[Path, str]) -> List[str]:
    """
    Extract text from a plain text file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        List containing the text content (single item for text files)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If extraction fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    try:
        logger.info(f"Reading text file: {file_path.name}")
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        text_content = None
        
        for encoding in encodings:
            try:
                text_content = file_path.read_text(encoding=encoding)
                logger.debug(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if text_content is None:
            raise RuntimeError("Could not decode text file with any supported encoding")
        
        # Clean up the text a bit
        text_content = text_content.strip()
        
        if not text_content:
            logger.warning("Text file appears to be empty")
            return [""]
        
        logger.info(f"Extracted {len(text_content)} characters from text file")
        return [text_content]
        
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        raise RuntimeError(f"Text extraction failed: {e}")


class TXTExtractor:
    """Simple text file extractor."""
    
    def __init__(self):
        pass
    
    def extract(self, file_path: Union[Path, str]) -> List[str]:
        """Extract text from text file."""
        return extract_text(file_path)
    
    def extract_with_metadata(self, file_path: Union[Path, str]) -> dict:
        """Extract text with metadata."""
        text_pages = self.extract(file_path)
        
        return {
            "text_pages": text_pages,
            "metadata": {
                "total_pages": len(text_pages),
                "file_size": Path(file_path).stat().st_size,
                "extraction_method": "text",
                "file_format": "txt"
            }
        }