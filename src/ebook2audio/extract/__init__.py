"""
Text extraction modules for different ebook formats.

Provides extractors for:
- PDF files (PyMuPDF + OCR fallback)
- EPUB files (ebooklib)
- MOBI/AZW files (Calibre wrapper)
"""

from .pdf import PDFExtractor, extract_text as extract_pdf_text
from .epub import EPUBExtractor, extract_text as extract_epub_text
from .mobi import MOBIExtractor, extract_text as extract_mobi_text
from pathlib import Path
from typing import List
from loguru import logger

__all__ = [
    "PDFExtractor",
    "EPUBExtractor", 
    "MOBIExtractor",
    "extract_text",
    "get_extractor_for_file",
]

# Format to extractor mapping
EXTRACTORS = {
    '.pdf': PDFExtractor,
    '.epub': EPUBExtractor,
    '.mobi': MOBIExtractor,
    '.azw': MOBIExtractor,
    '.azw3': MOBIExtractor,
    '.azw4': MOBIExtractor,
    '.prc': MOBIExtractor,
}

# Format to extract function mapping
EXTRACT_FUNCTIONS = {
    '.pdf': extract_pdf_text,
    '.epub': extract_epub_text,
    '.mobi': extract_mobi_text,
    '.azw': extract_mobi_text,
    '.azw3': extract_mobi_text,
    '.azw4': extract_mobi_text,
    '.prc': extract_mobi_text,
}


def get_extractor_for_file(file_path: Path) -> type:
    """
    Get the appropriate extractor class for a file.
    
    Args:
        file_path: Path to the ebook file
        
    Returns:
        Extractor class for the file format
        
    Raises:
        ValueError: If file format is not supported
    """
    extension = file_path.suffix.lower()
    
    if extension not in EXTRACTORS:
        supported = ', '.join(EXTRACTORS.keys())
        raise ValueError(f"Unsupported file format '{extension}'. Supported formats: {supported}")
    
    return EXTRACTORS[extension]


def extract_text(file_path: Path) -> List[str]:
    """
    Extract text from any supported ebook format.
    
    This is a unified interface that automatically detects the file format
    and uses the appropriate extractor.
    
    Args:
        file_path: Path to the ebook file
        
    Returns:
        List of page/chapter texts
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
        RuntimeError: If extraction fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = file_path.suffix.lower()
    
    if extension not in EXTRACT_FUNCTIONS:
        supported = ', '.join(EXTRACT_FUNCTIONS.keys())
        raise ValueError(f"Unsupported file format '{extension}'. Supported formats: {supported}")
    
    logger.info(f"Extracting text from {extension.upper()} file: {file_path.name}")
    
    extract_function = EXTRACT_FUNCTIONS[extension]
    return extract_function(file_path)


def get_supported_formats() -> List[str]:
    """
    Get list of supported ebook formats.
    
    Returns:
        List of supported file extensions
    """
    return list(EXTRACTORS.keys())