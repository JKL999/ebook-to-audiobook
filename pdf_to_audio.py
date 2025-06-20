#!/usr/bin/env python3
"""
Legacy PDF to audio conversion script - Updated to use modern parsing.

This script has been updated to use the new ebook2audio parsing system
with PyMuPDF instead of PyPDF2 for better performance and OCR support.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from ebook2audio.extract import extract_text
    from loguru import logger
    
    # Configure logger for simple output
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    def main():
        """Main function to extract text from PDF."""
        pdf_file = Path("example.pdf")
        
        if not pdf_file.exists():
            logger.error(f"PDF file not found: {pdf_file}")
            logger.info("Please ensure 'example.pdf' exists in the current directory")
            return 1
        
        try:
            logger.info(f"Extracting text from {pdf_file}")
            
            # Use modern parsing system
            pages = extract_text(pdf_file)
            
            logger.info(f"Successfully extracted {len(pages)} pages")
            
            # Print extracted text
            for i, page_text in enumerate(pages, 1):
                print(f"\n{'='*50}")
                print(f"PAGE {i}")
                print(f"{'='*50}")
                print(page_text)
                print(f"{'='*50}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"Error: Could not import ebook2audio modules: {e}")
    print("Make sure you have installed the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
