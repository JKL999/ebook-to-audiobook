#!/usr/bin/env python3
"""
Minimal ebook to audiobook converter - MVP version.

Usage:
    python convert.py input.pdf output.mp3
    python convert.py input.txt output.mp3
"""

import sys
from pathlib import Path
from gtts import gTTS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_file(file_path):
    """Extract text from various file formats."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.txt':
        logger.info(f"Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif suffix == '.pdf':
        logger.info(f"Extracting text from PDF: {file_path}")
        # For MVP, use simple PDF extraction without OCR
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            text = ""
            for page_num, page in enumerate(doc, 1):
                text += page.get_text()
                if page_num % 10 == 0:
                    logger.info(f"Processed {page_num}/{len(doc)} pages")
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            raise
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

def text_to_speech(text, output_path, lang='en'):
    """Convert text to speech using gTTS."""
    logger.info(f"Converting {len(text)} characters to speech...")
    
    # Split into chunks if text is too long
    max_chars = 5000  # gTTS can handle more, but let's be safe
    
    if len(text) <= max_chars:
        # Simple conversion for short text
        tts = gTTS(text=text, lang=lang)
        tts.save(str(output_path))
    else:
        # Split into chunks and combine
        from pydub import AudioSegment
        import tempfile
        import os
        
        chunks = []
        for i in range(0, len(text), max_chars):
            chunk_text = text[i:i+max_chars]
            
            # Create temp file for this chunk
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                temp_path = tmp.name
            
            tts = gTTS(text=chunk_text, lang=lang)
            tts.save(temp_path)
            chunks.append(temp_path)
            
            logger.info(f"Processed chunk {len(chunks)}/{(len(text) + max_chars - 1) // max_chars}")
        
        # Combine all chunks
        combined = AudioSegment.empty()
        for chunk_path in chunks:
            audio = AudioSegment.from_mp3(chunk_path)
            combined += audio
            os.unlink(chunk_path)  # Clean up temp file
        
        # Export combined audio
        combined.export(str(output_path), format='mp3')
    
    logger.info(f"Audio saved to: {output_path}")

def main():
    """Main conversion function."""
    if len(sys.argv) < 3:
        print("Usage: python convert.py input_file output.mp3")
        print("Supported formats: .txt, .pdf")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    try:
        # Extract text
        text = extract_text_from_file(input_file)
        logger.info(f"Extracted {len(text)} characters")
        
        if not text.strip():
            logger.error("No text extracted from file")
            sys.exit(1)
        
        # Convert to speech
        text_to_speech(text, output_file)
        
        logger.info("Conversion complete!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()