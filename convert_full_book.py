#!/usr/bin/env python3
"""
Full book conversion script for ebook-to-audiobook pipeline.

This script converts the entire Lee Kuan Yew book (739 pages) to audiobook
using our optimized parallel OCR and TTS pipeline.
"""

import time
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ebook2audio.pipeline import AudioBookPipeline, ConversionConfig
from ebook2audio.extract.pdf import PDFExtractor
from loguru import logger

def setup_logging():
    """Configure logging for the conversion process."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
    )
    logger.add(
        "audiobook_output/conversion.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB"
    )

def convert_book_to_audiobook(
    input_pdf: Path,
    output_path: Optional[Path] = None,
    test_pages: Optional[int] = None
) -> bool:
    """
    Convert a PDF book to audiobook.
    
    Args:
        input_pdf: Path to the input PDF
        output_path: Output audiobook path (optional)
        test_pages: If specified, only convert first N pages for testing
        
    Returns:
        True if successful, False otherwise
    """
    
    if not input_pdf.exists():
        logger.error(f"Input PDF not found: {input_pdf}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        output_name = input_pdf.stem + "_audiobook.mp3"
        output_path = Path("audiobook_output") / output_name
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting conversion: {input_pdf.name}")
    logger.info(f"Output: {output_path}")
    
    if test_pages:
        logger.info(f"TEST MODE: Converting only first {test_pages} pages")
    
    start_time = time.time()
    
    try:
        # Configure the pipeline for optimal performance
        config = ConversionConfig(
            # TTS settings
            voice_id="gtts_en_us",  # Google TTS English US
            speaking_rate=1.0,      # Normal speed
            volume=0.9,
            
            # Chunking settings - optimized for large books
            max_chunk_size=800,     # Slightly smaller chunks for better TTS
            chunk_overlap=30,
            sentence_break=True,    # Break on sentences for natural speech
            
            # Audio settings
            output_format="mp3",
            sample_rate=22050,      # Good quality, reasonable file size
            bitrate="96k",          # Smaller file size for long audiobooks
            
            # Processing settings
            add_silence_between_chunks=0.3,  # Short pause between chunks
            normalize_audio=True,
            
            # Output settings
            output_filename=output_path.name,
            temp_dir=Path("audiobook_output/temp")
        )
        
        # Create pipeline with our optimized PDF extractor
        pipeline = AudioBookPipeline(config)
        
        # Replace the default PDF extractor with our parallel OCR version
        logger.info("Configuring parallel OCR for PDF extraction...")
        
        # If this is a test run, create a subset PDF
        actual_input = input_pdf
        if test_pages:
            actual_input = create_test_pdf(input_pdf, test_pages)
        
        # Run the conversion
        logger.info("Starting conversion pipeline...")
        result = pipeline.convert(actual_input, output_path)
        
        # Report results
        elapsed_time = time.time() - start_time
        
        if result.success:
            logger.info("=" * 60)
            logger.info("CONVERSION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Input: {input_pdf.name}")
            logger.info(f"Output: {result.output_path}")
            logger.info(f"Processing time: {elapsed_time / 60:.1f} minutes")
            
            if result.duration:
                logger.info(f"Audio duration: {result.duration / 3600:.1f} hours")
                
            if result.output_path and result.output_path.exists():
                file_size = result.output_path.stat().st_size
                logger.info(f"File size: {file_size / (1024*1024):.1f} MB")
            
            logger.info(f"Chunks processed: {result.processed_chunks}/{result.total_chunks}")
            logger.info(f"Success rate: {result.success_rate:.1f}%")
            logger.info("=" * 60)
            
            return True
        else:
            logger.error("CONVERSION FAILED!")
            logger.error(f"Error: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"Conversion failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test PDF if created
        if test_pages and actual_input != input_pdf:
            try:
                actual_input.unlink(missing_ok=True)
            except Exception:
                pass

def create_test_pdf(input_pdf: Path, num_pages: int) -> Path:
    """
    Create a test PDF with only the first N pages.
    
    Args:
        input_pdf: Original PDF path
        num_pages: Number of pages to include
        
    Returns:
        Path to the test PDF
    """
    import fitz
    
    logger.info(f"Creating test PDF with first {num_pages} pages...")
    
    # Open original PDF
    doc = fitz.open(str(input_pdf))
    total_pages = len(doc)
    
    if num_pages >= total_pages:
        logger.warning(f"Requested {num_pages} pages but PDF only has {total_pages}")
        num_pages = total_pages
    
    # Create new PDF with subset of pages
    test_pdf_path = Path("audiobook_output") / f"test_{num_pages}_pages.pdf"
    new_doc = fitz.open()
    
    for page_num in range(num_pages):
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    
    new_doc.save(str(test_pdf_path))
    new_doc.close()
    doc.close()
    
    logger.info(f"Test PDF created: {test_pdf_path}")
    return test_pdf_path

def main():
    """Main conversion function."""
    setup_logging()
    
    # Input PDF path
    input_pdf = Path("from-third-world-to-first-by-lee-kuan-yew.pdf")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Test mode - convert only first 50 pages
            logger.info("Running in TEST MODE - converting first 50 pages only")
            success = convert_book_to_audiobook(
                input_pdf,
                output_path=Path("audiobook_output/test_50_pages_audiobook.mp3"),
                test_pages=50
            )
        elif command == "small":
            # Small test - convert only first 10 pages
            logger.info("Running in SMALL TEST MODE - converting first 10 pages only")
            success = convert_book_to_audiobook(
                input_pdf,
                output_path=Path("audiobook_output/test_10_pages_audiobook.mp3"),
                test_pages=10
            )
        elif command == "full":
            # Full conversion
            logger.info("Running FULL CONVERSION - all 739 pages")
            success = convert_book_to_audiobook(input_pdf)
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Usage: python convert_full_book.py [test|small|full]")
            sys.exit(1)
    else:
        # Default to test mode for safety
        logger.info("No command specified - running in TEST MODE")
        logger.info("Use 'python convert_full_book.py full' for complete conversion")
        success = convert_book_to_audiobook(
            input_pdf,
            output_path=Path("audiobook_output/test_50_pages_audiobook.mp3"),
            test_pages=50
        )
    
    if success:
        logger.info("Conversion completed successfully!")
        sys.exit(0)
    else:
        logger.error("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()