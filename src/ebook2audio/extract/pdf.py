"""
PDF text extraction using PyMuPDF with OCR fallback.

This module provides modern, high-performance PDF text extraction using PyMuPDF
(fitz) with automatic OCR fallback for scanned documents using ocrmypdf.
"""

import io
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import hashlib

import fitz  # PyMuPDF
from loguru import logger

from .ocr_parallel import ParallelOCRProcessor, OCRBatchResult
from .ocr_cache import OCRCache, ResumableProcessor
from .progress import ProgressTracker, create_simple_callback


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""
    pages: List[str]
    total_pages: int
    extraction_method: str  # 'text' or 'ocr'
    ocr_pages: List[int]  # Page numbers that required OCR


class PDFExtractor:
    """
    Modern PDF text extractor using PyMuPDF with OCR fallback.
    
    Features:
    - 10x faster than PyPDF2
    - Better layout preservation
    - Automatic OCR for scanned documents
    - Handles both text-based and image-based PDFs
    """
    
    def __init__(self, 
                 ocr_threshold: float = 0.1, 
                 enable_ocr: bool = True,
                 enable_parallel_ocr: bool = True,
                 enable_caching: bool = True,
                 cache_dir: Optional[Path] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize PDF extractor.
        
        Args:
            ocr_threshold: Minimum text density to avoid OCR (chars per pixel)
            enable_ocr: Whether to enable OCR fallback for scanned documents
            enable_parallel_ocr: Whether to use parallel OCR processing
            enable_caching: Whether to enable OCR result caching
            cache_dir: Directory for cache storage (default: ~/.cache/ebook2audio)
            max_workers: Number of parallel OCR workers (default: CPU count - 1)
        """
        self.ocr_threshold = ocr_threshold
        self.enable_ocr = enable_ocr
        self.enable_parallel_ocr = enable_parallel_ocr
        self.enable_caching = enable_caching
        
        # Initialize caching
        if self.enable_caching:
            if cache_dir is None:
                cache_dir = Path.home() / ".cache" / "ebook2audio"
            self.cache = OCRCache(cache_dir)
        else:
            self.cache = None
        
        # Initialize parallel OCR processor
        if self.enable_parallel_ocr and self.enable_ocr:
            self.parallel_processor = ParallelOCRProcessor(max_workers=max_workers)
        else:
            self.parallel_processor = None
        
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            # Check PyMuPDF
            fitz.open()
        except Exception as e:
            raise RuntimeError(f"PyMuPDF not available: {e}")
        
        if self.enable_ocr:
            try:
                # Check ocrmypdf
                result = subprocess.run(
                    ["ocrmypdf", "--version"], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    logger.warning("ocrmypdf not available, OCR disabled")
                    self.enable_ocr = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("ocrmypdf not found, OCR disabled")
                self.enable_ocr = False
    
    def extract_text(self, file_path: Path) -> List[str]:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of page texts
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
            RuntimeError: If extraction fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        result = self.extract_text_detailed(file_path)
        return result.pages
    
    def extract_text_detailed(self, file_path: Path) -> PDFExtractionResult:
        """
        Extract text from PDF with detailed extraction information.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDFExtractionResult with extraction details
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
            RuntimeError: If extraction fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.info(f"Extracting text from PDF: {file_path}")
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            
            if total_pages == 0:
                raise ValueError("PDF has no pages")
            
            logger.info(f"PDF has {total_pages} pages")
            
            # Check if we should use parallel processing
            if (self.enable_parallel_ocr and 
                self.parallel_processor and 
                total_pages >= 2):  # Use parallel for 2+ pages for testing
                return self._extract_text_parallel(doc, file_path)
            else:
                return self._extract_text_sequential(doc)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}") from e
    
    def _extract_text_sequential(self, doc: fitz.Document) -> PDFExtractionResult:
        """
        Extract text using sequential processing (original method).
        
        Args:
            doc: Opened PyMuPDF document
            
        Returns:
            PDFExtractionResult with extraction details
        """
        total_pages = len(doc)
        pages = []
        ocr_pages = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Try text extraction first
            text = page.get_text()
            
            # Check if page needs OCR (low text density)
            if self._needs_ocr(page, text):
                if self.enable_ocr:
                    logger.info(f"Page {page_num + 1} needs OCR (low text density)")
                    ocr_text = self._extract_page_ocr(page, page_num)
                    if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        ocr_pages.append(page_num + 1)
                else:
                    logger.warning(f"Page {page_num + 1} appears to be scanned but OCR is disabled")
            
            pages.append(text.strip())
        
        doc.close()
        
        # Determine extraction method
        extraction_method = "ocr" if ocr_pages else "text"
        if ocr_pages and len(ocr_pages) < total_pages:
            extraction_method = "mixed"
        
        result = PDFExtractionResult(
            pages=pages,
            total_pages=total_pages,
            extraction_method=extraction_method,
            ocr_pages=ocr_pages
        )
        
        logger.info(f"Sequential extraction complete: {extraction_method} method, "
                   f"{len(ocr_pages)} pages required OCR")
        
        return result
    
    def _extract_text_parallel(self, doc: fitz.Document, file_path: Path) -> PDFExtractionResult:
        """
        Extract text using parallel OCR processing for efficiency.
        
        Args:
            doc: Opened PyMuPDF document
            file_path: Path to PDF file for caching
            
        Returns:
            PDFExtractionResult with extraction details
        """
        total_pages = len(doc)
        pages = [""] * total_pages  # Pre-allocate list
        ocr_pages = []
        
        # Step 1: Identify pages that need OCR vs regular text extraction
        text_pages = []  # Pages with sufficient text
        ocr_needed_pages = []  # Pages that need OCR
        
        logger.info("Analyzing pages to determine OCR requirements...")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if self._needs_ocr(page, text) and self.enable_ocr:
                ocr_needed_pages.append(page_num + 1)  # 1-indexed
            else:
                text_pages.append((page_num, text.strip()))
        
        logger.info(f"Found {len(text_pages)} text-extractable pages, "
                   f"{len(ocr_needed_pages)} pages need OCR")
        
        # Step 2: Process text-extractable pages quickly
        for page_num, text in text_pages:
            pages[page_num] = text
        
        # Step 3: Process OCR pages in parallel if any
        if ocr_needed_pages:
            ocr_results = self._process_ocr_pages_parallel(
                file_path, ocr_needed_pages, total_pages
            )
            
            # Merge OCR results
            for result in ocr_results.results:
                if result.success and result.text.strip():
                    pages[result.page_num - 1] = result.text.strip()  # Convert to 0-indexed
                    ocr_pages.append(result.page_num)
                elif not pages[result.page_num - 1]:  # Only if no text was found
                    # Fallback to whatever text was extracted originally
                    page = doc[result.page_num - 1]
                    pages[result.page_num - 1] = page.get_text().strip()
        
        doc.close()
        
        # Determine extraction method
        extraction_method = "ocr" if ocr_pages else "text"
        if ocr_pages and len(ocr_pages) < total_pages:
            extraction_method = "mixed"
        
        result = PDFExtractionResult(
            pages=pages,
            total_pages=total_pages,
            extraction_method=extraction_method,
            ocr_pages=ocr_pages
        )
        
        logger.info(f"Parallel extraction complete: {extraction_method} method, "
                   f"{len(ocr_pages)} pages required OCR")
        
        return result
    
    def _process_ocr_pages_parallel(self, 
                                   file_path: Path, 
                                   page_numbers: List[int],
                                   total_pages: int) -> OCRBatchResult:
        """
        Process OCR pages in parallel with caching and progress tracking.
        
        Args:
            file_path: Path to PDF file
            page_numbers: List of page numbers that need OCR (1-indexed)
            total_pages: Total pages in document for progress tracking
            
        Returns:
            OCRBatchResult with processing results
        """
        if not self.parallel_processor:
            raise RuntimeError("Parallel processor not initialized")
        
        # Set up progress tracking
        with ProgressTracker(
            total_items=len(page_numbers),
            description=f"OCR Processing ({len(page_numbers)} pages)",
            enable_tqdm=True
        ) as progress:
            
            # Add simple logging callback
            progress.add_callback(create_simple_callback(log_interval=20))
            
            def progress_callback(completed: int, total: int) -> None:
                progress.update(completed)
            
            # Process pages with caching if enabled
            if self.enable_caching and self.cache:
                return self._process_with_caching(
                    file_path, page_numbers, progress_callback
                )
            else:
                return self.parallel_processor.process_batches(
                    file_path, page_numbers, progress_callback
                )
    
    def _process_with_caching(self, 
                             file_path: Path, 
                             page_numbers: List[int],
                             progress_callback: Optional[callable]) -> OCRBatchResult:
        """
        Process OCR pages with caching support.
        
        Args:
            file_path: Path to PDF file
            page_numbers: List of page numbers to process
            progress_callback: Progress update callback
            
        Returns:
            OCRBatchResult with processing results
        """
        if not self.cache:
            raise RuntimeError("Cache not initialized")
        
        # Check cache for existing results
        # For simplicity, we'll skip complex cache checking in this implementation
        # and just use parallel processing with basic caching
        
        document_id = hashlib.md5(str(file_path).encode()).hexdigest()
        resumable = ResumableProcessor(self.cache, document_id)
        
        # Check for resumable processing
        remaining_pages = resumable.get_remaining_pages(page_numbers)
        
        if len(remaining_pages) < len(page_numbers):
            logger.info(f"Resuming processing: {len(remaining_pages)}/{len(page_numbers)} pages remaining")
        
        # Process remaining pages
        if remaining_pages:
            result = self.parallel_processor.process_batches(
                file_path, remaining_pages, progress_callback
            )
            
            # Save checkpoint
            completed_pages = [p for p in page_numbers if p not in remaining_pages]
            completed_pages.extend([r.page_num for r in result.results if r.success])
            resumable.save_checkpoint(completed_pages, len(page_numbers))
            
            return result
        else:
            # All pages already processed - return empty result
            from .ocr_parallel import OCRBatchResult, OCRPageResult
            return OCRBatchResult(
                results=[],
                total_pages=0,
                success_count=0,
                failure_count=0,
                total_time=0.0,
                avg_page_time=0.0
            )
    
    def _needs_ocr(self, page: fitz.Page, text: str) -> bool:
        """
        Determine if a page needs OCR based on text density.
        
        Args:
            page: PyMuPDF page object
            text: Extracted text from page
            
        Returns:
            True if page likely needs OCR
        """
        if not text or len(text.strip()) < 10:
            return True
        
        # Calculate text density (characters per pixel)
        rect = page.rect
        page_area = rect.width * rect.height
        text_density = len(text.strip()) / page_area if page_area > 0 else 0
        
        return text_density < self.ocr_threshold
    
    def _extract_page_ocr(self, page: fitz.Page, page_num: int) -> Optional[str]:
        """
        Extract text from a single page using OCR.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            
        Returns:
            OCR text or None if OCR fails
        """
        if not self.enable_ocr:
            return None
        
        try:
            # Render page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
                img_file.write(img_data)
                img_path = img_file.name
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_file:
                pdf_path = pdf_file.name
            
            try:
                # Run OCR using ocrmypdf
                result = subprocess.run([
                    "ocrmypdf",
                    "--pages", f"{page_num + 1}",
                    "--output-type", "pdf",
                    "--force-ocr",
                    "--quiet",
                    str(img_path),
                    str(pdf_path)
                ], 
                capture_output=True, 
                text=True,
                timeout=60
                )
                
                if result.returncode == 0:
                    # Extract text from OCR'd PDF
                    ocr_doc = fitz.open(pdf_path)
                    if len(ocr_doc) > 0:
                        ocr_text = ocr_doc[0].get_text()
                        ocr_doc.close()
                        return ocr_text
                    ocr_doc.close()
                else:
                    logger.warning(f"OCR failed for page {page_num + 1}: {result.stderr}")
                    
            finally:
                # Clean up temporary files
                Path(img_path).unlink(missing_ok=True)
                Path(pdf_path).unlink(missing_ok=True)
                
        except subprocess.TimeoutExpired:
            logger.warning(f"OCR timeout for page {page_num + 1}")
        except Exception as e:
            logger.warning(f"OCR error for page {page_num + 1}: {e}")
        
        return None
    
    def extract_metadata(self, file_path: Path) -> dict:
        """
        Extract metadata from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            doc = fitz.open(str(file_path))
            metadata = doc.metadata
            doc.close()
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            return {}
    
    def get_page_count(self, file_path: Path) -> int:
        """
        Get the number of pages in a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Number of pages
        """
        try:
            doc = fitz.open(str(file_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.error(f"Failed to get page count: {e}")
            return 0


# Compatibility with pipeline interface
def extract_text(file_path: Path) -> List[str]:
    """
    Simple interface for pipeline compatibility.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        List of page texts
    """
    extractor = PDFExtractor()
    return extractor.extract_text(file_path)