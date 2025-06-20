"""
PDF text extraction using PyMuPDF with OCR fallback.

This module provides modern, high-performance PDF text extraction using PyMuPDF
(fitz) with automatic OCR fallback for scanned documents using ocrmypdf.
"""

import io
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import fitz  # PyMuPDF
from loguru import logger


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
    
    def __init__(self, ocr_threshold: float = 0.1, enable_ocr: bool = True):
        """
        Initialize PDF extractor.
        
        Args:
            ocr_threshold: Minimum text density to avoid OCR (chars per pixel)
            enable_ocr: Whether to enable OCR fallback for scanned documents
        """
        self.ocr_threshold = ocr_threshold
        self.enable_ocr = enable_ocr
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
            
            # Extract text from each page
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
            
            logger.info(f"Extraction complete: {extraction_method} method, "
                       f"{len(ocr_pages)} pages required OCR")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}") from e
    
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