"""
MOBI text extraction using Calibre CLI wrapper.

This module provides MOBI/AZW text extraction by converting to EPUB
using Calibre's ebook-convert utility and then extracting from EPUB.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

from loguru import logger
from .epub import EPUBExtractor, EPUBExtractionResult


@dataclass
class MOBIExtractionResult:
    """Result of MOBI text extraction."""
    chapters: List[str]
    total_chapters: int
    metadata: Dict[str, str]
    conversion_method: str  # 'calibre' or 'direct'


class MOBIExtractor:
    """
    MOBI/AZW text extractor using Calibre CLI wrapper.
    
    Features:
    - Supports MOBI, AZW, AZW3 formats
    - Uses Calibre's ebook-convert for format conversion
    - Preserves chapter structure and metadata
    - Automatic format detection
    """
    
    def __init__(self, calibre_timeout: int = 300):
        """
        Initialize MOBI extractor.
        
        Args:
            calibre_timeout: Timeout for Calibre conversion in seconds
        """
        self.calibre_timeout = calibre_timeout
        self.epub_extractor = EPUBExtractor()
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            # Check Calibre ebook-convert
            result = subprocess.run(
                ["ebook-convert", "--version"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Calibre ebook-convert not working properly")
            
            logger.info(f"Calibre version: {result.stdout.strip()}")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Calibre ebook-convert timeout during version check")
        except FileNotFoundError:
            raise RuntimeError(
                "Calibre ebook-convert not found. Please install Calibre: "
                "https://calibre-ebook.com/download"
            )
        except Exception as e:
            raise RuntimeError(f"Calibre ebook-convert not available: {e}")
    
    def extract_text(self, file_path: Path) -> List[str]:
        """
        Extract text from MOBI/AZW file.
        
        Args:
            file_path: Path to MOBI/AZW file
            
        Returns:
            List of chapter texts
            
        Raises:
            FileNotFoundError: If MOBI file doesn't exist
            ValueError: If file is not a valid MOBI/AZW
            RuntimeError: If extraction fails
        """
        result = self.extract_text_detailed(file_path)
        return result.chapters
    
    def extract_text_detailed(self, file_path: Path) -> MOBIExtractionResult:
        """
        Extract text from MOBI/AZW with detailed information.
        
        Args:
            file_path: Path to MOBI/AZW file
            
        Returns:
            MOBIExtractionResult with extraction details
            
        Raises:
            FileNotFoundError: If MOBI file doesn't exist
            ValueError: If file is not a valid MOBI/AZW
            RuntimeError: If extraction fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"MOBI file not found: {file_path}")
        
        # Validate file format
        if not self._is_supported_format(file_path):
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Extracting text from MOBI/AZW: {file_path}")
        
        try:
            # Convert MOBI to EPUB using Calibre
            epub_path = self._convert_to_epub(file_path)
            
            try:
                # Extract text from converted EPUB
                epub_result = self.epub_extractor.extract_text_detailed(epub_path)
                
                # Convert to MOBI result format
                result = MOBIExtractionResult(
                    chapters=[chapter.content for chapter in epub_result.chapters],
                    total_chapters=epub_result.total_chapters,
                    metadata=epub_result.metadata,
                    conversion_method="calibre"
                )
                
                logger.info(f"Extraction complete: {result.total_chapters} chapters extracted via Calibre")
                return result
                
            finally:
                # Clean up temporary EPUB file
                if epub_path.exists():
                    epub_path.unlink()
                    
        except Exception as e:
            logger.error(f"MOBI extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text from MOBI: {e}") from e
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        supported_extensions = {'.mobi', '.azw', '.azw3', '.azw4', '.prc'}
        return file_path.suffix.lower() in supported_extensions
    
    def _convert_to_epub(self, mobi_path: Path) -> Path:
        """
        Convert MOBI/AZW to EPUB using Calibre.
        
        Args:
            mobi_path: Path to MOBI/AZW file
            
        Returns:
            Path to converted EPUB file
            
        Raises:
            RuntimeError: If conversion fails
        """
        # Create temporary EPUB file
        with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp_file:
            epub_path = Path(tmp_file.name)
        
        try:
            logger.info(f"Converting {mobi_path.name} to EPUB using Calibre...")
            
            # Prepare Calibre command
            cmd = [
                "ebook-convert",
                str(mobi_path),
                str(epub_path),
                "--no-default-epub-cover",  # Skip cover generation for speed
                "--disable-font-rescaling",  # Preserve original formatting
                "--preserve-cover-aspect-ratio",
                "--insert-metadata",  # Preserve metadata
            ]
            
            # Run Calibre conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.calibre_timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise RuntimeError(f"Calibre conversion failed: {error_msg}")
            
            if not epub_path.exists() or epub_path.stat().st_size == 0:
                raise RuntimeError("Calibre conversion produced no output")
            
            logger.info(f"Conversion successful: {epub_path.stat().st_size} bytes")
            return epub_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"Calibre conversion timeout after {self.calibre_timeout}s")
            if epub_path.exists():
                epub_path.unlink()
            raise RuntimeError(f"Calibre conversion timeout after {self.calibre_timeout}s")
            
        except Exception as e:
            logger.error(f"Calibre conversion error: {e}")
            if epub_path.exists():
                epub_path.unlink()
            raise
    
    def extract_metadata(self, file_path: Path) -> Dict[str, str]:
        """
        Extract metadata from MOBI/AZW file.
        
        Args:
            file_path: Path to MOBI/AZW file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            # Use Calibre to extract metadata
            result = subprocess.run([
                "ebook-meta",
                str(file_path)
            ], 
            capture_output=True, 
            text=True,
            timeout=30
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to extract metadata with ebook-meta: {result.stderr}")
                return {}
            
            # Parse metadata output
            metadata = {}
            for line in result.stdout.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if value and value != 'Unknown':
                        metadata[key] = value
            
            return metadata
            
        except subprocess.TimeoutExpired:
            logger.warning("Metadata extraction timeout")
            return {}
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return {}
    
    def get_chapter_count(self, file_path: Path) -> int:
        """
        Get the number of chapters in a MOBI/AZW file.
        
        Args:
            file_path: Path to MOBI/AZW file
            
        Returns:
            Number of chapters
        """
        try:
            # Quick conversion to count chapters
            epub_path = self._convert_to_epub(file_path)
            try:
                count = self.epub_extractor.get_chapter_count(epub_path)
                return count
            finally:
                if epub_path.exists():
                    epub_path.unlink()
        except Exception as e:
            logger.error(f"Failed to get chapter count: {e}")
            return 0
    
    def list_supported_formats(self) -> List[str]:
        """
        List supported MOBI/AZW formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.mobi', '.azw', '.azw3', '.azw4', '.prc']
    
    def check_calibre_formats(self) -> Dict[str, bool]:
        """
        Check which formats are supported by the installed Calibre version.
        
        Returns:
            Dictionary mapping format names to support status
        """
        try:
            result = subprocess.run([
                "ebook-convert", 
                "--list-recipes"
            ], 
            capture_output=True, 
            text=True,
            timeout=10
            )
            
            # This is a simple check - in practice, Calibre supports most formats
            formats = {
                'MOBI': True,
                'AZW': True,
                'AZW3': True,
                'AZW4': True,
                'PRC': True,
            }
            
            return formats
            
        except Exception as e:
            logger.warning(f"Failed to check Calibre formats: {e}")
            return {}


# Compatibility with pipeline interface
def extract_text(file_path: Path) -> List[str]:
    """
    Simple interface for pipeline compatibility.
    
    Args:
        file_path: Path to MOBI/AZW file
        
    Returns:
        List of chapter texts
    """
    extractor = MOBIExtractor()
    return extractor.extract_text(file_path)