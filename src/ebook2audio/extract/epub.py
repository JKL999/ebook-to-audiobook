"""
EPUB text extraction using ebooklib with chapter detection.

This module provides EPUB text extraction with proper chapter/section detection
and HTML content processing.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import html

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class EPUBChapter:
    """EPUB chapter information."""
    title: str
    content: str
    order: int
    file_name: str


@dataclass
class EPUBExtractionResult:
    """Result of EPUB text extraction."""
    chapters: List[EPUBChapter]
    total_chapters: int
    metadata: Dict[str, str]
    toc_structure: List[Tuple[str, int]]  # (title, order) pairs


class EPUBExtractor:
    """
    EPUB text extractor using ebooklib.
    
    Features:
    - Proper chapter detection and ordering
    - HTML content cleaning
    - Table of contents parsing
    - Metadata extraction
    - Support for EPUB 2 and EPUB 3
    """
    
    def __init__(self, clean_html: bool = True, preserve_formatting: bool = False):
        """
        Initialize EPUB extractor.
        
        Args:
            clean_html: Whether to clean HTML tags and entities
            preserve_formatting: Whether to preserve basic formatting (paragraphs, etc.)
        """
        self.clean_html = clean_html
        self.preserve_formatting = preserve_formatting
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        try:
            # Test ebooklib
            ebooklib.__version__
        except Exception as e:
            raise RuntimeError(f"ebooklib not available: {e}")
        
        try:
            # Test BeautifulSoup
            BeautifulSoup("", "html.parser")
        except Exception as e:
            raise RuntimeError(f"BeautifulSoup not available: {e}")
    
    def extract_text(self, file_path: Path) -> List[str]:
        """
        Extract text from EPUB file.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            List of chapter texts
            
        Raises:
            FileNotFoundError: If EPUB file doesn't exist
            ValueError: If file is not a valid EPUB
            RuntimeError: If extraction fails
        """
        result = self.extract_text_detailed(file_path)
        return [chapter.content for chapter in result.chapters]
    
    def extract_text_detailed(self, file_path: Path) -> EPUBExtractionResult:
        """
        Extract text from EPUB with detailed chapter information.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            EPUBExtractionResult with chapter details
            
        Raises:
            FileNotFoundError: If EPUB file doesn't exist
            ValueError: If file is not a valid EPUB
            RuntimeError: If extraction fails
        """
        if not file_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {file_path}")
        
        logger.info(f"Extracting text from EPUB: {file_path}")
        
        try:
            # Open EPUB file
            book = epub.read_epub(str(file_path))
            
            # Extract metadata
            metadata = self._extract_metadata(book)
            logger.info(f"EPUB metadata: title='{metadata.get('title', 'Unknown')}', "
                       f"author='{metadata.get('creator', 'Unknown')}'")
            
            # Get spine items (reading order)
            spine_items = self._get_spine_items(book)
            logger.info(f"Found {len(spine_items)} spine items")
            
            # Extract chapters from spine
            chapters = []
            toc_structure = []
            
            for order, item in enumerate(spine_items):
                try:
                    chapter = self._extract_chapter(book, item, order)
                    if chapter and chapter.content.strip():
                        chapters.append(chapter)
                        toc_structure.append((chapter.title, order))
                        logger.debug(f"Extracted chapter {order + 1}: '{chapter.title[:50]}...'")
                    else:
                        logger.debug(f"Skipped empty chapter {order + 1}")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract chapter {order + 1}: {e}")
                    continue
            
            if not chapters:
                raise ValueError("No readable chapters found in EPUB")
            
            result = EPUBExtractionResult(
                chapters=chapters,
                total_chapters=len(chapters),
                metadata=metadata,
                toc_structure=toc_structure
            )
            
            logger.info(f"Extraction complete: {len(chapters)} chapters extracted")
            return result
            
        except Exception as e:
            logger.error(f"EPUB extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text from EPUB: {e}") from e
    
    def _extract_metadata(self, book: epub.EpubBook) -> Dict[str, str]:
        """Extract metadata from EPUB book."""
        metadata = {}
        
        try:
            # Standard Dublin Core metadata
            for key in ['title', 'creator', 'contributor', 'subject', 'description',
                       'publisher', 'date', 'type', 'format', 'identifier',
                       'source', 'language', 'relation', 'coverage', 'rights']:
                value = book.get_metadata('DC', key)
                if value:
                    # Handle list of metadata entries
                    if isinstance(value, list) and value:
                        metadata[key] = ', '.join([str(v[0]) for v in value if v])
                    elif value:
                        metadata[key] = str(value)
                        
        except Exception as e:
            logger.warning(f"Failed to extract some metadata: {e}")
        
        return metadata
    
    def _get_spine_items(self, book: epub.EpubBook) -> List[epub.EpubHtml]:
        """Get ordered spine items from EPUB."""
        spine_items = []
        
        for item_id, linear in book.spine:
            try:
                item = book.get_item_with_id(item_id)
                if isinstance(item, epub.EpubHtml):
                    spine_items.append(item)
            except Exception as e:
                logger.warning(f"Failed to get spine item {item_id}: {e}")
                continue
        
        return spine_items
    
    def _extract_chapter(self, book: epub.EpubBook, item: epub.EpubHtml, order: int) -> Optional[EPUBChapter]:
        """Extract text from a single EPUB chapter."""
        try:
            # Get chapter content
            content = item.get_content().decode('utf-8', errors='ignore')
            
            # Parse HTML content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title from various sources
            title = self._extract_chapter_title(soup, item, order)
            
            # Extract and clean text content
            text_content = self._extract_text_content(soup)
            
            if not text_content.strip():
                return None
            
            return EPUBChapter(
                title=title,
                content=text_content,
                order=order,
                file_name=item.file_name
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract chapter content: {e}")
            return None
    
    def _extract_chapter_title(self, soup: BeautifulSoup, item: epub.EpubHtml, order: int) -> str:
        """Extract chapter title from various sources."""
        # Try to find title in HTML
        title_tags = ['h1', 'h2', 'h3', 'title']
        for tag in title_tags:
            title_elem = soup.find(tag)
            if title_elem and title_elem.get_text(strip=True):
                return title_elem.get_text(strip=True)
        
        # Use filename as fallback
        if hasattr(item, 'file_name') and item.file_name:
            title = Path(item.file_name).stem
            # Clean up filename
            title = re.sub(r'[_-]', ' ', title)
            title = re.sub(r'\d+', '', title).strip()
            if title:
                return title.title()
        
        # Final fallback
        return f"Chapter {order + 1}"
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract and clean text content from HTML."""
        if not self.clean_html:
            return soup.get_text()
        
        # Remove script and style elements
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()
        
        if self.preserve_formatting:
            # Preserve paragraph breaks
            for p in soup.find_all('p'):
                p.append('\n\n')
            
            # Preserve line breaks
            for br in soup.find_all('br'):
                br.replace_with('\n')
            
            # Get text with preserved formatting
            text = soup.get_text()
        else:
            # Get clean text
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Decode HTML entities
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_toc(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Extract table of contents from EPUB.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            List of (title, href) tuples
        """
        try:
            book = epub.read_epub(str(file_path))
            toc_entries = []
            
            def process_toc_item(item):
                if isinstance(item, epub.Link):
                    toc_entries.append((item.title, item.href))
                elif isinstance(item, tuple) and len(item) >= 2:
                    # Handle nested TOC structure
                    if isinstance(item[1], list):
                        for sub_item in item[1]:
                            process_toc_item(sub_item)
                    else:
                        process_toc_item(item[1])
            
            for item in book.toc:
                process_toc_item(item)
            
            return toc_entries
            
        except Exception as e:
            logger.warning(f"Failed to extract TOC: {e}")
            return []
    
    def get_chapter_count(self, file_path: Path) -> int:
        """
        Get the number of chapters in an EPUB file.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Number of chapters
        """
        try:
            book = epub.read_epub(str(file_path))
            spine_items = self._get_spine_items(book)
            return len(spine_items)
        except Exception as e:
            logger.error(f"Failed to get chapter count: {e}")
            return 0


# Compatibility with pipeline interface
def extract_text(file_path: Path) -> List[str]:
    """
    Simple interface for pipeline compatibility.
    
    Args:
        file_path: Path to EPUB file
        
    Returns:
        List of chapter texts
    """
    extractor = EPUBExtractor()
    return extractor.extract_text(file_path)