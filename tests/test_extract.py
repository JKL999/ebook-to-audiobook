"""Tests for text extraction functionality."""

import pytest
from pathlib import Path
from ebook2audio.extract import extract_text
from ebook2audio.extract.pdf import PDFExtractor
from ebook2audio.extract.epub import EPUBExtractor
from ebook2audio.extract.txt import TXTExtractor

class TestTextExtraction:
    """Test text extraction from various formats."""
    
    def test_extract_text_from_pdf(self, sample_pdf_path):
        """Test PDF text extraction."""
        if sample_pdf_path.exists():
            text = extract_text(sample_pdf_path)
            assert isinstance(text, str)
            assert len(text) > 0
    
    def test_extract_text_from_epub(self, sample_epub_path):
        """Test EPUB text extraction."""
        if sample_epub_path.exists():
            text = extract_text(sample_epub_path)
            assert isinstance(text, str)
            assert len(text) > 0
    
    def test_extract_text_from_txt(self, sample_txt_path):
        """Test TXT text extraction."""
        if sample_txt_path.exists():
            text = extract_text(sample_txt_path)
            assert isinstance(text, str)
            assert len(text) > 0
    
    def test_extract_text_unsupported_format(self, temp_dir):
        """Test extraction from unsupported format."""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("test content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            extract_text(unsupported_file)

class TestPDFExtractor:
    """Test PDF extractor specifically."""
    
    def test_pdf_extractor_init(self):
        """Test PDF extractor initialization."""
        extractor = PDFExtractor()
        assert extractor is not None
    
    def test_pdf_with_ocr_fallback(self, temp_dir):
        """Test PDF extraction with OCR fallback."""
        # This would test OCR functionality if we had a scanned PDF
        pass

class TestEPUBExtractor:
    """Test EPUB extractor specifically."""
    
    def test_epub_extractor_init(self):
        """Test EPUB extractor initialization."""
        extractor = EPUBExtractor()
        assert extractor is not None

class TestTXTExtractor:
    """Test TXT extractor specifically."""
    
    def test_txt_extractor_with_encoding(self, temp_dir):
        """Test TXT extraction with different encodings."""
        test_file = temp_dir / "test_utf8.txt"
        test_content = "Hello, 世界! Привет!"
        test_file.write_text(test_content, encoding='utf-8')
        
        extractor = TXTExtractor()
        text = extractor.extract(test_file)
        assert text == test_content