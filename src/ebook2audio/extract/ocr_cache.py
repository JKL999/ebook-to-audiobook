"""
OCR caching system for PDF extraction.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class OCRCacheEntry:
    """OCR cache entry."""
    page_number: int
    text: str
    timestamp: float
    file_hash: str

class OCRCache:
    """OCR result caching system."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".ebook2audio" / "ocr_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cached_ocr(self, file_path: Path, page_number: int) -> Optional[str]:
        """Get cached OCR result for a page."""
        cache_file = self._get_cache_file(file_path)
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            page_key = str(page_number)
            if page_key in cache_data:
                return cache_data[page_key]["text"]
        except Exception:
            pass
        
        return None
    
    def cache_ocr(self, file_path: Path, page_number: int, text: str):
        """Cache OCR result for a page."""
        cache_file = self._get_cache_file(file_path)
        
        # Load existing cache
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except Exception:
                pass
        
        # Add new entry
        cache_data[str(page_number)] = {
            "text": text,
            "timestamp": time.time(),
            "file_hash": self._get_file_hash(file_path)
        }
        
        # Save cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass
    
    def _get_cache_file(self, file_path: Path) -> Path:
        """Get cache file path for a PDF."""
        file_hash = self._get_file_hash(file_path)
        return self.cache_dir / f"{file_hash}.json"
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for cache key."""
        return hashlib.md5(str(file_path).encode()).hexdigest()

class ResumableProcessor:
    """Resumable OCR processing."""
    
    def __init__(self, cache: OCRCache, *args, **kwargs):
        self.cache = cache
    
    def get_remaining_pages(self, file_path: Path, all_pages: List[int]) -> List[int]:
        """Get pages that still need OCR processing."""
        remaining = []
        for page_num in all_pages:
            if not self.cache.get_cached_ocr(file_path, page_num):
                remaining.append(page_num)
        return remaining
    
    def save_checkpoint(self, completed_pages: List[int], total_pages: int):
        """Save processing checkpoint."""
        # For now, this is handled by the cache itself
        pass