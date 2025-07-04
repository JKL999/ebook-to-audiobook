"""
Progress tracking for long-running operations.
"""

import time
from typing import Optional, Callable
from contextlib import contextmanager

class ProgressTracker:
    """Progress tracker with callback support."""
    
    def __init__(self, 
                 total_items: int,
                 description: str = "Processing",
                 enable_tqdm: bool = True):
        self.total_items = total_items
        self.description = description
        self.enable_tqdm = enable_tqdm
        self.current_item = 0
        self.callbacks = []
        self.start_time = None
    
    def add_callback(self, callback: Callable):
        """Add progress callback."""
        self.callbacks.append(callback)
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current_item += increment
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(self.current_item, self.total_items)
            except Exception:
                pass
    
    def __enter__(self):
        """Context manager entry."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"{self.description} completed in {elapsed:.1f}s")

def create_simple_callback(description: str = "Processing", log_interval: int = 10) -> Callable:
    """Create a simple progress callback."""
    def callback(current: int, total: int):
        if total > 0 and (current % log_interval == 0 or current == total):
            percentage = (current / total) * 100
            print(f"{description}: {current}/{total} ({percentage:.1f}%)")
    
    return callback

@contextmanager
def progress_tracker(total_items: int, description: str = "Processing"):
    """Context manager for progress tracking."""
    tracker = ProgressTracker(total_items, description)
    try:
        yield tracker
    finally:
        pass