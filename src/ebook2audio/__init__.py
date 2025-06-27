"""
Ebook2Audio - Convert ebooks to audiobooks using TTS.

This package provides a complete pipeline for converting ebooks (PDF, EPUB, MOBI, TXT)
to audiobooks using various TTS engines including Google TTS, system TTS, and advanced
neural TTS models.

Main Components:
- Text extraction from multiple ebook formats
- Voice management and TTS engine integration
- Audio pipeline with chunking and concatenation
- CLI interface for easy command-line usage
- Rich progress bars and user feedback

Usage Examples:
    # Simple conversion
    from ebook2audio import convert_ebook_to_audiobook
    result = convert_ebook_to_audiobook("book.pdf", "audiobook.mp3")
    
    # Advanced pipeline usage
    from ebook2audio.pipeline import AudioBookPipeline, ConversionConfig
    config = ConversionConfig(voice_id="gtts_en_us", output_format="mp3")
    pipeline = AudioBookPipeline(config)
    result = pipeline.convert("book.pdf", "audiobook.mp3")
    
    # Voice management
    from ebook2audio.voices import list_voices, get_voice
    voices = list_voices()
    voice = get_voice("gtts_en_us")
"""

__version__ = "1.0.0"
__author__ = "Ebook2Audio Team"
__license__ = "MIT"

# Main pipeline imports
from .pipeline import (
    AudioBookPipeline,
    ConversionConfig,
    ConversionResult,
    create_pipeline,
    convert_ebook_to_audiobook,
    PipelineStatus,
    TextChunk
)

# Voice management imports
from .voices import (
    list_voices,
    get_voice,
    get_voice_manager,
    register_voice,
    preview_voice,
    VoiceManager,
    VoiceCatalog,
    VoiceMetadata,
    Voice,
    BUILTIN_VOICES
)

# Text extraction imports
from .extract import extract_text

# Utility imports
from .utils import (
    setup_logging,
    format_duration,
    format_file_size,
    ProgressManager,
    FileUtils
)

# Configuration imports
from .config import *

__all__ = [
    # Main pipeline
    "AudioBookPipeline",
    "ConversionConfig", 
    "ConversionResult",
    "create_pipeline",
    "convert_ebook_to_audiobook",
    "PipelineStatus",
    "TextChunk",
    
    # Voice management
    "list_voices",
    "get_voice",
    "get_voice_manager",
    "register_voice",
    "preview_voice", 
    "VoiceManager",
    "VoiceCatalog",
    "VoiceMetadata",
    "Voice",
    "BUILTIN_VOICES",
    
    # Text extraction
    "extract_text",
    
    # Utilities
    "setup_logging",
    "format_duration",
    "format_file_size",
    "ProgressManager",
    "FileUtils",
    
    # Package info
    "__version__",
    "__author__",
    "__license__",
]