"""
Voice management and TTS engine integrations.

This module provides comprehensive voice management capabilities including:
- Voice catalog system with JSON-based voice registration
- Built-in and custom voice handling
- Voice discovery and validation
- TTS engine abstraction
- Voice preview and testing capabilities

The voice management system supports multiple TTS engines:
- XTTS-v2: Fast voice cloning with 6-second samples
- Bark: Expressive voice cloning with 5-12 second samples
- OpenVoice: Tone/accent control with emotional control
- Tortoise-TTS: High-quality audiobook synthesis
"""

from .base import BaseTTSProvider, Voice, VoiceMetadata
from .manager import VoiceManager, VoiceCatalog
from .catalog import load_voice_catalog, save_voice_catalog, validate_voice_config
from .builtin import BUILTIN_VOICES, get_builtin_voice

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseTTSProvider",
    "Voice", 
    "VoiceMetadata",
    
    # Voice management
    "VoiceManager",
    "VoiceCatalog",
    
    # Catalog utilities
    "load_voice_catalog",
    "save_voice_catalog", 
    "validate_voice_config",
    
    # Built-in voices
    "BUILTIN_VOICES",
    "get_builtin_voice",
]

# Initialize the global voice manager instance
_voice_manager = None

def get_voice_manager() -> VoiceManager:
    """Get the global VoiceManager instance."""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager()
        # Register TTS providers
        _initialize_providers(_voice_manager)
    return _voice_manager

def _initialize_providers(voice_manager: VoiceManager) -> None:
    """Initialize and register TTS providers."""
    try:
        # Import and register Google TTS provider
        from .gtts_provider import GoogleTTSProvider
        gtts_provider = GoogleTTSProvider()
        if gtts_provider.initialize():
            voice_manager.register_provider(gtts_provider)
        
        # Import and register System TTS provider
        from .pyttsx3_provider import SystemTTSProvider
        system_provider = SystemTTSProvider()
        if system_provider.initialize():
            voice_manager.register_provider(system_provider)
        
        # Import and register GPT-SoVITS provider
        from .gpt_sovits_provider import GPTSoVITSProvider
        gpt_sovits_provider = GPTSoVITSProvider()
        if gpt_sovits_provider.initialize():
            voice_manager.register_provider(gpt_sovits_provider)
            
    except Exception as e:
        from loguru import logger
        logger.warning(f"Failed to initialize some TTS providers: {e}")

def list_voices() -> dict:
    """List all available voices (built-in and custom)."""
    return get_voice_manager().list_voices()

def get_voice(voice_id: str) -> Voice:
    """Get a voice by ID."""
    return get_voice_manager().get_voice(voice_id)

def register_voice(voice: Voice) -> bool:
    """Register a custom voice."""
    return get_voice_manager().register_voice(voice)

def preview_voice(voice_id: str, text: str = "Hello, this is a voice preview.") -> str:
    """Generate a preview sample for a voice."""
    return get_voice_manager().preview_voice(voice_id, text)