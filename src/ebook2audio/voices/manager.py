"""
Voice Manager - Central voice management and orchestration.

This module provides the VoiceManager class which serves as the main interface
for voice discovery, loading, registration, and synthesis coordination.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Type
from datetime import datetime

from .base import Voice, VoiceMetadata, BaseTTSProvider, TTSEngine, VoiceType
from .catalog import VoiceCatalog, VoiceCatalogError

logger = logging.getLogger(__name__)


class VoiceManagerError(Exception):
    """Voice manager related errors."""
    pass


class VoiceManager:
    """
    Central voice management system.
    
    Coordinates voice discovery, loading, registration, and synthesis
    across multiple TTS engines and voice types.
    """
    
    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize voice manager.
        
        Args:
            catalog_path: Path to voices.json catalog file
        """
        self.catalog = VoiceCatalog(catalog_path)
        self._providers: Dict[TTSEngine, BaseTTSProvider] = {}
        self._loaded_voices: Dict[str, Voice] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the voice manager and load catalog."""
        try:
            self.catalog.load()
            self._initialized = True
            logger.info("Voice manager initialized successfully")
        except VoiceCatalogError as e:
            raise VoiceManagerError(f"Failed to initialize voice manager: {e}")
    
    def register_provider(self, provider: BaseTTSProvider) -> None:
        """
        Register a TTS provider.
        
        Args:
            provider: TTS provider instance
        """
        self._providers[provider.engine_type] = provider
        logger.info(f"Registered TTS provider: {provider.engine_name}")
    
    def get_provider(self, engine: TTSEngine) -> Optional[BaseTTSProvider]:
        """
        Get TTS provider for engine.
        
        Args:
            engine: TTS engine type
            
        Returns:
            TTS provider instance or None if not available
        """
        return self._providers.get(engine)
    
    def list_voices(self, 
                   engine: Optional[TTSEngine] = None,
                   language: Optional[str] = None,
                   voice_type: Optional[VoiceType] = None,
                   available_only: bool = True) -> Dict[str, VoiceMetadata]:
        """
        List available voices with optional filtering.
        
        Args:
            engine: Filter by TTS engine
            language: Filter by language
            voice_type: Filter by voice type
            available_only: Only return available voices
            
        Returns:
            Dictionary of voice_id -> VoiceMetadata
        """
        if not self._initialized:
            self.initialize()
        
        voices = self.catalog.get_all_voices()
        
        # Apply filters
        if engine:
            voices = {vid: v for vid, v in voices.items() if v.engine == engine}
        
        if language:
            voices = {vid: v for vid, v in voices.items() if v.language == language}
            
        if voice_type:
            voices = {vid: v for vid, v in voices.items() if v.voice_type == voice_type}
        
        if available_only:
            voices = {vid: v for vid, v in voices.items() if v.is_available()}
        
        return voices
    
    def get_voice(self, voice_id: str, load_if_needed: bool = True) -> Optional[Voice]:
        """
        Get a voice by ID.
        
        Args:
            voice_id: Voice identifier
            load_if_needed: Load voice if not already loaded
            
        Returns:
            Voice instance or None if not found
        """
        if not self._initialized:
            self.initialize()
        
        # Return cached voice if available
        if voice_id in self._loaded_voices:
            return self._loaded_voices[voice_id]
        
        # Get voice metadata from catalog
        voice_metadata = self.catalog.get_voice(voice_id)
        if not voice_metadata:
            logger.warning(f"Voice not found in catalog: {voice_id}")
            return None
        
        if load_if_needed:
            return self._load_voice(voice_metadata)
        else:
            # Return voice without provider (metadata only)
            return Voice(metadata=voice_metadata)
    
    def _load_voice(self, voice_metadata: VoiceMetadata) -> Optional[Voice]:
        """
        Load a voice with its TTS provider.
        
        Args:
            voice_metadata: Voice configuration
            
        Returns:
            Loaded Voice instance or None if loading failed
        """
        # Get TTS provider for this voice
        provider = self.get_provider(voice_metadata.engine)
        if not provider:
            logger.warning(f"No provider available for engine: {voice_metadata.engine}")
            return None
        
        # Check if provider is available
        if not provider.is_available():
            logger.warning(f"Provider not available: {voice_metadata.engine}")
            return None
        
        # Initialize provider if needed
        if not provider._initialized:
            if not provider.initialize():
                logger.error(f"Failed to initialize provider: {voice_metadata.engine}")
                return None
        
        # Load voice in provider
        if not provider.load_voice(voice_metadata):
            logger.error(f"Failed to load voice in provider: {voice_metadata.voice_id}")
            return None
        
        # Create voice instance
        voice = Voice(metadata=voice_metadata, provider=provider)
        
        # Cache loaded voice
        self._loaded_voices[voice_metadata.voice_id] = voice
        
        logger.info(f"Successfully loaded voice: {voice_metadata.voice_id}")
        return voice
    
    def register_voice(self, voice: Union[Voice, VoiceMetadata]) -> bool:
        """
        Register a new voice.
        
        Args:
            voice: Voice or VoiceMetadata to register
            
        Returns:
            bool: True if registration successful
        """
        if not self._initialized:
            self.initialize()
        
        # Extract metadata
        if isinstance(voice, Voice):
            metadata = voice.metadata
        else:
            metadata = voice
        
        try:
            # Validate voice configuration
            from .catalog import validate_voice_config
            errors = validate_voice_config(metadata.to_dict())
            if errors:
                logger.error(f"Voice validation failed: {errors}")
                return False
            
            # Add to catalog
            self.catalog.add_voice(metadata)
            self.catalog.save()
            
            # If it's a full Voice instance, cache it
            if isinstance(voice, Voice):
                self._loaded_voices[metadata.voice_id] = voice
            
            logger.info(f"Successfully registered voice: {metadata.voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register voice {metadata.voice_id}: {e}")
            return False
    
    def unregister_voice(self, voice_id: str) -> bool:
        """
        Unregister a voice.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            bool: True if unregistration successful
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Remove from catalog
            if not self.catalog.remove_voice(voice_id):
                logger.warning(f"Voice not found in catalog: {voice_id}")
                return False
            
            self.catalog.save()
            
            # Remove from cache
            if voice_id in self._loaded_voices:
                del self._loaded_voices[voice_id]
            
            logger.info(f"Successfully unregistered voice: {voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister voice {voice_id}: {e}")
            return False
    
    def preview_voice(self, voice_id: str, text: str = "Hello, this is a voice preview.") -> Optional[Path]:
        """
        Generate a preview sample for a voice.
        
        Args:
            voice_id: Voice identifier
            text: Preview text
            
        Returns:
            Path to generated preview audio file, or None if failed
        """
        voice = self.get_voice(voice_id, load_if_needed=True)
        if not voice or not voice.provider:
            logger.error(f"Cannot preview voice - not available: {voice_id}")
            return None
        
        try:
            return voice.provider.preview_voice(voice, text)
        except Exception as e:
            logger.error(f"Failed to preview voice {voice_id}: {e}")
            return None
    
    def validate_voice_availability(self, voice_id: str) -> bool:
        """
        Check if a voice is available for synthesis.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            bool: True if voice is available
        """
        voice = self.get_voice(voice_id, load_if_needed=False)
        if not voice:
            return False
        
        # Check metadata availability
        if not voice.metadata.is_available():
            return False
        
        # Check provider availability
        provider = self.get_provider(voice.metadata.engine)
        if not provider or not provider.is_available():
            return False
        
        return True
    
    def get_voice_info(self, voice_id: str) -> Optional[Dict]:
        """
        Get detailed information about a voice.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Dictionary with voice information
        """
        voice_metadata = self.catalog.get_voice(voice_id)
        if not voice_metadata:
            return None
        
        provider = self.get_provider(voice_metadata.engine)
        
        return {
            "voice_id": voice_id,
            "name": voice_metadata.name,
            "description": voice_metadata.description,
            "type": voice_metadata.voice_type.value,
            "engine": voice_metadata.engine.value,
            "language": voice_metadata.language,
            "accent": voice_metadata.accent,
            "quality": voice_metadata.quality.value,
            "sample_rate": voice_metadata.sample_rate,
            "available": voice_metadata.is_available(),
            "provider_available": provider is not None and provider.is_available() if provider else False,
            "loaded": voice_id in self._loaded_voices,
            "gender": voice_metadata.gender,
            "age_range": voice_metadata.age_range,
            "speaking_style": voice_metadata.speaking_style,
            "created_at": voice_metadata.created_at.isoformat() if voice_metadata.created_at else None,
            "quality_score": voice_metadata.quality_score
        }
    
    def get_engine_voices(self, engine: TTSEngine) -> List[str]:
        """
        Get voice IDs for a specific engine.
        
        Args:
            engine: TTS engine
            
        Returns:
            List of voice IDs
        """
        voices = self.list_voices(engine=engine)
        return list(voices.keys())
    
    def get_language_voices(self, language: str) -> List[str]:
        """
        Get voice IDs for a specific language.
        
        Args:
            language: Language code
            
        Returns:
            List of voice IDs
        """
        voices = self.list_voices(language=language)
        return list(voices.keys())
    
    def reload_catalog(self) -> None:
        """Reload the voice catalog from disk."""
        try:
            self.catalog.load()
            # Clear cached voices since catalog may have changed
            self._loaded_voices.clear()
            logger.info("Voice catalog reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload catalog: {e}")
            raise VoiceManagerError(f"Failed to reload catalog: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Clean up providers
        for provider in self._providers.values():
            try:
                provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up provider {provider.engine_name}: {e}")
        
        # Clear caches
        self._loaded_voices.clear()
        self._providers.clear()
        self._initialized = False
        
        logger.info("Voice manager cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()