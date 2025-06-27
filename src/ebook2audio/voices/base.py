"""
Base classes for voice management and TTS engine abstraction.

This module defines the core interfaces and data structures for the voice system:
- VoiceMetadata: Voice configuration and metadata
- Voice: Complete voice specification with provider and metadata
- BaseTTSProvider: Abstract interface for TTS engines
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class VoiceType(Enum):
    """Voice type classification."""
    BUILTIN = "builtin"
    CUSTOM = "custom"
    TRAINED = "trained"


class TTSEngine(Enum):
    """Supported TTS engines."""
    XTTS = "xtts"
    BARK = "bark"
    OPENVOICE = "openvoice"  
    TORTOISE = "tortoise"
    GTTS = "gtts"
    PYTTSX3 = "pyttsx3"


class VoiceQuality(Enum):
    """Voice quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class VoiceMetadata:
    """
    Voice metadata and configuration.
    
    This contains all the information needed to identify, load, and use a voice
    including engine-specific configuration parameters.
    """
    # Basic identification
    voice_id: str
    name: str
    description: str
    voice_type: VoiceType
    
    # Engine configuration
    engine: TTSEngine
    language: str
    accent: Optional[str] = None
    
    # Quality and performance
    quality: VoiceQuality = VoiceQuality.MEDIUM
    sample_rate: int = 22050
    quality_score: Optional[float] = None
    
    # File paths and resources
    model_path: Optional[Path] = None
    sample_audio_path: Optional[Path] = None
    training_data_path: Optional[Path] = None
    
    # Training information (for custom voices)
    training_samples_count: Optional[int] = None
    training_duration: Optional[float] = None  # in seconds
    created_at: Optional[datetime] = None
    trained_at: Optional[datetime] = None
    
    # Engine-specific configuration
    engine_config: Dict[str, Any] = None
    
    # Voice characteristics
    gender: Optional[str] = None
    age_range: Optional[str] = None
    speaking_style: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.engine_config is None:
            self.engine_config = {}
            
        # Convert string enums back to enum objects if needed
        if isinstance(self.voice_type, str):
            self.voice_type = VoiceType(self.voice_type)
        if isinstance(self.engine, str):
            self.engine = TTSEngine(self.engine)
        if isinstance(self.quality, str):
            self.quality = VoiceQuality(self.quality)
            
        # Convert path strings to Path objects
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        if isinstance(self.sample_audio_path, str):
            self.sample_audio_path = Path(self.sample_audio_path)
        if isinstance(self.training_data_path, str):
            self.training_data_path = Path(self.training_data_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        
        # Convert enums to strings
        data['voice_type'] = self.voice_type.value
        data['engine'] = self.engine.value
        data['quality'] = self.quality.value
        
        # Convert paths to strings
        if self.model_path:
            data['model_path'] = str(self.model_path)
        if self.sample_audio_path:
            data['sample_audio_path'] = str(self.sample_audio_path)
        if self.training_data_path:
            data['training_data_path'] = str(self.training_data_path)
            
        # Convert datetime to ISO string
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.trained_at:
            data['trained_at'] = self.trained_at.isoformat()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceMetadata':
        """Create VoiceMetadata from dictionary."""
        # Convert datetime strings back to datetime objects
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'trained_at' in data and data['trained_at']:
            data['trained_at'] = datetime.fromisoformat(data['trained_at'])
            
        return cls(**data)
    
    def is_available(self) -> bool:
        """Check if this voice is available (model files exist, etc.)."""
        if self.voice_type == VoiceType.BUILTIN:
            return True  # Built-in voices are always available
            
        # For custom voices, check if model path exists
        if self.model_path and not self.model_path.exists():
            return False
            
        return True
    
    def get_engine_specific_config(self, key: str, default: Any = None) -> Any:
        """Get engine-specific configuration value."""
        return self.engine_config.get(key, default)


@dataclass 
class Voice:
    """
    Complete voice specification.
    
    Combines voice metadata with the actual TTS provider instance
    for synthesis capabilities.
    """
    metadata: VoiceMetadata
    provider: Optional['BaseTTSProvider'] = None
    
    @property
    def voice_id(self) -> str:
        """Get voice ID."""
        return self.metadata.voice_id
    
    @property
    def name(self) -> str:
        """Get voice name."""
        return self.metadata.name
    
    @property
    def engine(self) -> TTSEngine:
        """Get TTS engine."""
        return self.metadata.engine
    
    @property
    def language(self) -> str:
        """Get voice language."""
        return self.metadata.language
    
    @property
    def quality(self) -> VoiceQuality:
        """Get voice quality."""
        return self.metadata.quality
    
    def is_available(self) -> bool:
        """Check if voice is available for synthesis."""
        return self.metadata.is_available() and self.provider is not None
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file, or None if synthesis failed
        """
        if not self.provider:
            raise ValueError(f"No TTS provider available for voice {self.voice_id}")
            
        return self.provider.synthesize(text, self, output_path)


class BaseTTSProvider(ABC):
    """
    Abstract base class for TTS engine providers.
    
    Each TTS engine (XTTS, Bark, OpenVoice, Tortoise) implements this interface
    to provide a consistent API for voice synthesis.
    """
    
    def __init__(self, engine_type: TTSEngine):
        self.engine_type = engine_type
        self._initialized = False
    
    @property
    def engine_name(self) -> str:
        """Get engine name."""
        return self.engine_type.value
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the TTS engine.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the TTS engine is available.
        
        Returns:
            bool: True if engine can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def load_voice(self, voice_metadata: VoiceMetadata) -> bool:
        """
        Load a voice for synthesis.
        
        Args:
            voice_metadata: Voice configuration to load
            
        Returns:
            bool: True if voice loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def synthesize(self, text: str, voice: Voice, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Synthesize speech from text using the specified voice.
        
        Args:
            text: Text to synthesize
            voice: Voice to use for synthesis
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file, or None if synthesis failed
        """
        pass
    
    @abstractmethod
    def preview_voice(self, voice: Voice, preview_text: str = "Hello, this is a voice preview.") -> Optional[Path]:
        """
        Generate a preview sample for a voice.
        
        Args:
            voice: Voice to preview
            preview_text: Text to use for preview
            
        Returns:
            Path to generated preview audio file, or None if preview failed
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of language codes (e.g., ['en', 'es', 'fr'])
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()