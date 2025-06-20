"""
Configuration management for ebook2audio.

Handles user configuration files, default settings, and configuration validation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from loguru import logger


class LogLevel(Enum):
    """Available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class TTSConfig:
    """TTS-specific configuration."""
    engine: str = "xtts"
    voice: str = "built-in/en_us_vctk_16"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    format: str = "mp3"
    quality: str = "high"  # low, medium, high
    bitrate: str = "128k"
    sample_rate: int = 22050
    channels: int = 1  # mono


@dataclass
class ProcessingConfig:
    """Processing-specific configuration."""
    parallel_jobs: int = 4
    chunk_size: int = 1000
    resume: bool = True
    temp_dir: Optional[str] = None


@dataclass
class UserConfig:
    """Complete user configuration."""
    tts: TTSConfig = None
    audio: AudioConfig = None
    processing: ProcessingConfig = None
    log_level: LogLevel = LogLevel.INFO
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.tts is None:
            self.tts = TTSConfig()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()


class ConfigManager:
    """Manages user configuration files and settings."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "ebook2audio"
        self.config_file = self.config_dir / "config.json"
        self.voices_file = self.config_dir / "voices.json"
        self._config: Optional[UserConfig] = None
        
    def ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self) -> UserConfig:
        """Load configuration from file or create default."""
        if self._config is not None:
            return self._config
            
        self.ensure_config_dir()
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                
                # Create config objects from dict data
                tts_config = TTSConfig(**data.get('tts', {}))
                audio_config = AudioConfig(**data.get('audio', {}))
                processing_config = ProcessingConfig(**data.get('processing', {}))
                
                self._config = UserConfig(
                    tts=tts_config,
                    audio=audio_config,
                    processing=processing_config,
                    log_level=LogLevel(data.get('log_level', 'INFO')),
                    output_dir=data.get('output_dir')
                )
                
                logger.debug(f"Loaded configuration from {self.config_file}")
                
            except Exception as e:
                logger.warning(f"Failed to load config, using defaults: {e}")
                self._config = UserConfig()
        else:
            logger.debug("No config file found, using defaults")
            self._config = UserConfig()
            
        return self._config
    
    def save_config(self, config: UserConfig) -> None:
        """Save configuration to file."""
        self.ensure_config_dir()
        
        # Convert to dict for JSON serialization
        config_dict = {
            'tts': asdict(config.tts),
            'audio': asdict(config.audio),
            'processing': asdict(config.processing),
            'log_level': config.log_level.value,
            'output_dir': config.output_dir
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self._config = config
            logger.debug(f"Saved configuration to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def get_config(self) -> UserConfig:
        """Get current configuration."""
        return self.load_config()
    
    def update_config(self, **kwargs) -> None:
        """Update specific configuration values."""
        config = self.load_config()
        
        # Update fields if provided
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.save_config(config)
    
    def load_voices(self) -> Dict[str, Any]:
        """Load voice catalog from file."""
        if not self.voices_file.exists():
            return self._create_default_voices()
        
        try:
            with open(self.voices_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load voices, using defaults: {e}")
            return self._create_default_voices()
    
    def save_voices(self, voices: Dict[str, Any]) -> None:
        """Save voice catalog to file."""
        self.ensure_config_dir()
        
        try:
            with open(self.voices_file, 'w') as f:
                json.dump(voices, f, indent=2)
            logger.debug(f"Saved voices to {self.voices_file}")
        except Exception as e:
            logger.error(f"Failed to save voices: {e}")
            raise
    
    def _create_default_voices(self) -> Dict[str, Any]:
        """Create default voice catalog."""
        default_voices = {
            "built-in": {
                "en_us_vctk_16": {
                    "name": "English US VCTK 16",
                    "language": "en-US",
                    "gender": "mixed",
                    "engine": "xtts",
                    "quality": "high",
                    "description": "Default English voice with American accent"
                },
                "en_uk_vctk_8": {
                    "name": "English UK VCTK 8", 
                    "language": "en-UK",
                    "gender": "mixed",
                    "engine": "xtts",
                    "quality": "high",
                    "description": "British English voice"
                }
            },
            "custom": {},
            "last_updated": "2025-06-20"
        }
        
        # Save default voices
        self.save_voices(default_voices)
        return default_voices
    
    def list_voices(self) -> List[str]:
        """List all available voice IDs."""
        voices = self.load_voices()
        voice_ids = []
        
        # Add built-in voices
        for voice_id in voices.get("built-in", {}):
            voice_ids.append(f"built-in/{voice_id}")
        
        # Add custom voices
        for voice_id in voices.get("custom", {}):
            voice_ids.append(voice_id)
        
        return sorted(voice_ids)
    
    def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific voice."""
        voices = self.load_voices()
        
        if voice_id.startswith("built-in/"):
            voice_name = voice_id.replace("built-in/", "")
            return voices.get("built-in", {}).get(voice_name)
        else:
            return voices.get("custom", {}).get(voice_id)
    
    def add_custom_voice(self, voice_id: str, voice_info: Dict[str, Any]) -> None:
        """Add a custom voice to the catalog."""
        voices = self.load_voices()
        
        if "custom" not in voices:
            voices["custom"] = {}
        
        voices["custom"][voice_id] = voice_info
        voices["last_updated"] = "2025-06-20"
        
        self.save_voices(voices)
        logger.info(f"Added custom voice: {voice_id}")


# Global config manager instance
config_manager = ConfigManager()