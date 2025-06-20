"""
Voice catalog system for managing voices.json configuration.

This module handles:
- Loading and saving voice catalog JSON files
- Validating voice configurations
- Voice discovery and registration
- Catalog schema validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .base import VoiceMetadata, VoiceType, TTSEngine, VoiceQuality

logger = logging.getLogger(__name__)

# Default voices catalog path
DEFAULT_CATALOG_PATH = Path.home() / ".ebook2audio" / "voices.json"
PROJECT_CATALOG_PATH = Path(__file__).parent.parent.parent.parent / "voices.json"

class VoiceCatalogError(Exception):
    """Voice catalog related errors."""
    pass

class VoiceCatalog:
    """
    Voice catalog management class.
    
    Handles loading, saving, and managing the voices.json configuration file
    that contains all built-in and custom voice definitions.
    """
    
    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize voice catalog.
        
        Args:
            catalog_path: Path to voices.json file. If None, uses default location.
        """
        if catalog_path is None:
            # Try project catalog first, then user catalog
            if PROJECT_CATALOG_PATH.exists():
                self.catalog_path = PROJECT_CATALOG_PATH
            else:
                self.catalog_path = DEFAULT_CATALOG_PATH
        else:
            self.catalog_path = Path(catalog_path)
            
        self._catalog_data: Dict[str, Any] = {}
        self._loaded = False
    
    def ensure_catalog_dir(self) -> None:
        """Ensure catalog directory exists."""
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self, create_if_missing: bool = True) -> Dict[str, Any]:
        """
        Load the voice catalog from JSON file.
        
        Args:
            create_if_missing: Create default catalog if file doesn't exist
            
        Returns:
            Dictionary containing catalog data
            
        Raises:
            VoiceCatalogError: If catalog cannot be loaded
        """
        try:
            if not self.catalog_path.exists():
                if create_if_missing:
                    logger.info(f"Creating default voice catalog at {self.catalog_path}")
                    self._catalog_data = self._create_default_catalog()
                    self.save()
                else:
                    raise VoiceCatalogError(f"Voice catalog not found: {self.catalog_path}")
            else:
                with open(self.catalog_path, 'r', encoding='utf-8') as f:
                    self._catalog_data = json.load(f)
                    
                # Validate catalog structure
                self._validate_catalog_structure()
                
            self._loaded = True
            logger.info(f"Loaded voice catalog from {self.catalog_path}")
            return self._catalog_data
            
        except json.JSONDecodeError as e:
            raise VoiceCatalogError(f"Invalid JSON in catalog file: {e}")
        except Exception as e:
            raise VoiceCatalogError(f"Failed to load catalog: {e}")
    
    def save(self) -> None:
        """
        Save the voice catalog to JSON file.
        
        Raises:
            VoiceCatalogError: If catalog cannot be saved
        """
        try:
            self.ensure_catalog_dir()
            
            # Add metadata
            self._catalog_data["metadata"] = {
                "version": "1.0.0",
                "last_updated": datetime.now().isoformat(),
                "description": "Ebook2Audio Voice Catalog"
            }
            
            with open(self.catalog_path, 'w', encoding='utf-8') as f:
                json.dump(self._catalog_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved voice catalog to {self.catalog_path}")
            
        except Exception as e:
            raise VoiceCatalogError(f"Failed to save catalog: {e}")
    
    def _create_default_catalog(self) -> Dict[str, Any]:
        """Create default voice catalog structure."""
        return {
            "metadata": {
                "version": "1.0.0",
                "last_updated": datetime.now().isoformat(),
                "description": "Ebook2Audio Voice Catalog"
            },
            "built_in_voices": {
                "en_us_vctk_16": {
                    "voice_id": "en_us_vctk_16",
                    "name": "English US (VCTK-16)",
                    "description": "High-quality English (US) voice using VCTK corpus speaker 16",
                    "voice_type": "builtin",
                    "engine": "xtts",
                    "language": "en",
                    "accent": "us",
                    "quality": "high",
                    "sample_rate": 22050,
                    "gender": "female",
                    "age_range": "adult",
                    "speaking_style": "neutral",
                    "engine_config": {
                        "temperature": 0.75,
                        "length_penalty": 1.0,
                        "repetition_penalty": 1.1
                    }
                },
                "en_uk_vctk_92": {
                    "voice_id": "en_uk_vctk_92",
                    "name": "English UK (VCTK-92)",
                    "description": "High-quality British English voice using VCTK corpus speaker 92",
                    "voice_type": "builtin",
                    "engine": "xtts",
                    "language": "en",
                    "accent": "uk",
                    "quality": "high",
                    "sample_rate": 22050,
                    "gender": "male",
                    "age_range": "adult",
                    "speaking_style": "neutral",
                    "engine_config": {
                        "temperature": 0.75,
                        "length_penalty": 1.0,
                        "repetition_penalty": 1.1
                    }
                },
                "bark_speaker_v2_en_0": {
                    "voice_id": "bark_speaker_v2_en_0",
                    "name": "Bark English Speaker 0",
                    "description": "Expressive English voice using Bark TTS",
                    "voice_type": "builtin",
                    "engine": "bark",
                    "language": "en",
                    "accent": "us",
                    "quality": "medium",
                    "sample_rate": 24000,
                    "gender": "neutral",
                    "age_range": "adult",
                    "speaking_style": "expressive",
                    "engine_config": {
                        "text_temp": 0.7,
                        "waveform_temp": 0.7
                    }
                }
            },
            "custom_voices": {},
            "voice_collections": {
                "default": {
                    "name": "Default Voices",
                    "description": "Recommended voices for general use",
                    "voices": ["en_us_vctk_16", "en_uk_vctk_92"]
                },
                "expressive": {
                    "name": "Expressive Voices",
                    "description": "Voices with emotional expression",
                    "voices": ["bark_speaker_v2_en_0"]
                }
            }
        }
    
    def _validate_catalog_structure(self) -> None:
        """Validate catalog has required structure."""
        required_keys = ["built_in_voices", "custom_voices"]
        
        for key in required_keys:
            if key not in self._catalog_data:
                self._catalog_data[key] = {}
                
        # Ensure voice_collections exists
        if "voice_collections" not in self._catalog_data:
            self._catalog_data["voice_collections"] = {}
    
    def get_all_voices(self) -> Dict[str, VoiceMetadata]:
        """Get all voices as VoiceMetadata objects."""
        if not self._loaded:
            self.load()
            
        voices = {}
        
        # Load built-in voices
        for voice_id, voice_data in self._catalog_data.get("built_in_voices", {}).items():
            try:
                voices[voice_id] = VoiceMetadata.from_dict(voice_data)
            except Exception as e:
                logger.warning(f"Failed to load built-in voice {voice_id}: {e}")
        
        # Load custom voices  
        for voice_id, voice_data in self._catalog_data.get("custom_voices", {}).items():
            try:
                voices[voice_id] = VoiceMetadata.from_dict(voice_data)
            except Exception as e:
                logger.warning(f"Failed to load custom voice {voice_id}: {e}")
        
        return voices
    
    def get_voice(self, voice_id: str) -> Optional[VoiceMetadata]:
        """Get a specific voice by ID."""
        voices = self.get_all_voices()
        return voices.get(voice_id)
    
    def add_voice(self, voice_metadata: VoiceMetadata) -> None:
        """Add a voice to the catalog."""
        if not self._loaded:
            self.load()
            
        voice_data = voice_metadata.to_dict()
        
        if voice_metadata.voice_type == VoiceType.BUILTIN:
            self._catalog_data["built_in_voices"][voice_metadata.voice_id] = voice_data
        else:
            self._catalog_data["custom_voices"][voice_metadata.voice_id] = voice_data
    
    def remove_voice(self, voice_id: str) -> bool:
        """Remove a voice from the catalog."""
        if not self._loaded:
            self.load()
            
        # Try built-in voices first
        if voice_id in self._catalog_data.get("built_in_voices", {}):
            del self._catalog_data["built_in_voices"][voice_id]
            return True
            
        # Try custom voices
        if voice_id in self._catalog_data.get("custom_voices", {}):
            del self._catalog_data["custom_voices"][voice_id]
            return True
            
        return False
    
    def list_by_engine(self, engine: TTSEngine) -> List[VoiceMetadata]:
        """List voices by TTS engine."""
        voices = self.get_all_voices()
        return [voice for voice in voices.values() if voice.engine == engine]
    
    def list_by_language(self, language: str) -> List[VoiceMetadata]:
        """List voices by language."""
        voices = self.get_all_voices()
        return [voice for voice in voices.values() if voice.language == language]
    
    def get_collection(self, collection_name: str) -> List[str]:
        """Get voice IDs from a collection."""
        if not self._loaded:
            self.load()
            
        collections = self._catalog_data.get("voice_collections", {})
        collection = collections.get(collection_name, {})
        return collection.get("voices", [])


def load_voice_catalog(catalog_path: Optional[Path] = None) -> VoiceCatalog:
    """
    Load voice catalog from file.
    
    Args:
        catalog_path: Path to catalog file, uses default if None
        
    Returns:
        VoiceCatalog instance
    """
    catalog = VoiceCatalog(catalog_path)
    catalog.load()
    return catalog


def save_voice_catalog(catalog: VoiceCatalog) -> None:
    """
    Save voice catalog to file.
    
    Args:
        catalog: VoiceCatalog instance to save
    """
    catalog.save()


def validate_voice_config(voice_data: Dict[str, Any]) -> List[str]:
    """
    Validate voice configuration data.
    
    Args:
        voice_data: Voice configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = [
        "voice_id", "name", "description", "voice_type", 
        "engine", "language"
    ]
    
    for field in required_fields:
        if field not in voice_data:
            errors.append(f"Missing required field: {field}")
        elif not voice_data[field]:
            errors.append(f"Empty required field: {field}")
    
    # Validate enums
    try:
        if "voice_type" in voice_data:
            VoiceType(voice_data["voice_type"])
    except ValueError:
        errors.append(f"Invalid voice_type: {voice_data.get('voice_type')}")
    
    try:    
        if "engine" in voice_data:
            TTSEngine(voice_data["engine"])
    except ValueError:
        errors.append(f"Invalid engine: {voice_data.get('engine')}")
    
    try:
        if "quality" in voice_data:
            VoiceQuality(voice_data["quality"])
    except ValueError:
        errors.append(f"Invalid quality: {voice_data.get('quality')}")
    
    # Validate sample_rate
    if "sample_rate" in voice_data:
        try:
            sample_rate = int(voice_data["sample_rate"])
            if sample_rate < 8000 or sample_rate > 48000:
                errors.append("Sample rate must be between 8000 and 48000 Hz")
        except (ValueError, TypeError):
            errors.append("Sample rate must be a valid integer")
    
    # Validate quality_score
    if "quality_score" in voice_data and voice_data["quality_score"] is not None:
        try:
            score = float(voice_data["quality_score"])
            if not 0.0 <= score <= 1.0:
                errors.append("Quality score must be between 0.0 and 1.0")
        except (ValueError, TypeError):
            errors.append("Quality score must be a valid number")
    
    return errors