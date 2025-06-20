"""
Custom voice registration and management utilities.

This module handles the creation, validation, and management of custom voices
including voice cloning from audio samples and trained model registration.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import hashlib

from .base import VoiceMetadata, VoiceType, TTSEngine, VoiceQuality
from .catalog import validate_voice_config

logger = logging.getLogger(__name__)

class CustomVoiceError(Exception):
    """Custom voice related errors."""
    pass


class CustomVoiceBuilder:
    """
    Builder class for creating custom voice configurations.
    
    Provides a fluent interface for configuring custom voices
    with validation and best practice defaults.
    """
    
    def __init__(self, voice_id: str, name: str):
        """
        Initialize custom voice builder.
        
        Args:
            voice_id: Unique voice identifier
            name: Human-readable voice name
        """
        self.voice_id = voice_id
        self.name = name
        self._config = {
            "voice_id": voice_id,
            "name": name,
            "voice_type": VoiceType.CUSTOM,
            "created_at": datetime.now()
        }
    
    def description(self, desc: str) -> 'CustomVoiceBuilder':
        """Set voice description."""
        self._config["description"] = desc
        return self
    
    def engine(self, engine: TTSEngine) -> 'CustomVoiceBuilder':
        """Set TTS engine."""
        self._config["engine"] = engine
        return self
    
    def language(self, lang: str) -> 'CustomVoiceBuilder':
        """Set voice language."""
        self._config["language"] = lang
        return self
    
    def accent(self, accent: str) -> 'CustomVoiceBuilder':
        """Set voice accent."""
        self._config["accent"] = accent
        return self
    
    def quality(self, quality: VoiceQuality) -> 'CustomVoiceBuilder':
        """Set voice quality."""
        self._config["quality"] = quality
        return self
    
    def sample_rate(self, rate: int) -> 'CustomVoiceBuilder':
        """Set audio sample rate."""
        if rate < 8000 or rate > 48000:
            raise ValueError("Sample rate must be between 8000 and 48000 Hz")
        self._config["sample_rate"] = rate
        return self
    
    def model_path(self, path: Union[str, Path]) -> 'CustomVoiceBuilder':
        """Set model file path."""
        self._config["model_path"] = Path(path)
        return self
    
    def sample_audio_path(self, path: Union[str, Path]) -> 'CustomVoiceBuilder':
        """Set sample audio file path."""
        self._config["sample_audio_path"] = Path(path)
        return self
    
    def training_data_path(self, path: Union[str, Path]) -> 'CustomVoiceBuilder':
        """Set training data directory path."""
        self._config["training_data_path"] = Path(path)
        return self
    
    def training_info(self, samples_count: int, duration: float) -> 'CustomVoiceBuilder':
        """Set training information."""
        self._config["training_samples_count"] = samples_count
        self._config["training_duration"] = duration
        return self
    
    def quality_score(self, score: float) -> 'CustomVoiceBuilder':
        """Set quality assessment score (0.0-1.0)."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")
        self._config["quality_score"] = score
        return self
    
    def gender(self, gender: str) -> 'CustomVoiceBuilder':
        """Set voice gender."""
        self._config["gender"] = gender
        return self
    
    def age_range(self, age_range: str) -> 'CustomVoiceBuilder':
        """Set voice age range."""
        self._config["age_range"] = age_range
        return self
    
    def speaking_style(self, style: str) -> 'CustomVoiceBuilder':
        """Set speaking style."""
        self._config["speaking_style"] = style
        return self
    
    def engine_config(self, config: Dict) -> 'CustomVoiceBuilder':
        """Set engine-specific configuration."""
        self._config["engine_config"] = config
        return self
    
    def build(self) -> VoiceMetadata:
        """
        Build and validate the voice metadata.
        
        Returns:
            VoiceMetadata instance
            
        Raises:
            CustomVoiceError: If configuration is invalid
        """
        # Validate required fields
        required_fields = ["voice_id", "name", "engine", "language"]
        missing_fields = [field for field in required_fields if field not in self._config]
        
        if missing_fields:
            raise CustomVoiceError(f"Missing required fields: {missing_fields}")
        
        # Set defaults
        if "description" not in self._config:
            self._config["description"] = f"Custom voice: {self.name}"
        
        if "quality" not in self._config:
            self._config["quality"] = VoiceQuality.MEDIUM
            
        if "sample_rate" not in self._config:
            # Set default sample rate based on engine
            engine_defaults = {
                TTSEngine.XTTS: 22050,
                TTSEngine.BARK: 24000,
                TTSEngine.OPENVOICE: 24000,
                TTSEngine.TORTOISE: 22050
            }
            self._config["sample_rate"] = engine_defaults.get(self._config["engine"], 22050)
        
        # Validate configuration
        errors = validate_voice_config(self._config)
        if errors:
            raise CustomVoiceError(f"Voice configuration validation failed: {errors}")
        
        return VoiceMetadata.from_dict(self._config)


class CustomVoiceManager:
    """
    Manager for custom voice operations.
    
    Handles custom voice creation, validation, registration,
    and file management.
    """
    
    def __init__(self, voices_dir: Optional[Path] = None):
        """
        Initialize custom voice manager.
        
        Args:
            voices_dir: Directory for storing custom voice files
        """
        if voices_dir is None:
            voices_dir = Path.home() / ".ebook2audio" / "voices"
        
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
    
    def create_voice_directory(self, voice_id: str) -> Path:
        """
        Create directory structure for a custom voice.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Path to voice directory
        """
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (voice_dir / "models").mkdir(exist_ok=True)
        (voice_dir / "samples").mkdir(exist_ok=True)
        (voice_dir / "training_data").mkdir(exist_ok=True)
        (voice_dir / "previews").mkdir(exist_ok=True)
        
        return voice_dir
    
    def validate_audio_sample(self, audio_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate audio sample for voice cloning.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not audio_path.exists():
            errors.append(f"Audio file not found: {audio_path}")
            return False, errors
        
        # Check file extension
        valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        if audio_path.suffix.lower() not in valid_extensions:
            errors.append(f"Unsupported audio format: {audio_path.suffix}")
        
        # Check file size (should be reasonable for voice samples)
        file_size = audio_path.stat().st_size
        if file_size < 10000:  # Less than 10KB
            errors.append("Audio file too small (likely corrupted)")
        elif file_size > 100_000_000:  # More than 100MB
            errors.append("Audio file too large for voice sample")
        
        # TODO: Add audio format validation using librosa/pydub
        # - Check sample rate
        # - Check audio duration
        # - Check audio quality
        # - Detect silence/noise issues
        
        return len(errors) == 0, errors
    
    def prepare_training_samples(self, 
                               samples_dir: Path, 
                               voice_id: str,
                               target_sample_rate: int = 22050) -> Tuple[bool, Dict]:
        """
        Prepare audio samples for training.
        
        Args:
            samples_dir: Directory containing training samples
            voice_id: Voice identifier
            target_sample_rate: Target sample rate for processed audio
            
        Returns:
            Tuple of (success, preparation_info)
        """
        if not samples_dir.exists():
            return False, {"error": f"Samples directory not found: {samples_dir}"}
        
        # Find audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(samples_dir.glob(f"*{ext}"))
        
        if not audio_files:
            return False, {"error": "No audio files found in samples directory"}
        
        # Create processed samples directory
        voice_dir = self.create_voice_directory(voice_id)
        processed_dir = voice_dir / "training_data" / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        preparation_info = {
            "total_files": len(audio_files),
            "processed_files": 0,
            "total_duration": 0.0,
            "sample_rate": target_sample_rate,
            "errors": []
        }
        
        # TODO: Implement audio preprocessing
        # - Convert to target sample rate
        # - Normalize audio levels
        # - Trim silence
        # - Split long files if needed
        # - Generate spectrograms if required
        
        # For now, just copy files and validate
        for audio_file in audio_files:
            try:
                is_valid, errors = self.validate_audio_sample(audio_file)
                if is_valid:
                    # Copy to processed directory
                    dest_path = processed_dir / audio_file.name
                    shutil.copy2(audio_file, dest_path)
                    preparation_info["processed_files"] += 1
                else:
                    preparation_info["errors"].extend(errors)
            except Exception as e:
                preparation_info["errors"].append(f"Error processing {audio_file.name}: {e}")
        
        success = preparation_info["processed_files"] > 0
        return success, preparation_info
    
    def register_pretrained_voice(self,
                                voice_id: str,
                                name: str,
                                engine: TTSEngine,
                                model_path: Path,
                                sample_audio_path: Optional[Path] = None,
                                **kwargs) -> VoiceMetadata:
        """
        Register a pre-trained custom voice model.
        
        Args:
            voice_id: Unique voice identifier
            name: Human-readable name
            engine: TTS engine for this voice
            model_path: Path to trained model file
            sample_audio_path: Optional sample audio file
            **kwargs: Additional voice configuration
            
        Returns:
            VoiceMetadata for the registered voice
            
        Raises:
            CustomVoiceError: If registration fails
        """
        if not model_path.exists():
            raise CustomVoiceError(f"Model file not found: {model_path}")
        
        # Create voice directory and copy model
        voice_dir = self.create_voice_directory(voice_id)
        
        # Copy model file
        model_dest = voice_dir / "models" / model_path.name
        shutil.copy2(model_path, model_dest)
        
        # Copy sample audio if provided
        if sample_audio_path and sample_audio_path.exists():
            sample_dest = voice_dir / "samples" / sample_audio_path.name
            shutil.copy2(sample_audio_path, sample_dest)
        else:
            sample_dest = None
        
        # Build voice metadata
        builder = CustomVoiceBuilder(voice_id, name)
        builder.engine(engine)
        builder.model_path(model_dest)
        
        if sample_dest:
            builder.sample_audio_path(sample_dest)
        
        # Apply additional configuration
        for key, value in kwargs.items():
            if hasattr(builder, key):
                getattr(builder, key)(value)
        
        return builder.build()
    
    def clone_voice_from_samples(self,
                               voice_id: str,
                               name: str,
                               samples_path: Path,
                               engine: TTSEngine = TTSEngine.XTTS,
                               **kwargs) -> Tuple[bool, Union[VoiceMetadata, str]]:
        """
        Create a voice clone from audio samples.
        
        Args:
            voice_id: Unique voice identifier
            name: Human-readable name
            samples_path: Path to audio samples (file or directory)
            engine: TTS engine to use for cloning
            **kwargs: Additional voice configuration
            
        Returns:
            Tuple of (success, VoiceMetadata or error_message)
        """
        try:
            # Prepare samples
            if samples_path.is_file():
                # Single sample file
                is_valid, errors = self.validate_audio_sample(samples_path)
                if not is_valid:
                    return False, f"Sample validation failed: {errors}"
                
                # Create voice directory and copy sample
                voice_dir = self.create_voice_directory(voice_id)
                sample_dest = voice_dir / "samples" / samples_path.name
                shutil.copy2(samples_path, sample_dest)
                
                preparation_info = {
                    "processed_files": 1,
                    "sample_path": sample_dest
                }
            else:
                # Directory of samples
                success, preparation_info = self.prepare_training_samples(
                    samples_path, voice_id
                )
                if not success:
                    return False, preparation_info.get("error", "Sample preparation failed")
            
            # Build voice metadata
            builder = CustomVoiceBuilder(voice_id, name)
            builder.engine(engine)
            builder.voice_type = VoiceType.CUSTOM
            
            if "sample_path" in preparation_info:
                builder.sample_audio_path(preparation_info["sample_path"])
            
            if "processed_files" in preparation_info:
                builder.training_info(
                    preparation_info["processed_files"],
                    preparation_info.get("total_duration", 0.0)
                )
            
            # Apply additional configuration
            for key, value in kwargs.items():
                if hasattr(builder, key):
                    getattr(builder, key)(value)
            
            voice_metadata = builder.build()
            
            # Save voice configuration
            self._save_voice_config(voice_metadata)
            
            return True, voice_metadata
            
        except Exception as e:
            logger.error(f"Failed to clone voice {voice_id}: {e}")
            return False, str(e)
    
    def _save_voice_config(self, voice_metadata: VoiceMetadata) -> None:
        """Save voice configuration to JSON file."""
        voice_dir = self.voices_dir / voice_metadata.voice_id
        config_path = voice_dir / "voice_config.json"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(voice_metadata.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load_voice_config(self, voice_id: str) -> Optional[VoiceMetadata]:
        """Load voice configuration from JSON file."""
        config_path = self.voices_dir / voice_id / "voice_config.json"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return VoiceMetadata.from_dict(config_data)
        except Exception as e:
            logger.error(f"Failed to load voice config for {voice_id}: {e}")
            return None
    
    def delete_custom_voice(self, voice_id: str) -> bool:
        """
        Delete a custom voice and all its files.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            bool: True if deletion successful
        """
        voice_dir = self.voices_dir / voice_id
        
        if not voice_dir.exists():
            return False
        
        try:
            shutil.rmtree(voice_dir)
            logger.info(f"Deleted custom voice: {voice_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete custom voice {voice_id}: {e}")
            return False
    
    def list_custom_voices(self) -> List[str]:
        """List all custom voice IDs."""
        voice_ids = []
        
        if not self.voices_dir.exists():
            return voice_ids
        
        for item in self.voices_dir.iterdir():
            if item.is_dir() and (item / "voice_config.json").exists():
                voice_ids.append(item.name)
        
        return voice_ids
    
    def get_voice_info(self, voice_id: str) -> Optional[Dict]:
        """Get detailed information about a custom voice."""
        voice_metadata = self.load_voice_config(voice_id)
        if not voice_metadata:
            return None
        
        voice_dir = self.voices_dir / voice_id
        
        # Gather file information
        file_info = {}
        for subdir in ["models", "samples", "training_data", "previews"]:
            subdir_path = voice_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                file_info[subdir] = {
                    "count": len(files),
                    "files": [f.name for f in files if f.is_file()]
                }
            else:
                file_info[subdir] = {"count": 0, "files": []}
        
        return {
            "voice_id": voice_id,
            "metadata": voice_metadata.to_dict(),
            "file_info": file_info,
            "directory": str(voice_dir)
        }


def create_custom_voice(voice_id: str, name: str) -> CustomVoiceBuilder:
    """
    Create a new custom voice builder.
    
    Args:
        voice_id: Unique voice identifier
        name: Human-readable voice name
        
    Returns:
        CustomVoiceBuilder instance
    """
    return CustomVoiceBuilder(voice_id, name)