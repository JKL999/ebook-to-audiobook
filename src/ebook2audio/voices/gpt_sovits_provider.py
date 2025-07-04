"""
GPT-SoVITS TTS Provider for high-quality voice cloning.

This module provides integration with GPT-SoVITS for advanced voice synthesis
using trained voice models.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
from loguru import logger

# Add LKY custom inference script path
lky_inference_script = Path(__file__).parent.parent.parent.parent / "lky_audiobook_inference" / "inference.py"
logger.info(f"Looking for LKY inference script at: {lky_inference_script}")
logger.info(f"Script exists: {lky_inference_script.exists()}")

# Use subprocess approach to call the working standalone inference
GPT_SOVITS_AVAILABLE = lky_inference_script.exists()

from .base import BaseTTSProvider, Voice, VoiceMetadata, TTSEngine


class GPTSoVITSProvider(BaseTTSProvider):
    """
    GPT-SoVITS TTS provider for high-quality voice synthesis.
    
    This provider uses trained GPT-SoVITS models to synthesize speech
    with voice cloning capabilities.
    """
    
    def __init__(self):
        """Initialize GPT-SoVITS provider."""
        super().__init__(TTSEngine.GPT_SOVITS)
        self.inference_script_path: Optional[Path] = None
        self.current_voice_id: Optional[str] = None
        self.model_cache: Dict[str, Any] = {}
        
    def initialize(self) -> bool:
        """
        Initialize the GPT-SoVITS engine.
        
        Returns:
            bool: True if initialization successful
        """
        if not GPT_SOVITS_AVAILABLE:
            logger.error("LKY inference script not available")
            return False
            
        try:
            self.inference_script_path = lky_inference_script
            self._initialized = True
            logger.info("LKY GPT-SoVITS provider initialized with subprocess approach")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LKY GPT-SoVITS: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPT-SoVITS is available."""
        return GPT_SOVITS_AVAILABLE and self._initialized
    
    def load_voice(self, voice_metadata: VoiceMetadata) -> bool:
        """
        Load a voice (simplified for subprocess approach).
        
        Args:
            voice_metadata: Voice configuration
            
        Returns:
            bool: True if voice configuration is valid
        """
        if not self.is_available():
            return False
            
        try:
            # Since we're using subprocess, we just validate the voice config
            # The actual model loading happens in the subprocess
            if voice_metadata.voice_id == "lky_en_trained":
                self.current_voice_id = voice_metadata.voice_id
                logger.info(f"Voice {voice_metadata.voice_id} configured for subprocess inference")
                return True
            else:
                logger.warning(f"Voice {voice_metadata.voice_id} not supported by LKY inference")
                return False
            
        except Exception as e:
            logger.error(f"Failed to configure voice {voice_metadata.voice_id}: {e}")
            return False
    
    def synthesize(self, text: str, voice: Voice, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Synthesize speech using LKY inference script via subprocess.
        
        Args:
            text: Text to synthesize
            voice: Voice to use for synthesis
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file, or None if synthesis failed
        """
        if not self.is_available() or not self.inference_script_path:
            logger.error("LKY inference script not available")
            return None
            
        try:
            # Generate output path if not provided
            if output_path is None:
                temp_dir = Path(tempfile.gettempdir()) / "lky_audio"
                temp_dir.mkdir(exist_ok=True)
                output_path = temp_dir / f"lky_{voice.voice_id}_{hash(text)}.wav"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run the LKY inference script via subprocess
            logger.info(f"Synthesizing {len(text)} characters with LKY voice via subprocess")
            
            cmd = [
                "python3", str(self.inference_script_path),
                "--text", text,
                "--output", str(output_path)
            ]
            
            # Change to the inference script directory for proper imports
            inference_dir = self.inference_script_path.parent
            
            result = subprocess.run(
                cmd,
                cwd=str(inference_dir),
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                logger.error(f"LKY inference failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return None
            
            # Check if output file was created
            if not output_path.exists():
                logger.error(f"Output file not created: {output_path}")
                return None
            
            logger.info(f"LKY audio synthesis successful: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("LKY inference timed out after 60 seconds")
            return None
        except Exception as e:
            logger.error(f"LKY synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_voice_settings(self) -> Dict[str, Any]:
        """Get available voice settings for GPT-SoVITS."""
        return {
            "gpt_model_path": {
                "type": "path",
                "description": "Path to GPT model checkpoint (.ckpt)",
                "required": True
            },
            "sovits_model_path": {
                "type": "path", 
                "description": "Path to SoVITS model checkpoint (.pth)",
                "required": True
            },
            "ref_audio_path": {
                "type": "path",
                "description": "Reference audio for voice cloning",
                "required": True
            },
            "prompt_text": {
                "type": "string",
                "description": "Text content of reference audio",
                "required": False
            },
            "prompt_lang": {
                "type": "string",
                "description": "Language of prompt text",
                "default": "en",
                "options": ["en", "zh", "ja"]
            },
            "text_lang": {
                "type": "string",
                "description": "Language of text to synthesize",
                "default": "en",
                "options": ["en", "zh", "ja"]
            },
            "top_k": {
                "type": "integer",
                "description": "Top-k sampling parameter",
                "default": 5,
                "min": 1,
                "max": 20
            },
            "top_p": {
                "type": "float",
                "description": "Top-p sampling parameter",
                "default": 1.0,
                "min": 0.1,
                "max": 1.0
            },
            "temperature": {
                "type": "float",
                "description": "Sampling temperature",
                "default": 1.0,
                "min": 0.1,
                "max": 2.0
            },
            "speed_factor": {
                "type": "float",
                "description": "Speech speed factor",
                "default": 1.0,
                "min": 0.5,
                "max": 2.0
            },
            "device": {
                "type": "string",
                "description": "Compute device",
                "default": "cpu",
                "options": ["cpu", "cuda"]
            },
            "is_half": {
                "type": "boolean",
                "description": "Use half precision",
                "default": False
            }
        }
    
    def stop(self):
        """Stop any ongoing synthesis."""
        if self.tts_model:
            self.tts_model.stop_flag = True
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["en", "zh", "ja"]
    
    def preview_voice(self, voice: Voice, preview_text: str = "Hello, this is a voice preview.") -> Optional[Path]:
        """Generate a preview sample for a voice."""
        return self.synthesize(preview_text, voice)
    
    def cleanup(self):
        """Clean up resources."""
        self.inference_script_path = None
        self.model_cache.clear()
        self.current_voice_id = None
        self._initialized = False