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

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GPT_SOVITS_DIR = PROJECT_ROOT / "GPT-SoVITS"
INFERENCE_CLI = GPT_SOVITS_DIR / "GPT_SoVITS" / "inference_cli.py"

logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"GPT-SoVITS dir: {GPT_SOVITS_DIR}")
logger.info(f"Inference CLI: {INFERENCE_CLI}")
logger.info(f"Inference CLI exists: {INFERENCE_CLI.exists()}")

# Check if GPT-SoVITS is available
GPT_SOVITS_AVAILABLE = INFERENCE_CLI.exists()

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
        self.inference_cli_path: Optional[Path] = None
        self.current_voice_id: Optional[str] = None
        self.model_cache: Dict[str, Any] = {}
        # Model paths for LKY voice
        self.s1_model_path: Optional[Path] = None
        self.s2_model_path: Optional[Path] = None
        self.ref_audio_path: Optional[Path] = None
        
    def initialize(self) -> bool:
        """
        Initialize the GPT-SoVITS engine.
        
        Returns:
            bool: True if initialization successful
        """
        if not GPT_SOVITS_AVAILABLE:
            logger.error("GPT-SoVITS inference CLI not available")
            return False
            
        try:
            self.inference_cli_path = INFERENCE_CLI
            
            # Set up model paths for LKY voice
            logs_dir = GPT_SOVITS_DIR / "logs/lky_en_enhanced"
            
            # Find the enhanced S1 model
            s1_models = list(logs_dir.glob("**/*e50.ckpt"))
            if not s1_models:
                s1_models = list(logs_dir.glob("**/*.ckpt"))
            
            if s1_models:
                self.s1_model_path = max(s1_models, key=os.path.getctime)
                logger.info(f"Found S1 model: {self.s1_model_path}")
            else:
                logger.error("No S1 models found")
                return False
            
            # Use pretrained S2 model
            self.s2_model_path = GPT_SOVITS_DIR / "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
            if not self.s2_model_path.exists():
                logger.error(f"S2 model not found: {self.s2_model_path}")
                return False
            
            # Set default reference audio (must be 3-10 seconds)
            # Using trimmed 5-second version of segment 0125
            self.ref_audio_path = GPT_SOVITS_DIR / "output/lky_training_data_enhanced/audio_segments/lky_segment_0125_5sec.wav"
            
            self._initialized = True
            logger.info("GPT-SoVITS provider initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GPT-SoVITS: {e}")
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
        Synthesize speech using GPT-SoVITS inference CLI via subprocess.
        
        Args:
            text: Text to synthesize
            voice: Voice to use for synthesis
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file, or None if synthesis failed
        """
        if not self.is_available() or not self.inference_cli_path:
            logger.error("GPT-SoVITS inference CLI not available")
            return None
            
        if not all([self.s1_model_path, self.s2_model_path, self.ref_audio_path]):
            logger.error("Model paths not properly initialized")
            return None
            
        try:
            # Generate output path if not provided
            if output_path is None:
                temp_dir = Path(tempfile.gettempdir()) / "gpt_sovits_audio"
                temp_dir.mkdir(exist_ok=True)
                output_path = temp_dir / f"lky_{voice.voice_id}_{hash(text)}.wav"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run the GPT-SoVITS inference CLI via subprocess
            logger.info(f"Synthesizing {len(text)} characters with GPT-SoVITS")
            
            # Set up environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{GPT_SOVITS_DIR}:{GPT_SOVITS_DIR}/GPT_SoVITS"
            
            # Build command using the actual inference_cli.py argument names
            # First, create temporary text files for ref_text and target_text
            import tempfile
            
            ref_text = "The odds were against our survival, and we had to ensure our own existence, which we had previously taken for granted."
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as ref_file:
                ref_file.write(ref_text)
                ref_text_path = ref_file.name
                
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as target_file:
                target_file.write(text)
                target_text_path = target_file.name
            
            cmd = [
                "python3", str(self.inference_cli_path),
                "--gpt_model", str(self.s1_model_path),
                "--sovits_model", str(self.s2_model_path),
                "--ref_audio", str(self.ref_audio_path),
                "--ref_text", ref_text_path,
                "--ref_language", "英文",
                "--target_text", target_text_path,
                "--target_language", "英文",
                "--output_path", str(output_path.parent)  # It expects a directory
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(GPT_SOVITS_DIR),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout for longer texts
                )
                
                if result.returncode != 0:
                    logger.error(f"GPT-SoVITS inference failed with return code {result.returncode}")
                    logger.error(f"stderr: {result.stderr}")
                    logger.error(f"stdout: {result.stdout}")
                    return None
                
                # Log stdout for debugging even on success
                logger.info(f"Command completed with return code: {result.returncode}")
                logger.info(f"stdout: {result.stdout}")
                
                # The inference_cli.py saves as ./output.wav relative to the working directory
                actual_output = GPT_SOVITS_DIR / "output.wav"
                
                # Check if output file was created
                if not actual_output.exists():
                    logger.error(f"Output file not created: {actual_output}")
                    # Check if it's in the specified output path directory
                    output_in_dir = Path(str(output_path.parent)) / "output.wav"
                    if output_in_dir.exists():
                        logger.info(f"Found output at: {output_in_dir}")
                        actual_output = output_in_dir
                    else:
                        logger.error(f"No output.wav found in working directory or output directory")
                        return None
                
                # Move to the expected location
                if actual_output != output_path:
                    actual_output.rename(output_path)
                
                logger.info(f"GPT-SoVITS synthesis successful: {output_path}")
                return output_path
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(ref_text_path)
                    os.unlink(target_text_path)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            logger.error("GPT-SoVITS inference timed out after 120 seconds")
            return None
        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis failed: {e}")
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
        self.inference_cli_path = None
        self.s1_model_path = None
        self.s2_model_path = None
        self.ref_audio_path = None
        self.model_cache.clear()
        self.current_voice_id = None
        self._initialized = False