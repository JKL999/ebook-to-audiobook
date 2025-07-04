"""
GPT-SoVITS TTS Provider for high-quality voice cloning.

This module provides integration with GPT-SoVITS for advanced voice synthesis
using trained voice models.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
from loguru import logger

# Add GPT-SoVITS to Python path
gpt_sovits_path = Path(__file__).parent.parent.parent.parent / "GPT_SoVITS"
if gpt_sovits_path.exists():
    sys.path.insert(0, str(gpt_sovits_path))

try:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    GPT_SOVITS_AVAILABLE = True
except ImportError:
    GPT_SOVITS_AVAILABLE = False
    logger.warning("GPT-SoVITS not available. Please ensure GPT_SoVITS is in the project root.")

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
        self.tts_model: Optional[TTS] = None
        self.current_voice_id: Optional[str] = None
        self.model_cache: Dict[str, Any] = {}
        
    def initialize(self) -> bool:
        """
        Initialize the GPT-SoVITS engine.
        
        Returns:
            bool: True if initialization successful
        """
        if not GPT_SOVITS_AVAILABLE:
            logger.error("GPT-SoVITS is not available")
            return False
            
        try:
            # Initialize with default config
            # Models will be loaded when a specific voice is loaded
            self._initialized = True
            logger.info("GPT-SoVITS provider initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GPT-SoVITS: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if GPT-SoVITS is available."""
        return GPT_SOVITS_AVAILABLE and self._initialized
    
    def load_voice(self, voice_metadata: VoiceMetadata) -> bool:
        """
        Load a GPT-SoVITS voice model.
        
        Args:
            voice_metadata: Voice configuration containing model paths
            
        Returns:
            bool: True if voice loaded successfully
        """
        if not self.is_available():
            return False
            
        try:
            # Check if this voice is already loaded
            if self.current_voice_id == voice_metadata.voice_id:
                logger.info(f"Voice {voice_metadata.voice_id} already loaded")
                return True
            
            # Get model paths from engine config
            engine_config = voice_metadata.engine_config or {}
            gpt_model_path = engine_config.get("gpt_model_path")
            sovits_model_path = engine_config.get("sovits_model_path")
            
            if not gpt_model_path or not sovits_model_path:
                logger.error("GPT and SoVITS model paths are required in engine_config")
                return False
            
            # Create TTS config
            tts_config = {
                "device": engine_config.get("device", "cpu"),
                "is_half": engine_config.get("is_half", False),
                "t2s_weights_path": gpt_model_path,
                "vits_weights_path": sovits_model_path,
                "cnhuhbert_base_path": engine_config.get(
                    "cnhuhbert_base_path", 
                    "GPT_SoVITS/pretrained_models/chinese-hubert-base"
                ),
                "bert_base_path": engine_config.get(
                    "bert_base_path",
                    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
                ),
            }
            
            # Initialize TTS model
            logger.info(f"Loading GPT-SoVITS models for voice {voice_metadata.voice_id}")
            self.tts_model = TTS(tts_config)
            self.current_voice_id = voice_metadata.voice_id
            
            # Cache voice-specific settings
            self.model_cache[voice_metadata.voice_id] = {
                "ref_audio_path": engine_config.get("ref_audio_path"),
                "prompt_text": engine_config.get("prompt_text"),
                "prompt_lang": engine_config.get("prompt_lang", "en"),
            }
            
            logger.info(f"Successfully loaded voice {voice_metadata.voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load voice {voice_metadata.voice_id}: {e}")
            return False
    
    def synthesize(self, text: str, voice: Voice, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Synthesize speech using GPT-SoVITS.
        
        Args:
            text: Text to synthesize
            voice: Voice to use for synthesis
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file, or None if synthesis failed
        """
        if not self.is_available() or not self.tts_model:
            logger.error("GPT-SoVITS not initialized or model not loaded")
            return None
            
        # Load voice if not current
        if self.current_voice_id != voice.voice_id:
            if not self.load_voice(voice.metadata):
                return None
        
        try:
            # Get voice-specific settings
            voice_settings = self.model_cache.get(voice.voice_id, {})
            ref_audio_path = voice_settings.get("ref_audio_path")
            prompt_text = voice_settings.get("prompt_text", "")
            prompt_lang = voice_settings.get("prompt_lang", "en")
            
            if not ref_audio_path:
                logger.error("Reference audio path is required for GPT-SoVITS")
                return None
            
            # Get synthesis parameters from voice metadata
            engine_config = voice.metadata.engine_config or {}
            
            # Prepare TTS inputs
            tts_inputs = {
                "text": text,
                "text_lang": engine_config.get("text_lang", "en"),
                "ref_audio_path": ref_audio_path,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "top_k": engine_config.get("top_k", 5),
                "top_p": engine_config.get("top_p", 1.0),
                "temperature": engine_config.get("temperature", 1.0),
                "text_split_method": engine_config.get("text_split_method", "cut5"),
                "batch_size": engine_config.get("batch_size", 1),
                "speed_factor": engine_config.get("speed_factor", 1.0),
                "seed": engine_config.get("seed", -1),
            }
            
            # Run TTS inference
            logger.info(f"Synthesizing {len(text)} characters with voice {voice.voice_id}")
            results = list(self.tts_model.run(tts_inputs))
            
            if not results:
                logger.error("No audio generated")
                return None
            
            # Get the last (complete) audio result
            sample_rate, audio_data = results[-1]
            
            # Ensure audio data is in the correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio if needed
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Generate output path if not provided
            if output_path is None:
                temp_dir = Path(tempfile.gettempdir()) / "gpt_sovits_audio"
                temp_dir.mkdir(exist_ok=True)
                output_path = temp_dir / f"gpt_sovits_{voice.voice_id}_{hash(text)}.wav"
            
            # Save audio file
            sf.write(str(output_path), audio_data, sample_rate)
            logger.info(f"Audio saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
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
    
    def cleanup(self):
        """Clean up resources."""
        self.tts_model = None
        self.model_cache.clear()
        self.current_voice_id = None
        self._initialized = False