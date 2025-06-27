"""
Google Text-to-Speech (gTTS) provider implementation.

This module provides TTS synthesis using Google's Text-to-Speech service,
which offers high-quality voices with internet connectivity required.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from io import BytesIO

try:
    from gtts import gTTS
    from pydub import AudioSegment
    GTTS_AVAILABLE = True
except ImportError as e:
    print(f"gTTS import error: {e}")
    GTTS_AVAILABLE = False

from .base import BaseTTSProvider, TTSEngine, Voice, VoiceMetadata

logger = logging.getLogger(__name__)


class GoogleTTSProvider(BaseTTSProvider):
    """
    Google Text-to-Speech provider.
    
    Uses Google's gTTS service to synthesize speech from text.
    Requires internet connectivity but provides high-quality voices
    in multiple languages.
    """
    
    def __init__(self):
        super().__init__(TTSEngine.GTTS)
        self._available_languages = None
        
    def initialize(self) -> bool:
        """Initialize the gTTS provider."""
        if not GTTS_AVAILABLE:
            logger.error("gTTS library not available. Install with: pip install gTTS")
            return False
            
        try:
            # Test basic functionality with a simple synthesis
            test_tts = gTTS(text="test", lang='en', slow=False)
            # If no exception, gTTS is working
            self._initialized = True
            logger.info("gTTS provider initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize gTTS provider: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if gTTS is available."""
        return GTTS_AVAILABLE and self._initialized
    
    def load_voice(self, voice_metadata: VoiceMetadata) -> bool:
        """
        Load a voice for gTTS synthesis.
        
        For gTTS, voices are defined by language codes and don't need
        explicit loading, so this just validates the voice configuration.
        """
        if voice_metadata.engine != TTSEngine.GTTS:
            logger.error(f"Voice {voice_metadata.voice_id} is not a gTTS voice")
            return False
            
        # Validate language code
        if not self._is_language_supported(voice_metadata.language):
            logger.error(f"Language not supported by gTTS: {voice_metadata.language}")
            return False
            
        return True
    
    def synthesize(self, text: str, voice: Voice, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Synthesize speech using gTTS.
        
        Args:
            text: Text to synthesize
            voice: Voice configuration to use
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file, or None if synthesis failed
        """
        if not self.is_available():
            logger.error("gTTS provider not available")
            return None
            
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None
            
        try:
            # Get language and TLD from voice metadata
            lang = voice.metadata.language
            tld = voice.metadata.get_engine_specific_config('tld', 'com')
            slow = voice.metadata.get_engine_specific_config('slow', False)
            
            # Create gTTS instance
            tts = gTTS(text=text, lang=lang, slow=slow, tld=tld)
            
            # Generate output path if not provided
            if output_path is None:
                output_path = Path(tempfile.mktemp(suffix='.mp3'))
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio to file
            tts.save(str(output_path))
            
            # Convert to target sample rate if specified
            target_sample_rate = voice.metadata.sample_rate
            if target_sample_rate and target_sample_rate != 22050:  # gTTS default is ~22kHz
                try:
                    audio = AudioSegment.from_mp3(str(output_path))
                    audio = audio.set_frame_rate(target_sample_rate)
                    audio.export(str(output_path), format="mp3")
                except Exception as e:
                    logger.warning(f"Failed to adjust sample rate: {e}")
            
            logger.info(f"Successfully synthesized audio to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to synthesize with gTTS: {e}")
            return None
    
    def preview_voice(self, voice: Voice, preview_text: str = "Hello, this is a preview of this voice.") -> Optional[Path]:
        """Generate a preview sample for a gTTS voice."""
        try:
            preview_path = Path(tempfile.mktemp(suffix='_preview.mp3'))
            return self.synthesize(preview_text, voice, preview_path)
        except Exception as e:
            logger.error(f"Failed to generate voice preview: {e}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        if self._available_languages is None:
            try:
                from gtts.lang import tts_langs
                self._available_languages = list(tts_langs().keys())
            except Exception as e:
                logger.warning(f"Failed to get gTTS languages: {e}")
                # Fallback to common languages
                self._available_languages = [
                    'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
                    'ar', 'hi', 'nl', 'sv', 'da', 'no', 'fi', 'pl', 'tr', 'cs'
                ]
        
        return self._available_languages.copy()
    
    def _is_language_supported(self, language: str) -> bool:
        """Check if a language is supported by gTTS."""
        supported_langs = self.get_supported_languages()
        return language in supported_langs
    
    def get_available_tlds(self) -> List[str]:
        """Get available top-level domains for different regional accents."""
        return [
            'com',    # Default
            'co.uk',  # British English
            'com.au', # Australian English  
            'ca',     # Canadian English
            'co.in',  # Indian English
            'ie',     # Irish English
            'co.za',  # South African English
            'fr',     # French
            'de',     # German
            'es',     # Spanish
            'com.br', # Brazilian Portuguese
            'pt',     # European Portuguese
            'com.mx', # Mexican Spanish
        ]
    
    def cleanup(self) -> None:
        """Clean up gTTS provider resources."""
        self._available_languages = None
        super().cleanup()