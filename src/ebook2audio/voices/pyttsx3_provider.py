"""
Pyttsx3 offline TTS provider implementation.

This module provides TTS synthesis using pyttsx3, which uses
system-installed TTS engines for offline speech synthesis.
"""

import logging
import tempfile
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    import pyttsx3
    from pydub import AudioSegment
    PYTTSX3_AVAILABLE = True
except ImportError as e:
    print(f"pyttsx3 import error: {e}")
    PYTTSX3_AVAILABLE = False

from .base import BaseTTSProvider, TTSEngine, Voice, VoiceMetadata

logger = logging.getLogger(__name__)


class SystemTTSProvider(BaseTTSProvider):
    """
    System TTS provider using pyttsx3.
    
    Uses system-installed TTS engines (SAPI on Windows, NSSpeechSynthesizer on macOS,
    espeak on Linux) for offline speech synthesis.
    """
    
    def __init__(self):
        super().__init__(TTSEngine.PYTTSX3)
        self._engine = None
        self._available_voices = None
        self._current_voice_id = None
        
    def initialize(self) -> bool:
        """Initialize the pyttsx3 provider."""
        if not PYTTSX3_AVAILABLE:
            logger.error("pyttsx3 library not available. Install with: pip install pyttsx3")
            return False
            
        try:
            # Initialize pyttsx3 engine
            self._engine = pyttsx3.init()
            
            # Test basic functionality
            voices = self._engine.getProperty('voices')
            if not voices:
                logger.warning("No system voices found")
                return False
                
            self._initialized = True
            logger.info(f"pyttsx3 provider initialized with {len(voices)} system voices")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 provider: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if pyttsx3 is available."""
        return PYTTSX3_AVAILABLE and self._initialized and self._engine is not None
    
    def load_voice(self, voice_metadata: VoiceMetadata) -> bool:
        """
        Load a voice for pyttsx3 synthesis.
        
        Sets the voice in the pyttsx3 engine based on voice metadata.
        """
        if voice_metadata.engine != TTSEngine.PYTTSX3:
            logger.error(f"Voice {voice_metadata.voice_id} is not a pyttsx3 voice")
            return False
            
        if not self.is_available():
            logger.error("pyttsx3 provider not available")
            return False
            
        try:
            # Get system voice ID from voice metadata
            system_voice_id = voice_metadata.get_engine_specific_config('system_voice_id')
            
            if system_voice_id:
                # Set specific voice by ID
                self._engine.setProperty('voice', system_voice_id)
                self._current_voice_id = system_voice_id
            else:
                # Find voice by characteristics
                voices = self._get_available_voices()
                matching_voice = self._find_matching_voice(voice_metadata, voices)
                
                if matching_voice:
                    self._engine.setProperty('voice', matching_voice['id'])
                    self._current_voice_id = matching_voice['id']
                else:
                    logger.warning(f"No matching system voice found for {voice_metadata.voice_id}")
                    # Use default voice
                    default_voices = self._engine.getProperty('voices')
                    if default_voices:
                        self._engine.setProperty('voice', default_voices[0].id)
                        self._current_voice_id = default_voices[0].id
            
            # Set voice properties
            rate = voice_metadata.get_engine_specific_config('rate', 200)
            volume = voice_metadata.get_engine_specific_config('volume', 0.9)
            
            self._engine.setProperty('rate', rate)
            self._engine.setProperty('volume', volume)
            
            logger.info(f"Loaded pyttsx3 voice: {voice_metadata.voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pyttsx3 voice {voice_metadata.voice_id}: {e}")
            return False
    
    def synthesize(self, text: str, voice: Voice, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Synthesize speech using pyttsx3.
        
        Args:
            text: Text to synthesize
            voice: Voice configuration to use
            output_path: Optional output file path
            
        Returns:
            Path to generated audio file, or None if synthesis failed
        """
        if not self.is_available():
            logger.error("pyttsx3 provider not available")
            return None
            
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None
            
        try:
            # Load voice if needed
            if not self.load_voice(voice.metadata):
                logger.error("Failed to load voice for synthesis")
                return None
            
            # Generate output path if not provided
            if output_path is None:
                output_path = Path(tempfile.mktemp(suffix='.wav'))
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio to file
            self._engine.save_to_file(text, str(output_path))
            self._engine.runAndWait()
            
            # Check if file was created
            if not output_path.exists():
                logger.error("pyttsx3 failed to create audio file")
                return None
            
            # Convert to target format/sample rate if needed
            try:
                target_sample_rate = voice.metadata.sample_rate
                if target_sample_rate:
                    audio = AudioSegment.from_wav(str(output_path))
                    audio = audio.set_frame_rate(target_sample_rate)
                    
                    # Convert to MP3 if desired
                    if output_path.suffix.lower() == '.mp3':
                        audio.export(str(output_path), format="mp3")
                    else:
                        audio.export(str(output_path), format="wav")
                        
            except Exception as e:
                logger.warning(f"Failed to process audio format: {e}")
            
            logger.info(f"Successfully synthesized audio to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to synthesize with pyttsx3: {e}")
            return None
    
    def preview_voice(self, voice: Voice, preview_text: str = "Hello, this is a preview of this voice.") -> Optional[Path]:
        """Generate a preview sample for a pyttsx3 voice."""
        try:
            preview_path = Path(tempfile.mktemp(suffix='_preview.wav'))
            return self.synthesize(preview_text, voice, preview_path)
        except Exception as e:
            logger.error(f"Failed to generate voice preview: {e}")
            return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages based on available system voices."""
        if not self.is_available():
            return []
            
        languages = set()
        voices = self._get_available_voices()
        
        for voice in voices:
            # Extract language from voice name/locale
            lang = self._extract_language_from_voice(voice)
            if lang:
                languages.add(lang)
        
        return list(languages)
    
    def _get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available system voices."""
        if not self.is_available():
            return []
            
        if self._available_voices is None:
            voices = []
            try:
                system_voices = self._engine.getProperty('voices')
                for voice in system_voices:
                    voice_info = {
                        'id': voice.id,
                        'name': voice.name,
                        'languages': getattr(voice, 'languages', []),
                        'gender': getattr(voice, 'gender', None),
                        'age': getattr(voice, 'age', None)
                    }
                    voices.append(voice_info)
                    
                self._available_voices = voices
                
            except Exception as e:
                logger.error(f"Failed to get available voices: {e}")
                return []
        
        return self._available_voices
    
    def _find_matching_voice(self, voice_metadata: VoiceMetadata, available_voices: List[Dict]) -> Optional[Dict]:
        """Find system voice that best matches the voice metadata."""
        if not available_voices:
            return None
        
        # Try to match by language and gender
        for voice in available_voices:
            voice_lang = self._extract_language_from_voice(voice)
            if voice_lang == voice_metadata.language:
                # Check gender if available
                if voice_metadata.gender and voice.get('gender'):
                    if voice_metadata.gender.lower() in voice['gender'].lower():
                        return voice
                else:
                    return voice
        
        # Fallback: return first voice with matching language
        for voice in available_voices:
            voice_lang = self._extract_language_from_voice(voice)
            if voice_lang == voice_metadata.language:
                return voice
        
        # Last resort: return first available voice
        return available_voices[0] if available_voices else None
    
    def _extract_language_from_voice(self, voice_info: Dict[str, Any]) -> Optional[str]:
        """Extract language code from voice information."""
        # Try languages attribute first
        if voice_info.get('languages'):
            languages = voice_info['languages']
            if languages:
                # Return first language code
                lang = languages[0]
                if isinstance(lang, str) and len(lang) >= 2:
                    return lang[:2].lower()
        
        # Fallback: parse from voice name
        name = voice_info.get('name', '').lower()
        
        # Common language patterns in voice names
        language_patterns = {
            'english': 'en',
            'spanish': 'es', 
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'portuguese': 'pt',
            'russian': 'ru',
            'japanese': 'ja',
            'chinese': 'zh',
            'korean': 'ko',
            'dutch': 'nl',
            'swedish': 'sv',
            'danish': 'da',
            'norwegian': 'no',
            'finnish': 'fi'
        }
        
        for lang_name, lang_code in language_patterns.items():
            if lang_name in name:
                return lang_code
        
        # Default to English if no language detected
        return 'en'
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about available system voices."""
        if not self.is_available():
            return {}
        
        voices = self._get_available_voices()
        system_info = {
            'platform': platform.system(),
            'total_voices': len(voices),
            'voices': voices
        }
        
        return system_info
    
    def cleanup(self) -> None:
        """Clean up pyttsx3 provider resources."""
        if self._engine:
            try:
                self._engine.stop()
            except:
                pass
            self._engine = None
        
        self._available_voices = None
        self._current_voice_id = None
        super().cleanup()