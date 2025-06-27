"""
Built-in voice presets and utilities.

This module provides pre-configured voice definitions for popular TTS engines
and common voice characteristics. These voices are immediately available
without additional training or setup.
"""

from datetime import datetime
from typing import Dict, Optional, List
from .base import VoiceMetadata, VoiceType, TTSEngine, VoiceQuality

# Built-in voice definitions
BUILTIN_VOICES: Dict[str, VoiceMetadata] = {
    # XTTS-v2 Built-in Voices
    "en_us_vctk_16": VoiceMetadata(
        voice_id="en_us_vctk_16",
        name="English US (VCTK-16)",
        description="High-quality English (US) female voice using VCTK corpus speaker 16. Clear pronunciation with neutral American accent.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.XTTS,
        language="en",
        accent="us",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="female",
        age_range="adult",
        speaking_style="neutral",
        engine_config={
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 1.1,
            "speed": 1.0
        }
    ),
    
    "en_uk_vctk_92": VoiceMetadata(
        voice_id="en_uk_vctk_92",
        name="English UK (VCTK-92)",
        description="High-quality British English male voice using VCTK corpus speaker 92. Refined pronunciation with clear RP accent.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.XTTS,
        language="en",
        accent="uk",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="male",
        age_range="adult",
        speaking_style="neutral",
        engine_config={
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 1.1,
            "speed": 1.0
        }
    ),
    
    "en_au_vctk_60": VoiceMetadata(
        voice_id="en_au_vctk_60",
        name="English AU (VCTK-60)",
        description="High-quality Australian English female voice using VCTK corpus speaker 60. Warm tone with Australian accent.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.XTTS,
        language="en",
        accent="au",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="female",
        age_range="adult",
        speaking_style="warm",
        engine_config={
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 1.1,
            "speed": 1.0
        }
    ),
    
    # Spanish voices
    "es_vctk_79": VoiceMetadata(
        voice_id="es_vctk_79",
        name="Spanish (VCTK-79)",
        description="High-quality Spanish voice using VCTK corpus speaker 79. Clear pronunciation with neutral Spanish accent.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.XTTS,
        language="es",
        accent="neutral",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="male",
        age_range="adult",
        speaking_style="neutral",
        engine_config={
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 1.1,
            "speed": 1.0
        }
    ),
    
    # Bark Built-in Voices
    "bark_speaker_v2_en_0": VoiceMetadata(
        voice_id="bark_speaker_v2_en_0",
        name="Bark English Speaker 0",
        description="Expressive English voice using Bark TTS. Natural intonation with emotional expression capabilities.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.BARK,
        language="en",
        accent="us",
        quality=VoiceQuality.MEDIUM,
        sample_rate=24000,
        gender="neutral",
        age_range="adult",
        speaking_style="expressive",
        engine_config={
            "text_temp": 0.7,
            "waveform_temp": 0.7,
            "speaker_id": "v2/en_speaker_0"
        }
    ),
    
    "bark_speaker_v2_en_1": VoiceMetadata(
        voice_id="bark_speaker_v2_en_1",
        name="Bark English Speaker 1",
        description="Expressive English voice using Bark TTS. Animated delivery with good emotional range.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.BARK,
        language="en",
        accent="us",
        quality=VoiceQuality.MEDIUM,
        sample_rate=24000,
        gender="neutral",
        age_range="adult",
        speaking_style="animated",
        engine_config={
            "text_temp": 0.7,
            "waveform_temp": 0.7,
            "speaker_id": "v2/en_speaker_1"
        }
    ),
    
    "bark_speaker_v2_en_narrator": VoiceMetadata(
        voice_id="bark_speaker_v2_en_narrator",
        name="Bark English Narrator",
        description="Narrative-optimized English voice using Bark TTS. Excellent for storytelling and audiobooks.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.BARK,
        language="en",
        accent="us",
        quality=VoiceQuality.MEDIUM,
        sample_rate=24000,
        gender="neutral",
        age_range="adult",
        speaking_style="narrative",
        engine_config={
            "text_temp": 0.6,
            "waveform_temp": 0.6,
            "speaker_id": "v2/en_speaker_narrator"
        }
    ),
    
    # OpenVoice Built-in Voices  
    "openvoice_v1_en_default": VoiceMetadata(
        voice_id="openvoice_v1_en_default",
        name="OpenVoice English Default",
        description="Versatile English voice with tone and emotion control. Good for varied content types.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.OPENVOICE,
        language="en",
        accent="us",
        quality=VoiceQuality.MEDIUM,
        sample_rate=24000,
        gender="neutral",
        age_range="adult",
        speaking_style="versatile",
        engine_config={
            "tone": "neutral",
            "emotion": "neutral",
            "speed": 1.0,
            "voice_conversion": False
        }
    ),
    
    "openvoice_v1_en_cheerful": VoiceMetadata(
        voice_id="openvoice_v1_en_cheerful",
        name="OpenVoice English Cheerful",
        description="Upbeat English voice with cheerful tone. Great for educational and uplifting content.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.OPENVOICE,
        language="en",
        accent="us",
        quality=VoiceQuality.MEDIUM,
        sample_rate=24000,
        gender="neutral",
        age_range="adult",
        speaking_style="cheerful",
        engine_config={
            "tone": "cheerful",
            "emotion": "happy",
            "speed": 1.0,
            "voice_conversion": False
        }
    ),
    
    # Tortoise TTS Built-in Voices (if available)
    "tortoise_angie": VoiceMetadata(
        voice_id="tortoise_angie",
        name="Tortoise Angie",
        description="High-quality female voice optimized for audiobook narration. Excellent clarity and consistency.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.TORTOISE,
        language="en",
        accent="us",
        quality=VoiceQuality.VERY_HIGH,
        sample_rate=22050,
        gender="female",
        age_range="adult",
        speaking_style="professional",
        engine_config={
            "preset": "ultra_fast",
            "voice_diversity_intelligibility_slider": 0.5,
            "voice_fixer": True
        }
    ),
    
    "tortoise_tom": VoiceMetadata(
        voice_id="tortoise_tom",
        name="Tortoise Tom",
        description="High-quality male voice optimized for audiobook narration. Deep, authoritative tone.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.TORTOISE,
        language="en",
        accent="us",
        quality=VoiceQuality.VERY_HIGH,
        sample_rate=22050,
        gender="male",
        age_range="adult",
        speaking_style="authoritative",
        engine_config={
            "preset": "ultra_fast",
            "voice_diversity_intelligibility_slider": 0.5,
            "voice_fixer": True
        }
    ),
    
    # Google TTS (gTTS) Built-in Voices
    "gtts_en_us": VoiceMetadata(
        voice_id="gtts_en_us",
        name="Google TTS English (US)",
        description="High-quality English (US) voice using Google Text-to-Speech. Requires internet connection.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.GTTS,
        language="en",
        accent="us",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="neutral",
        age_range="adult",
        speaking_style="natural",
        engine_config={
            "tld": "com",
            "slow": False
        }
    ),
    
    "gtts_en_uk": VoiceMetadata(
        voice_id="gtts_en_uk",
        name="Google TTS English (UK)",
        description="High-quality British English voice using Google Text-to-Speech. Requires internet connection.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.GTTS,
        language="en",
        accent="uk",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="neutral",
        age_range="adult",
        speaking_style="natural",
        engine_config={
            "tld": "co.uk",
            "slow": False
        }
    ),
    
    "gtts_en_au": VoiceMetadata(
        voice_id="gtts_en_au",
        name="Google TTS English (AU)",
        description="High-quality Australian English voice using Google Text-to-Speech. Requires internet connection.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.GTTS,
        language="en",
        accent="au",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="neutral",
        age_range="adult",
        speaking_style="natural",
        engine_config={
            "tld": "com.au",
            "slow": False
        }
    ),
    
    "gtts_es": VoiceMetadata(
        voice_id="gtts_es",
        name="Google TTS Spanish",
        description="High-quality Spanish voice using Google Text-to-Speech. Requires internet connection.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.GTTS,
        language="es",
        accent="neutral",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="neutral",
        age_range="adult",
        speaking_style="natural",
        engine_config={
            "tld": "es",
            "slow": False
        }
    ),
    
    "gtts_fr": VoiceMetadata(
        voice_id="gtts_fr",
        name="Google TTS French",
        description="High-quality French voice using Google Text-to-Speech. Requires internet connection.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.GTTS,
        language="fr",
        accent="neutral",
        quality=VoiceQuality.HIGH,
        sample_rate=22050,
        gender="neutral",
        age_range="adult",
        speaking_style="natural",
        engine_config={
            "tld": "fr",
            "slow": False
        }
    ),
    
    # System TTS (pyttsx3) Built-in Voices
    "system_en_default": VoiceMetadata(
        voice_id="system_en_default",
        name="System English Default",
        description="Default system English voice using pyttsx3. Works offline with system TTS engines.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.PYTTSX3,
        language="en",
        accent="system",
        quality=VoiceQuality.MEDIUM,
        sample_rate=22050,
        gender="neutral",
        age_range="adult",
        speaking_style="neutral",
        engine_config={
            "rate": 200,
            "volume": 0.9,
            "system_voice_id": None  # Auto-select best match
        }
    ),
    
    "system_en_female": VoiceMetadata(
        voice_id="system_en_female",
        name="System English Female",
        description="Female system English voice using pyttsx3. Works offline with system TTS engines.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.PYTTSX3,
        language="en",
        accent="system",
        quality=VoiceQuality.MEDIUM,
        sample_rate=22050,
        gender="female",
        age_range="adult",
        speaking_style="neutral",
        engine_config={
            "rate": 200,
            "volume": 0.9,
            "system_voice_id": None  # Auto-select female voice
        }
    ),
    
    "system_en_male": VoiceMetadata(
        voice_id="system_en_male",
        name="System English Male",
        description="Male system English voice using pyttsx3. Works offline with system TTS engines.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.PYTTSX3,
        language="en",
        accent="system",
        quality=VoiceQuality.MEDIUM,
        sample_rate=22050,
        gender="male",
        age_range="adult",
        speaking_style="neutral",
        engine_config={
            "rate": 200,
            "volume": 0.9,
            "system_voice_id": None  # Auto-select male voice
        }
    ),
    
    "system_en_slow": VoiceMetadata(
        voice_id="system_en_slow",
        name="System English Slow",
        description="Slower system English voice for better comprehension. Works offline with system TTS engines.",
        voice_type=VoiceType.BUILTIN,
        engine=TTSEngine.PYTTSX3,
        language="en",
        accent="system",
        quality=VoiceQuality.MEDIUM,
        sample_rate=22050,
        gender="neutral",
        age_range="adult",
        speaking_style="deliberate",
        engine_config={
            "rate": 150,
            "volume": 0.9,
            "system_voice_id": None
        }
    )
}

# Voice collections for easy selection
VOICE_COLLECTIONS = {
    "default": {
        "name": "Default Voices",
        "description": "Recommended voices for general use",
        "voices": ["gtts_en_us", "system_en_default", "en_us_vctk_16", "bark_speaker_v2_en_narrator"]
    },
    "high_quality": {
        "name": "High Quality Voices", 
        "description": "Best quality voices for professional audiobooks",
        "voices": ["tortoise_angie", "tortoise_tom", "gtts_en_us", "gtts_en_uk", "en_us_vctk_16", "en_uk_vctk_92"]
    },
    "expressive": {
        "name": "Expressive Voices",
        "description": "Voices with emotional expression capabilities",
        "voices": ["bark_speaker_v2_en_0", "bark_speaker_v2_en_1", "openvoice_v1_en_cheerful"]
    },
    "fast": {
        "name": "Fast Synthesis",
        "description": "Voices optimized for speed",
        "voices": ["gtts_en_us", "system_en_default", "en_us_vctk_16", "bark_speaker_v2_en_0"]
    },
    "offline": {
        "name": "Offline Voices",
        "description": "Voices that work without internet connection",
        "voices": ["system_en_default", "system_en_female", "system_en_male", "en_us_vctk_16", "bark_speaker_v2_en_narrator"]
    },
    "online": {
        "name": "Online Voices", 
        "description": "High-quality voices requiring internet connection",
        "voices": ["gtts_en_us", "gtts_en_uk", "gtts_en_au", "gtts_es", "gtts_fr"]
    },
    "multilingual": {
        "name": "Multi-language Support",
        "description": "Voices supporting multiple languages",
        "voices": ["gtts_es", "gtts_fr", "es_vctk_79", "bark_speaker_v2_en_0"]
    },
    "audiobook": {
        "name": "Audiobook Optimized",
        "description": "Voices specifically optimized for long-form narration",
        "voices": ["tortoise_angie", "tortoise_tom", "gtts_en_us", "bark_speaker_v2_en_narrator", "system_en_slow"]
    },
    "simple": {
        "name": "Simple TTS",
        "description": "Easy-to-use TTS engines with minimal setup",
        "voices": ["gtts_en_us", "gtts_en_uk", "system_en_default", "system_en_female", "system_en_male"]
    }
}


def get_builtin_voice(voice_id: str) -> Optional[VoiceMetadata]:
    """
    Get a built-in voice by ID.
    
    Args:
        voice_id: Built-in voice identifier
        
    Returns:
        VoiceMetadata instance or None if not found
    """
    return BUILTIN_VOICES.get(voice_id)


def list_builtin_voices(
    language: Optional[str] = None,
    engine: Optional[TTSEngine] = None,
    quality: Optional[VoiceQuality] = None
) -> Dict[str, VoiceMetadata]:
    """
    List built-in voices with optional filtering.
    
    Args:
        language: Filter by language code
        engine: Filter by TTS engine
        quality: Filter by voice quality
        
    Returns:
        Dictionary of voice_id -> VoiceMetadata
    """
    voices = BUILTIN_VOICES.copy()
    
    if language:
        voices = {vid: v for vid, v in voices.items() if v.language == language}
    
    if engine:
        voices = {vid: v for vid, v in voices.items() if v.engine == engine}
        
    if quality:
        voices = {vid: v for vid, v in voices.items() if v.quality == quality}
    
    return voices


def get_voice_collection(collection_name: str) -> List[str]:
    """
    Get voice IDs from a collection.
    
    Args:
        collection_name: Collection name
        
    Returns:
        List of voice IDs
    """
    collection = VOICE_COLLECTIONS.get(collection_name, {})
    return collection.get("voices", [])


def list_voice_collections() -> Dict[str, Dict]:
    """
    List all available voice collections.
    
    Returns:
        Dictionary of collection definitions
    """
    return VOICE_COLLECTIONS.copy()


def get_recommended_voice(
    language: str = "en",
    quality: VoiceQuality = VoiceQuality.HIGH,
    style: str = "neutral"
) -> Optional[str]:
    """
    Get a recommended voice ID based on criteria.
    
    Args:
        language: Target language
        quality: Desired quality level
        style: Speaking style preference
        
    Returns:
        Recommended voice ID or None
    """
    # Filter voices by language
    candidates = {vid: v for vid, v in BUILTIN_VOICES.items() 
                 if v.language == language}
    
    if not candidates:
        return None
    
    # Prefer exact quality match, but allow fallback
    quality_matches = {vid: v for vid, v in candidates.items() 
                      if v.quality == quality}
    
    if quality_matches:
        candidates = quality_matches
    
    # Try to match speaking style
    style_matches = {vid: v for vid, v in candidates.items() 
                    if v.speaking_style and style.lower() in v.speaking_style.lower()}
    
    if style_matches:
        candidates = style_matches
    
    # Return first match (they're ordered by preference in the dict)
    return next(iter(candidates.keys())) if candidates else None


def validate_builtin_voice(voice_id: str) -> bool:
    """
    Validate that a voice ID refers to a valid built-in voice.
    
    Args:
        voice_id: Voice identifier to validate
        
    Returns:
        bool: True if valid built-in voice
    """
    return voice_id in BUILTIN_VOICES


def get_voice_by_characteristics(
    gender: Optional[str] = None,
    accent: Optional[str] = None,
    engine: Optional[TTSEngine] = None,
    quality_min: VoiceQuality = VoiceQuality.MEDIUM
) -> List[str]:
    """
    Find voices matching specific characteristics.
    
    Args:
        gender: Preferred gender ("male", "female", "neutral")
        accent: Preferred accent ("us", "uk", "au", etc.)
        engine: Preferred TTS engine
        quality_min: Minimum quality level
        
    Returns:
        List of matching voice IDs
    """
    matches = []
    
    for voice_id, voice in BUILTIN_VOICES.items():
        # Check quality threshold
        quality_values = [q.value for q in VoiceQuality]
        if quality_values.index(voice.quality.value) < quality_values.index(quality_min.value):
            continue
        
        # Check gender
        if gender and voice.gender != gender:
            continue
            
        # Check accent
        if accent and voice.accent != accent:
            continue
            
        # Check engine
        if engine and voice.engine != engine:
            continue
        
        matches.append(voice_id)
    
    return matches


# Default voice mappings for common use cases
DEFAULT_VOICES = {
    "english_us_female": "system_en_female",
    "english_uk_male": "gtts_en_uk", 
    "english_expressive": "bark_speaker_v2_en_narrator",
    "high_quality_female": "tortoise_angie",
    "high_quality_male": "tortoise_tom",
    "fast_synthesis": "gtts_en_us",
    "offline_default": "system_en_default",
    "online_default": "gtts_en_us",
    "simple_female": "system_en_female",
    "simple_male": "system_en_male",
    "google_us": "gtts_en_us",
    "google_uk": "gtts_en_uk",
    "system_default": "system_en_default"
}