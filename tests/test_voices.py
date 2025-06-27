"""Tests for voice management functionality."""

import pytest
from ebook2audio.voices import (
    list_voices,
    get_voice,
    VoiceManager,
    VoiceCatalog,
    VoiceMetadata
)
from ebook2audio.voices.base import Voice
from ebook2audio.voices.builtin import BUILTIN_VOICES

class TestVoiceManagement:
    """Test voice management functionality."""
    
    def test_list_voices(self):
        """Test listing available voices."""
        voices = list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        
        # Check if builtin voices are included
        builtin_ids = [v.id for v in BUILTIN_VOICES]
        listed_ids = [v.id for v in voices]
        for builtin_id in builtin_ids:
            assert builtin_id in listed_ids
    
    def test_get_voice_builtin(self):
        """Test getting a builtin voice."""
        # Test with a known builtin voice
        voice = get_voice("gtts_en_us")
        assert voice is not None
        assert isinstance(voice, Voice)
        assert voice.metadata.id == "gtts_en_us"
    
    def test_get_voice_invalid(self):
        """Test getting an invalid voice."""
        voice = get_voice("invalid_voice_id")
        assert voice is None

class TestVoiceManager:
    """Test VoiceManager class."""
    
    def test_voice_manager_init(self):
        """Test VoiceManager initialization."""
        manager = VoiceManager()
        assert manager is not None
        assert len(manager.list_voices()) > 0
    
    def test_voice_manager_preview(self, temp_dir):
        """Test voice preview functionality."""
        manager = VoiceManager()
        output_path = temp_dir / "preview.mp3"
        
        # Test with a gtts voice (should work without additional setup)
        result = manager.preview_voice(
            "gtts_en_us", 
            "Hello, this is a test.",
            str(output_path)
        )
        
        # Check if preview was created (may fail if gtts not available)
        if result:
            assert output_path.exists()

class TestVoiceCatalog:
    """Test VoiceCatalog functionality."""
    
    def test_voice_catalog_builtin(self):
        """Test builtin voice catalog."""
        catalog = VoiceCatalog()
        voices = catalog.list_builtin_voices()
        
        assert len(voices) > 0
        for voice in voices:
            assert isinstance(voice, VoiceMetadata)
            assert voice.id
            assert voice.name
            assert voice.provider

class TestVoiceMetadata:
    """Test VoiceMetadata class."""
    
    def test_voice_metadata_creation(self):
        """Test creating voice metadata."""
        metadata = VoiceMetadata(
            id="test_voice",
            name="Test Voice",
            provider="test",
            language="en-US",
            gender="neutral",
            description="A test voice"
        )
        
        assert metadata.id == "test_voice"
        assert metadata.name == "Test Voice"
        assert metadata.provider == "test"
        assert metadata.language == "en-US"
        assert metadata.gender == "neutral"
        assert metadata.description == "A test voice"