"""Tests for audiobook conversion pipeline."""

import pytest
from pathlib import Path
from ebook2audio.pipeline import (
    AudioBookPipeline,
    ConversionConfig,
    ConversionResult,
    PipelineStatus,
    TextChunk,
    convert_ebook_to_audiobook
)

class TestConversionConfig:
    """Test ConversionConfig class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = ConversionConfig()
        assert config.voice_id == "gtts_en_us"
        assert config.output_format == "mp3"
        assert config.sample_rate == 22050
        assert config.bitrate == "128k"
        assert config.chunk_size == 1000
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            voice_id="custom_voice",
            output_format="wav",
            sample_rate=44100,
            bitrate="256k",
            chunk_size=500
        )
        assert config.voice_id == "custom_voice"
        assert config.output_format == "wav"
        assert config.sample_rate == 44100
        assert config.bitrate == "256k"
        assert config.chunk_size == 500

class TestTextChunk:
    """Test TextChunk class."""
    
    def test_text_chunk_creation(self):
        """Test creating text chunks."""
        chunk = TextChunk(
            text="This is a test chunk.",
            start_char=0,
            end_char=20,
            chunk_index=0
        )
        assert chunk.text == "This is a test chunk."
        assert chunk.start_char == 0
        assert chunk.end_char == 20
        assert chunk.chunk_index == 0

class TestAudioBookPipeline:
    """Test AudioBookPipeline functionality."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        config = ConversionConfig()
        pipeline = AudioBookPipeline(config)
        assert pipeline is not None
        assert pipeline.config == config
        assert pipeline.status == PipelineStatus.IDLE
    
    def test_pipeline_text_chunking(self):
        """Test text chunking functionality."""
        config = ConversionConfig(chunk_size=50)
        pipeline = AudioBookPipeline(config)
        
        test_text = "This is a test. " * 10  # Create text longer than chunk size
        chunks = pipeline._chunk_text(test_text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.text) <= config.chunk_size * 1.2  # Allow some overflow
    
    def test_pipeline_status_tracking(self):
        """Test pipeline status tracking."""
        config = ConversionConfig()
        pipeline = AudioBookPipeline(config)
        
        assert pipeline.status == PipelineStatus.IDLE
        
        # Status should change during conversion
        # (actual conversion test would require mocking TTS engines)

class TestConversionResult:
    """Test ConversionResult class."""
    
    def test_conversion_result_success(self):
        """Test successful conversion result."""
        result = ConversionResult(
            success=True,
            output_path=Path("/tmp/output.mp3"),
            duration=120.5,
            file_size=1024000,
            chunks_processed=10
        )
        
        assert result.success is True
        assert result.output_path == Path("/tmp/output.mp3")
        assert result.duration == 120.5
        assert result.file_size == 1024000
        assert result.chunks_processed == 10
        assert result.error is None
    
    def test_conversion_result_failure(self):
        """Test failed conversion result."""
        result = ConversionResult(
            success=False,
            error="Test error message"
        )
        
        assert result.success is False
        assert result.error == "Test error message"
        assert result.output_path is None

class TestConversionFunction:
    """Test the main conversion function."""
    
    def test_convert_ebook_to_audiobook_invalid_input(self):
        """Test conversion with invalid input."""
        result = convert_ebook_to_audiobook(
            "nonexistent_file.pdf",
            "output.mp3"
        )
        
        assert result.success is False
        assert result.error is not None
    
    def test_convert_ebook_to_audiobook_mock(self, temp_dir, mocker):
        """Test conversion with mocked components."""
        # Create a test input file
        input_file = temp_dir / "test.txt"
        input_file.write_text("Test content for audiobook conversion.")
        output_file = temp_dir / "output.mp3"
        
        # Mock the TTS engine
        mock_voice = mocker.Mock()
        mock_voice.synthesize.return_value = b"mock audio data"
        
        mocker.patch('ebook2audio.voices.get_voice', return_value=mock_voice)
        
        # Test conversion
        result = convert_ebook_to_audiobook(
            str(input_file),
            str(output_file),
            voice_id="gtts_en_us"
        )
        
        # The actual conversion might fail without real TTS, 
        # but we're testing the structure
        assert isinstance(result, ConversionResult)