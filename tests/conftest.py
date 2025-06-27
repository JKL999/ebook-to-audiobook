"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_pdf_path() -> Path:
    """Return path to sample PDF file."""
    return Path(__file__).parent / "data" / "sample.pdf"

@pytest.fixture
def sample_epub_path() -> Path:
    """Return path to sample EPUB file."""
    return Path(__file__).parent / "data" / "sample.epub"

@pytest.fixture
def sample_txt_path() -> Path:
    """Return path to sample text file."""
    return Path(__file__).parent / "data" / "sample.txt"

@pytest.fixture
def sample_audio_path() -> Path:
    """Return path to sample audio file for voice cloning."""
    return Path(__file__).parent / "data" / "sample_voice.wav"

@pytest.fixture
def mock_tts_engine(mocker):
    """Mock TTS engine for testing."""
    mock = mocker.Mock()
    mock.synthesize.return_value = b"mock audio data"
    mock.get_voice_info.return_value = {"name": "Mock Voice", "language": "en"}
    return mock