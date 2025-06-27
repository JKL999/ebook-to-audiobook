"""
Audio pipeline for converting ebooks to audiobooks.

This module provides the main AudioBookPipeline class that orchestrates
the complete conversion process from ebook to audiobook:

1. Text extraction from ebook files
2. Text chunking for optimal TTS processing
3. Sequential audio synthesis using TTS providers
4. Audio concatenation and post-processing
5. Progress tracking and error handling
"""

import re
import time
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum

try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from loguru import logger
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn

from .extract import extract_text
from .voices import get_voice_manager, get_voice, VoiceManager
from .voices.gtts_provider import GoogleTTSProvider
from .voices.pyttsx3_provider import SystemTTSProvider
from .utils import ProgressManager, format_duration, format_file_size


class PipelineStatus(Enum):
    """Status of the pipeline conversion."""
    IDLE = "idle"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    SYNTHESIZING = "synthesizing"
    CONCATENATING = "concatenating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TextChunk:
    """Represents a chunk of text for TTS processing."""
    index: int
    text: str
    estimated_duration: float = 0.0
    audio_path: Optional[Path] = None
    
    @property
    def word_count(self) -> int:
        """Get word count of the chunk."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Get character count of the chunk."""
        return len(self.text)


@dataclass
class ConversionConfig:
    """Configuration for audiobook conversion."""
    # TTS settings
    voice_id: str = "gtts_en_us"
    speaking_rate: float = 1.0
    volume: float = 0.9
    
    # Chunking settings
    max_chunk_size: int = 1000  # characters
    chunk_overlap: int = 50     # characters
    sentence_break: bool = True
    
    # Audio settings
    output_format: str = "mp3"
    sample_rate: int = 22050
    bitrate: str = "128k"
    
    # Processing settings
    add_silence_between_chunks: float = 0.5  # seconds
    normalize_audio: bool = True
    
    # Output settings
    output_filename: Optional[str] = None
    temp_dir: Optional[Path] = None


@dataclass
class ConversionResult:
    """Result of audiobook conversion."""
    success: bool
    output_path: Optional[Path] = None
    duration: Optional[float] = None  # seconds
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    processing_time: Optional[float] = None  # seconds
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100


class TextChunker:
    """Handles text chunking for optimal TTS processing."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        
    def chunk_text(self, text_pages: List[str]) -> List[TextChunk]:
        """
        Chunk text into optimal sizes for TTS processing.
        
        Args:
            text_pages: List of page/chapter texts
            
        Returns:
            List of text chunks
        """
        logger.info(f"Chunking text from {len(text_pages)} pages")
        
        # Combine all pages into single text
        full_text = "\n\n".join(text_pages)
        
        # Clean and normalize text
        full_text = self._clean_text(full_text)
        
        # Split into chunks
        chunks = self._split_into_chunks(full_text)
        
        # Create TextChunk objects
        text_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = TextChunk(
                index=i,
                text=chunk_text,
                estimated_duration=self._estimate_duration(chunk_text)
            )
            text_chunks.append(chunk)
        
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for TTS."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Normalize punctuation
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{2,}', '—', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        
        return text.strip()
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of appropriate size."""
        chunks = []
        
        if self.config.sentence_break:
            # Split by sentences for better TTS quality
            chunks = self._split_by_sentences(text)
        else:
            # Split by character count
            chunks = self._split_by_characters(text)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences, respecting chunk size limits."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with current sentence
                if len(sentence) <= self.config.max_chunk_size:
                    current_chunk = sentence
                else:
                    # Sentence is too long, split it by characters
                    char_chunks = self._split_by_characters(sentence)
                    chunks.extend(char_chunks[:-1])  # Add all but last
                    current_chunk = char_chunks[-1] if char_chunks else ""
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_by_characters(self, text: str) -> List[str]:
        """Split text by character count."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.max_chunk_size
            
            # If we're at the end, take the rest
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            # Try to find a good break point (space, punctuation)
            break_point = end
            for i in range(end, max(start, end - self.config.chunk_overlap), -1):
                if text[i] in ' \n\t.,!?;:':
                    break_point = i
                    break
            
            chunks.append(text[start:break_point].strip())
            start = break_point
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration for text (rough estimate)."""
        # Rough estimate: 150 words per minute average reading speed
        word_count = len(text.split())
        duration_minutes = word_count / 150
        return duration_minutes * 60  # Convert to seconds


class AudioBookPipeline:
    """Main pipeline for converting ebooks to audiobooks."""
    
    def __init__(self, config: ConversionConfig = None, console: Console = None):
        self.config = config or ConversionConfig()
        self.console = console or Console()
        self.progress_manager = ProgressManager(self.console)
        
        # Initialize components
        self.chunker = TextChunker(self.config)
        self.voice_manager = get_voice_manager()
        
        # State
        self.status = PipelineStatus.IDLE
        self.temp_files: List[Path] = []
        
        # Initialize TTS providers
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize TTS providers."""
        try:
            # Initialize Google TTS provider
            gtts_provider = GoogleTTSProvider()
            if gtts_provider.initialize():
                logger.info("Google TTS provider initialized")
            
            # Initialize System TTS provider
            system_provider = SystemTTSProvider()
            if system_provider.initialize():
                logger.info("System TTS provider initialized")
                
        except Exception as e:
            logger.warning(f"Failed to initialize some TTS providers: {e}")
    
    def convert(self, input_path: Union[str, Path], output_path: Union[str, Path] = None) -> ConversionResult:
        """
        Convert an ebook to audiobook.
        
        Args:
            input_path: Path to the ebook file
            output_path: Path for the output audiobook (optional)
            
        Returns:
            ConversionResult with success status and details
        """
        start_time = time.time()
        input_path = Path(input_path)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.with_suffix(f'.{self.config.output_format}')
        else:
            output_path = Path(output_path)
        
        logger.info(f"Starting conversion: {input_path.name} -> {output_path.name}")
        
        try:
            # Step 1: Extract text
            self.status = PipelineStatus.EXTRACTING
            text_pages = self._extract_text(input_path)
            
            # Step 2: Chunk text
            self.status = PipelineStatus.CHUNKING
            text_chunks = self._chunk_text(text_pages)
            
            # Step 3: Synthesize audio
            self.status = PipelineStatus.SYNTHESIZING
            audio_files = self._synthesize_audio(text_chunks)
            
            # Step 4: Concatenate audio
            self.status = PipelineStatus.CONCATENATING
            final_audio = self._concatenate_audio(audio_files, output_path)
            
            # Calculate results
            processing_time = time.time() - start_time
            audio_duration = self._get_audio_duration(final_audio)
            
            self.status = PipelineStatus.COMPLETED
            
            result = ConversionResult(
                success=True,
                output_path=final_audio,
                duration=audio_duration,
                total_chunks=len(text_chunks),
                processed_chunks=len(audio_files),
                processing_time=processing_time
            )
            
            logger.info(f"Conversion completed successfully in {format_duration(processing_time)}")
            return result
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            error_msg = f"Conversion failed: {str(e)}"
            logger.error(error_msg)
            
            return ConversionResult(
                success=False,
                error_message=error_msg,
                processing_time=time.time() - start_time
            )
        
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files()
    
    def _extract_text(self, input_path: Path) -> List[str]:
        """Extract text from ebook file."""
        logger.info(f"Extracting text from {input_path.name}")
        
        try:
            text_pages = extract_text(input_path)
            logger.info(f"Extracted {len(text_pages)} pages")
            return text_pages
        except Exception as e:
            raise RuntimeError(f"Failed to extract text: {e}")
    
    def _chunk_text(self, text_pages: List[str]) -> List[TextChunk]:
        """Chunk text for TTS processing."""
        logger.info("Chunking text for TTS processing")
        
        try:
            return self.chunker.chunk_text(text_pages)
        except Exception as e:
            raise RuntimeError(f"Failed to chunk text: {e}")
    
    def _synthesize_audio(self, text_chunks: List[TextChunk]) -> List[Path]:
        """Synthesize audio for all text chunks."""
        logger.info(f"Synthesizing audio for {len(text_chunks)} chunks")
        
        if not PYDUB_AVAILABLE:
            raise RuntimeError("pydub is required for audio processing. Install with: pip install pydub")
        
        # Get voice
        try:
            voice = get_voice(self.config.voice_id)
        except Exception as e:
            raise RuntimeError(f"Failed to get voice '{self.config.voice_id}': {e}")
        
        # Setup progress tracking
        self.progress_manager.start("Synthesizing Audio")
        task_id = self.progress_manager.add_task("synthesis", "Synthesizing chunks...", len(text_chunks))
        
        audio_files = []
        temp_dir = self.config.temp_dir or Path(tempfile.mkdtemp())
        
        try:
            for i, chunk in enumerate(text_chunks):
                # Generate audio for chunk
                chunk_audio_path = temp_dir / f"chunk_{i:04d}.mp3"
                
                try:
                    synthesized_path = voice.synthesize(chunk.text, chunk_audio_path)
                    if synthesized_path and synthesized_path.exists():
                        chunk.audio_path = synthesized_path
                        audio_files.append(synthesized_path)
                        self.temp_files.append(synthesized_path)
                    else:
                        logger.warning(f"Failed to synthesize chunk {i}")
                        
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i}: {e}")
                
                # Update progress
                self.progress_manager.update_task(task_id, advance=1)
            
            logger.info(f"Successfully synthesized {len(audio_files)} audio chunks")
            return audio_files
            
        finally:
            self.progress_manager.stop()
    
    def _concatenate_audio(self, audio_files: List[Path], output_path: Path) -> Path:
        """Concatenate audio files into final audiobook."""
        logger.info(f"Concatenating {len(audio_files)} audio files")
        
        if not audio_files:
            raise RuntimeError("No audio files to concatenate")
        
        try:
            # Load first audio file
            combined_audio = AudioSegment.from_file(str(audio_files[0]))
            
            # Add silence between chunks if configured
            silence_duration = int(self.config.add_silence_between_chunks * 1000)  # Convert to ms
            silence = AudioSegment.silent(duration=silence_duration)
            
            # Concatenate remaining files
            for audio_file in audio_files[1:]:
                audio_segment = AudioSegment.from_file(str(audio_file))
                combined_audio = combined_audio + silence + audio_segment
            
            # Normalize audio if configured
            if self.config.normalize_audio:
                combined_audio = combined_audio.normalize()
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export final audio
            export_params = {
                "format": self.config.output_format,
                "parameters": ["-ar", str(self.config.sample_rate)]
            }
            
            if self.config.output_format == "mp3":
                export_params["bitrate"] = self.config.bitrate
            
            combined_audio.export(str(output_path), **export_params)
            
            logger.info(f"Exported final audiobook to {output_path}")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to concatenate audio: {e}")
    
    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get duration of audio file in seconds."""
        try:
            audio = AudioSegment.from_file(str(audio_path))
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return None
    
    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        self.temp_files.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "status": self.status.value,
            "temp_files": len(self.temp_files),
            "config": {
                "voice_id": self.config.voice_id,
                "output_format": self.config.output_format,
                "max_chunk_size": self.config.max_chunk_size
            }
        }


def create_pipeline(voice_id: str = "gtts_en_us", output_format: str = "mp3") -> AudioBookPipeline:
    """
    Create a pre-configured audio pipeline.
    
    Args:
        voice_id: Voice to use for TTS
        output_format: Output audio format
        
    Returns:
        Configured AudioBookPipeline instance
    """
    config = ConversionConfig(
        voice_id=voice_id,
        output_format=output_format
    )
    
    return AudioBookPipeline(config)


# Convenience function for simple usage
def convert_ebook_to_audiobook(
    input_path: Union[str, Path],
    output_path: Union[str, Path] = None,
    voice_id: str = "gtts_en_us",
    output_format: str = "mp3"
) -> ConversionResult:
    """
    Convert an ebook to audiobook with simple interface.
    
    Args:
        input_path: Path to the ebook file
        output_path: Path for the output audiobook (optional)
        voice_id: Voice to use for TTS
        output_format: Output audio format
        
    Returns:
        ConversionResult with success status and details
    """
    pipeline = create_pipeline(voice_id, output_format)
    return pipeline.convert(input_path, output_path)