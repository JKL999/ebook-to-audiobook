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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    parallel_synthesis: bool = True
    max_workers: int = 4  # Number of concurrent TTS workers
    
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
        """Synthesize audio for all text chunks with resume capability and parallel processing."""
        logger.info(f"Synthesizing audio for {len(text_chunks)} chunks")
        
        if not PYDUB_AVAILABLE:
            raise RuntimeError("pydub is required for audio processing. Install with: pip install pydub")
        
        # Setup temp directory
        temp_dir = self.config.temp_dir or Path(tempfile.mkdtemp())
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing chunks (resume capability)
        existing_chunks = self._scan_existing_chunks(temp_dir, len(text_chunks))
        skipped_count = len(existing_chunks)
        
        if skipped_count > 0:
            logger.info(f"Found {skipped_count} existing audio chunks, resuming synthesis from chunk {skipped_count}")
        
        # Choose synthesis method based on configuration
        if self.config.parallel_synthesis and len(text_chunks) - skipped_count > 1:
            return self._synthesize_audio_parallel(text_chunks, temp_dir, existing_chunks, skipped_count)
        else:
            return self._synthesize_audio_sequential(text_chunks, temp_dir, existing_chunks, skipped_count)
    
    def _synthesize_audio_sequential(self, text_chunks: List[TextChunk], temp_dir: Path, 
                                   existing_chunks: List[Path], skipped_count: int) -> List[Path]:
        """Sequential synthesis (original method)."""
        # Get voice
        try:
            voice = get_voice(self.config.voice_id)
        except Exception as e:
            raise RuntimeError(f"Failed to get voice '{self.config.voice_id}': {e}")
        
        # Setup progress tracking
        remaining_chunks = len(text_chunks) - skipped_count
        self.progress_manager.start("Synthesizing Audio")
        task_id = self.progress_manager.add_task("synthesis", 
                                                f"Synthesizing chunks (resuming from {skipped_count})...", 
                                                remaining_chunks)
        
        audio_files = list(existing_chunks)  # Start with existing chunks
        
        try:
            for i, chunk in enumerate(text_chunks):
                chunk_audio_path = temp_dir / f"chunk_{i:04d}.mp3"
                
                # Skip if chunk already exists
                if chunk_audio_path.exists() and chunk_audio_path.stat().st_size > 0:
                    chunk.audio_path = chunk_audio_path
                    if chunk_audio_path not in audio_files:
                        audio_files.append(chunk_audio_path)
                        self.temp_files.append(chunk_audio_path)
                    continue
                
                try:
                    synthesized_path = voice.synthesize(chunk.text, chunk_audio_path)
                    if synthesized_path and synthesized_path.exists():
                        chunk.audio_path = synthesized_path
                        audio_files.append(synthesized_path)
                        self.temp_files.append(synthesized_path)
                        logger.debug(f"Synthesized chunk {i}")
                    else:
                        logger.warning(f"Failed to synthesize chunk {i}")
                        
                except Exception as e:
                    logger.error(f"Error synthesizing chunk {i}: {e}")
                
                # Update progress only for newly processed chunks
                if i >= skipped_count:
                    self.progress_manager.update_task(task_id, advance=1)
            
            logger.info(f"Successfully synthesized {len(audio_files)} total audio chunks ({len(audio_files) - skipped_count} new, {skipped_count} resumed)")
            return audio_files
            
        finally:
            self.progress_manager.stop()
    
    def _synthesize_audio_parallel(self, text_chunks: List[TextChunk], temp_dir: Path, 
                                 existing_chunks: List[Path], skipped_count: int) -> List[Path]:
        """Parallel synthesis using ThreadPoolExecutor for improved performance."""
        logger.info(f"Using parallel synthesis with {self.config.max_workers} workers")
        
        # Setup progress tracking
        remaining_chunks = len(text_chunks) - skipped_count
        self.progress_manager.start("Synthesizing Audio (Parallel)")
        task_id = self.progress_manager.add_task("synthesis", 
                                                f"Parallel synthesis (resuming from {skipped_count})...", 
                                                remaining_chunks)
        
        # Initialize results array with existing chunks
        audio_files = [None] * len(text_chunks)
        for i, chunk_path in enumerate(existing_chunks):
            audio_files[i] = chunk_path
            self.temp_files.append(chunk_path)
        
        # Create list of chunks that need synthesis
        chunks_to_process = []
        for i, chunk in enumerate(text_chunks):
            chunk_audio_path = temp_dir / f"chunk_{i:04d}.mp3"
            if not (chunk_audio_path.exists() and chunk_audio_path.stat().st_size > 0):
                chunks_to_process.append((i, chunk, chunk_audio_path))
        
        # Thread-safe progress counter
        progress_lock = threading.Lock()
        completed_count = 0
        
        def synthesize_chunk(chunk_data):
            """Worker function for parallel synthesis with provider fallback."""
            nonlocal completed_count
            i, chunk, chunk_audio_path = chunk_data
            
            try:
                # Primary synthesis attempt
                voice = get_voice(self.config.voice_id)
                synthesized_path = voice.synthesize(chunk.text, chunk_audio_path)
                
                if synthesized_path and synthesized_path.exists():
                    chunk.audio_path = synthesized_path
                    logger.debug(f"Synthesized chunk {i} with primary voice")
                    
                    with progress_lock:
                        nonlocal completed_count
                        completed_count += 1
                        self.progress_manager.update_task(task_id, advance=1)
                    
                    return i, synthesized_path
                else:
                    # Try fallback if primary fails
                    return self._try_fallback_synthesis(i, chunk, chunk_audio_path, progress_lock, task_id)
                    
            except Exception as e:
                logger.warning(f"Primary synthesis failed for chunk {i}: {e}")
                # Try fallback on exception
                return self._try_fallback_synthesis(i, chunk, chunk_audio_path, progress_lock, task_id)
        
        try:
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {executor.submit(synthesize_chunk, chunk_data): chunk_data 
                                 for chunk_data in chunks_to_process}
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_data = future_to_chunk[future]
                    try:
                        chunk_index, synthesized_path = future.result()
                        if synthesized_path:
                            audio_files[chunk_index] = synthesized_path
                            self.temp_files.append(synthesized_path)
                    except Exception as e:
                        logger.error(f"Exception in worker thread: {e}")
            
            # Filter out None values and ensure proper ordering
            final_audio_files = [f for f in audio_files if f is not None]
            
            new_chunks = len(final_audio_files) - skipped_count
            logger.info(f"Successfully synthesized {len(final_audio_files)} total audio chunks ({new_chunks} new, {skipped_count} resumed)")
            return final_audio_files
            
        finally:
            self.progress_manager.stop()
    
    def _scan_existing_chunks(self, temp_dir: Path, total_chunks: int) -> List[Path]:
        """Scan for existing audio chunks that can be reused for resume capability."""
        existing_chunks = []
        
        for i in range(total_chunks):
            chunk_path = temp_dir / f"chunk_{i:04d}.mp3"
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                existing_chunks.append(chunk_path)
            else:
                # Stop at first missing chunk to maintain sequence
                break
        
        return existing_chunks
    
    def _try_fallback_synthesis(self, chunk_index: int, chunk: TextChunk, chunk_audio_path: Path, 
                               progress_lock: threading.Lock, task_id) -> tuple:
        """Attempt synthesis with fallback provider (pyttsx3)."""
        try:
            # Try pyttsx3 as fallback - use a more generic approach
            from .voices.pyttsx3_provider import SystemTTSProvider
            from .voices.base import Voice, VoiceMetadata, TTSEngine
            
            # Create a basic system voice directly
            system_provider = SystemTTSProvider()
            if system_provider.initialize():
                # Create a basic voice metadata for system TTS
                voice_metadata = VoiceMetadata(
                    voice_id="system_fallback",
                    display_name="System TTS Fallback",
                    engine=TTSEngine.PYTTSX3,
                    language="en"
                )
                fallback_voice = Voice(voice_metadata, system_provider)
            else:
                logger.error(f"System TTS provider not available for fallback")
                return chunk_index, None
            synthesized_path = fallback_voice.synthesize(chunk.text, chunk_audio_path)
            
            if synthesized_path and synthesized_path.exists():
                chunk.audio_path = synthesized_path
                logger.info(f"Synthesized chunk {chunk_index} with fallback voice (pyttsx3)")
                
                with progress_lock:
                    self.progress_manager.update_task(task_id, advance=1)
                
                return chunk_index, synthesized_path
            else:
                logger.error(f"Fallback synthesis also failed for chunk {chunk_index}")
                return chunk_index, None
                
        except Exception as e:
            logger.error(f"Fallback synthesis failed for chunk {chunk_index}: {e}")
            return chunk_index, None
    
    def _concatenate_audio(self, audio_files: List[Path], output_path: Path) -> Path:
        """Concatenate audio files into final audiobook with memory-efficient streaming."""
        logger.info(f"Concatenating {len(audio_files)} audio files")
        
        if not audio_files:
            raise RuntimeError("No audio files to concatenate")
        
        # For large audiobooks, use streaming concatenation to avoid memory issues
        if len(audio_files) > 100:  # Large audiobook threshold
            return self._concatenate_audio_streaming(audio_files, output_path)
        else:
            return self._concatenate_audio_memory(audio_files, output_path)
    
    def _concatenate_audio_memory(self, audio_files: List[Path], output_path: Path) -> Path:
        """Memory-based concatenation for smaller audiobooks (original method)."""
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
    
    def _concatenate_audio_streaming(self, audio_files: List[Path], output_path: Path) -> Path:
        """Streaming concatenation for large audiobooks to minimize memory usage."""
        import subprocess
        import tempfile
        
        try:
            logger.info(f"Using streaming concatenation for large audiobook ({len(audio_files)} chunks)")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary file list for ffmpeg
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                filelist_path = Path(f.name)
                
                # Add silence between chunks if configured
                silence_duration = self.config.add_silence_between_chunks
                
                for i, audio_file in enumerate(audio_files):
                    # Write the audio file
                    f.write(f"file '{audio_file.absolute()}'\n")
                    
                    # Add silence between chunks (except after the last chunk)
                    if i < len(audio_files) - 1 and silence_duration > 0:
                        # Create a temporary silence file
                        silence_file = audio_file.parent / f"silence_{i}.mp3"
                        silence_audio = AudioSegment.silent(duration=int(silence_duration * 1000))
                        silence_audio.export(str(silence_file), format="mp3")
                        f.write(f"file '{silence_file.absolute()}'\n")
            
            # Build ffmpeg command for concatenation
            ffmpeg_cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(filelist_path),
                '-c', 'copy',  # Copy without re-encoding for speed
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            # If format conversion is needed, add encoding parameters
            if self.config.output_format != "mp3":
                ffmpeg_cmd.extend(['-f', self.config.output_format])
            
            if self.config.output_format == "mp3":
                ffmpeg_cmd.extend(['-b:a', self.config.bitrate])
            
            # Execute ffmpeg
            logger.info("Running ffmpeg for streaming concatenation...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            
            # Clean up temporary files
            filelist_path.unlink()
            
            # Clean up silence files
            for i in range(len(audio_files) - 1):
                silence_file = audio_files[0].parent / f"silence_{i}.mp3"
                if silence_file.exists():
                    silence_file.unlink()
            
            logger.info(f"Successfully concatenated {len(audio_files)} chunks to {output_path}")
            return output_path
            
        except Exception as e:
            # Fallback to memory-based concatenation if streaming fails
            logger.warning(f"Streaming concatenation failed, falling back to memory concatenation: {e}")
            return self._concatenate_audio_memory(audio_files, output_path)
    
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