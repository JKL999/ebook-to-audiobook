"""
Command-line interface for ebook2audio.

This module provides a comprehensive CLI for converting ebooks to audiobooks with
support for various TTS engines, voice selection, progress tracking, and batch processing.

Features:
- Convert ebooks to audiobooks with progress bars
- List and filter available voices
- Voice preview functionality
- Batch processing support
- Rich output formatting
- Comprehensive error handling and user feedback
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from rich.prompt import Confirm, Prompt
from loguru import logger

from . import (
    AudioBookPipeline,
    ConversionConfig,
    ConversionResult,
    list_voices,
    get_voice,
    get_voice_manager,
    setup_logging,
    format_duration,
    format_file_size,
    BUILTIN_VOICES,
)
from .voices.base import TTSEngine, VoiceQuality
from .voices.builtin import VOICE_COLLECTIONS, list_builtin_voices
from .utils import FileUtils

# Initialize Typer app and Rich console
app = typer.Typer(
    name="ebook2audio",
    help="Convert ebooks to audiobooks using various TTS engines",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

# Subcommands
voices_app = typer.Typer(
    name="voices",
    help="Manage and list TTS voices",
    add_completion=False
)
app.add_typer(voices_app, name="voices")


def setup_cli_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Setup logging for CLI usage."""
    if quiet:
        level = "ERROR"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    
    setup_logging(level=level, console=console)


def validate_input_file(file_path: Path) -> Path:
    """Validate input file exists and is supported format."""
    if not file_path.exists():
        console.print(f"[red]Error: Input file does not exist: {file_path}[/red]")
        raise typer.Exit(1)
    
    if not file_path.is_file():
        console.print(f"[red]Error: Input path is not a file: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Check supported formats
    supported_extensions = {'.pdf', '.epub', '.mobi', '.azw', '.azw3', '.txt'}
    if file_path.suffix.lower() not in supported_extensions:
        console.print(f"[yellow]Warning: File extension '{file_path.suffix}' may not be supported.[/yellow]")
        console.print(f"Supported formats: {', '.join(supported_extensions)}")
        
        if not Confirm.ask("Continue anyway?", default=False):
            raise typer.Exit(0)
    
    return file_path


def validate_voice_id(voice_id: str) -> str:
    """Validate voice ID exists."""
    try:
        voice = get_voice(voice_id)
        if voice is None:
            available_voices = list(list_voices().keys())
            console.print(f"[red]Error: Voice '{voice_id}' not found.[/red]")
            console.print(f"Available voices: {', '.join(available_voices[:10])}")
            if len(available_voices) > 10:
                console.print(f"... and {len(available_voices) - 10} more. Use 'ebook2audio voices list' to see all.")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error validating voice '{voice_id}': {e}[/red]")
        raise typer.Exit(1)
    
    return voice_id


def generate_output_path(input_path: Path, output_format: str) -> Path:
    """Generate output path based on input file."""
    return input_path.with_suffix(f'.{output_format}')


def display_conversion_summary(result: ConversionResult, input_path: Path) -> None:
    """Display conversion result summary."""
    if result.success:
        # Success summary
        panel_content = f"""[green]✓ Conversion completed successfully![/green]

[bold]Input:[/bold] {input_path.name}
[bold]Output:[/bold] {result.output_path.name if result.output_path else 'N/A'}
[bold]Duration:[/bold] {format_duration(result.duration) if result.duration else 'Unknown'}
[bold]Processing Time:[/bold] {format_duration(result.processing_time) if result.processing_time else 'Unknown'}
[bold]Chunks Processed:[/bold] {result.processed_chunks}/{result.total_chunks} ({result.success_rate:.1f}%)"""

        if result.output_path and result.output_path.exists():
            file_size = format_file_size(result.output_path.stat().st_size)
            panel_content += f"\n[bold]Output Size:[/bold] {file_size}"

        console.print(Panel(panel_content, title="Conversion Results", border_style="green"))
        
        # Show output file location
        if result.output_path:
            console.print(f"\n[bold]Audiobook saved to:[/bold] [blue]{result.output_path.absolute()}[/blue]")
    else:
        # Error summary
        panel_content = f"""[red]✗ Conversion failed![/red]

[bold]Input:[/bold] {input_path.name}
[bold]Error:[/bold] {result.error_message or 'Unknown error'}
[bold]Processing Time:[/bold] {format_duration(result.processing_time) if result.processing_time else 'Unknown'}
[bold]Chunks Processed:[/bold] {result.processed_chunks}/{result.total_chunks}"""

        console.print(Panel(panel_content, title="Conversion Failed", border_style="red"))


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Path to the ebook file (PDF, EPUB, MOBI, TXT)"),
    output_file: Optional[Path] = typer.Argument(None, help="Output audiobook file path (optional)"),
    voice: Optional[str] = typer.Option("gtts_en_us", "--voice", "-v", help="Voice ID to use for TTS"),
    output_format: str = typer.Option("mp3", "--format", "-f", help="Output audio format (mp3, wav, flac)"),
    speaking_rate: float = typer.Option(1.0, "--rate", "-r", help="Speaking rate (0.5-2.0)", min=0.5, max=2.0),
    volume: float = typer.Option(0.9, "--volume", help="Audio volume (0.0-1.0)", min=0.0, max=1.0),
    chunk_size: int = typer.Option(1000, "--chunk-size", help="Text chunk size for TTS processing", min=100, max=5000),
    add_silence: float = typer.Option(0.5, "--silence", help="Silence between chunks in seconds", min=0.0, max=5.0),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Normalize audio levels"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode - minimal output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without actually converting"),
) -> None:
    """
    Convert an ebook to audiobook using Text-to-Speech.
    
    This command converts ebooks in various formats (PDF, EPUB, MOBI, TXT) to audiobooks
    using the specified TTS voice and settings.
    
    Examples:
    
        # Basic conversion
        ebook2audio convert book.pdf audiobook.mp3
        
        # With specific voice
        ebook2audio convert book.pdf audiobook.mp3 --voice gtts_en_uk
        
        # Custom settings
        ebook2audio convert book.pdf audiobook.wav --rate 1.2 --chunk-size 800
        
        # Dry run to preview
        ebook2audio convert book.pdf --dry-run
    """
    setup_cli_logging(verbose=verbose, quiet=quiet)
    
    # Validate inputs
    input_file = validate_input_file(input_file)
    voice_id = validate_voice_id(voice)
    
    # Generate output path if not provided
    if output_file is None:
        output_file = generate_output_path(input_file, output_format)
    
    # Check if output file already exists
    if output_file.exists() and not dry_run:
        console.print(f"[yellow]Warning: Output file already exists: {output_file}[/yellow]")
        if not Confirm.ask("Overwrite existing file?", default=False):
            console.print("Conversion cancelled.")
            raise typer.Exit(0)
    
    # Display conversion plan
    console.print(Panel(f"""[bold]Conversion Plan[/bold]

[bold]Input File:[/bold] {input_file.name} ({format_file_size(input_file.stat().st_size)})
[bold]Output File:[/bold] {output_file.name}
[bold]Voice:[/bold] {voice_id}
[bold]Format:[/bold] {output_format.upper()}
[bold]Speaking Rate:[/bold] {speaking_rate}x
[bold]Volume:[/bold] {volume}
[bold]Chunk Size:[/bold] {chunk_size} characters
[bold]Silence Between Chunks:[/bold] {add_silence}s
[bold]Normalize Audio:[/bold] {"Yes" if normalize else "No"}""", 
                        title="Conversion Settings", border_style="blue"))
    
    if dry_run:
        console.print("[yellow]Dry run completed. No files were created.[/yellow]")
        return
    
    # Create conversion configuration
    config = ConversionConfig(
        voice_id=voice_id,
        speaking_rate=speaking_rate,
        volume=volume,
        max_chunk_size=chunk_size,
        output_format=output_format,
        add_silence_between_chunks=add_silence,
        normalize_audio=normalize,
        output_filename=str(output_file)
    )
    
    # Initialize pipeline
    console.print("\n[blue]Initializing conversion pipeline...[/blue]")
    pipeline = AudioBookPipeline(config=config, console=console)
    
    # Start conversion
    start_time = time.time()
    console.print(f"[green]Starting conversion of {input_file.name}...[/green]\n")
    
    try:
        result = pipeline.convert(input_file, output_file)
        
        # Display results
        display_conversion_summary(result, input_file)
        
        if result.success:
            if not quiet:
                console.print(f"\n[green]🎉 Conversion completed in {format_duration(time.time() - start_time)}![/green]")
        else:
            console.print(f"\n[red]💥 Conversion failed after {format_duration(time.time() - start_time)}[/red]")
            raise typer.Exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Conversion interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error during conversion: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@voices_app.command("list")
def list_voices_command(
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Filter by TTS engine"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Filter by language"),
    quality: Optional[str] = typer.Option(None, "--quality", "-q", help="Filter by voice quality"),
    gender: Optional[str] = typer.Option(None, "--gender", "-g", help="Filter by gender"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Show voices from a collection"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed voice information"),
    available_only: bool = typer.Option(False, "--available", "-a", help="Show only available voices"),
) -> None:
    """
    List available TTS voices with optional filtering.
    
    This command displays all available voices with their characteristics.
    You can filter by various criteria to find the perfect voice for your needs.
    
    Examples:
    
        # List all voices
        ebook2audio voices list
        
        # Filter by engine
        ebook2audio voices list --engine GTTS
        
        # Filter by language and quality
        ebook2audio voices list --language en --quality HIGH
        
        # Show collection voices
        ebook2audio voices list --collection audiobook
        
        # Detailed information
        ebook2audio voices list --detailed
    """
    try:
        # Get voices based on filters
        if collection:
            # Show voices from specific collection
            if collection not in VOICE_COLLECTIONS:
                console.print(f"[red]Collection '{collection}' not found.[/red]")
                console.print(f"Available collections: {', '.join(VOICE_COLLECTIONS.keys())}")
                raise typer.Exit(1)
            
            collection_info = VOICE_COLLECTIONS[collection]
            voice_ids = collection_info.get("voices", [])
            voices = {vid: get_voice(vid).metadata for vid in voice_ids if get_voice(vid)}
            
            console.print(f"\n[bold]Collection: {collection_info['name']}[/bold]")
            console.print(f"[dim]{collection_info['description']}[/dim]\n")
        else:
            # Get all voices and apply filters
            voices = list_voices()
            
            # Apply filters
            if engine:
                try:
                    engine_enum = TTSEngine(engine.upper())
                    voices = {vid: v for vid, v in voices.items() if v.engine == engine_enum}
                except ValueError:
                    console.print(f"[red]Invalid engine: {engine}[/red]")
                    console.print(f"Valid engines: {', '.join([e.value for e in TTSEngine])}")
                    raise typer.Exit(1)
            
            if language:
                voices = {vid: v for vid, v in voices.items() if v.language == language}
            
            if quality:
                try:
                    quality_enum = VoiceQuality(quality.upper())
                    voices = {vid: v for vid, v in voices.items() if v.quality == quality_enum}
                except ValueError:
                    console.print(f"[red]Invalid quality: {quality}[/red]")
                    console.print(f"Valid qualities: {', '.join([q.value for q in VoiceQuality])}")
                    raise typer.Exit(1)
            
            if gender:
                voices = {vid: v for vid, v in voices.items() if v.gender == gender}
        
        if not voices:
            console.print("[yellow]No voices found matching the specified criteria.[/yellow]")
            return
        
        # Filter by availability if requested
        if available_only:
            available_voices = {}
            for vid, voice_meta in voices.items():
                try:
                    voice = get_voice(vid)
                    if voice and voice.is_available():
                        available_voices[vid] = voice_meta
                except:
                    pass
            voices = available_voices
        
        if not voices:
            console.print("[yellow]No available voices found matching the criteria.[/yellow]")
            return
        
        # Display voices
        if detailed:
            # Detailed view with individual panels
            for voice_id, voice_meta in sorted(voices.items()):
                status = "[green]✓[/green]"
                try:
                    voice = get_voice(voice_id)
                    if not voice or not voice.is_available():
                        status = "[red]✗[/red]"
                except:
                    status = "[red]✗[/red]"
                
                panel_content = f"""[bold]Description:[/bold] {voice_meta.description}
[bold]Engine:[/bold] {voice_meta.engine.value}
[bold]Language:[/bold] {voice_meta.language} ({voice_meta.accent})
[bold]Quality:[/bold] {voice_meta.quality.value}
[bold]Gender:[/bold] {voice_meta.gender}
[bold]Style:[/bold] {voice_meta.speaking_style}
[bold]Sample Rate:[/bold] {voice_meta.sample_rate} Hz
[bold]Available:[/bold] {status}"""
                
                console.print(Panel(panel_content, title=f"[bold]{voice_meta.name}[/bold] ({voice_id})", 
                                  border_style="green" if status == "[green]✓[/green]" else "red"))
                console.print()
        else:
            # Table view
            table = Table(title=f"Available Voices ({len(voices)} found)")
            table.add_column("Voice ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Engine", style="blue")
            table.add_column("Language", style="yellow")
            table.add_column("Quality", style="magenta")
            table.add_column("Gender", style="white")
            table.add_column("Available", style="green")
            
            for voice_id, voice_meta in sorted(voices.items()):
                status = "✓"
                status_style = "green"
                try:
                    voice = get_voice(voice_id)
                    if not voice or not voice.is_available():
                        status = "✗"
                        status_style = "red"
                except:
                    status = "✗"
                    status_style = "red"
                
                table.add_row(
                    voice_id,
                    voice_meta.name,
                    voice_meta.engine.value,
                    f"{voice_meta.language}-{voice_meta.accent}",
                    voice_meta.quality.value,
                    voice_meta.gender,
                    f"[{status_style}]{status}[/{status_style}]"
                )
            
            console.print(table)
        
        # Show summary
        total_voices = len(voices)
        available_count = 0
        for voice_id in voices.keys():
            try:
                voice = get_voice(voice_id)
                if voice and voice.is_available():
                    available_count += 1
            except:
                pass
        
        console.print(f"\n[bold]Summary:[/bold] {total_voices} voices found, {available_count} available")
        
    except Exception as e:
        console.print(f"[red]Error listing voices: {e}[/red]")
        raise typer.Exit(1)


@voices_app.command("collections")
def list_collections_command() -> None:
    """
    List available voice collections.
    
    Voice collections are curated groups of voices optimized for specific use cases
    like audiobooks, fast synthesis, high quality, etc.
    """
    console.print(Panel("[bold]Voice Collections[/bold]\n" +
                       "Collections are curated groups of voices for specific use cases.",
                       title="Voice Collections", border_style="blue"))
    
    table = Table(title="Available Collections")
    table.add_column("Collection", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    table.add_column("Voices", style="yellow")
    
    for collection_id, collection_info in VOICE_COLLECTIONS.items():
        voice_count = len(collection_info.get("voices", []))
        table.add_row(
            collection_id,
            collection_info.get("name", "Unknown"),
            collection_info.get("description", "No description"),
            str(voice_count)
        )
    
    console.print(table)
    console.print(f"\n[dim]Use 'ebook2audio voices list --collection <name>' to see voices in a collection.[/dim]")


@voices_app.command("test")
def test_voice_command(
    voice_id: str = typer.Argument(..., help="Voice ID to test"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Custom text to synthesize"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save test audio to file"),
) -> None:
    """
    Test a voice by generating a sample audio.
    
    This command generates a short audio sample using the specified voice,
    which can be used to preview how the voice sounds before conversion.
    
    Examples:
    
        # Test default voice
        ebook2audio voices test gtts_en_us
        
        # Test with custom text
        ebook2audio voices test gtts_en_uk --text "Hello, this is a test."
        
        # Save test audio to file
        ebook2audio voices test system_en_female --output test.wav
    """
    # Validate voice
    voice_id = validate_voice_id(voice_id)
    
    # Default test text
    if not text:
        text = "Hello! This is a test of the text-to-speech voice. The quick brown fox jumps over the lazy dog."
    
    console.print(f"[blue]Testing voice: {voice_id}[/blue]")
    console.print(f"[dim]Text: {text}[/dim]\n")
    
    try:
        # Get voice
        voice = get_voice(voice_id)
        if not voice.is_available():
            console.print(f"[red]Voice '{voice_id}' is not available.[/red]")
            raise typer.Exit(1)
        
        # Generate output path if not provided
        if output_file is None:
            output_file = Path(f"voice_test_{voice_id}.wav")
        
        # Synthesize test audio
        console.print(f"[yellow]Generating test audio...[/yellow]")
        result_path = voice.synthesize(text, output_file)
        
        if result_path and result_path.exists():
            console.print(f"[green]✓ Test audio generated successfully![/green]")
            console.print(f"[bold]File:[/bold] {result_path.absolute()}")
            console.print(f"[bold]Size:[/bold] {format_file_size(result_path.stat().st_size)}")
            
            # Try to get duration
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(str(result_path))
                duration = len(audio) / 1000.0
                console.print(f"[bold]Duration:[/bold] {format_duration(duration)}")
            except:
                pass
        else:
            console.print(f"[red]Failed to generate test audio[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error testing voice: {e}[/red]")
        raise typer.Exit(1)


@app.command("info")
def info_command() -> None:
    """
    Show information about ebook2audio and available TTS engines.
    """
    from . import __version__
    
    console.print(Panel(f"""[bold]Ebook2Audio v{__version__}[/bold]

Convert ebooks to audiobooks using various Text-to-Speech engines.

[bold]Supported Input Formats:[/bold]
• PDF (Portable Document Format)
• EPUB (Electronic Publication)
• MOBI (Mobipocket)
• AZW/AZW3 (Amazon Kindle)
• TXT (Plain Text)

[bold]Supported Output Formats:[/bold]
• MP3 (MPEG Audio Layer 3)
• WAV (Waveform Audio)
• FLAC (Free Lossless Audio Codec)
• M4A (MPEG-4 Audio)

[bold]Available TTS Engines:[/bold]
• Google TTS (gTTS) - High quality, requires internet
• System TTS (pyttsx3) - Offline, uses system voices
• XTTS-v2 - Advanced neural TTS with voice cloning
• Bark - Expressive TTS with emotional control
• OpenVoice - Versatile voice synthesis
• Tortoise TTS - High-quality audiobook synthesis""", 
                        title="About Ebook2Audio", border_style="blue"))
    
    # Show system info
    voice_manager = get_voice_manager()
    total_voices = len(list_voices())
    available_voices = 0
    
    for voice_id in list_voices().keys():
        try:
            voice = get_voice(voice_id)
            if voice and voice.is_available():
                available_voices += 1
        except:
            pass
    
    console.print(f"\n[bold]System Status:[/bold]")
    console.print(f"• Total voices: {total_voices}")
    console.print(f"• Available voices: {available_voices}")
    console.print(f"• Voice collections: {len(VOICE_COLLECTIONS)}")


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", help="Show version and exit"),
) -> None:
    """
    Ebook2Audio - Convert ebooks to audiobooks using Text-to-Speech.
    
    Transform your ebooks into audiobooks using various TTS engines with rich progress
    tracking, voice selection, and high-quality audio output.
    """
    if version:
        from . import __version__
        console.print(f"ebook2audio {__version__}")
        raise typer.Exit()


def cli_main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()