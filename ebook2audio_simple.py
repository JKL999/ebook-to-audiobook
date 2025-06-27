#!/usr/bin/env python3
"""
Simple ebook2audio CLI - MVP version.

A minimal command-line interface for converting ebooks to audiobooks.
"""

import sys
from pathlib import Path
import typer
from rich.console import Console

# Initialize
app = typer.Typer()
console = Console()

@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input ebook file"),
    output_file: Path = typer.Argument(..., help="Output audio file"),
    voice: str = typer.Option("en", "--voice", "-v", help="Voice/language to use"),
):
    """Convert an ebook to audiobook using gTTS."""
    
    # Check input file exists
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Converting {input_file} to {output_file}...[/green]")
    
    try:
        # Import here to avoid long startup time
        from gtts import gTTS
        import fitz  # PyMuPDF
        
        # Extract text based on file type
        text = ""
        if input_file.suffix.lower() == '.txt':
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif input_file.suffix.lower() == '.pdf':
            doc = fitz.open(str(input_file))
            for page in doc:
                text += page.get_text()
            doc.close()
        else:
            console.print(f"[red]Error: Unsupported file type: {input_file.suffix}[/red]")
            raise typer.Exit(1)
        
        if not text.strip():
            console.print("[red]Error: No text extracted from file[/red]")
            raise typer.Exit(1)
        
        console.print(f"Extracted {len(text)} characters")
        
        # Convert to speech
        tts = gTTS(text=text, lang=voice)
        tts.save(str(output_file))
        
        console.print(f"[green]✓ Audio saved to: {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def list_voices():
    """List available voice/language codes."""
    console.print("[bold]Available voices (language codes):[/bold]")
    console.print("• en - English")
    console.print("• es - Spanish") 
    console.print("• fr - French")
    console.print("• de - German")
    console.print("• it - Italian")
    console.print("• pt - Portuguese")
    console.print("• ru - Russian")
    console.print("• ja - Japanese")
    console.print("• ko - Korean")
    console.print("• zh - Chinese")
    console.print("\nFor more languages, see: https://gtts.readthedocs.io/")

if __name__ == "__main__":
    app()