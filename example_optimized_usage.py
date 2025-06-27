#!/usr/bin/env python3
"""
Example usage of the optimized audio generation pipeline.

This demonstrates how to use the new optimization features for efficient
ebook to audiobook conversion.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ebook2audio.pipeline import AudioBookPipeline, ConversionConfig

def convert_with_optimizations(input_pdf, output_path=None):
    """
    Convert an ebook to audiobook using all optimizations.
    
    Args:
        input_pdf: Path to the PDF file to convert
        output_path: Optional output path for the audiobook
    """
    # Create optimized configuration
    config = ConversionConfig(
        # TTS settings
        voice_id="gtts_en_us",  # Primary voice (will fallback to pyttsx3 if needed)
        
        # Optimization settings
        parallel_synthesis=True,    # Enable parallel processing
        max_workers=4,             # Use 4 concurrent workers (adjust based on your CPU)
        
        # Audio quality settings
        output_format="mp3",
        sample_rate=22050,
        bitrate="128k",
        
        # Processing settings
        add_silence_between_chunks=0.3,  # Shorter silence for faster processing
        normalize_audio=True,
        
        # Resume capability (will automatically detect existing chunks)
        temp_dir=Path("audiobook_output/temp"),  # Persistent temp directory
    )
    
    # Create pipeline with optimized configuration
    pipeline = AudioBookPipeline(config)
    
    # Convert the ebook
    print("Converting {} with optimizations...".format(input_pdf))
    print("Features enabled:")
    print("✓ Resume capability (skips existing chunks)")
    print("✓ Parallel synthesis (4x faster)")
    print("✓ Rate limiting with retry logic")
    print("✓ Automatic fallback to offline TTS")
    print("✓ Memory-efficient concatenation")
    print()
    
    result = pipeline.convert(input_pdf, output_path)
    
    if result.success:
        print("🎉 Conversion completed successfully!")
        print("📁 Output: {}".format(result.output_path))
        print("⏱️  Processing time: {:.1f} minutes".format(result.processing_time/60))
        print("🎵 Audio duration: {:.1f} hours".format(result.duration/3600))
        print("📊 Chunks processed: {}/{}".format(result.processed_chunks, result.total_chunks))
        print("✅ Success rate: {:.1f}%".format(result.success_rate))
    else:
        print("❌ Conversion failed!")
        print("Error: {}".format(result.error_message))

def resume_existing_conversion():
    """
    Resume an existing conversion that was interrupted.
    
    This will automatically detect existing chunks and continue where it left off.
    """
    print("Resuming existing conversion...")
    
    # Use the same temp directory and configuration
    config = ConversionConfig(
        parallel_synthesis=True,
        max_workers=6,  # Increase workers for faster completion
        temp_dir=Path("audiobook_output/temp"),
    )
    
    pipeline = AudioBookPipeline(config)
    
    # The pipeline will automatically detect and skip existing chunks
    result = pipeline.convert(
        "from-third-world-to-first-by-lee-kuan-yew.pdf",
        "audiobook_output/from-third-world-to-first-RESUMED.mp3"
    )
    
    return result

def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized ebook to audiobook conversion")
    parser.add_argument("input", nargs="?", help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output audiobook file")
    parser.add_argument("--resume", action="store_true", help="Resume existing conversion")
    parser.add_argument("--test", action="store_true", help="Run test with 2 pages")
    
    args = parser.parse_args()
    
    if args.resume:
        print("Resuming existing conversion with optimizations...")
        result = resume_existing_conversion()
    elif args.test:
        print("Running test conversion with 2 pages...")
        convert_with_optimizations("audiobook_output/test_2_pages.pdf", "audiobook_output/test_optimized.mp3")
    elif args.input:
        convert_with_optimizations(args.input, args.output)
    else:
        print("Usage examples:")
        print(f"  {sys.argv[0]} book.pdf                    # Convert with optimizations")
        print(f"  {sys.argv[0]} book.pdf -o audiobook.mp3  # Specify output")
        print(f"  {sys.argv[0]} --resume                    # Resume interrupted conversion")
        print(f"  {sys.argv[0]} --test                      # Test with 2 pages")

if __name__ == "__main__":
    main()