# 📚➡️🎧 Ebook to Audiobook Converter

A high-performance Python application that converts ebooks (PDF, EPUB, etc.) into audiobooks with AI-powered text-to-speech narration. Features intelligent optimization for speed, reliability, and quality.

## ✨ Key Features

### 🚀 **Performance Optimizations**
- **4-8x Faster Processing** with parallel TTS synthesis
- **Resume Capability** - automatically continues interrupted conversions
- **Memory-Efficient** streaming for large books (1000+ pages)
- **Smart Rate Limiting** with exponential backoff and retry logic

### 🎯 **Intelligent Processing**
- **Automatic Provider Fallback** - switches from online to offline TTS when rate limited
- **OCR Support** for scanned PDFs with parallel processing
- **Smart Text Chunking** - sentence-aware splitting for natural speech
- **Progress Tracking** with detailed conversion metrics

### 🎵 **Audio Quality**
- **Multiple TTS Engines** - Google TTS (gTTS) and System TTS (pyttsx3)
- **Configurable Audio Settings** - bitrate, sample rate, format options
- **Automatic Normalization** and silence management between chunks
- **Professional Output** in MP3 format optimized for audiobooks

## 🛠️ Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install system dependencies (macOS)
brew install ffmpeg

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Install system dependencies (Windows)
# Download ffmpeg from https://ffmpeg.org/download.html
```

### Python Dependencies
```bash
# Clone the repository
git clone https://github.com/JKL999/ebook-to-audiobook.git
cd ebook-to-audiobook

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage
```bash
# Convert a PDF to audiobook with optimizations
python example_optimized_usage.py your-book.pdf

# Specify output file
python example_optimized_usage.py your-book.pdf -o my-audiobook.mp3

# Resume an interrupted conversion
python example_optimized_usage.py --resume
```

### Advanced Usage
```python
from ebook2audio.pipeline import AudioBookPipeline, ConversionConfig

# Create optimized configuration
config = ConversionConfig(
    # Performance settings
    parallel_synthesis=True,        # Enable parallel processing
    max_workers=4,                 # Concurrent TTS workers
    
    # Quality settings
    voice_id="gtts_en_us",         # Primary voice
    output_format="mp3",
    sample_rate=22050,
    bitrate="128k",
    
    # Resume capability
    temp_dir=Path("audiobook_output/temp")
)

# Convert with automatic optimizations
pipeline = AudioBookPipeline(config)
result = pipeline.convert("book.pdf", "audiobook.mp3")

if result.success:
    print(f"✅ Conversion completed in {result.processing_time/60:.1f} minutes")
    print(f"🎵 Audio duration: {result.duration/3600:.1f} hours")
    print(f"📊 Success rate: {result.success_rate:.1f}%")
```

## 📋 Usage Examples

### Test with Sample Pages
```bash
# Test with first 10 pages
python convert_full_book.py small

# Test with first 50 pages  
python convert_full_book.py test

# Full book conversion
python convert_full_book.py full
```

### Resume Interrupted Conversion
```bash
# The system automatically detects existing chunks and continues
python example_optimized_usage.py your-book.pdf
# Will skip existing chunks and continue where it left off
```

## ⚙️ Configuration Options

### TTS Settings
```python
ConversionConfig(
    voice_id="gtts_en_us",          # Primary voice (falls back to pyttsx3)
    speaking_rate=1.0,              # Speech speed multiplier
    volume=0.9,                     # Audio volume (0.0-1.0)
)
```

### Performance Settings
```python
ConversionConfig(
    parallel_synthesis=True,         # Enable concurrent processing
    max_workers=4,                  # Number of worker threads
    max_chunk_size=800,             # Characters per TTS chunk
    chunk_overlap=30,               # Overlap for natural flow
)
```

### Audio Quality
```python
ConversionConfig(
    output_format="mp3",            # mp3, wav, m4a
    sample_rate=22050,              # Audio sample rate
    bitrate="128k",                 # MP3 bitrate
    normalize_audio=True,           # Automatic volume normalization
    add_silence_between_chunks=0.3, # Seconds of silence between chunks
)
```

## 🔧 Supported Features

### Input Formats
- ✅ **PDF** (text-based and scanned with OCR)
- ✅ **EPUB** (planned)
- ✅ **TXT** (plain text)

### Output Formats
- ✅ **MP3** (recommended for audiobooks)
- ✅ **WAV** (uncompressed)
- ✅ **M4A** (Apple format)

### TTS Engines
- ✅ **Google TTS (gTTS)** - High quality, requires internet
- ✅ **System TTS (pyttsx3)** - Offline, always available
- ✅ **Automatic Fallback** - Switches when rate limited

## 📊 Performance Benchmarks

### Speed Improvements
- **Sequential Processing**: ~1 chunk/second
- **Parallel Processing**: ~4-8 chunks/second (4-8x speedup)
- **Resume Capability**: Skip existing chunks instantly
- **Memory Usage**: <500MB even for 1000+ page books

### Real-World Example
```
Book: "From Third World to First" (470 pages, ~1,879 chunks)
- Original: ~31 minutes processing time
- Optimized: ~8 minutes processing time (4x faster)
- With resume: <1 minute to continue existing conversion
```

## 🛡️ Error Handling & Reliability

### Automatic Rate Limit Handling
- **Detection**: Identifies 429 errors, quota exceeded messages
- **Retry Logic**: Exponential backoff (1s → 2s → 4s → 8s → 16s)
- **Fallback**: Automatic switch to offline TTS when limits hit
- **Recovery**: Seamless continuation without data loss

### Resume & Recovery
- **Checkpoint System**: Automatically saves progress after each chunk
- **Smart Detection**: Scans for existing audio chunks on restart
- **Validation**: Ensures chunks are complete and valid before skipping
- **Fault Tolerance**: Handles interrupted conversions gracefully

## 📁 Output Structure

```
audiobook_output/
├── your-book_audiobook.mp3      # Final audiobook
├── temp/                        # Temporary chunks (for resume)
│   ├── chunk_0000.mp3
│   ├── chunk_0001.mp3
│   └── ...
├── conversion.log               # Detailed processing log
└── cache/                       # OCR cache (for PDFs)
    └── pdf_cache.db
```

## 🐛 Troubleshooting

### Common Issues

**Rate Limited by Google TTS**
```bash
# The system automatically handles this, but you can also:
# 1. Wait 24 hours for limits to reset
# 2. Use offline mode: voice_id="pyttsx3_default"
# 3. Resume existing conversion (skips completed chunks)
```

**OCR Not Working**
```bash
# Install OCR dependencies
pip install ocrmypdf
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu
```

**Memory Issues with Large Books**
```bash
# The system automatically uses streaming for 100+ chunks
# But you can force it:
config = ConversionConfig(streaming_threshold=50)
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Enables detailed logging for troubleshooting
```

## 📈 Recent Updates

### v2.0.0 - Performance & Reliability Overhaul
- ✅ **4-8x Speed Improvement** with parallel TTS synthesis
- ✅ **Resume Capability** for interrupted conversions  
- ✅ **Smart Rate Limiting** with exponential backoff
- ✅ **Automatic Fallback** from online to offline TTS
- ✅ **Memory Optimization** for large audiobooks
- ✅ **Enhanced OCR** with parallel processing and caching

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/JKL999/ebook-to-audiobook.git
cd ebook-to-audiobook
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Tests
```bash
# Test optimizations
python demo_optimizations.py

# Test with sample book
python test_optimizations.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

- 📧 **Issues**: [GitHub Issues](https://github.com/JKL999/ebook-to-audiobook/issues)
- 📖 **Documentation**: This README and inline code documentation
- 💡 **Feature Requests**: Open an issue with the "enhancement" label

## 🔮 Future Vision: Custom Voice Cloning

The next major evolution of this project will integrate **state-of-the-art voice cloning technology** to eliminate API rate limits and provide unlimited, personalized audiobook generation.

### **Coming in Phase 3: Voice Cloning Integration**
- **🎯 Train Custom Voices**: Create personalized narrators from 1-minute audio samples
- **⚡ Local Processing**: Eliminate API rate limits with local GPU acceleration  
- **🎨 Professional Quality**: 80-95% voice similarity using GPT-SoVITS/XTTS v2
- **💰 Cost Efficient**: No ongoing API costs, unlimited audiobook generation
- **🔒 Privacy First**: All processing stays local on your hardware

**Hardware Requirements:**
- **Training**: RTX 3070+ or similar (8GB+ VRAM)
- **Inference**: M3 MacBook Air or CPU-optimized models

**See [ROADMAP.md](ROADMAP.md) for detailed implementation plan and technology research.**

---

**🎯 Ready to convert your ebook library to audiobooks?**
```bash
python example_optimized_usage.py your-favorite-book.pdf
```