# 📚 Ebook to Audiobook Converter

A Python application that converts ebooks into audiobooks with AI-powered text-to-speech. Features parallel TTS synthesis, voice cloning via GPT-SoVITS, and intelligent processing for books of any size.

## Key Features

- **4-8x Faster Processing** with parallel TTS synthesis
- **Resume Capability** — automatically continues interrupted conversions
- **Voice Cloning** — GPT-SoVITS integration for custom narrator voices
- **Multiple TTS Engines** — Google TTS, System TTS (pyttsx3), with automatic fallback
- **OCR Support** — parallel OCR for scanned PDFs with caching
- **Memory-Efficient** — streaming concatenation for large books (1000+ pages)
- **Smart Rate Limiting** — exponential backoff and automatic provider switching

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Package Manager | Poetry |
| Text Extraction | PyMuPDF (fitz), ocrmypdf, ebooklib |
| TTS Engines | gTTS, pyttsx3, GPT-SoVITS |
| Voice Cloning | GPT-SoVITS + Chinese HuBERT + RoBERTa |
| Audio Processing | pydub, ffmpeg, librosa, torchaudio |
| CLI Framework | Typer + Rich |
| ML Runtime | PyTorch (CUDA / Apple MPS / CPU) |

## Installation

```bash
# Prerequisites: Python 3.10+, ffmpeg
brew install ffmpeg       # macOS
# or: sudo apt-get install ffmpeg  # Ubuntu/Debian

# Clone and install
git clone https://github.com/JKL999/ebook-to-audiobook.git
cd ebook-to-audiobook

# Install with Poetry
pip install poetry
poetry install
```

## Quick Start

### CLI Usage

```bash
# Convert a PDF to audiobook
poetry run ebook2audio convert your-book.pdf -o audiobook.mp3

# List available voices
poetry run ebook2audio voices list

# Test a voice
poetry run ebook2audio voices test gtts_en_us
```

### Convert Script (with test modes)

```bash
# Test with first 10 pages
python convert_full_book.py small

# Test with first 50 pages
python convert_full_book.py test

# Full book conversion
python convert_full_book.py full
```

### Python API

```python
from ebook2audio.pipeline import AudioBookPipeline, ConversionConfig

config = ConversionConfig(
    parallel_synthesis=True,
    max_workers=4,
    voice_id="gtts_en_us",
    output_format="mp3",
    sample_rate=22050,
    bitrate="128k",
    temp_dir=Path("audiobook_output/temp"),
)

pipeline = AudioBookPipeline(config)
result = pipeline.convert("book.pdf", "audiobook.mp3")

if result.success:
    print(f"Completed in {result.processing_time/60:.1f} min")
    print(f"Duration: {result.duration/3600:.1f} hours")
```

## Architecture

```
src/ebook2audio/
├── pipeline.py          # Central orchestrator (extraction → chunking → TTS → concat)
├── cli.py               # Typer CLI with Rich progress bars
├── config.py             # User configuration manager
├── utils.py              # Progress tracking, checkpoints, batch processing
├── extract/
│   ├── pdf.py            # PyMuPDF + OCR fallback
│   ├── epub.py           # EPUB support (ebooklib)
│   ├── mobi.py           # MOBI/AZW (Calibre wrapper)
│   ├── txt.py            # Plain text
│   ├── ocr_parallel.py   # Parallel OCR processing
│   └── ocr_cache.py      # OCR result caching
└── voices/
    ├── base.py            # Voice abstractions + TTSEngine enum
    ├── manager.py         # Voice orchestration + provider registry
    ├── catalog.py         # JSON-based voice catalog
    ├── gtts_provider.py   # Google TTS (with rate limiting + retry)
    ├── pyttsx3_provider.py # System TTS (offline fallback)
    └── gpt_sovits_provider.py  # Voice cloning inference
```

## Supported Formats

### Input
- **PDF** — text-based and scanned (with OCR)
- **EPUB** — via ebooklib
- **MOBI/AZW** — via Calibre
- **TXT** — plain text

### Output
- **MP3** (recommended)
- **WAV** (uncompressed)
- **M4A** (Apple format)

## Voice Cloning (GPT-SoVITS)

The voice cloning pipeline trains a custom narrator voice from audio samples:

```bash
# 1. Extract training segments from raw audio
python extract_lky_segments.py

# 2. Download pretrained models
python setup_gpt_sovits.py

# 3. Train the voice model
python train_lky_enhanced.py

# 4. Convert with the trained voice
python convert_full_book.py test
```

**Hardware Requirements:**
- Training: GPU with 8GB+ VRAM (RTX 3070+)
- Inference: CPU or Apple Silicon (M1+)

## Performance

```
Book: 470 pages, ~1,879 chunks
Sequential:  ~31 minutes
Parallel:    ~8 minutes (4x faster)
With resume: <1 minute to continue
Memory:      <500MB for 1000+ page books
```

## Error Handling

- **Rate Limiting**: Exponential backoff (1s → 2s → 4s → 8s → 16s) with automatic fallback to offline TTS
- **Checkpoints**: Progress saved after each chunk — resume interrupted conversions automatically
- **Validation**: Chunks verified before skipping on resume

## Running Tests

```bash
poetry run pytest tests/
```

## License

MIT

## Author

Tim Zhang ([JKL999](https://github.com/JKL999))
