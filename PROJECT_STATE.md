# Ebook2Audio Project State

## Project Overview
Modernizing a basic PDF text extraction script into a production-ready ebook-to-audiobook converter with AI-powered voice cloning capabilities.

### Original State
- Basic Python script using PyPDF2==1.28.6
- Only extracts text from hardcoded 'example.pdf'
- No TTS, audio conversion, or user interface

### Target State
- Modern Python package with Poetry/pyproject.toml
- Multi-format ebook support (PDF, EPUB, MOBI)
- Voice cloning from sample audio (6-second samples)
- Professional CLI with Typer and rich progress bars
- Multiple TTS engines (XTTS-v2, Bark, OpenVoice, Tortoise)
- Parallel processing and resume capability

## Technology Stack (per AI Agent Blueprint)

| Component | Technology | Version | Why Chosen |
|-----------|------------|---------|------------|
| **PDF Processing** | PyMuPDF (pymupdf) | ≥1.24 | 10x faster than PyPDF2, better layout preservation |
| **EPUB Support** | ebooklib | latest | Standard EPUB processing |
| **MOBI Support** | Calibre CLI | latest | ebook-convert for MOBI/AZW3 |
| **OCR** | ocrmypdf | 16+ | Tesseract wrapper for scanned PDFs |
| **TTS Engine 1** | coqui-TTS (XTTS-v2) | 0.22 | 6-second voice cloning, <150ms latency |
| **TTS Engine 2** | Bark-with-Voice-Clone | latest | 5-12s samples, expressive, multilingual |
| **TTS Engine 3** | MyShell OpenVoice v1 | 0.1.4 | Tone/accent control, emotional control |
| **TTS Engine 4** | Tortoise-TTS | latest | Highest audiobook quality (fallback) |
| **Audio Processing** | pydub + FFmpeg | 0.27 + 6 | Audio manipulation, format conversion |
| **CLI Framework** | Typer | 0.12 | Type-hinted, Click-powered CLI |
| **Progress/Logging** | rich | 13 | Beautiful progress bars and logging |
| **Packaging** | Poetry | 1.8 | Modern Python dependency management |
| **Testing** | pytest + ruff | latest | Testing and linting |

## Target Architecture
```
ebook2audio/
├─ pyproject.toml           # Poetry + tool configs
├─ src/ebook2audio/
│   ├─ __init__.py
│   ├─ cli.py               # Typer app entry-point
│   ├─ extract/
│   │   ├─ pdf.py           # PyMuPDF + ocrmypdf
│   │   ├─ epub.py          # ebooklib
│   │   └─ mobi.py          # calibre wrapper
│   ├─ voices/
│   │   ├─ base.py          # common interface
│   │   ├─ xtts.py
│   │   ├─ bark.py
│   │   └─ openvoice.py
│   ├─ training/
│   │   └─ preprocess.py    # FFmpeg/SoX helpers
│   ├─ pipeline.py          # Orchestrates extraction→TTS→merge
│   └─ utils.py             # logging, chunking, tqdm hooks
└─ tests/
    └─ …
```

## Agent Assignments

### Agent 1: Project Structure (feature/project-structure)
- **Status**: Pending
- **Subtree Branch**: feature/project-structure
- **Tasks**:
  - Create pyproject.toml with Poetry configuration
  - Set up src/ebook2audio/ directory structure
  - Initialize pytest testing framework
  - Create basic __init__.py files
- **Dependencies**: None (foundational)

### Agent 2: Parsing Layer (feature/parsing)
- **Status**: Pending
- **Subtree Branch**: feature/parsing
- **Tasks**:
  - Replace PyPDF2 with PyMuPDF
  - Implement EPUB support via ebooklib
  - Add MOBI support via Calibre CLI wrapper
  - Create OCR fallback with ocrmypdf
- **Dependencies**: Project structure must be complete

### Agent 3: CLI Interface (feature/cli)
- **Status**: Pending
- **Subtree Branch**: feature/cli
- **Tasks**:
  - Implement Typer-based CLI with commands: convert, clone, train, batch
  - Add rich progress bars and logging
  - Create comprehensive CLI options and help
- **Dependencies**: Project structure

### Agent 4: Voice Management (feature/voice-management)
- **Status**: Pending
- **Subtree Branch**: feature/voice-management
- **Tasks**:
  - Create voice catalog system (voices.json)
  - Implement voice registration and management utilities
  - Support built-in and custom voice handling
- **Dependencies**: Project structure

### Agent 5: TTS Engines (feature/tts-engines)
- **Status**: Pending
- **Subtree Branch**: feature/tts-engines
- **Tasks**:
  - Implement XTTS-v2 integration
  - Add Bark-with-Voice-Clone support
  - Include OpenVoice integration
  - Add Tortoise-TTS (optional)
  - Create common TTS interface
- **Dependencies**: Project structure, voice management

### Agent 6: Voice Training (feature/voice-training)
- **Status**: Pending
- **Subtree Branch**: feature/voice-training
- **Tasks**:
  - Audio preprocessing pipeline (16kHz mono, silence trimming)
  - Model-specific training workflows
  - Voice cloning from sample audio files
- **Dependencies**: Project structure, TTS engines

### Agent 7: Audio Pipeline (feature/audio-pipeline)
- **Status**: Pending
- **Subtree Branch**: feature/audio-pipeline
- **Tasks**:
  - Text chunking and sentence segmentation
  - Parallel synthesis with asyncio
  - Audio post-processing with pydub/FFmpeg
  - Multiple output format support (MP3, M4B, WAV)
- **Dependencies**: Parsing layer, TTS engines

## Key CLI Commands (Target)
```bash
ebook2audio convert input.pdf --voice built-in/en_us_vctk_16
ebook2audio clone train ./samples/*.wav --name author
ebook2audio convert example.pdf --voice author --out /audiobooks/example
ebook2audio batch ./library/**/*.pdf --voice default --jobs 4
```

## Voice Cloning Specifications

| Model | Min Sample | Training Notes | Quality |
|-------|------------|----------------|---------|
| XTTS-v2 | ≈6 seconds | No training, inference-time embedding | Good timbre/accent |
| Bark w/ voice-clone | 5-12s | History prompt trick, optional RVC | Very expressive |
| OpenVoice | 10-20s | Style/emotion tokens | Best accent/emotion control |
| Tortoise | ≥2 min | Full finetune (24h on 3090) | Highest audiobook quality |

## Current Status
- **Phase**: Initial setup
- **Active Work**: Creating central state tracking
- **Next**: Initialize git subtrees and launch parallel agents

## Known Issues
- None yet (project just starting)

## Next Steps
1. Complete state tracking setup
2. Initialize git subtree branches
3. Launch Agent 1 (Project Structure) - foundational
4. Launch remaining agents in parallel once structure is ready
5. Coordinate integration and testing

## Integration Notes
- All agents must update this document and AGENT_PROGRESS.json with progress
- Use git subtree for parallel development
- Final integration will merge all feature branches
- Comprehensive testing before v1.0.0 release

---
Last Updated: 2025-01-XX by Main Coordinator