# Ebook-to-Audiobook File Structure Reference

## Project Overview
This document provides a comprehensive reference to the file structure and associations within the ebook-to-audiobook project.

## Directory Structure

### Root Directory (`/home/tim/Projects/ebook-to-audiobook/`)

```
.
├── src/                          # Main source code directory
├── GPT-SoVITS/                  # GPT-SoVITS voice cloning system (submodule)
├── raw-training-data/           # Raw audio data for voice training
├── audiobook_output/            # Generated audiobooks and temp files
├── lky_audiobook_inference/     # LKY-specific inference setup
├── models/                      # Trained voice models
├── tests/                       # Test suite
├── docs/                        # Documentation
├── venv/                        # Python virtual environment
└── Various Python scripts and configs
```

## Main Entry Points

| File | Purpose |
|------|---------|
| `convert_full_book.py` | Main script for converting full books to audiobooks |
| `train_lky_voice.py` | Script for training LKY voice model |
| `train_lky_enhanced.py` | Enhanced training script with 30 samples |
| `train_s2_enhanced.py` | Stage 2 training script |
| `extract_lky_segments.py` | Extract audio segments for training |
| `test_enhanced_inference.py` | Test enhanced model inference |
| `test_s1_model.py` | Test Stage 1 model |
| `setup_gpt_sovits.py` | Setup script for GPT-SoVITS |

## Source Code Structure (`src/ebook2audio/`)

### Core Modules
- `__init__.py` - Package initialization
- `cli.py` - Command-line interface
- `config.py` - Configuration management
- `pipeline.py` - Main AudioBookPipeline class
- `utils.py` - Utility functions

### Extract Module (`src/ebook2audio/extract/`)
Text extraction from various formats:
- `pdf.py` - PDF extraction with OCR support
- `epub.py` - EPUB format extraction
- `mobi.py` - MOBI format extraction
- `txt.py` - Plain text extraction
- `ocr_parallel.py` - Parallel OCR processing
- `ocr_cache.py` - OCR caching system
- `progress.py` - Progress tracking

### Voices Module (`src/ebook2audio/voices/`)
TTS voice provider system:
- `base.py` - Base voice provider interface
- `manager.py` - Voice management system
- `catalog.py` - Voice catalog
- `builtin.py` - Built-in voices
- `custom.py` - Custom voice support
- `gtts_provider.py` - Google TTS provider
- `pyttsx3_provider.py` - System TTS provider
- `gpt_sovits_provider.py` - GPT-SoVITS provider (KEY FILE)
- `custom_trained_provider.py` - Custom trained voice provider

### Training Module (`src/ebook2audio/training/`)
Training-related functionality (to be expanded)

## GPT-SoVITS Structure (`GPT-SoVITS/`)

### Main Module (`GPT-SoVITS/GPT_SoVITS/`)
- `AR/` - Autoregressive model components
- `BigVGAN/` - BigVGAN vocoder
- `TTS_infer_pack/` - TTS inference package
- `configs/` - Configuration files
  - `lky_*.yaml` - LKY-specific configs
  - `s1*.yaml` - Stage 1 configs
  - `s2*.json` - Stage 2 configs
- `feature_extractor/` - Feature extraction
- `module/` - Core modules
- `prepare_datasets/` - Dataset preparation
- `pretrained_models/` - Pre-trained models
  - `chinese-hubert-base/`
  - `chinese-roberta-wwm-ext-large/`
  - `gsv-v2final-pretrained/`
- `text/` - Text processing

### Training Data (`GPT-SoVITS/output/`)
- `lky_training_data/` - Initial training data
- `lky_training_data_enhanced/` - Enhanced training data
  - `audio_segments/` - 30 processed segments
  - `lky_training_list.txt` - Training file list

### Training Logs (`GPT-SoVITS/logs/`)
- `lky_en_cli/` - Initial training logs
- `lky_en_enhanced/` - Enhanced training logs
  - `ckpt/` - Model checkpoints
  - `lky_en_enhanced-e*.ckpt` - Epoch checkpoints

## Configuration Files

### Project Configuration
- `pyproject.toml` - Poetry project configuration
- `poetry.lock` - Poetry dependency lock file
- `voices.json` - Voice configuration database

### GPT-SoVITS Configuration
- Various YAML and JSON files in `GPT-SoVITS/configs/`
- LKY-specific configurations
- Stage-specific training configurations

## Key File Associations

### Bug 1: GPTSoVITS Integration
Primary files:
- `src/ebook2audio/voices/gpt_sovits_provider.py`
- `src/ebook2audio/voices/custom_trained_provider.py`
- `train_lky_voice.py`
- `train_lky_enhanced.py`
- `GPT-SoVITS/GPT_SoVITS/inference.py`
- `GPT-SoVITS/GPT_SoVITS/TTS_infer_pack/`

### Bug 2: Repository Organization
Affected areas:
- Path references in training scripts
- Configuration file locations
- Model file locations
- Documentation scattered in root

### Bug 3: Memory Issues
Related files:
- `train_s2_enhanced.py`
- `GPT-SoVITS/GPT_SoVITS/configs/s2*.json`
- Training scripts in `GPT-SoVITS/`

## Model Storage

### Trained Models
- `GPT-SoVITS/logs/lky_en_enhanced/ckpt/` - Checkpoints
- `models/` - Final models (to be organized)
- `lky_audiobook_inference/` - Inference setup

### Pre-trained Models
- `GPT-SoVITS/pretrained_models/` - Base models for fine-tuning

## Documentation Files
All documentation has been moved to `docs/`:
- `bug-documentation.md` - Current bug tracking
- `file-structure-reference.md` - This file
- Various project status and planning documents