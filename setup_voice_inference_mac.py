#!/usr/bin/env python3
"""
Setup voice inference on M3 MacBook Air.

This script installs and tests lightweight TTS models optimized for 
CPU/Apple Silicon inference while the desktop handles training.
"""

import subprocess
import sys
from pathlib import Path

def setup_voice_inference():
    """Set up voice inference environment for Mac."""
    
    print("🎤 Setting up Voice Inference on M3 MacBook Air")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    print("\n📦 Installing voice inference libraries...")
    
    # Install MeloTTS (CPU-optimized, most popular on HuggingFace)
    packages = [
        "melo-tts",  # Lightweight, CPU-optimized
        "TTS",       # For XTTS v2 comparison
        "torch",     # PyTorch for Mac
        "torchaudio",
        "soundfile",
        "librosa"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
            print("You may need to install it manually")
    
    print("\n✅ Voice inference setup complete!")
    print("\nNext steps:")
    print("1. Run test_voice_inference.py to validate setup")
    print("2. Wait for desktop agent to complete training")
    print("3. Test inference with custom trained models")

if __name__ == "__main__":
    setup_voice_inference()