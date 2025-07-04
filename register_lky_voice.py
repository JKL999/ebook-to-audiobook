#!/usr/bin/env python3
"""
Register the trained LKY voice model in the voice catalog.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ebook2audio.voices.catalog import VoiceCatalog
from ebook2audio.voices.base import VoiceMetadata, VoiceType, TTSEngine, VoiceQuality


def register_lky_voice():
    """Register the LKY voice in the catalog."""
    
    # Create voice metadata for LKY
    lky_voice = VoiceMetadata(
        voice_id="lky_en_trained",
        name="Lee Kuan Yew (Trained)",
        description="High-quality voice clone of Lee Kuan Yew trained on his audiobook samples",
        voice_type=VoiceType.TRAINED,
        engine=TTSEngine.GPT_SOVITS,
        language="en",
        accent="Singaporean English",
        quality=VoiceQuality.VERY_HIGH,
        sample_rate=32000,
        quality_score=9.5,
        model_path=Path("GPT_SoVITS/logs/lky_en_cli"),
        created_at=datetime.now(),
        trained_at=datetime.now(),
        engine_config={
            # Model paths
            "gpt_model_path": "GPT_SoVITS/logs/lky_en_cli/logs_s1/8-15200.ckpt",
            "sovits_model_path": "GPT_SoVITS/logs/lky_en_cli/logs_s2_v2/G_233333333333.pth",
            
            # Reference audio for voice cloning
            "ref_audio_path": "GPT_SoVITS/logs/lky_en_cli/2-wav16k/lky_segment_00.wav",
            "prompt_text": "and would pay for because their prices were competitive. They all believed that there were short cuts to prosperity and they thought the best way was by state intervention. This was a mistake that otherwise well intentioned leaders like Julie",
            "prompt_lang": "en",
            
            # Language settings
            "text_lang": "en",
            
            # Model settings
            "device": "cpu",  # Change to "cuda" if GPU available
            "is_half": False,
            
            # Inference parameters
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": 1.0,
            
            # Paths to pretrained models
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        }
    )
    
    # Load or create catalog
    catalog_path = Path("voices.json")
    catalog = VoiceCatalog(catalog_path)
    
    try:
        # Load existing catalog or create new
        catalog.load(create_if_missing=True)
        
        # Add the LKY voice
        catalog.add_voice(lky_voice)
        
        # Save the updated catalog
        catalog.save()
        
        print(f"✅ Successfully registered LKY voice: {lky_voice.voice_id}")
        print(f"📁 Voice catalog saved to: {catalog_path}")
        print(f"🎤 Voice details:")
        print(f"   - Name: {lky_voice.name}")
        print(f"   - Engine: {lky_voice.engine.value}")
        print(f"   - Quality: {lky_voice.quality.value}")
        print(f"   - Language: {lky_voice.language} ({lky_voice.accent})")
        
    except Exception as e:
        print(f"❌ Failed to register voice: {e}")
        return False
    
    return True


if __name__ == "__main__":
    register_lky_voice()