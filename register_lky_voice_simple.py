#!/usr/bin/env python3
"""
Simple script to register the LKY voice by creating/updating voices.json directly.
"""

import json
from pathlib import Path
from datetime import datetime

def register_lky_voice():
    """Register the LKY voice by creating/updating voices.json."""
    
    # LKY voice configuration
    lky_voice = {
        "voice_id": "lky_en_trained",
        "name": "Lee Kuan Yew (Trained)",
        "description": "High-quality voice clone of Lee Kuan Yew trained on his audiobook samples",
        "voice_type": "trained",
        "engine": "gpt_sovits",
        "language": "en",
        "accent": "Singaporean English",
        "quality": "very_high",
        "sample_rate": 32000,
        "quality_score": 9.5,
        "model_path": "GPT_SoVITS/logs/lky_en_cli",
        "created_at": datetime.now().isoformat(),
        "trained_at": datetime.now().isoformat(),
        "engine_config": {
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
    }
    
    # Load existing catalog or create new
    catalog_path = Path("voices.json")
    
    if catalog_path.exists():
        with open(catalog_path, 'r') as f:
            catalog = json.load(f)
    else:
        # Create default catalog structure
        catalog = {
            "version": "1.0",
            "built_in_voices": {},
            "custom_voices": {},
            "voice_collections": {
                "english_voices": {
                    "name": "English Voices",
                    "description": "English language voices",
                    "voices": []
                }
            }
        }
    
    # Add LKY voice to custom voices
    catalog["custom_voices"]["lky_en_trained"] = lky_voice
    
    # Add to English voices collection if not already there
    english_collection = catalog["voice_collections"]["english_voices"]["voices"]
    if "lky_en_trained" not in english_collection:
        english_collection.append("lky_en_trained")
    
    # Save updated catalog
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"✅ Successfully registered LKY voice")
    print(f"📁 Voice catalog saved to: {catalog_path}")
    print(f"🎤 Voice ID: lky_en_trained")
    print(f"   - Name: Lee Kuan Yew (Trained)")
    print(f"   - Engine: gpt_sovits")
    print(f"   - Quality: very_high")
    print(f"   - Language: en (Singaporean English)")


if __name__ == "__main__":
    register_lky_voice()