#!/usr/bin/env python3
"""
Create LKY audiobook using the existing ebook-to-audiobook infrastructure.

This script provides instructions for using the LKY voice model with the
existing tools, since the full integration requires proper dependencies.
"""

import os
from pathlib import Path

def check_setup():
    """Check if everything is set up correctly."""
    print("📚 LKY Audiobook Creation Guide")
    print("=" * 60)
    
    # Check files
    checks = {
        "PDF Book": Path("from-third-world-to-first-by-lee-kuan-yew.pdf"),
        "Voice Catalog": Path("voices.json"),
        "GPT Model": Path("GPT-SoVITS/GPT_SoVITS/logs/lky_en_cli/s1_train/ckpt/epoch=19-step=220.ckpt"),
        "SoVITS Model": Path("GPT-SoVITS/GPT_SoVITS/logs/lky_en_cli/logs_s2_v2/G_233333333333.pth"),
        "Reference Audio": Path("GPT-SoVITS/GPT_SoVITS/logs/lky_en_cli/0-get-audio/lky_segment_00.wav"),
    }
    
    all_good = True
    print("\n✅ Checking prerequisites:")
    for name, path in checks.items():
        if path.exists():
            print(f"   ✓ {name}: Found")
        else:
            print(f"   ✗ {name}: Missing ({path})")
            all_good = False
    
    return all_good


def show_manual_steps():
    """Show manual steps to create the audiobook."""
    print("\n📋 Manual Steps to Create LKY Audiobook:")
    print("-" * 60)
    
    print("\n1️⃣  Extract text from PDF:")
    print("   python3 -m PyPDF2 from-third-world-to-first-by-lee-kuan-yew.pdf > book_text.txt")
    
    print("\n2️⃣  Start GPT-SoVITS WebUI:")
    print("   cd GPT-SoVITS")
    print("   python webui.py")
    
    print("\n3️⃣  In the WebUI:")
    print("   a) Load GPT model: logs/lky_en_cli/s1_train/ckpt/epoch=19-step=220.ckpt")
    print("   b) Load SoVITS model: logs/lky_en_cli/logs_s2_v2/G_233333333333.pth")
    print("   c) Set reference audio: logs/lky_en_cli/0-get-audio/lky_segment_00.wav")
    print("   d) Paste book text and generate")
    
    print("\n4️⃣  Or use the API:")
    print("   cd GPT-SoVITS")
    print("   python api_v2.py")
    print("   # Then send requests with the book text")


def show_quick_test():
    """Show quick test command."""
    print("\n🧪 Quick Voice Test:")
    print("-" * 60)
    
    test_text = "Singapore's survival as an independent nation was in doubt."
    
    print("\nSave this as test_request.json:")
    print(f'''{{
    "text": "{test_text}",
    "text_lang": "en", 
    "ref_audio_path": "GPT_SoVITS/logs/lky_en_cli/0-get-audio/lky_segment_00.wav",
    "prompt_text": "and would pay for because their prices were competitive.",
    "prompt_lang": "en",
    "top_k": 5,
    "top_p": 1,
    "temperature": 1,
    "text_split_method": "cut5",
    "speed_factor": 1.0,
    "media_type": "wav"
}}''')
    
    print("\nThen run:")
    print("curl -X POST http://127.0.0.1:9880/tts \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d @test_request.json \\")
    print("  -o lky_test.wav")


def main():
    """Main function."""
    if check_setup():
        print("\n✅ All files are ready!")
        
        # Show options
        print("\n🎯 Options for creating the audiobook:")
        print("\nA) Full Integration (requires dependencies):")
        print("   python convert_full_book.py --voice lky_en_trained")
        
        print("\nB) Manual Generation:")
        show_manual_steps()
        
        print("\nC) Quick Test First:")
        show_quick_test()
        
        print("\n" + "=" * 60)
        print("💡 Recommendation: Start with option C to test the voice quality,")
        print("   then proceed with A or B for the full book.")
        
    else:
        print("\n❌ Some files are missing. Please check the paths above.")


if __name__ == "__main__":
    main()