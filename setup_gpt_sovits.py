#!/usr/bin/env python3
"""
Setup script for GPT-SoVITS pretrained models and training environment
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        print(f"Success: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

def setup_pretrained_models():
    """Download and setup pretrained models"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    pretrained_dir = base_dir / "GPT_SoVITS" / "pretrained_models"
    
    # Create directories
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    # We'll use the existing pretrained models from our LKY inference directory
    lky_pretrained = Path(__file__).parent / "lky_audiobook_inference" / "pretrained_models"
    
    if lky_pretrained.exists():
        print("Copying existing pretrained models from LKY inference...")
        
        # Copy chinese-roberta-wwm-ext-large
        roberta_src = lky_pretrained / "chinese-roberta-wwm-ext-large"
        roberta_dst = pretrained_dir / "chinese-roberta-wwm-ext-large"
        if roberta_src.exists() and not roberta_dst.exists():
            run_command(f"cp -r {roberta_src} {roberta_dst}")
        
        # Copy chinese-hubert-base
        hubert_src = lky_pretrained / "chinese-hubert-base" 
        hubert_dst = pretrained_dir / "chinese-hubert-base"
        if hubert_src.exists() and not hubert_dst.exists():
            run_command(f"cp -r {hubert_src} {hubert_dst}")
    
    # Download GPT-SoVITS pretrained weights (v2 version - most stable)
    weights_dir = pretrained_dir / "gsv-v2final-pretrained"
    weights_dir.mkdir(exist_ok=True)
    
    # Download v2 pretrained models if they don't exist
    gpt_weight = weights_dir / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    sovits_weight = weights_dir / "s2G2333k.pth"
    
    if not gpt_weight.exists():
        print("Downloading GPT pretrained model...")
        run_command(f"wget -O {gpt_weight} https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch%3D12-step%3D369668.ckpt")
    
    if not sovits_weight.exists():
        print("Downloading SoVITS pretrained model...")
        run_command(f"wget -O {sovits_weight} https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth")
    
    print("Pretrained models setup complete!")

def setup_training_config():
    """Create training configuration for LKY voice"""
    config_path = Path(__file__).parent / "GPT-SoVITS" / "GPT_SoVITS" / "configs" / "lky_training.yaml"
    
    config_content = """
# LKY Voice Training Configuration
exp_name: lky_en_cli
version: v2
pretrained_s1: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
pretrained_s2G: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth
bert_pretrained_dir: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
ssl_pretrained_dir: GPT_SoVITS/pretrained_models/chinese-hubert-base

# Training data paths
inp_text: output/lky_training_data/lky_training_list.txt
inp_wav_dir: output/lky_training_data/
opt_dir: logs/lky_en_cli

# Device settings  
device: cuda
is_half: true
batch_size: 4
epochs: 50
learning_rate: 0.0002

# Training settings
save_every_epoch: 5
eval_every_epoch: 1
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    print(f"Training config created at: {config_path}")

if __name__ == "__main__":
    print("Setting up GPT-SoVITS for LKY voice training...")
    
    # Change to GPT-SoVITS directory
    gpt_sovits_dir = Path(__file__).parent / "GPT-SoVITS"
    os.chdir(gpt_sovits_dir)
    
    # Setup pretrained models
    setup_pretrained_models()
    
    # Setup training config
    setup_training_config()
    
    print("\nGPT-SoVITS setup complete!")
    print("Ready to start LKY voice model training.")