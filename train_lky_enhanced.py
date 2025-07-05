#!/usr/bin/env python3
"""
Enhanced LKY Voice Model Training Script using GPT-SoVITS
Uses the newly extracted 30 LKY speech segments for improved training
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None, env=None):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        if env:
            full_env = os.environ.copy()
            full_env.update(env)
        else:
            full_env = None
            
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd, env=full_env)
        print(f"Success: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        print(f"Output: {e.stdout}")
        return None

def prepare_enhanced_training_environment():
    """Prepare the enhanced training environment"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    
    # Set environment variables for enhanced dataset
    env_vars = {
        "exp_name": "lky_en_enhanced",
        "inp_text": str(base_dir / "output/lky_training_data_enhanced/lky_training_list.txt"),
        "inp_wav_dir": str(base_dir / "output/lky_training_data_enhanced/audio_segments/"),
        "opt_dir": str(base_dir / "logs/lky_en_enhanced"),
        "bert_pretrained_dir": str(base_dir / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        "ssl_pretrained_dir": str(base_dir / "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
        "cnhubert_base_dir": str(base_dir / "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
        "bert_base_path": str(base_dir / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
        "pretrained_s1": str(base_dir / "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        "pretrained_s2G": str(base_dir / "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"),
        "s2config_path": str(base_dir / "GPT_SoVITS/configs/s2.json"),
        "i_part": "0",
        "all_parts": "1",
        "is_half": "True",
        "version": "v2",
        "_CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": f"{base_dir}/GPT_SoVITS:{base_dir}"
    }
    
    return env_vars

def prepare_enhanced_dataset():
    """Run dataset preparation steps with enhanced data"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    env_vars = prepare_enhanced_training_environment()
    
    print("Step 1: Text preprocessing...")
    cmd1 = f"cd {base_dir} && python GPT_SoVITS/prepare_datasets/1-get-text.py"
    result1 = run_command(cmd1, env=env_vars)
    
    if not result1:
        print("Step 1 failed!")
        return False
    
    print("Step 2: Audio preprocessing (HuBERT features)...")
    cmd2 = f"cd {base_dir} && python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"
    result2 = run_command(cmd2, env=env_vars)
    
    if not result2:
        print("Step 2 failed!")
        return False
    
    print("Step 3: Semantic preprocessing...")
    cmd3 = f"cd {base_dir} && python GPT_SoVITS/prepare_datasets/3-get-semantic.py"
    result3 = run_command(cmd3, env=env_vars)
    
    if not result3:
        print("Step 3 failed!")
        return False
    
    print("Enhanced dataset preparation completed successfully!")
    return True

def create_enhanced_s1_config():
    """Create S1 training config for enhanced dataset"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    config_path = base_dir / "GPT_SoVITS/configs/lky_enhanced_s1_config.yaml"
    
    config_content = """train_semantic_path: logs/lky_en_enhanced/6-name2semantic-0.tsv
train_phoneme_path: logs/lky_en_enhanced/2-name2text-0.txt
dev_semantic_path: logs/lky_en_enhanced/6-name2semantic-0.tsv
dev_phoneme_path: logs/lky_en_enhanced/2-name2text-0.txt

data:
  max_eval_sample: 8
  max_sec: 54
  num_workers: 2
  pad_val: 1024
  
model:
  vocab_size: 1025
  phoneme_vocab_size: 512
  embedding_dim: 512
  hidden_dim: 512
  head: 8
  linear_units: 2048
  n_layer: 12
  dropout: 0.1
  EOS: 1024
  random_bert: 0

train:
  seed: 1234
  epochs: 50
  batch_size: 4
  save_every_n_epoch: 5
  precision: 16-mixed
  gradient_clip: 1.0
  if_save_latest: true
  if_save_every_weights: true
  half_weights_save_dir: logs/lky_en_enhanced
  exp_name: lky_en_enhanced

optimizer:
  lr: 0.005
  lr_init: 0.00001
  lr_end: 0.0001
  warmup_steps: 1000
  decay_steps: 20000

pretrained_model: GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
output_dir: logs/lky_en_enhanced

inference:
  top_k: 5"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Enhanced S1 config created at: {config_path}")
    return str(config_path)

def train_enhanced_s1_model():
    """Train Stage 1 (GPT) model with enhanced dataset"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    env_vars = prepare_enhanced_training_environment()
    
    print("Training Enhanced Stage 1 (GPT) model...")
    
    # Create enhanced config
    s1_config = create_enhanced_s1_config()
    
    # Additional training parameters for S1
    env_vars.update({
        "total_epoch": "50",
        "text_low_lr_rate": "0.4",
        "batch_size": "4",
        "lr": "0.005",
        "if_dpo": "False",
        "if_save_latest": "True",
        "if_save_every_weights": "True",
        "save_every_epoch": "5"
    })
    
    cmd = f"cd {base_dir} && python GPT_SoVITS/s1_train.py -c {s1_config}"
    result = run_command(cmd, env=env_vars)
    
    if result:
        print("Enhanced Stage 1 training completed!")
        return True
    else:
        print("Enhanced Stage 1 training failed!")
        return False

def train_enhanced_s2_model():
    """Train Stage 2 (SoVITS) model with enhanced dataset"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    env_vars = prepare_enhanced_training_environment()
    
    print("Training Enhanced Stage 2 (SoVITS) model...")
    
    # Additional training parameters for S2
    env_vars.update({
        "total_epoch": "50", 
        "batch_size": "4",
        "lr": "0.0002",
        "if_cache_data_in_gpu": "False",
        "if_save_latest": "True",
        "if_save_every_weights": "True",
        "save_every_epoch": "5"
    })
    
    cmd = f"cd {base_dir} && python GPT_SoVITS/s2_train.py"
    result = run_command(cmd, env=env_vars)
    
    if result:
        print("Enhanced Stage 2 training completed!")
        return True
    else:
        print("Enhanced Stage 2 training failed!")
        return False

def test_enhanced_model():
    """Test the enhanced trained model"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    
    # Find the latest trained models
    logs_dir = base_dir / "logs/lky_en_enhanced"
    
    # Look for S1 model
    s1_models = list(logs_dir.glob("**/s1*.ckpt"))
    if s1_models:
        latest_s1 = max(s1_models, key=os.path.getctime)
        print(f"Found Enhanced S1 model: {latest_s1}")
    
    # Look for S2 model  
    s2_models = list(logs_dir.glob("**/s2*.pth"))
    if s2_models:
        latest_s2 = max(s2_models, key=os.path.getctime)
        print(f"Found Enhanced S2 model: {latest_s2}")
    
    # Test inference
    if s1_models and s2_models:
        print("Testing enhanced trained model...")
        
        # Create a simple test script
        test_text = "Singapore's journey from third world to first required visionary leadership, pragmatic policies, and the collective will of our people to build a nation that could thrive against all odds."
        test_output = logs_dir / "enhanced_test_output.wav"
        
        # Use the inference CLI to test
        cmd = f"""cd {base_dir} && python GPT_SoVITS/inference_cli.py \\
            --t2s_weights_path {latest_s1} \\
            --vits_weights_path {latest_s2} \\
            --ref_audio_path {base_dir}/output/lky_training_data_enhanced/audio_segments/lky_segment_0033.wav \\
            --aux_ref_audio_paths "" \\
            --prompt_text "The characteristics they have in common were self-confidence" \\
            --prompt_language en \\
            --text "{test_text}" \\
            --text_language en \\
            --output_path {test_output}"""
            
        result = run_command(cmd)
        
        if result and test_output.exists():
            print(f"✅ Enhanced model test successful! Output saved to: {test_output}")
            return True
        else:
            print("❌ Enhanced model test failed!")
            return False
    else:
        print("❌ Could not find enhanced trained models!")
        return False

def main():
    """Main enhanced training pipeline"""
    print("🎤 Starting Enhanced LKY Voice Model Training Pipeline")
    print("Using 30 high-quality LKY speech segments")
    print("=" * 60)
    
    # Check if we have enhanced training data
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    training_list = base_dir / "output/lky_training_data_enhanced/lky_training_list.txt"
    audio_dir = base_dir / "output/lky_training_data_enhanced/audio_segments"
    
    if not training_list.exists() or not audio_dir.exists():
        print("❌ Enhanced training data not found!")
        print(f"Expected: {training_list}")
        print(f"Expected: {audio_dir}")
        return False
    
    # Count training files
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} training audio files")
    
    if len(audio_files) < 10:
        print("❌ Insufficient training data! Need at least 10 audio files")
        return False
    
    # Change to GPT-SoVITS directory
    os.chdir(base_dir)
    
    # Step 1: Prepare enhanced dataset
    print("\n📊 Step 1: Preparing enhanced dataset...")
    if not prepare_enhanced_dataset():
        print("❌ Enhanced dataset preparation failed!")
        return False
    
    # Step 2: Train S1 (GPT) model
    print("\n🤖 Step 2: Training Enhanced Stage 1 (GPT) model...")
    if not train_enhanced_s1_model():
        print("❌ Enhanced Stage 1 training failed!")
        return False
    
    # Step 3: Train S2 (SoVITS) model  
    print("\n🎵 Step 3: Training Enhanced Stage 2 (SoVITS) model...")
    if not train_enhanced_s2_model():
        print("❌ Enhanced Stage 2 training failed!")
        return False
    
    # Step 4: Test the enhanced trained model
    print("\n🧪 Step 4: Testing enhanced trained model...")
    if not test_enhanced_model():
        print("❌ Enhanced model testing failed!")
        return False
    
    print("\n🎉 Enhanced LKY Voice Model Training Complete!")
    print("The enhanced trained models are ready for integration with the audiobook pipeline.")
    print("Quality should be significantly improved with 30 training samples vs. 1 sample.")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)