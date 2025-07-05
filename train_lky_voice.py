#!/usr/bin/env python3
"""
LKY Voice Model Training Script using GPT-SoVITS
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

def prepare_training_environment():
    """Prepare the training environment"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    
    # Set environment variables
    env_vars = {
        "exp_name": "lky_en_cli",
        "inp_text": str(base_dir / "output/lky_training_data/lky_training_list.txt"),
        "inp_wav_dir": str(base_dir / "output/lky_training_data/"),
        "opt_dir": str(base_dir / "logs/lky_en_cli"),
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
        "_CUDA_VISIBLE_DEVICES": "0"
    }
    
    return env_vars

def prepare_dataset():
    """Run dataset preparation steps"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    env_vars = prepare_training_environment()
    
    # Set Python path for GPT-SoVITS modules
    env_vars["PYTHONPATH"] = f"{base_dir}/GPT_SoVITS:{base_dir}"
    
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
    
    print("Dataset preparation completed successfully!")
    return True

def train_s1_model():
    """Train Stage 1 (GPT) model"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    env_vars = prepare_training_environment()
    
    print("Training Stage 1 (GPT) model...")
    
    # Additional training parameters for S1
    env_vars.update({
        "total_epoch": "50",
        "text_low_lr_rate": "0.4",
        "batch_size": "4",
        "lr": "0.0002",
        "if_dpo": "False",
        "if_save_latest": "True",
        "if_save_every_weights": "True",
        "save_every_epoch": "5"
    })
    
    s1_config = base_dir / "GPT_SoVITS/configs/lky_s1_config.yaml"
    cmd = f"cd {base_dir} && python GPT_SoVITS/s1_train.py -c {s1_config}"
    result = run_command(cmd, env=env_vars)
    
    if result:
        print("Stage 1 training completed!")
        return True
    else:
        print("Stage 1 training failed!")
        return False

def train_s2_model():
    """Train Stage 2 (SoVITS) model"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    env_vars = prepare_training_environment()
    
    print("Training Stage 2 (SoVITS) model...")
    
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
        print("Stage 2 training completed!")
        return True
    else:
        print("Stage 2 training failed!")
        return False

def test_trained_model():
    """Test the trained model"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    
    # Find the latest trained models
    logs_dir = base_dir / "logs/lky_en_cli"
    
    # Look for S1 model
    s1_models = list(logs_dir.glob("**/s1*.ckpt"))
    if s1_models:
        latest_s1 = max(s1_models, key=os.path.getctime)
        print(f"Found S1 model: {latest_s1}")
    
    # Look for S2 model  
    s2_models = list(logs_dir.glob("**/s2*.pth"))
    if s2_models:
        latest_s2 = max(s2_models, key=os.path.getctime)
        print(f"Found S2 model: {latest_s2}")
    
    # Test inference
    if s1_models and s2_models:
        print("Testing trained model...")
        
        # Create a simple test script
        test_text = "This is a test of the trained LKY voice model."
        test_output = base_dir / "logs/lky_en_cli/test_output.wav"
        
        # Use the inference CLI to test
        cmd = f"""cd {base_dir} && python GPT_SoVITS/inference_cli.py \\
            --t2s_weights_path {latest_s1} \\
            --vits_weights_path {latest_s2} \\
            --ref_audio_path {base_dir}/output/lky_training_data/lky_ref_audio.wav \\
            --aux_ref_audio_paths "" \\
            --prompt_text "When we became independent in 1965" \\
            --prompt_language en \\
            --text "{test_text}" \\
            --text_language en \\
            --output_path {test_output}"""
            
        result = run_command(cmd)
        
        if result and test_output.exists():
            print(f"✅ Model test successful! Output saved to: {test_output}")
            return True
        else:
            print("❌ Model test failed!")
            return False
    else:
        print("❌ Could not find trained models!")
        return False

def main():
    """Main training pipeline"""
    print("🎤 Starting LKY Voice Model Training Pipeline")
    print("=" * 50)
    
    # Check if we have training data
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    training_list = base_dir / "output/lky_training_data/lky_training_list.txt"
    training_audio = base_dir / "output/lky_training_data/lky_ref_audio.wav"
    
    if not training_list.exists() or not training_audio.exists():
        print("❌ Training data not found!")
        print(f"Expected: {training_list}")
        print(f"Expected: {training_audio}")
        return False
    
    # Change to GPT-SoVITS directory
    os.chdir(base_dir)
    
    # Step 1: Prepare dataset
    print("\n📊 Step 1: Preparing dataset...")
    if not prepare_dataset():
        print("❌ Dataset preparation failed!")
        return False
    
    # Step 2: Train S1 (GPT) model
    print("\n🤖 Step 2: Training Stage 1 (GPT) model...")
    if not train_s1_model():
        print("❌ Stage 1 training failed!")
        return False
    
    # Step 3: Train S2 (SoVITS) model  
    print("\n🎵 Step 3: Training Stage 2 (SoVITS) model...")
    if not train_s2_model():
        print("❌ Stage 2 training failed!")
        return False
    
    # Step 4: Test the trained model
    print("\n🧪 Step 4: Testing trained model...")
    if not test_trained_model():
        print("❌ Model testing failed!")
        return False
    
    print("\n🎉 LKY Voice Model Training Complete!")
    print("The trained models are ready for integration with the audiobook pipeline.")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)