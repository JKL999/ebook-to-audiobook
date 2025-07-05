#!/usr/bin/env python3
"""
Enhanced S2 (SoVITS) Training Script
Run S2 training after successful S1 training
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
        "s2config_path": str(base_dir / "configs/lky_enhanced_s2.json"),
        "i_part": "0",
        "all_parts": "1",
        "is_half": "True",
        "version": "v2",
        "_CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": f"{base_dir}/GPT_SoVITS:{base_dir}"
    }
    
    return env_vars

def train_enhanced_s2_model():
    """Train Stage 2 (SoVITS) model with enhanced dataset"""
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    env_vars = prepare_enhanced_training_environment()
    
    print("Training Enhanced Stage 2 (SoVITS) model...")
    
    # Additional training parameters for S2
    env_vars.update({
        "total_epoch": "50", 
        "batch_size": "1",  # Reduced from 4 to 1 for memory optimization
        "lr": "0.0002",
        "if_cache_data_in_gpu": "False",
        "if_save_latest": "True",
        "if_save_every_weights": "True",
        "save_every_epoch": "5"
    })
    
    # Set config file path and run S2 training 
    config_file = str(base_dir / "configs/lky_enhanced_s2.json")
    cmd = f"cd {base_dir} && python3 GPT_SoVITS/s2_train.py --config {config_file}"
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
        cmd = f"""cd {base_dir} && python3 GPT_SoVITS/inference_cli.py \\
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
            
            # Check file size as basic validation
            file_size = test_output.stat().st_size
            if file_size > 1000:  # At least 1KB for a valid audio file
                print(f"Generated audio file size: {file_size} bytes")
                return True
            else:
                print(f"Generated audio file too small: {file_size} bytes")
                return False
        else:
            print("❌ Enhanced model test failed!")
            return False
    else:
        print("❌ Could not find enhanced trained models!")
        print(f"S1 models found: {len(s1_models)}")
        print(f"S2 models found: {len(s2_models)}")
        return False

def main():
    """Main S2 training pipeline"""
    print("🎵 Starting Enhanced S2 (SoVITS) Training")
    print("=" * 50)
    
    base_dir = Path(__file__).parent / "GPT-SoVITS"
    
    # Change to GPT-SoVITS directory
    os.chdir(base_dir)
    
    # Train S2 (SoVITS) model  
    print("\n🎵 Training Enhanced Stage 2 (SoVITS) model...")
    if not train_enhanced_s2_model():
        print("❌ Enhanced Stage 2 training failed!")
        return False
    
    # Test the enhanced trained model
    print("\n🧪 Testing enhanced trained model...")
    if not test_enhanced_model():
        print("❌ Enhanced model testing failed!")
        return False
    
    print("\n🎉 Enhanced S2 Training Complete!")
    print("Both S1 and S2 models are now trained and ready!")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)