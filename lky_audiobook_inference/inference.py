import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import traceback

# Add the new src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import necessary modules from the new src structure
from feature_extractor.cnhubert import CNHubert
from module.models import SynthesizerTrn, SynthesizerTrnV3, Generator
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from TTS_infer_pack.TTS import TTS_Config, TTS
from TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from TTS_infer_pack.TextPreprocessor import TextPreprocessor
from tools.i18n.i18n import I18nAuto
from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from module.mel_processing import mel_spectrogram_torch, spectrogram_torch

# Define absolute paths for models and reference audio
GPT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lky", "lky_gpt_model.ckpt")
SOVITS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "lky", "lky_sovits_model.pth")
REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "models", "lky", "lky_ref_audio.wav")
BERT_BASE_PATH = os.path.join(os.path.dirname(__file__), "pretrained_models", "chinese-roberta-wwm-ext-large")
CNHUBERT_BASE_PATH = os.path.join(os.path.dirname(__file__), "pretrained_models", "chinese-hubert-base")

# Initialize i18n (can be simplified if only English is needed)
i18n = I18nAuto()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = True if torch.cuda.is_available() else False # Use half precision on GPU

# Load BERT and CNHubert models
# These are loaded within TTS_Config and TTS classes, so we'll rely on that.

def synthesize_audio(text_to_synthesize: str, output_filename: str = "output.wav"):
    # Initialize TTS_Config with absolute paths
    tts_config = TTS_Config()
    tts_config.t2s_weights_path = GPT_MODEL_PATH
    tts_config.vits_weights_path = SOVITS_MODEL_PATH
    tts_config.bert_base_path = BERT_BASE_PATH
    tts_config.cnhuhbert_base_path = CNHUBERT_BASE_PATH
    tts_config.device = device
    tts_config.is_half = is_half
    tts_config.version = "v2" # Assuming LKY model is v2 compatible

    # Initialize TTS pipeline
    tts_pipeline = TTS(tts_config)

    # Prepare request for TTS
    req = {
        "text": text_to_synthesize,
        "text_lang": "en",
        "ref_audio_path": REF_AUDIO_PATH,
        "prompt_lang": "en",
        "prompt_text": "and would pay for because their prices were competitive. They all believed that there were short cuts to prosperity and they thought the best way was by state intervention. This was a mistake that otherwise well intentioned leaders like Julie", # From voices.json
        "top_k": 5,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": 1.0,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False,
    }

    # Perform synthesis
    try:
        sr, audio_data = next(tts_pipeline.run(req))
        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        sf.write(output_path, audio_data, sr)
        print(f"Audio saved to {output_path}")
    except Exception as e:
        print(f"Error during synthesis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LKY Voice TTS Inference")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--output", help="Output audio file path")
    
    args = parser.parse_args()
    
    if args.text and args.output:
        synthesize_audio(args.text, args.output)
    else:
        # Default sample
        sample_text = "Singapore's survival as an independent nation was in doubt."
        synthesize_audio(sample_text, "lky_audio_sample.wav")
