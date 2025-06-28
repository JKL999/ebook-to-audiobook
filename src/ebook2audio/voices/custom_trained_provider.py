"""
Custom Trained Voice Provider for Desktop-Trained Models

This provider integrates custom voice models trained on the desktop RTX 3070
with the existing ebook-to-audiobook pipeline for Mac inference.

Compatible with models from:
- GPT-SoVITS training environment 
- Desktop agent training workflow
- Phase 3.2 voice cloning integration
"""

import torch
import torchaudio
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import BaseTTSProvider, Voice, VoiceMetadata, TTSEngine, VoiceType, VoiceQuality


class CustomTrainedProvider(BaseTTSProvider):
    """TTS Provider for custom trained voice models from desktop training."""
    
    def __init__(self):
        super().__init__(TTSEngine.XTTS)  # Use XTTS engine type for compatibility
        self.device = None
        self.loaded_models = {}
        self.model_cache = {}
        
    def initialize(self) -> bool:
        """Initialize the custom trained voice provider."""
        try:
            # Determine best device for inference
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🍎 Using MPS device for Apple Silicon optimization")
            else:
                self.device = torch.device("cpu")
                print("🔄 Using CPU device for inference")
            
            # Test device with simple tensor operation
            test_tensor = torch.randn(10, 10, device=self.device)
            _ = torch.matmul(test_tensor, test_tensor.t())
            
            self._initialized = True
            print(f"✅ Custom Trained Provider initialized on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Custom Trained Provider: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self._initialized and torch.is_available
    
    def load_voice(self, voice_metadata: VoiceMetadata) -> bool:
        """Load a custom trained voice model."""
        try:
            if not voice_metadata.model_path or not voice_metadata.model_path.exists():
                print(f"❌ Model file not found: {voice_metadata.model_path}")
                return False
            
            print(f"📥 Loading custom voice model: {voice_metadata.name}")
            
            # Load the model checkpoint
            checkpoint = torch.load(
                voice_metadata.model_path, 
                map_location=self.device,
                weights_only=False  # Allow loading full checkpoints
            )
            
            # Extract model components based on structure
            if isinstance(checkpoint, dict):
                model_data = checkpoint.get('model_state_dict', checkpoint)
                config = checkpoint.get('config', {})
                model_info = checkpoint.get('model_info', {})
                
                print(f"✅ Loaded model with config: {config.get('voice_name', 'unknown')}")
                print(f"📊 Quality score: {model_info.get('quality_score', 'N/A')}")
                
                # Store loaded model
                self.loaded_models[voice_metadata.voice_id] = {
                    'model_data': model_data,
                    'config': config,
                    'model_info': model_info,
                    'metadata': voice_metadata
                }
                
                return True
            else:
                print("❌ Unexpected model format")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load voice model: {e}")
            return False
    
    def synthesize(self, text: str, voice: Voice, output_path: Optional[Path] = None) -> Optional[Path]:
        """Synthesize speech using custom trained voice model."""
        try:
            if voice.voice_id not in self.loaded_models:
                if not self.load_voice(voice.metadata):
                    return None
            
            model_info = self.loaded_models[voice.voice_id]
            
            print(f"🎤 Synthesizing with {voice.name}: '{text[:50]}...'")
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = int(time.time())
                output_path = Path(f"test_outputs/model_tests/synthesis_{voice.voice_id}_{timestamp}.wav")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simulate synthesis process (actual implementation would depend on model format)
            # For now, create a simple audio file as placeholder
            synthesis_start = time.time()
            
            # Create a simple sine wave as placeholder audio
            sample_rate = voice.metadata.sample_rate or 22050
            duration = max(0.5, len(text) * 0.05)  # Rough estimate based on text length
            
            # Generate placeholder audio
            num_samples = int(sample_rate * duration)
            t = torch.linspace(0, duration, num_samples, device=self.device)
            
            # Create a pleasant tone (placeholder for actual voice synthesis)
            frequency = 440.0  # A4 note
            audio = 0.3 * torch.sin(2 * torch.pi * frequency * t)
            
            # Add some variation to make it more voice-like
            audio += 0.1 * torch.sin(2 * torch.pi * frequency * 1.5 * t)
            audio = audio.unsqueeze(0)  # Add channel dimension
            
            # Save audio file
            torchaudio.save(str(output_path), audio.cpu(), sample_rate)
            
            synthesis_time = time.time() - synthesis_start
            print(f"⚡ Synthesis completed in {synthesis_time:.3f} seconds")
            print(f"🎵 Audio saved to: {output_path}")
            
            # Log synthesis metadata
            self._log_synthesis_stats(voice, text, synthesis_time, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            return None
    
    def preview_voice(self, voice: Voice, preview_text: str = "Hello, this is a voice preview.") -> Optional[Path]:
        """Generate a preview sample for the voice."""
        print(f"🎧 Generating preview for {voice.name}")
        
        preview_path = Path(f"test_outputs/model_tests/preview_{voice.voice_id}.wav")
        return self.synthesize(preview_text, voice, preview_path)
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages for custom trained models."""
        # Custom trained models support the language they were trained on
        # For now, return common languages that could be supported
        return ["en", "en-US", "en-GB"]
    
    def _log_synthesis_stats(self, voice: Voice, text: str, synthesis_time: float, output_path: Path):
        """Log synthesis statistics for performance monitoring."""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'voice_id': voice.voice_id,
            'voice_name': voice.name,
            'text_length': len(text),
            'synthesis_time': synthesis_time,
            'output_path': str(output_path),
            'device': str(self.device),
            'chars_per_second': len(text) / synthesis_time if synthesis_time > 0 else 0
        }
        
        # Log to file for performance tracking
        log_path = Path("test_outputs/synthesis_stats.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "a") as f:
            import json
            f.write(json.dumps(stats) + "\n")
    
    def get_model_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a loaded model."""
        if voice_id not in self.loaded_models:
            return None
            
        model_info = self.loaded_models[voice_id]
        return {
            'voice_id': voice_id,
            'config': model_info['config'],
            'model_info': model_info['model_info'],
            'device': str(self.device),
            'loaded_at': datetime.now().isoformat()
        }
    
    def list_loaded_models(self) -> List[str]:
        """List all currently loaded voice models."""
        return list(self.loaded_models.keys())
    
    def unload_voice(self, voice_id: str) -> bool:
        """Unload a voice model to free memory."""
        if voice_id in self.loaded_models:
            del self.loaded_models[voice_id]
            print(f"🗑️ Unloaded voice model: {voice_id}")
            return True
        return False
    
    def cleanup(self) -> None:
        """Clean up resources and unload all models."""
        self.loaded_models.clear()
        self.model_cache.clear()
        super().cleanup()
        print("🧹 Custom Trained Provider cleanup complete")


def create_custom_voice_from_model(model_path: Path, voice_name: str, voice_id: Optional[str] = None) -> Optional[Voice]:
    """
    Create a Voice object from a trained model file.
    
    This utility function helps create Voice objects for desktop-trained models
    with appropriate metadata configuration.
    """
    try:
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return None
        
        # Generate voice ID if not provided
        if voice_id is None:
            voice_id = f"custom_{model_path.stem}"
        
        # Try to extract metadata from model file
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
            model_info = checkpoint.get('model_info', {}) if isinstance(checkpoint, dict) else {}
        except:
            config = {}
            model_info = {}
        
        # Create voice metadata
        metadata = VoiceMetadata(
            voice_id=voice_id,
            name=voice_name,
            description=f"Custom trained voice: {voice_name}",
            voice_type=VoiceType.TRAINED,
            engine=TTSEngine.XTTS,  # Use XTTS for compatibility
            language=config.get('language', 'en'),
            quality=VoiceQuality.HIGH,
            sample_rate=config.get('sample_rate', 22050),
            quality_score=model_info.get('quality_score'),
            model_path=model_path,
            training_samples_count=model_info.get('training_samples'),
            created_at=datetime.now(),
            engine_config={
                'model_type': config.get('model_type', 'custom'),
                'trained_on': config.get('trained_on', 'desktop'),
                'compatibility': config.get('compatibility', 'Mac_CPU_MPS')
            }
        )
        
        # Create provider
        provider = CustomTrainedProvider()
        if not provider.initialize():
            print("❌ Failed to initialize provider")
            return None
        
        # Create voice object
        voice = Voice(metadata=metadata, provider=provider)
        
        print(f"✅ Created custom voice: {voice_name} ({voice_id})")
        return voice
        
    except Exception as e:
        print(f"❌ Failed to create custom voice: {e}")
        return None


def discover_trained_models(model_directory: Path = None) -> List[Voice]:
    """
    Discover and create Voice objects for all trained models in a directory.
    
    Scans for .pth files and attempts to create Voice objects for each.
    """
    if model_directory is None:
        model_directory = Path("models/custom_voices")
    
    if not model_directory.exists():
        print(f"📁 Model directory not found: {model_directory}")
        return []
    
    voices = []
    model_files = list(model_directory.glob("*.pth")) + list(model_directory.glob("*.pt"))
    
    print(f"🔍 Discovering models in {model_directory}")
    print(f"📦 Found {len(model_files)} model files")
    
    for model_file in model_files:
        voice_name = model_file.stem.replace("_", " ").title()
        voice = create_custom_voice_from_model(model_file, voice_name)
        
        if voice:
            voices.append(voice)
    
    print(f"✅ Successfully created {len(voices)} custom voices")
    return voices