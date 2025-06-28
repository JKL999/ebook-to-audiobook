# 🖥️ Desktop Agent Tasks: Voice Cloning Training Setup

## 🎯 Mission: Set Up Voice Training Environment on RTX 3070 Desktop

### Context
The Mac agent has completed the optimized ebook-to-audiobook MVP with 4-8x performance improvements. We're now entering **Phase 3: Voice Cloning Integration** to eliminate API rate limits and enable custom voice training.

### Your Hardware
- **GPU**: RTX 3070 (8GB VRAM) - Perfect for voice training
- **RAM**: 32GB - Excellent for batch processing
- **CPU**: i7 - Good for preprocessing

### Primary Task: Install and Test GPT-SoVITS

Based on our research, **GPT-SoVITS** is our recommended choice because:
- ✅ Only needs 1 minute of voice data
- ✅ 8GB VRAM is sufficient (you have this!)
- ✅ 80-95% voice similarity with just 5-second samples
- ✅ Includes complete WebUI for easy training
- ✅ Supports English, Japanese, and Chinese

### Step-by-Step Instructions

#### 1. Clone and Install GPT-SoVITS
```bash
# Clone the repository
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

#### 2. Download Required Models
- Follow the instructions in the GPT-SoVITS README to download base models
- The WebUI should guide you through this process

#### 3. Prepare Test Voice Sample
Create a test voice sample:
- Record or find 1-2 minutes of clear speech
- Save as `test_voice_sample.wav` (mono, 16kHz recommended)
- Ensure good quality: no background noise, consistent volume

#### 4. Launch WebUI and Train
```bash
python webui.py
```
- Navigate to the training interface
- Upload your voice sample
- Start training (should take 30-60 minutes)
- Save the trained model as `custom_voice_v1.pth`

#### 5. Test Voice Generation
Once training completes:
- Test with sample text: "Hello, this is a test of my custom voice clone."
- Try different emotions and speaking styles
- Export some test audio files

#### 6. Document Results
Create `VOICE_TRAINING_RESULTS.md` with:
- Training time taken
- GPU memory usage
- Voice quality assessment (1-10)
- Any issues encountered
- Inference speed benchmarks

### Alternative Setup (If GPT-SoVITS Has Issues)

Try **XTTS v2** as backup:
```bash
pip install TTS
# Test with:
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
```

### Integration Notes for Mac Agent

The Mac agent will be setting up lightweight inference using:
- **MeloTTS** (CPU-optimized)
- **XTTS v2** (for comparison)

Make sure to:
1. Export your trained voice model to a portable format
2. Test that the model file works independently
3. Document the model format and requirements

### Success Criteria
- [ ] GPT-SoVITS installed and WebUI running
- [ ] Successfully trained a voice from 1-minute sample
- [ ] Generated test audio with custom voice
- [ ] Documented training process and results
- [ ] Model file ready for Mac inference testing

### Troubleshooting Tips
- If CUDA errors: Ensure NVIDIA drivers are updated
- If memory errors: Try reducing batch size in training config
- If quality is poor: Check audio sample quality and try longer samples

### Communication
Once complete, commit your results:
```bash
git add VOICE_TRAINING_RESULTS.md
git add models/custom_voice_v1.pth  # If not too large
git commit -m "feat: complete voice training setup on desktop"
git push origin main
```

The Mac agent will then pull and test inference with your trained model!

---

**Questions?** Check the [ROADMAP.md](ROADMAP.md) for the full voice cloning integration plan.

**Good luck, Desktop Agent! 🚀**