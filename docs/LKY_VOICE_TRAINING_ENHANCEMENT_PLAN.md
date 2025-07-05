# LKY Voice Model Training Data Enhancement Plan

## Current Situation Analysis

I've analyzed your setup and found:

1. **Raw Training Data Available:**
   - `LKY-speech-For Third World Leaders Hope or Despair.mp3` (full recording, ~15 minutes of LKY starts around 15-minute mark)
   - `LKY-speech-.tsv` (timestamp file with 673 segments, from 0-5830 seconds)
   - Single `lky_ref_audio.wav` (current limited training data)

2. **Whisper-Transcriber Tool Available:**
   - Full transcription and speaker diarization capabilities
   - Supports segmentation with precise timestamps
   - Can export in multiple formats (TXT, SRT, JSON)

3. **GPT-SoVITS Training Setup:**
   - Environment configured and working
   - Dataset preparation pipeline functional
   - Training failed due to insufficient data (only 1 sample)

## Comprehensive Plan: Generate Sufficient Training Data

### Phase 1: Audio Processing and Segmentation (30-60 minutes)

**1.1 Extract LKY Speech Segments**
- Use your whisper-transcriber to process the full MP3 file
- Extract clean speech segments from 15:00+ mark where LKY speaks
- Filter segments by:
  - Duration (2-10 seconds for optimal training)
  - Audio quality (remove segments with background noise/applause)
  - Speech clarity (exclude overlapping speech)

**1.2 Audio Preprocessing**
- Convert segments to WAV format at 22050Hz (GPT-SoVITS standard)
- Normalize audio levels
- Remove silence padding
- Target: Extract 50-100 clean LKY speech segments

### Phase 2: Transcription and Alignment (30-45 minutes)

**2.1 Accurate Transcription**
- Use whisper-transcriber with speaker diarization
- Generate precise text transcriptions for each audio segment
- Manual review and correction of transcripts for accuracy

**2.2 Training Data Generation**
- Create GPT-SoVITS compatible training format:
  ```
  filename.wav|lky|en|transcription_text
  ```
- Generate training list file with all segments
- Organize files in proper directory structure

### Phase 3: Enhanced Training Dataset (15-30 minutes)

**3.1 Data Augmentation**
- Select best 20-30 segments with diverse:
  - Sentence lengths (short, medium, long)
  - Speaking styles (formal speech, Q&A responses)
  - Emotional tones (confident, explanatory, emphatic)

**3.2 Quality Validation**
- Listen to each selected segment
- Verify transcription accuracy
- Ensure audio quality standards

### Phase 4: Model Training (60-120 minutes)

**4.1 Dataset Preparation**
- Run GPT-SoVITS preprocessing pipeline with new dataset
- Verify all preparation steps complete successfully

**4.2 Model Training**
- Train S1 (GPT) model with 20-30 samples (sufficient for voice cloning)
- Train S2 (SoVITS) model 
- Monitor training progress and adjust parameters if needed

**4.3 Model Testing**
- Test trained model with sample text
- Verify voice quality and similarity to original LKY
- Generate test audio samples

### Phase 5: Integration and Validation (30 minutes)

**5.1 Pipeline Integration**
- Update voice configuration to use new trained models
- Test integration with ebook-to-audiobook pipeline
- Verify end-to-end functionality

**5.2 Quality Assessment**
- Generate sample audiobook segments
- Compare with original LKY voice recordings
- Document model performance metrics

## Expected Outcomes

**Training Data:**
- 20-30 high-quality LKY voice samples
- Accurate transcriptions for each sample
- Properly formatted training dataset

**Model Quality:**
- Significant improvement over single-sample training
- Natural-sounding LKY voice synthesis
- Good pronunciation and intonation matching

**Integration:**
- Fully functional LKY voice in audiobook pipeline
- Compatible with existing ebook processing workflow
- Production-ready voice model

## Technical Implementation

**Tools to Use:**
1. **whisper-transcriber**: Audio processing and transcription
2. **FFmpeg**: Audio format conversion and segmentation  
3. **GPT-SoVITS**: Model training pipeline
4. **Custom scripts**: Data preparation and validation

**Key Advantages:**
- Leverage existing proven transcription tool
- Use real LKY speech data for authentic voice cloning
- Systematic approach to ensure sufficient training data
- Quality validation at each step

This plan transforms the limited training data issue into a comprehensive dataset using your actual LKY recording, ensuring the voice model will have sufficient and diverse training examples for high-quality synthesis.

---

## Progress Tracking

- [ ] Phase 1: Audio Processing and Segmentation
- [ ] Phase 2: Transcription and Alignment  
- [ ] Phase 3: Enhanced Training Dataset
- [ ] Phase 4: Model Training
- [ ] Phase 5: Integration and Validation