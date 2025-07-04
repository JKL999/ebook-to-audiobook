#!/usr/bin/env python3
"""
LKY Speech Segment Extraction Tool

This script processes the raw LKY speech recording to extract clean segments
suitable for GPT-SoVITS voice training.
"""

import os
import sys
import json
import subprocess
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

class LKYSegmentExtractor:
    def __init__(self, audio_path: str, tsv_path: str, output_dir: str):
        """
        Initialize the LKY segment extractor
        
        Args:
            audio_path: Path to the raw LKY MP3 file
            tsv_path: Path to the timestamp TSV file  
            output_dir: Directory to save extracted segments
        """
        self.audio_path = Path(audio_path)
        self.tsv_path = Path(tsv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.audio_segments_dir = self.output_dir / "audio_segments"
        self.audio_segments_dir.mkdir(exist_ok=True)
        
        # Load audio file
        print(f"Loading audio file: {self.audio_path}")
        self.audio, self.sr = librosa.load(str(self.audio_path), sr=22050)
        print(f"Audio loaded: {len(self.audio)/self.sr:.1f} seconds at {self.sr}Hz")
        
        # Load timestamp data
        if self.tsv_path.exists():
            print(f"Loading timestamp data: {self.tsv_path}")
            self.timestamps = pd.read_csv(self.tsv_path, sep='\t')
            print(f"Found {len(self.timestamps)} timestamp entries")
        else:
            print("No timestamp file found, will use Whisper for segmentation")
            self.timestamps = None
    
    def extract_segments_from_timestamps(self, min_duration: float = 2.0, max_duration: float = 15.0, start_time_filter: float = 700.0) -> List[Dict]:
        """
        Extract audio segments based on existing timestamp data
        
        Args:
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            start_time_filter: Only consider segments after this time (in seconds)
            
        Returns:
            List of segment dictionaries with audio data and metadata
        """
        segments = []
        
        if self.timestamps is None:
            print("No timestamp data available")
            return segments
        
        print(f"Extracting segments with duration {min_duration}-{max_duration} seconds...")
        
        # Debug: show first few rows
        print("First 5 timestamp entries:")
        print(self.timestamps.head())
        print("Columns:", list(self.timestamps.columns))
        
        valid_segments = 0
        duration_filtered = 0
        text_filtered = 0
        quality_filtered = 0
        
        for idx, row in self.timestamps.iterrows():
            try:
                # Parse timestamps (assuming milliseconds)
                start_ms = int(row['start'])
                end_ms = int(row['end'])
                text = str(row['text']).strip()
                
                # Convert to seconds
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                duration = end_sec - start_sec
                
                # Debug first few entries with real content
                if len(text) > 10 and idx < 10:
                    print(f"Row {idx}: {start_sec:.1f}-{end_sec:.1f}s ({duration:.1f}s) - '{text[:50]}...'")
                
                # Skip segments before LKY starts speaking (around 12 minutes)
                if start_sec < start_time_filter:
                    continue
                
                # Skip empty or placeholder text
                if not text or text == '.' or len(text) < 10:
                    text_filtered += 1
                    continue
                
                # Filter by duration
                if duration < min_duration or duration > max_duration:
                    duration_filtered += 1
                    continue
                
                # Extract audio segment
                start_sample = int(start_sec * self.sr)
                end_sample = int(end_sec * self.sr)
                
                if start_sample >= len(self.audio) or end_sample > len(self.audio):
                    continue
                
                segment_audio = self.audio[start_sample:end_sample]
                
                # Check audio quality (basic silence detection)
                if not self.is_good_quality(segment_audio):
                    quality_filtered += 1
                    continue
                
                valid_segments += 1
                segment_info = {
                    'id': f"lky_segment_{idx:04d}",
                    'start_time': start_sec,
                    'end_time': end_sec,
                    'duration': duration,
                    'text': text,
                    'audio': segment_audio,
                    'filename': f"lky_segment_{idx:04d}.wav"
                }
                segments.append(segment_info)
                    
            except (ValueError, KeyError) as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Filtering results:")
        print(f"  Total entries: {len(self.timestamps)}")
        print(f"  Text filtered: {text_filtered}")
        print(f"  Duration filtered: {duration_filtered}")
        print(f"  Quality filtered: {quality_filtered}")
        print(f"  Valid segments: {valid_segments}")
        
        return segments
    
    def is_good_quality(self, audio_segment: np.ndarray, silence_threshold: float = 0.8, min_energy: float = 0.00001) -> bool:
        """
        Check if audio segment meets quality criteria
        
        Args:
            audio_segment: Audio data as numpy array
            silence_threshold: Maximum allowed silence ratio
            min_energy: Minimum required energy level
            
        Returns:
            True if segment meets quality criteria
        """
        # Check for minimum energy
        rms_energy = np.sqrt(np.mean(audio_segment**2))
        if rms_energy < min_energy:
            return False
        
        # Check silence ratio (more lenient)
        silence_frames = np.sum(np.abs(audio_segment) < 0.005)
        silence_ratio = silence_frames / len(audio_segment)
        if silence_ratio > silence_threshold:
            return False
        
        return True
    
    def save_segments(self, segments: List[Dict]) -> str:
        """
        Save audio segments and create training list file
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            Path to the generated training list file
        """
        training_list = []
        
        print(f"Saving {len(segments)} segments...")
        
        for segment in segments:
            # Save audio file
            audio_path = self.audio_segments_dir / segment['filename']
            sf.write(str(audio_path), segment['audio'], self.sr)
            
            # Add to training list
            training_entry = f"{segment['filename']}|lky|en|{segment['text']}"
            training_list.append(training_entry)
            
            print(f"Saved: {segment['filename']} ({segment['duration']:.1f}s) - {segment['text'][:50]}...")
        
        # Save training list file
        training_list_path = self.output_dir / "lky_training_list.txt"
        with open(training_list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(training_list))
        
        print(f"Training list saved to: {training_list_path}")
        return str(training_list_path)
    
    def run_whisper_transcription(self) -> Dict:
        """
        Use Whisper to transcribe the full audio with timestamps
        
        Returns:
            Whisper transcription result
        """
        print("Running Whisper transcription with timestamps...")
        
        try:
            import whisper
            
            # Load Whisper model
            model = whisper.load_model("base")
            
            # Transcribe with word-level timestamps
            result = model.transcribe(
                str(self.audio_path), 
                word_timestamps=True,
                language="en"
            )
            
            return result
            
        except ImportError:
            print("Whisper not available, using timestamp file only")
            return None
    
    def extract_segments_from_whisper(self, result: Dict, min_duration: float = 2.0, max_duration: float = 10.0) -> List[Dict]:
        """
        Extract segments from Whisper transcription results
        
        Args:
            result: Whisper transcription result
            min_duration: Minimum segment duration
            max_duration: Maximum segment duration
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        if not result:
            return segments
        
        print("Extracting segments from Whisper results...")
        
        for idx, segment in enumerate(result['segments']):
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            text = segment['text'].strip()
            
            # Filter segments
            if duration < min_duration or duration > max_duration:
                continue
            
            if len(text) < 10:  # Skip very short text
                continue
            
            # Extract audio
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            
            if start_sample >= len(self.audio) or end_sample > len(self.audio):
                continue
            
            segment_audio = self.audio[start_sample:end_sample]
            
            if self.is_good_quality(segment_audio):
                segment_info = {
                    'id': f"lky_whisper_{idx:04d}",
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'text': text,
                    'audio': segment_audio,
                    'filename': f"lky_whisper_{idx:04d}.wav"
                }
                segments.append(segment_info)
        
        print(f"Extracted {len(segments)} segments from Whisper")
        return segments


def main():
    """Main function to extract LKY speech segments"""
    
    # Paths
    audio_path = "raw-training-data/LKY-speech-For Third World Leaders Hope or Despair.mp3"
    tsv_path = "raw-training-data/LKY-speech-.tsv"
    output_dir = "GPT-SoVITS/output/lky_training_data_enhanced"
    
    # Initialize extractor
    extractor = LKYSegmentExtractor(audio_path, tsv_path, output_dir)
    
    # Extract segments using existing timestamps
    print("\n=== Phase 1: Extracting segments from timestamp data ===")
    timestamp_segments = extractor.extract_segments_from_timestamps(min_duration=2.0, max_duration=15.0, start_time_filter=700.0)
    
    # Skip Whisper transcription for now (had GPU issues)
    print("\n=== Phase 2: Skipping Whisper transcription (using timestamp data only) ===")
    all_segments = timestamp_segments
    
    print(f"\n=== Total segments found: {len(all_segments)} ===")
    
    if len(all_segments) == 0:
        print("No suitable segments found. Please check the audio file and timestamps.")
        return
    
    # Save best segments for training
    print("\n=== Phase 3: Saving segments for training ===")
    
    # Sort by quality/duration and take best segments
    best_segments = sorted(all_segments, key=lambda x: x['duration'], reverse=True)[:30]
    
    training_list_path = extractor.save_segments(best_segments)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Segments saved: {len(best_segments)}")
    print(f"Output directory: {output_dir}")
    print(f"Training list: {training_list_path}")
    print(f"Ready for GPT-SoVITS training!")


if __name__ == "__main__":
    main()