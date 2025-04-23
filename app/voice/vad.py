"""
Voice Activity Detection (VAD) Module

This module provides real-time voice activity detection to determine
when a person is speaking versus when there is silence.
"""

import logging
import numpy as np
from typing import List, Tuple
import time
import collections

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """Real-time voice activity detection for audio streams"""
    
    def __init__(self, 
                 sample_rate=8000, 
                 frame_duration=0.02,  # 20ms frames
                 threshold=0.02,
                 min_speech_duration=0.3,  # 300ms minimum for speech segment
                 min_silence_duration=0.5):  # 500ms of silence to consider speech ended
        """
        Initialize the VAD
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration: Duration of each frame in seconds
            threshold: Energy threshold for speech detection (0.0 to 1.0)
            min_speech_duration: Minimum duration of speech in seconds
            min_silence_duration: Minimum duration of silence to consider speech ended
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.threshold = threshold
        self.min_speech_frames = int(min_speech_duration / frame_duration)
        self.min_silence_frames = int(min_silence_duration / frame_duration)
        
        # State variables
        self.is_speech = False
        self.speech_start_time = 0
        self.energy_history = collections.deque(maxlen=100)  # About 2 seconds of history
        self.silence_counter = 0
        self.speech_counter = 0
        
        logger.info(f"Initialized VAD with threshold {threshold}, min_speech_frames={self.min_speech_frames}, min_silence_frames={self.min_silence_frames}")
    
    def process_frame(self, frame_data: bytes) -> Tuple[bool, bool, float]:
        """
        Process a single frame of audio
        
        Args:
            frame_data: Audio data in bytes (PCM)
            
        Returns:
            Tuple[bool, bool, float]: 
                - is_speech: Current frame contains speech
                - speech_end: Speech segment has ended
                - energy: Frame energy
        """
        # Convert bytes to numpy array (assuming 16-bit PCM)
        try:
            audio = np.frombuffer(frame_data, dtype=np.int16)
        except:
            # If conversion fails, assume it's 8-bit mu-law
            audio = np.frombuffer(frame_data, dtype=np.uint8)
            # Convert to float and normalize
            audio = (audio.astype(np.float32) - 128) / 128.0
            # Scale to 16-bit range
            audio = audio * 32767
        
        # Calculate energy
        energy = np.mean(np.abs(audio)) / 32767.0  # Normalize to 0.0-1.0
        
        # Add to history
        self.energy_history.append(energy)
        
        # Determine speech/silence
        frame_is_speech = energy > self.threshold
        speech_end = False
        
        # State machine
        if self.is_speech:
            if frame_is_speech:
                # Continue speech
                self.speech_counter += 1
                self.silence_counter = 0
            else:
                # Potential end of speech
                self.silence_counter += 1
                
                # Check if silence has been long enough to end speech
                if self.silence_counter >= self.min_silence_frames:
                    self.is_speech = False
                    speech_end = True
                    self.speech_counter = 0
                    logger.debug(f"End of speech detected after {self.silence_counter * self.frame_duration:.2f}s of silence")
        else:
            if frame_is_speech:
                # Potential start of speech
                self.speech_counter += 1
                
                # Check if speech has been long enough to start tracking
                if self.speech_counter >= self.min_speech_frames:
                    self.is_speech = True
                    self.speech_start_time = time.time()
                    self.silence_counter = 0
                    logger.debug(f"Start of speech detected with energy {energy:.4f}")
            else:
                # Continue silence
                self.speech_counter = 0
        
        return self.is_speech, speech_end, energy
    
    def reset(self):
        """Reset the detector state"""
        self.is_speech = False
        self.speech_start_time = 0
        self.energy_history.clear()
        self.silence_counter = 0
        self.speech_counter = 0
    
    def get_speech_duration(self) -> float:
        """
        Get the duration of the current speech segment
        
        Returns:
            float: Duration in seconds
        """
        if not self.is_speech or self.speech_start_time == 0:
            return 0.0
            
        return time.time() - self.speech_start_time
    
    def adapt_threshold(self):
        """Dynamically adapt the threshold based on audio history"""
        if len(self.energy_history) < 50:  # Need enough history
            return
            
        # Calculate statistics
        energy_array = np.array(self.energy_history)
        mean_energy = np.mean(energy_array)
        std_energy = np.std(energy_array)
        
        # Set threshold to mean + 1.5 * std (can be tuned)
        new_threshold = mean_energy + 1.5 * std_energy
        
        # Clamp to reasonable values
        new_threshold = max(0.01, min(0.1, new_threshold))
        
        # Only update if significantly different
        if abs(new_threshold - self.threshold) > 0.005:
            logger.info(f"Adapting VAD threshold: {self.threshold:.4f} -> {new_threshold:.4f}")
            self.threshold = new_threshold