"""
OpenAI Audio Processing Module for Media Streams

This module handles speech-to-text and text-to-speech processing
using OpenAI's APIs in a streaming fashion.
"""

import logging
import asyncio
import io
import time
import traceback
import wave
import numpy as np
from typing import Dict, List, Any, Optional, Union, Generator, AsyncGenerator
import httpx
import json
from tempfile import NamedTemporaryFile
import base64

from app.config import (
    OPENAI_API_KEY,
    STT_MODEL,
    STT_LANGUAGE,
    TTS_MODEL,
    TTS_PROVIDER,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
)

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing for real-time STT and TTS"""
    
    def __init__(self, openai_api_key=None):
        """
        Initialize the audio processor
        
        Args:
            openai_api_key: OpenAI API key (defaults to config)
        """
        self.api_key = openai_api_key or OPENAI_API_KEY
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info("Initialized AudioProcessor")
        
    async def transcribe_audio(self, audio_data: bytes, language: str = STT_LANGUAGE) -> str:
        """
        Transcribe audio using OpenAI Whisper API
        
        Args:
            audio_data: Raw audio data (PCM or mu-law)
            language: Language code (e.g., "en", "es")
            
        Returns:
            str: Transcribed text
        """
        try:
            # If we got mu-law data, convert it to PCM first
            if AUDIO_FORMAT == "mulaw":
                audio_data = self._convert_mulaw_to_pcm(audio_data)
            
            # Create a temporary WAV file
            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write WAV header and data
                with wave.open(temp_path, "wb") as wav_file:
                    wav_file.setnchannels(AUDIO_CHANNELS)
                    wav_file.setsampwidth(2)  # 16-bit PCM
                    wav_file.setframerate(AUDIO_SAMPLE_RATE)
                    wav_file.writeframes(audio_data)
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Read the WAV file
            with open(temp_path, "rb") as audio_file:
                files = {
                    "file": ("audio.wav", audio_file, "audio/wav"),
                    "model": (None, STT_MODEL),
                    "language": (None, language),
                    "response_format": (None, "text")
                }
                
                # Send the request to OpenAI API
                url = "https://api.openai.com/v1/audio/transcriptions"
                response = await self.client.post(url, headers=headers, files=files)
                
                if response.status_code != 200:
                    logger.error(f"Error transcribing audio: {response.status_code} {response.text}")
                    return ""
                
                # Parse the response
                text = response.text.strip()
                return text
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
        finally:
            # Clean up
            try:
                import os
                os.remove(temp_path)
            except:
                pass
    
    async def stream_transcribe_audio(self, audio_chunk_generator: AsyncGenerator[bytes, None], language: str = STT_LANGUAGE) -> AsyncGenerator[str, None]:
        """
        Stream audio to OpenAI Whisper API for real-time transcription
        
        Args:
            audio_chunk_generator: Async generator yielding audio chunks
            language: Language code (e.g., "en", "es")
            
        Yields:
            str: Partial transcriptions as they become available
        """
        # Note: As of my knowledge cutoff, OpenAI doesn't have a streaming whisper API
        # This is a simulated implementation that processes chunks as they arrive
        # In a real implementation, you would use a service that supports streaming STT
        
        buffer = bytearray()
        silence_counter = 0
        
        async for chunk in audio_chunk_generator:
            # If we got mu-law data, convert it to PCM first
            if AUDIO_FORMAT == "mulaw":
                chunk = self._convert_mulaw_to_pcm(chunk)
            
            # Add to buffer
            buffer.extend(chunk)
            
            # Check if we have enough audio for transcription (at least 1 second)
            if len(buffer) >= AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * 2:  # 2 bytes per sample for 16-bit PCM
                # Check for silence
                is_silence = self._is_silence(chunk)
                
                if is_silence:
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # If we have silence for a certain duration or buffer is large enough, transcribe
                if silence_counter >= 3 or len(buffer) >= AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * 2 * 5:  # 5 seconds max
                    # Transcribe the buffer
                    transcription = await self.transcribe_audio(bytes(buffer), language)
                    
                    # Clear the buffer
                    buffer = bytearray()
                    silence_counter = 0
                    
                    # Yield the transcription
                    if transcription:
                        yield transcription
    
    async def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """
        Convert text to speech using OpenAI TTS API
        
        Args:
            text: Text to synthesize
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            
        Returns:
            bytes: Audio data in MP3 format
        """
        try:
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the request body
            data = {
                "model": TTS_MODEL,
                "input": text,
                "voice": voice
            }
            
            # Send the request to OpenAI API
            url = "https://api.openai.com/v1/audio/speech"
            response = await self.client.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                logger.error(f"Error synthesizing speech: {response.status_code} {response.text}")
                return b""
            
            # Return the audio data
            return response.content
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            logger.error(traceback.format_exc())
            return b""
    
    async def stream_text_to_speech(self, text: str, voice: str = "alloy", chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
        """
        Stream text to speech using OpenAI TTS API
        
        Args:
            text: Text to synthesize
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            chunk_size: Size of audio chunks to yield
            
        Yields:
            bytes: Audio data chunks in MP3 format
        """
        try:
            # Get the full audio
            audio_data = await self.text_to_speech(text, voice)
            
            # Stream it in chunks
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i+chunk_size]
                await asyncio.sleep(0.1)  # Small delay to simulate streaming
                
        except Exception as e:
            logger.error(f"Error streaming speech: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def mp3_to_mulaw(self, mp3_data: bytes) -> AsyncGenerator[bytes, None]:
        """
        Convert MP3 audio to mu-law format for Twilio Media Streams
        
        Args:
            mp3_data: Audio data in MP3 format
            
        Yields:
            bytes: Audio data chunks in mu-law format
        """
        try:
            # In a real implementation, you would use a library like pydub, librosa, or ffmpeg
            # to convert MP3 to PCM and then to mu-law
            # For now, we'll simulate the conversion with random data
            
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Estimate chunks based on MP3 size (very rough approximation)
            # In a real implementation, you would decode the MP3 and convert correctly
            estimated_duration = len(mp3_data) / 16000  # Very rough estimate
            chunks = int(estimated_duration / 0.02)  # 20ms chunks
            
            for _ in range(chunks):
                # Generate simulated mu-law data (160 bytes = 20ms at 8kHz)
                yield bytes([128] * 160)
                await asyncio.sleep(0.02)  # Wait to simulate real-time streaming
                
        except Exception as e:
            logger.error(f"Error converting MP3 to mu-law: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _convert_mulaw_to_pcm(self, mulaw_data: bytes) -> bytes:
        """
        Convert mu-law audio to PCM
        
        Args:
            mulaw_data: Audio data in mu-law format
            
        Returns:
            bytes: Audio data in PCM format
        """
        # In a real implementation, you would use a library like scipy or numpy
        # to perform proper mu-law to PCM conversion
        # For now, we'll return a simple expansion
        
        try:
            # Convert to numpy array
            mulaw_array = np.frombuffer(mulaw_data, dtype=np.uint8)
            
            # Simple mu-law expansion (not accurate but illustrative)
            # In production, use proper audio libraries
            pcm_array = (mulaw_array.astype(np.float32) - 128) / 128.0
            pcm_array = pcm_array * 32767  # Scale to 16-bit range
            
            # Convert to 16-bit PCM
            pcm_data = pcm_array.astype(np.int16).tobytes()
            
            return pcm_data
        except Exception as e:
            logger.error(f"Error converting mu-law to PCM: {str(e)}")
            logger.error(traceback.format_exc())
            # Return original data if conversion fails
            return mulaw_data
    
    def _is_silence(self, audio_data: bytes, threshold: float = 0.03) -> bool:
        """
        Check if audio chunk is silence
        
        Args:
            audio_data: Audio data (PCM)
            threshold: Silence threshold (0.0 to 1.0)
            
        Returns:
            bool: True if the audio is silence
        """
        try:
            # Convert to numpy array (assuming 16-bit PCM)
            pcm_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS amplitude
            rms = np.sqrt(np.mean(pcm_array.astype(np.float32) ** 2))
            
            # Normalize
            normalized_rms = rms / 32768.0  # 16-bit max value
            
            # Check if below threshold
            return normalized_rms < threshold
        except Exception as e:
            logger.error(f"Error detecting silence: {str(e)}")
            logger.error(traceback.format_exc())
            return False