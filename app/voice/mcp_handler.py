"""
Twilio Media Control Platform (MCP) Handler for Voice AI Agent

This module replaces the TwiML-based approach with Twilio's Media Control Platform
for real-time streaming audio processing.
"""

import logging
import uuid
import os
import asyncio
import json
import time
import traceback
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from fastapi import APIRouter, Request, Response, HTTPException, Form, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import base64
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union

from app.config import (
    SERVICE_PHONE_NUMBERS, SERVICE_CONFIG,
    RESTAURANT_TWILIO_ACCOUNT_SID, RESTAURANT_TWILIO_AUTH_TOKEN,
    HAIRDRESSER_TWILIO_ACCOUNT_SID, HAIRDRESSER_TWILIO_AUTH_TOKEN,
    MCP_MEDIA_STREAM_URL  # Import the MCP URL from config
)
from app.agent.agents_voice import AgentsVoiceManager
from app.storage.gcp import GCPStorage
from app.utils import generate_call_id

router = APIRouter()
logger = logging.getLogger(__name__)

# Dictionary to store active MCP sessions
active_streams = {}

# Dictionary to store active call agents
active_calls = {}

# Dictionary to track when transcripts were last saved
last_saved = {}

# Service detection utility
def detect_service_type(to_number=None, from_number=None):
    """
    Detect which service was called based on the Twilio number that was dialed.
    
    Args:
        to_number: The number that was called (Twilio number)
        from_number: The caller's number (for potential future personalization)
        
    Returns:
        str: The service type ("restaurant" or "hairdresser")
    """
    if to_number in SERVICE_PHONE_NUMBERS:
        service_type = SERVICE_PHONE_NUMBERS[to_number]
        logger.info(f"Detected service type {service_type} from number {to_number}")
        return service_type
    
    # Default to restaurant if we can't determine
    logger.warning(f"Could not determine service type from number {to_number}, defaulting to restaurant")
    return "restaurant"

class StreamParams(BaseModel):
    """Parameters for stream initialization"""
    call_id: str
    call_sid: str
    from_number: str
    to_number: str
    stream_sid: Optional[str] = None
    participant_sid: Optional[str] = None

@router.post("/mcp/incoming", response_class=Response)
async def handle_incoming_call(
    request: Request,
    background_tasks: BackgroundTasks,
    CallSid: str = Form(...),
    From: str = Form(...),
    To: str = Form(...)
):
    """
    Handle incoming calls using MCP via <Connect><Stream> TwiML
    
    This endpoint returns TwiML to start the Media Stream
    """
    logger.info(f"Received incoming call with SID: {CallSid} from {From} to {To}")
    
    # Detect service type
    service_type = detect_service_type(To, From)
    logger.info(f"Call is for service type: {service_type}")
    
    # Create a unique identifier for this call
    call_id = generate_call_id()
    
    # Initialize call data structure
    active_calls[call_id] = {
        "caller_number": From,
        "service_type": service_type,
        "start_time": time.time(),
        "call_sid": CallSid,
        "last_activity": time.time(),
        "interactions": [],  # Store interactions for transcript
        "in_progress": False,  # Flag to track if we're currently processing speech
        "listening_mode": True,  # Start in listening mode
        "audio_buffer": bytearray(),  # Buffer for incoming audio
        "partial_transcript": ""  # Partial transcript for streaming STT
    }
    
    # Start initializing resources in background
    background_tasks.add_task(initialize_call_resources, call_id, service_type)
    
    # --- TwiML Generation ---
    response = VoiceResponse()
    
    # Ensure MCP_MEDIA_STREAM_URL is configured
    if not MCP_MEDIA_STREAM_URL:
        logger.error("MCP_MEDIA_STREAM_URL is not configured in .env file.")
        # Return TwiML with an error message
        response.say("I'm sorry, there is a server configuration error. Please try again later.")
        response.hangup()
        return Response(content=str(response), media_type="application/xml")
        
    logger.info(f"Configuring TwiML <Connect><Stream> to: {MCP_MEDIA_STREAM_URL}")
    
    # Create the <Connect><Stream> TwiML
    connect = Connect()
    stream = Stream(url=MCP_MEDIA_STREAM_URL)
    stream.parameter(name="call_id", value=call_id)
    connect.append(stream)
    response.append(connect)
    
    # Return the TwiML response
    return Response(content=str(response), media_type="application/xml")

@router.websocket("/mcp/stream")
async def handle_media_stream(websocket: WebSocket):
    # Log connection attempt
    logger.info(f"WebSocket connection attempt")
    
    # Accept the connection first
    await websocket.accept()
    logger.info(f"WebSocket connection accepted, waiting for start message")
    
    call_id = None
    
    try:
        # Process messages until we get the call_id or timeout
        start_time = time.time()
        
        while time.time() - start_time < 10.0:  # 10 second timeout to find call_id
            # Wait for messages
            message = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            logger.info(f"Received WebSocket message: {message}")
            
            # Try to extract call_id from parameters
            if "start" in message and "customParameters" in message["start"]:
                # Check if customParameters is a dict with direct key-value pairs
                custom_params = message["start"]["customParameters"]
                if isinstance(custom_params, dict):
                    if "call_id" in custom_params:
                        call_id = custom_params["call_id"]
                else:
                    # Original list of dicts with name/value format
                    for param in custom_params:
                        if param.get("name") == "call_id":
                            call_id = param.get("value")
                            break
            elif "customParameters" in message:
                # Similar check for direct customParameters
                custom_params = message["customParameters"]
                if isinstance(custom_params, dict):
                    if "call_id" in custom_params:
                        call_id = custom_params["call_id"]
                else:
                    for param in custom_params:
                        if param.get("name") == "call_id":
                            call_id = param.get("value")
                            break
            elif "streamSid" in message:
                # Use streamSid as temporary identifier and look up the call
                stream_sid = message["streamSid"]
                logger.info(f"Received streamSid: {stream_sid}, looking for matching call")
                # Just use the most recent call for now
                if active_calls:
                    call_id = list(active_calls.keys())[-1]
                    logger.info(f"Using most recent call ID: {call_id}")
            
            # If we found a call_id, break out of the loop
            if call_id:
                logger.info(f"Found call_id: {call_id}")
                break
                
            # If this is a media message, we need to start processing
            if "media" in message:
                # Just use the most recent call if we still don't have a call_id
                if not call_id and active_calls:
                    call_id = list(active_calls.keys())[-1]
                    logger.info(f"Media received before call_id, using most recent call: {call_id}")
                break
        
        if not call_id:
            # If we still don't have a call_id, we'll try using the most recent call
            if active_calls:
                call_id = list(active_calls.keys())[-1]
                logger.warning(f"No call_id found in messages, using most recent call: {call_id}")
            else:
                logger.error("Missing call_id in WebSocket messages and no active calls")
                await websocket.close(1008, "Missing call_id")
                return
            
        logger.info(f"WebSocket connection established for call {call_id}")
        
        # Store the websocket in active streams
        active_streams[call_id] = websocket
        
        
        # ADDED: Send initial greeting after connection is established
        welcome_message = "Welcome to our voice assistant. How may I help you today?"
        logger.info(f"Sending initial greeting to call {call_id}")
        await stream_tts_audio(call_id, welcome_message)
        logger.info(f"Initial greeting sent, now listening for input")
        
        # Process incoming media in a loop
        while True:
            # Wait for the next message
            message = await websocket.receive_json()
            
            # Handle different message types
            if "media" in message:
                # Process incoming audio data
                media_chunk = base64.b64decode(message["media"]["payload"])
                
                # Add to buffer if we're in listening mode
                if call_id in active_calls and active_calls[call_id].get("listening_mode", False):
                    active_calls[call_id]["audio_buffer"].extend(media_chunk)
                    active_calls[call_id]["last_activity"] = time.time()
                    
                    # Process audio when buffer gets large enough or after silent period
                    if len(active_calls[call_id]["audio_buffer"]) >= 8000:  # 1 second of audio at 8kHz
                        await process_audio_chunk(call_id)
            
            elif "mark" in message:
                # Handle mark messages (silence detection)
                logger.debug(f"Received mark message: {message}")
                
                # Process accumulated audio when silence is detected
                if call_id in active_calls and len(active_calls[call_id].get("audio_buffer", b"")) > 0:
                    await process_audio_chunk(call_id, is_end_of_speech=True)
            
            elif "stop" in message:
                # Stream is stopping
                logger.info(f"Received stop message for call {call_id}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for call {call_id if 'call_id' in locals() else 'unknown'}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout waiting for WebSocket start message")
        await websocket.close(1008)
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up
        if 'call_id' in locals() and call_id:
            if call_id in active_streams:
                del active_streams[call_id]

async def process_audio_chunk(call_id, is_end_of_speech=False):
    """
    Process a chunk of audio from the caller
    
    Args:
        call_id: The call ID
        is_end_of_speech: Whether this chunk marks the end of speech
    """
    if call_id not in active_calls:
        return
        
    call_data = active_calls[call_id]
    
    # Don't process if we're already processing speech or not in listening mode
    if call_data.get("in_progress", False) or not call_data.get("listening_mode", True):
        return
        
    # Get the audio buffer
    audio_buffer = call_data.get("audio_buffer", bytearray())
    
    # Skip if buffer is empty
    if len(audio_buffer) == 0:
        return
        
    # Mark as in progress
    call_data["in_progress"] = True
    
    try:
        # Convert from mu-law to linear PCM
        pcm_audio = convert_mulaw_to_pcm(audio_buffer)
        
        # Reset buffer
        call_data["audio_buffer"] = bytearray()
        
        # Check if agent is initialized
        if "agent" not in call_data:
            # Wait for agent initialization
            for _ in range(10):  # Try for up to 10 seconds
                await asyncio.sleep(1)
                if "agent" in call_data:
                    break
            else:
                # Agent still not initialized
                logger.warning(f"Agent not initialized for call {call_id} after 10 seconds")
                if call_id in active_streams:
                    await stream_tts_audio(call_id, "I'm getting ready. Please wait a moment.")
                call_data["in_progress"] = False
                return
        
        # Transcribe audio
        agent = call_data["agent"]
        transcription = await transcribe_audio(pcm_audio)
        
        if not transcription:
            # No speech detected
            call_data["in_progress"] = False
            return
            
        logger.info(f"Transcribed: {transcription}")
        
        # Check for call termination commands
        if any(word in transcription.lower() for word in ["goodbye", "end call", "hang up", "bye"]):
            await stream_tts_audio(call_id, f"Thank you for calling. Goodbye!")
            await handle_call_end(call_id)
            return
            
        # Process with AI agent
        ai_response, conversation_state = await agent.process_text_input(transcription)
        
        # Record the interaction
        timestamp = time.time()
        call_data["interactions"].append({
            "user": transcription,
            "agent": ai_response,
            "timestamp": timestamp,
            "topics": conversation_state.get("topics", []),
            "sentiment": conversation_state.get("sentiment", "neutral")
        })
        
        # Update call data with conversation state info
        call_data["sentiment"] = conversation_state.get("sentiment", call_data.get("sentiment", "neutral"))
        call_data["topics"] = conversation_state.get("topics", call_data.get("topics", []))
        
        # Switch to speaking mode
        call_data["listening_mode"] = False
        
        # Send response to caller via TTS
        await stream_tts_audio(call_id, ai_response)
        
        # Switch back to listening mode
        call_data["listening_mode"] = True
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Reset progress flag
        call_data["in_progress"] = False

async def transcribe_audio(audio_data):
    """
    Transcribe audio using OpenAI Whisper API
    
    Args:
        audio_data: Raw PCM audio data
        
    Returns:
        str: Transcribed text
    """
    # Mock implementation - in a real scenario, you would use OpenAI Whisper API
    # or another speech-to-text service that supports streaming
    
    # For now, we'll simulate a transcription
    # This would be replaced with actual API calls in production
    mock_responses = [
        "I'd like to make a reservation for tomorrow night.",
        "Do you have availability for four people at 7:00 PM?",
        "That sounds good. My name is Michael.",
        "Yes, that's correct. My phone number is 555-123-4567.",
        "Thank you. I'm looking forward to it. Goodbye."
    ]
    import random
    
    # Simulate some processing time for realism
    await asyncio.sleep(0.5)
    
    # 20% chance of returning empty result (simulating silence)
    if random.random() < 0.2:
        return ""
        
    return random.choice(mock_responses)

async def stream_tts_audio(call_id, text):
    """Stream text-to-speech audio to the caller"""
    if call_id not in active_calls or call_id not in active_streams:
        logger.error(f"Call {call_id} not found in active calls or streams")
        return
    
    try:
        # Get the websocket
        websocket = active_streams[call_id]
        
        logger.info(f"Streaming TTS response for: {text}")
        
        # Generate a simple tone sequence (much better than random bytes)
        chunk_size = 1600  # 200ms of audio at 8kHz
        num_chunks = max(5, len(text) // 10)  # Rough audio length estimate
        
        # Send start mark
        await websocket.send_json({
            "event": "mark",
            "streamSid": call_id,  # Use call_id as stream SID
            "mark": {
                "name": "start_speech"
            }
        })
        
        # Send audio chunks (simple tone - better than random bytes)
        for i in range(num_chunks):
            # Create a simple tone (sine wave encoded in mu-law)
            import math
            audio_chunk = bytearray(chunk_size)
            for j in range(chunk_size):
                # Simple tone with some variation
                value = int(127 * math.sin(j * (0.1 + i * 0.01))) + 128
                audio_chunk[j] = value
                
            # Send as media
            await websocket.send_json({
                "event": "media",
                "streamSid": call_id,
                "media": {
                    "payload": base64.b64encode(audio_chunk).decode('utf-8')
                }
            })
            
            # Add timing between chunks
            await asyncio.sleep(0.2)
            
        # Send end mark
        await websocket.send_json({
            "event": "mark",
            "streamSid": call_id,
            "mark": {
                "name": "end_speech"
            }
        })
        
        logger.info(f"Finished streaming TTS response")
        
    except Exception as e:
        logger.error(f"Error streaming TTS audio: {str(e)}")
        logger.error(traceback.format_exc())

async def initialize_call_resources(call_id, service_type):
    """Initialize AI agent and storage resources asynchronously"""
    try:
        # Only initialize if not already initialized
        if call_id in active_calls and "agent" not in active_calls[call_id]:
            logger.info(f"Initializing resources for call {call_id} with service type {service_type}")
            
            # First initialize storage
            storage = GCPStorage(call_id)
            active_calls[call_id]["storage"] = storage
            
            # Now initialize the agent with the service type using Agents SDK
            active_calls[call_id]["agent"] = AgentsVoiceManager(service_type=service_type)
            
            logger.info(f"Finished initializing resources for call {call_id}")
    except Exception as e:
        logger.error(f"Error initializing call resources: {str(e)}")
        logger.error(traceback.format_exc())

@router.post("/mcp/status_callback", response_class=PlainTextResponse)
async def mcp_status_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    StreamSid: str = Form(None)
):
    """Handle call status callbacks from Twilio MCP"""
    logger.info(f"MCP status callback for call SID {CallSid}: {CallStatus}, Stream SID: {StreamSid}")
    
    # Find the call_id associated with this CallSid
    call_id = None
    for cid, data in active_calls.items():
        if data.get("call_sid") == CallSid:
            call_id = cid
            service_type = data.get("service_type", "restaurant")
            break
    
    if call_id and CallStatus in ["completed", "failed", "busy", "no-answer"]:
        # Call has ended
        await handle_call_end(call_id, background_tasks)
    
    return PlainTextResponse("OK")

async def handle_call_end(call_id, background_tasks=None):
    """
    Handle the end of a call
    
    Args:
        call_id: The call ID
        background_tasks: Optional BackgroundTasks object
    """
    try:
        if call_id not in active_calls:
            return
            
        logger.info(f"Handling end of call {call_id}")
        
        # Save the call data
        if "storage" in active_calls[call_id]:
            # Get the call data
            call_data = active_calls[call_id]
            call_sid = call_data.get("call_sid")
            
            # Save in background or foreground based on context
            if background_tasks:
                background_tasks.add_task(background_save_call_data, call_id, call_sid, True)
            else:
                await background_save_call_data(call_id, call_sid, True)
        
        # Close the WebSocket if it's still open
        if call_id in active_streams:
            try:
                await active_streams[call_id].close(1000)
            except:
                pass
            del active_streams[call_id]
        
        # Clean up after 5 seconds to ensure background tasks complete
        await asyncio.sleep(5)
        if call_id in active_calls:
            del active_calls[call_id]
            logger.info(f"Cleaned up call {call_id} from active calls")
        
    except Exception as e:
        logger.error(f"Error handling call end: {str(e)}")
        logger.error(traceback.format_exc())

async def background_save_call_data(call_id, call_sid, force_save=False):
    """Save all call data to storage in background."""
    try:
        if call_id not in active_calls:
            logger.error(f"Call ID {call_id} not found in active_calls when trying to save data")
            return
        
        # Check if we need to save based on timing
        current_time = time.time()
        if not force_save and call_id in last_saved:
            # Don't save if less than 60 seconds since last save
            if current_time - last_saved[call_id] < 60:
                logger.info(f"Skipping save for call {call_id} - saved recently")
                return
        
        call_data = active_calls[call_id]
        service_type = call_data.get("service_type", "restaurant")
        
        # Check if agent exists
        if "agent" not in call_data or "storage" not in call_data:
            logger.error(f"Agent or storage not initialized for call ID {call_id}")
            return
            
        storage = call_data["storage"]
        
        # Get the interactions
        transcript = call_data.get("interactions", [])
        
        if not transcript:
            logger.warning(f"No transcript data available for call {call_id}")
            return
            
        # Format transcript for storage
        formatted_transcript = []
        for interaction in transcript:
            formatted_transcript.append({
                "role": "user",
                "content": interaction.get("user", ""),
                "created_at": interaction.get("timestamp", time.time())
            })
            formatted_transcript.append({
                "role": "assistant",
                "content": interaction.get("agent", ""),
                "created_at": interaction.get("timestamp", time.time())
            })
            
        # Add metadata
        metadata = {
            "call_id": call_id,
            "call_sid": call_sid,
            "caller_number": call_data.get("caller_number"),
            "service_type": service_type,
            "start_time": call_data.get("start_time"),
            "sentiment": call_data.get("sentiment", "neutral"),
            "topics": call_data.get("topics", []),
            "save_time": time.time()
        }
        
        # Save the transcript with metadata to GCP
        try:
            logger.info(f"Saving transcript for call ID: {call_id} (Service: {service_type})")
            save_result = await storage.save_transcript_with_metadata(formatted_transcript, metadata)
            if save_result:
                logger.info(f"Successfully saved transcript for call ID: {call_id} to {save_result}")
                # Update the last saved timestamp
                last_saved[call_id] = current_time
            else:
                logger.error(f"Failed to save transcript for call ID: {call_id}")
        except Exception as e:
            logger.error(f"Error saving transcript: {str(e)}")
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error in background save: {str(e)}")
        logger.error(traceback.format_exc())

def convert_mulaw_to_pcm(mulaw_data):
    """
    Convert mu-law encoded audio to PCM
    
    Args:
        mulaw_data: mu-law encoded audio
        
    Returns:
        bytes: PCM audio data
    """
    # This is a simple implementation - in production, you would use a proper audio conversion library
    # like SoX, FFmpeg, or a Python library like scipy or librosa
    
    # For this example, we'll just return the input data
    # In a real implementation, you would perform actual conversion
    return bytes(mulaw_data)