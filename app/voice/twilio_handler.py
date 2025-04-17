import logging
import uuid
import os
import asyncio
import json
import time
import traceback
from fastapi import APIRouter, Request, Response, HTTPException, Form, Depends, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse, Gather, Say
from twilio.rest import Client
import numpy as np

from app.config import (
    SERVICE_PHONE_NUMBERS, SERVICE_CONFIG,
    RESTAURANT_TWILIO_ACCOUNT_SID, RESTAURANT_TWILIO_AUTH_TOKEN,
    HAIRDRESSER_TWILIO_ACCOUNT_SID, HAIRDRESSER_TWILIO_AUTH_TOKEN
)
from app.agent.agents_voice import AgentsVoiceManager
from app.storage.gcp import GCPStorage
from app.utils import generate_call_id

router = APIRouter()
logger = logging.getLogger(__name__)

# Dictionary to store Twilio clients for each service
twilio_clients = {
    "restaurant": Client(RESTAURANT_TWILIO_ACCOUNT_SID, RESTAURANT_TWILIO_AUTH_TOKEN),
    "hairdresser": Client(HAIRDRESSER_TWILIO_ACCOUNT_SID, HAIRDRESSER_TWILIO_AUTH_TOKEN)
}

# Dictionary to store active call agents
active_calls = {}

# Dictionary to track when transcripts were last saved
last_saved = {}

# Common responses for quick replies (to reduce AI waiting time)
QUICK_RESPONSES = {
    "restaurant": {
        "greeting": "Hello! Thank you for calling Bella Cucina Restaurant. How can I help you today?",
        "not_heard": "I'm sorry, I didn't catch that. Could you please repeat?",
        "thinking": "I'm thinking about that...",
        "goodbye": "Thank you for calling Bella Cucina Restaurant. Goodbye!",
        "timeout": "I haven't heard from you in a while. Thank you for calling. Goodbye!",
        "anything_else": "Is there anything else I can help you with regarding your reservation?",
        "processing": "Let me process that for you."
    },
    "hairdresser": {
        "greeting": "Hello! Welcome to Style Studio Salon. How may I assist you with your appointment today?",
        "not_heard": "I'm sorry, I didn't quite catch that. Could you please repeat?",
        "thinking": "I'm just thinking about that...",
        "goodbye": "Thank you for calling Style Studio Salon. Have a wonderful day!",
        "timeout": "I haven't heard from you in a while. Thank you for calling Style Studio Salon. Goodbye!",
        "anything_else": "Is there anything else I can help you with regarding your appointment?",
        "processing": "I'm processing that information for you."
    }
}

# Enhanced voice settings
VOICE_OPTIONS = {
    "restaurant": {
        "voice": "Polly.Joanna",  # Amazon Polly voice
        "language": "en-US",
        "speed": "1.0",  # Normal speed for more natural feeling
        "pause_words": [".", ",", "?", "!"],
        "pause_lengths": {
            ".": 500,  # Slightly shorter pauses for more natural conversation
            ",": 200,
            "?": 500,
            "!": 500
        }
    },
    "hairdresser": {
        "voice": "Polly.Amy",  # British English voice
        "language": "en-GB",
        "speed": "1.0",
        "pause_words": [".", ",", "?", "!"],
        "pause_lengths": {
            ".": 500,
            ",": 200,
            "?": 500,
            "!": 500
        }
    }
}

# Pre-enhance common responses for instant availability
ENHANCED_QUICK_RESPONSES = {}

def enhance_speech_output(text, voice_config=None):
    if not voice_config:
        voice_config = VOICE_OPTIONS["restaurant"]
    
    # Add pauses based on punctuation
    for punct in voice_config["pause_words"]:
        if punct in voice_config["pause_lengths"]:
            pause_time = voice_config["pause_lengths"][punct]
            text = text.replace(punct, f'{punct}<break time="{pause_time}ms"/>')
    
    # Wrap in SSML tags
    ssml = f'<speak><prosody rate="{voice_config["speed"]}">{text}</prosody></speak>'
    return ssml

def setup_enhanced_responses():
    """Pre-compute enhanced SSML for quick responses"""
    for service_type, responses in QUICK_RESPONSES.items():
        ENHANCED_QUICK_RESPONSES[service_type] = {}
        for key, text in responses.items():
            voice_config = VOICE_OPTIONS.get(service_type, VOICE_OPTIONS["restaurant"])
            ENHANCED_QUICK_RESPONSES[service_type][key] = enhance_speech_output(text, voice_config)

# Call this at startup (AFTER the functions are defined)
setup_enhanced_responses()

def get_quick_response_ssml(key, service_type="restaurant"):
    """Get a pre-enhanced quick response with SSML
    
    Args:
        key: The key for the quick response
        service_type: The type of service (restaurant or hairdresser)
    """
    # Use the service type to look up the configuration
    voice_config = VOICE_OPTIONS.get(service_type, VOICE_OPTIONS["restaurant"])
    
    if (service_type in ENHANCED_QUICK_RESPONSES and 
        key in ENHANCED_QUICK_RESPONSES[service_type]):
        return ENHANCED_QUICK_RESPONSES[service_type][key]
    
    # Fallback if not found
    text = QUICK_RESPONSES.get(service_type, QUICK_RESPONSES["restaurant"]).get(key, "I'm listening.")
    return enhance_speech_output(text, voice_config)

def create_ssml_say(text, voice_config, is_ssml=False):
    """Helper to create a Say verb with the right voice settings"""
    if not is_ssml:
        text = enhance_speech_output(text, voice_config)
        is_ssml = True
    
    say = Say(
        message=text,
        voice=voice_config["voice"],
        language=voice_config["language"]
    )
    say.ssml = is_ssml
    return say

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
            
        agent = call_data["agent"]
        storage = call_data["storage"]
        
        # Get the full transcript
        # Note: With Agents SDK, we need to track conversations differently
        # We'll use the interactions list from call_data
        transcript = call_data.get("interactions", [])
        
        if not transcript:
            logger.warning(f"No transcript data available for call {call_id}")
            return
            
        # Format transcript in a similar way to what we had before
        formatted_transcript = []
        for interaction in transcript:
            formatted_transcript.append({
                "role": "user",
                "content": interaction.get("user", ""),
                "created_at": interaction.get("timestamp", datetime.now().isoformat() if datetime else time.time())
            })
            formatted_transcript.append({
                "role": "assistant",
                "content": interaction.get("agent", ""),
                "created_at": interaction.get("timestamp", datetime.now().isoformat() if datetime else time.time())
            })
            
        # Add metadata
        metadata = {
            "call_id": call_id,
            "call_sid": call_sid,
            "caller_number": call_data.get("caller_number"),
            "service_type": service_type,
            "start_time": call_data.get("start_time"),
            "voice_config": {
                "voice": call_data.get("voice_config", {}).get("voice"),
                "language": call_data.get("voice_config", {}).get("language")
            },
            "sentiment": call_data.get("sentiment", "neutral"),
            "topics": call_data.get("topics", []),
            "save_time": datetime.now().isoformat() if datetime else time.time()
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
        
        # If we have a recording, save it to GCP
        if "recording_sid" in call_data:
            try:
                recording_sid = call_data["recording_sid"]
                twilio_client = twilio_clients.get(service_type, twilio_clients["restaurant"])
                max_attempts = 5 if force_save else 2
                wait_time_base = 3

                for attempt in range(max_attempts):
                    wait_time = wait_time_base * (attempt + 1)
                    logger.info(f"Attempt {attempt+1}/{max_attempts} to save recording {recording_sid}, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)

                    try:
                        recording = twilio_client.recordings(recording_sid).fetch()

                        if recording.status == "completed":
                            account_sid = SERVICE_CONFIG[service_type]["account_sid"]
                            recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.mp3"
                            logger.info(f"Recording {recording_sid} is ready, saving to GCS...")
                            # Pass the metadata dictionary here
                            audio_result = await storage.save_audio(recording_url, metadata) # Pass metadata

                            if audio_result:
                                logger.info(f"Successfully saved audio for call ID: {call_id} to {audio_result}")
                                call_data["recording_saved"] = True
                                return
                            else:
                                logger.error(f"Failed to save audio for call ID: {call_id}")
                        else:
                            logger.warning(f"Recording not completed yet for SID {recording_sid}, status: {recording.status}")
                    except Exception as inner_e:
                        # Modify the error message here to be more specific
                        logger.error(f"Error fetching/saving recording {recording_sid}: {str(inner_e)}")
                        logger.error(traceback.format_exc())

                logger.error(f"Failed to save recording after {max_attempts} attempts")
            except Exception as e:
                logger.error(f"Error saving audio process: {str(e)}") # Clarify error source
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error in background save: {str(e)}")
        logger.error(traceback.format_exc())

@router.post("/voice", response_class=PlainTextResponse)
async def voice_webhook(
    request: Request, 
    background_tasks: BackgroundTasks,
    CallSid: str = Form(None), 
    From: str = Form(None),
    To: str = Form(None)
):
    """Handle incoming Twilio voice calls."""
    start_time = time.time()
    logger.info(f"Received voice call with SID: {CallSid} from {From} to {To}")
    
    # Detect which service this call is for
    service_type = detect_service_type(To, From)
    logger.info(f"Call is for service type: {service_type}")
    
    # Create a TwiML response
    response = VoiceResponse()
    
    # Create a unique identifier for this call
    call_id = CallSid or str(uuid.uuid4())
    
    # Initialize an AgentsVoiceManager for this call if it doesn't exist
    if call_id not in active_calls:
        # Select voice configuration for the service
        voice_config = VOICE_OPTIONS.get(service_type, VOICE_OPTIONS["restaurant"])
        
        # Only initialize the agent after responding to the user
        # This avoids delay in the initial greeting
        active_calls[call_id] = {
            "caller_number": From,
            "service_type": service_type,
            "voice_config": voice_config,
            "start_time": time.time(),
            "call_sid": CallSid,
            "last_activity": time.time(),
            "interactions": []  # Store interactions for transcript
        }
        
        # Use pre-enhanced greeting for faster response
        greeting_ssml = get_quick_response_ssml("greeting", service_type)
        
        # Say the greeting with enhanced voice
        say = Say(
            message=greeting_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # Initialize AgentsVoiceManager and Storage in background to avoid initial delay
        background_tasks.add_task(initialize_call_resources, call_id, voice_config, service_type)
    else:
        # Call already exists - ensure agent is initialized
        if "agent" not in active_calls[call_id]:
            # Add a brief delay message while agent initializes
            service_type = active_calls[call_id].get("service_type", "restaurant")
            voice_config = active_calls[call_id]["voice_config"]
            processing_ssml = get_quick_response_ssml("processing", service_type)
            
            say = Say(
                message=processing_ssml,
                voice=voice_config["voice"],
                language=voice_config["language"]
            )
            say.ssml = True
            response.append(say)
            
            # Initialize in background if not done already
            background_tasks.add_task(initialize_call_resources, call_id, voice_config, service_type)
    
    # Gather speech input from the user
    gather = Gather(
        input='speech',
        action=f'/twilio/process_speech?call_id={call_id}&service_type={service_type}',
        method='POST',
        speech_timeout='auto',
        language='en-US',
        hints='help, goodbye, thank you, yes, no',
        profanity_filter=False,  # Allows more natural speech
        enhanced=True  # Better speech recognition
    )
    
    # Add the gather to the response
    response.append(gather)
    
    # If no input received, redirect to the same endpoint
    response.redirect(f'/twilio/voice?CallSid={CallSid}')
    
    processing_time = time.time() - start_time
    logger.info(f"Voice webhook processed in {processing_time:.3f} seconds")
    
    return Response(content=str(response), media_type="application/xml")

async def initialize_call_resources(call_id, voice_config, service_type):
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

@router.post("/process_speech", response_class=PlainTextResponse)
async def process_speech(
    request: Request,
    background_tasks: BackgroundTasks,
    call_id: str,
    service_type: str = Query("restaurant"),
    SpeechResult: str = Form(None),
    CallSid: str = Form(None),
    Confidence: str = Form(None)
):
    """Process speech input from Twilio with optimized response time."""
    start_time = time.time()
    logger.info(f"Processing speech for call ID: {call_id} (Service: {service_type})")
    
    # Parse confidence to float
    confidence_value = 0.0
    try:
        if Confidence:
            confidence_value = float(Confidence)
    except ValueError:
        logger.warning(f"Could not parse confidence value: {Confidence}")
    
    logger.info(f"Speech result: {SpeechResult} (Confidence: {confidence_value})")
    
    # Create a TwiML response
    response = VoiceResponse()
    
    # Check if we have data for this call
    if call_id not in active_calls:
        logger.error(f"No active call found for ID: {call_id}")
        response.say("I'm sorry, there was an error processing your request.")
        return Response(content=str(response), media_type="application/xml")
    
    # Get the call data
    call_data = active_calls[call_id]
    stored_service_type = call_data.get("service_type", "restaurant")
    
    # Use the stored service type if it's different from the query parameter
    if stored_service_type != service_type:
        logger.warning(f"Service type mismatch for call {call_id}: stored={stored_service_type}, received={service_type}")
        service_type = stored_service_type
    
    voice_config = call_data["voice_config"]
    
    # Update activity timestamp
    call_data["last_activity"] = time.time()
    
    # Ensure call_sid is stored
    if CallSid and call_data.get("call_sid") != CallSid:
        call_data["call_sid"] = CallSid
        logger.info(f"Updated CallSid for call {call_id}: {CallSid}")
    
    # Record the call if it's not already being recorded
    if CallSid and "recording_sid" not in call_data:
        background_tasks.add_task(start_recording, CallSid, call_id, service_type)
    
    # Schedule periodic saving of transcript after every interaction
    if "agent" in call_data and "storage" in call_data:
        # Get current interaction count
        interaction_count = len(call_data.get("interactions", []))
        # Save data every 2 interactions instead of 3
        if interaction_count > 0 and interaction_count % 2 == 0:
            background_tasks.add_task(background_save_call_data, call_id, CallSid, False)
    
    # Check for low confidence speech recognition
    if confidence_value < 0.3 and SpeechResult:
        logger.warning(f"Low confidence speech detected: {SpeechResult} (Confidence: {confidence_value})")
        # Use pre-enhanced not heard response
        not_heard_ssml = get_quick_response_ssml("not_heard", service_type)
        say = Say(message=not_heard_ssml, voice=voice_config["voice"], language=voice_config["language"])
        say.ssml = True
        response.append(say)
        
        # Gather speech input again
        gather = Gather(
            input='speech',
            action=f'/twilio/process_speech?call_id={call_id}&service_type={service_type}',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            enhanced=True
        )
        response.append(gather)
        response.redirect(f'/twilio/check_call_status?call_id={call_id}&service_type={service_type}')
        return Response(content=str(response), media_type="application/xml")
    
    # Check for call termination commands - quick response path
    if SpeechResult and any(word in SpeechResult.lower() for word in ["goodbye", "end call", "hang up", "bye"]):
        # Say goodbye using pre-enhanced response
        goodbye_ssml = get_quick_response_ssml("goodbye", service_type)
        say = Say(
            message=goodbye_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # End the call
        response.hangup()
        
        # Save data in background with force_save=True to ensure it saves
        background_tasks.add_task(background_save_call_data, call_id, CallSid, True)
        
        processing_time = time.time() - start_time
        logger.info(f"Goodbye processed in {processing_time:.3f} seconds")
        
        return Response(content=str(response), media_type="application/xml")
    
    # If no speech detected
    if not SpeechResult:
        # Use pre-enhanced not heard response
        not_heard_ssml = get_quick_response_ssml("not_heard", service_type)
        say = Say(
            message=not_heard_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # Gather speech input again
        gather = Gather(
            input='speech',
            action=f'/twilio/process_speech?call_id={call_id}&service_type={service_type}',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            hints='help, goodbye, thank you, yes, no',
            enhanced=True
        )
        response.append(gather)
        
        # If still no input, check call status
        response.redirect(f'/twilio/check_call_status?call_id={call_id}&service_type={service_type}')
        
        processing_time = time.time() - start_time
        logger.info(f"No speech processed in {processing_time:.3f} seconds")
        
        return Response(content=str(response), media_type="application/xml")
    
    # Check if agent is initialized
    if "agent" not in call_data:
        # Add a brief message while agent initializes
        processing_ssml = get_quick_response_ssml("processing", service_type)
        say = Say(
            message=processing_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # Initialize in background if not done already
        background_tasks.add_task(initialize_call_resources, call_id, voice_config, service_type)
        
        # Gather speech input again after a moment
        gather = Gather(
            input='speech',
            action=f'/twilio/process_speech?call_id={call_id}&service_type={service_type}',
            method='POST',
            speech_timeout='auto',
            language='en-US'
        )
        response.append(gather)
        
        # If no input, check call status
        response.redirect(f'/twilio/check_call_status?call_id={call_id}&service_type={service_type}')
        
        return Response(content=str(response), media_type="application/xml")
    
    # Get the agent for this call
    agent = call_data["agent"]
    
    # Create a task to process the input with Agents SDK with a timeout
    ai_response = None
    conversation_state = {}
    
    # Set a timeout for AI response - increase significantly for o3-mini
    AI_TIMEOUT = 20.0  # Increased from 6.0 to 20.0
    
    try:
        # Use asyncio.wait_for to limit the time spent waiting for AI
        ai_response, conversation_state = await asyncio.wait_for(
            agent.process_text_input(SpeechResult),
            timeout=AI_TIMEOUT
        )
    except asyncio.TimeoutError:
        # If AI is taking too long, use a generic response but continue processing in background
        logger.warning(f"AI response timeout for call {call_id}")
        ai_response = "I understand. Let me think about that for a moment."
        
        # Continue processing in background
        background_tasks.add_task(process_ai_response_background, call_id, SpeechResult)
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        logger.error(traceback.format_exc())
        ai_response = "I understand. I'm processing your request."
    
    # Enhance the response with SSML
    ssml_response = enhance_speech_output(ai_response, voice_config)
    
    # Say the response with enhanced voice
    say = Say(
        message=ssml_response,
        voice=voice_config["voice"],
        language=voice_config["language"]
    )
    say.ssml = True
    response.append(say)
    
    # Save the interaction in the call data
    timestamp = datetime.now().isoformat() if datetime else time.time()
    call_data["interactions"] = call_data.get("interactions", []) + [
        {
            "user": SpeechResult, 
            "agent": ai_response, 
            "confidence": Confidence,
            "timestamp": timestamp,
            "topics": conversation_state.get("topics", []),
            "sentiment": conversation_state.get("sentiment", "neutral")
        }
    ]
    
    # Update call data with conversation state info
    call_data["sentiment"] = conversation_state.get("sentiment", call_data.get("sentiment", "neutral"))
    call_data["topics"] = conversation_state.get("topics", call_data.get("topics", []))
    
    # Gather speech input again
    gather = Gather(
        input='speech',
        action=f'/twilio/process_speech?call_id={call_id}&service_type={service_type}',
        method='POST',
        speech_timeout='auto',
        language='en-US',
        hints='help, goodbye, thank you, yes, no',
        enhanced=True
    )
    response.append(gather)
    
    # If no input received after timeout, check call status
    response.redirect(f'/twilio/check_call_status?call_id={call_id}&service_type={service_type}')
    
    processing_time = time.time() - start_time
    logger.info(f"Speech processed in {processing_time:.3f} seconds")
    
    return Response(content=str(response), media_type="application/xml")

async def process_ai_response_background(call_id, speech_input):
    """Process AI response in background to avoid delaying user experience"""
    try:
        if call_id not in active_calls or "agent" not in active_calls[call_id]:
            return
            
        call_data = active_calls[call_id]
        agent = call_data["agent"]
        
        # Process with the agent (this might take time, but it's in background)
        ai_response, conversation_state = await agent.process_text_input(speech_input)
        
        # Update the last interaction with the actual AI response
        if "interactions" in call_data and call_data["interactions"]:
            # Find the last interaction with this user input
            for interaction in reversed(call_data["interactions"]):
                if interaction.get("user") == speech_input:
                    # Update it with the actual AI response
                    interaction["agent"] = ai_response
                    interaction["topics"] = conversation_state.get("topics", [])
                    interaction["sentiment"] = conversation_state.get("sentiment", "neutral")
                    break
        
        logger.info(f"Background AI processing completed for call {call_id}")
        
    except Exception as e:
        logger.error(f"Error in background AI processing: {str(e)}")
        logger.error(traceback.format_exc())

async def start_recording(call_sid, call_id, service_type="restaurant"):
    """Start recording a call in the background"""
    try:
        if call_id not in active_calls:
            return
        
        # Get the right Twilio client based on service type
        twilio_client = twilio_clients.get(service_type, twilio_clients["restaurant"])
            
        recording = twilio_client.calls(call_sid).recordings.create()
        active_calls[call_id]["recording_sid"] = recording.sid
        logger.info(f"Started recording for call {call_sid} with recording SID {recording.sid} (Service: {service_type})")
    except Exception as e:
        logger.error(f"Error starting call recording: {str(e)}")
        logger.error(traceback.format_exc())

@router.post("/check_call_status", response_class=PlainTextResponse)
async def check_call_status(
    request: Request, 
    background_tasks: BackgroundTasks, 
    call_id: str,
    service_type: str = Query("restaurant")
):
    """Check if the call is still active after timeout."""
    start_time = time.time()
    logger.info(f"Checking call status for ID: {call_id} (Service: {service_type})")
    
    # Create a TwiML response
    response = VoiceResponse()
    
    # Check if the call exists
    if call_id in active_calls:
        call_data = active_calls[call_id]
        stored_service_type = call_data.get("service_type", "restaurant")
        
        # Use stored service type if different
        if stored_service_type != service_type:
            service_type = stored_service_type
            
        # If call has been inactive for too long, end it
        current_time = time.time()
        last_activity = call_data.get("last_activity", call_data.get("start_time", 0))
        
        if current_time - last_activity > 20:  # 20 seconds of inactivity (reduced from 30)
            logger.info(f"Call {call_id} inactive for too long, ending call")
            
            # Say goodbye
            voice_config = call_data["voice_config"]
            timeout_ssml = get_quick_response_ssml("timeout", service_type)
            say = Say(
                message=timeout_ssml,
                voice=voice_config["voice"],
                language=voice_config["language"]
            )
            say.ssml = True
            response.append(say)
            
            # End the call
            response.hangup()
            
            # Save data in background with force_save=True to ensure it saves
            background_tasks.add_task(background_save_call_data, call_id, call_data.get("call_sid"), True)
            
            processing_time = time.time() - start_time
            logger.info(f"Call status timeout processed in {processing_time:.3f} seconds")
            
            return Response(content=str(response), media_type="application/xml")
        
        # Otherwise, prompt again
        voice_config = call_data["voice_config"]
        anything_else_ssml = get_quick_response_ssml("anything_else", service_type)
        say = Say(
            message=anything_else_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # Update last activity time
        call_data["last_activity"] = current_time
        
        # Gather input again
        gather = Gather(
            input='speech',
            action=f'/twilio/process_speech?call_id={call_id}&service_type={service_type}',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            enhanced=True
        )
        response.append(gather)
        
        # If still no input, end the call
        response.redirect(f'/twilio/end_call?call_id={call_id}&service_type={service_type}')
    else:
        # Call doesn't exist, redirect to end call
        response.redirect(f'/twilio/end_call?call_id={call_id}&service_type={service_type}')
    
    processing_time = time.time() - start_time
    logger.info(f"Call status check processed in {processing_time:.3f} seconds")
    
    return Response(content=str(response), media_type="application/xml")

@router.post("/end_call", response_class=PlainTextResponse)
async def end_call(
    request: Request, 
    background_tasks: BackgroundTasks, 
    call_id: str,
    service_type: str = Query("restaurant")
):
    """End the call and save the conversation."""
    start_time = time.time()
    logger.info(f"Ending call with ID: {call_id} (Service: {service_type})")
    
    # Create a TwiML response
    response = VoiceResponse()
    
    # Check if we have an agent for this call
    if call_id in active_calls:
        call_data = active_calls[call_id]
        stored_service_type = call_data.get("service_type", "restaurant")
        
        # Use stored service type if different
        if stored_service_type != service_type:
            service_type = stored_service_type
            
        voice_config = call_data["voice_config"]
        
        # Save all call data in background with force_save=True to ensure it saves
        background_tasks.add_task(background_save_call_data, call_id, call_data.get("call_sid"), True)
        
        # Say goodbye with enhanced voice - use quick response
        goodbye_ssml = get_quick_response_ssml("goodbye", service_type)
        say = Say(
            message=goodbye_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # Clean up after ensuring data is saved
        # Wait a short time to let background task start
        await asyncio.sleep(1)
    else:
        # Default goodbye
        response.say("Thank you for calling. Goodbye!")
    
    response.hangup()
    
    processing_time = time.time() - start_time
    logger.info(f"End call processed in {processing_time:.3f} seconds")
    
    return Response(content=str(response), media_type="application/xml")

@router.post("/status_callback", response_class=PlainTextResponse)
async def status_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    CallSid: str = Form(None),
    CallStatus: str = Form(None),
    To: str = Form(None)
):
    """Handle call status callbacks from Twilio."""
    # Detect service type from the To number if available
    service_type = detect_service_type(To)
    logger.info(f"Call status callback for SID {CallSid}: {CallStatus} (Service: {service_type})")
    
    # Find the call_id associated with this CallSid
    call_id = None
    for cid, data in active_calls.items():
        if data.get("call_sid") == CallSid:
            call_id = cid
            service_type = data.get("service_type", service_type)  # Update with stored service type
            break
    
    if call_id and CallStatus in ["completed", "failed", "busy", "no-answer"]:
        logger.info(f"Call {CallSid} ended with status {CallStatus}. Saving data immediately. (Service: {service_type})")
        
        # Immediately save the call data with force_save=True
        try:
            # Run this synchronously to ensure it completes before responding
            await background_save_call_data(call_id, CallSid, True)
            logger.info(f"Successfully saved call data for call {call_id}")
        except Exception as e:
            logger.error(f"Error saving call data in status callback: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Clean up from active calls - but only after a delay to ensure data is saved
        async def delayed_cleanup(call_id):
            await asyncio.sleep(5)  # Wait 5 seconds to ensure background tasks complete
            if call_id in active_calls:
                del active_calls[call_id]
                logger.info(f"Cleaned up call {call_id} from active calls")
                
        background_tasks.add_task(delayed_cleanup, call_id)
    
    return PlainTextResponse("OK")

# Import datetime at the top level to avoid issues
try:
    from datetime import datetime
except ImportError:
    logger.warning("Could not import datetime, using time.time() for timestamps")
    datetime = None