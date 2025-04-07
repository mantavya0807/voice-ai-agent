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

from app.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
from app.agent.openai_agent import OpenAIAgent
from app.storage.gcp import GCPStorage
from app.utils import generate_call_id

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Dictionary to store active call agents
active_calls = {}

# Dictionary to track when transcripts were last saved
last_saved = {}

# Common responses for quick replies (to reduce AI waiting time)
QUICK_RESPONSES = {
    "greeting": "Hello! I'm your AI assistant. How can I help you today?",
    "not_heard": "I'm sorry, I didn't catch that. Could you please repeat?",
    "thinking": "I'm thinking about that...",
    "goodbye": "Thank you for calling. Goodbye!",
    "timeout": "I haven't heard from you in a while. Thank you for calling. Goodbye!",
    "anything_else": "Is there anything else I can help you with?",
    "processing": "Let me process that for you."
}

# Enhanced voice settings
VOICE_OPTIONS = {
    "default": {
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
    "british": {
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
        voice_config = VOICE_OPTIONS["default"]
    
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
    for key, text in QUICK_RESPONSES.items():
        ENHANCED_QUICK_RESPONSES[key] = {}
        for voice_name, voice_config in VOICE_OPTIONS.items():
            ENHANCED_QUICK_RESPONSES[key][voice_name] = enhance_speech_output(text, voice_config)

# Call this at startup (AFTER the functions are defined)
setup_enhanced_responses()

def get_quick_response_ssml(key, voice_preference="default"):
    """Get a pre-enhanced quick response with SSML
    
    Args:
        key: The key for the quick response
        voice_preference: The name of the voice preference
    """
    # Use the voice preference to look up the configuration
    voice_config = VOICE_OPTIONS.get(voice_preference, VOICE_OPTIONS["default"])
    
    if key in ENHANCED_QUICK_RESPONSES and voice_preference in ENHANCED_QUICK_RESPONSES[key]:
        return ENHANCED_QUICK_RESPONSES[key][voice_preference]
    
    # Fallback if not found
    text = QUICK_RESPONSES.get(key, "I'm listening.")
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
        
        # Check if agent exists
        if "agent" not in call_data or "storage" not in call_data:
            logger.error(f"Agent or storage not initialized for call ID {call_id}")
            return
            
        agent = call_data["agent"]
        storage = call_data["storage"]
        
        # Get the full transcript
        transcript = agent.get_transcript()
        
        if not transcript:
            logger.warning(f"No transcript data available for call {call_id}")
            return
            
        # Add metadata
        metadata = {
            "call_id": call_id,
            "call_sid": call_sid,
            "caller_number": call_data.get("caller_number"),
            "start_time": call_data.get("start_time"),
            "voice_config": {
                "voice": call_data.get("voice_config", {}).get("voice"),
                "language": call_data.get("voice_config", {}).get("language")
            },
            "interactions": call_data.get("interactions", []),
            "save_time": datetime.now().isoformat() if datetime else time.time()
        }
        
        # Save the transcript with metadata to GCP
        try:
            logger.info(f"Saving transcript for call ID: {call_id}")
            save_result = await storage.save_transcript_with_metadata(transcript, metadata)
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
                # If force_save is True, retry multiple times with longer waits
                max_attempts = 5 if force_save else 2
                wait_time_base = 3  # seconds
                
                for attempt in range(max_attempts):
                    # Increase wait time for each retry
                    wait_time = wait_time_base * (attempt + 1)
                    logger.info(f"Attempt {attempt+1}/{max_attempts} to save recording {recording_sid}, waiting {wait_time}s")
                    
                    # Wait for recording to complete
                    await asyncio.sleep(wait_time)
                    recording = twilio_client.recordings(recording_sid).fetch()
                    
                    if recording.status == "completed":
                        recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.mp3"
                        logger.info(f"Recording {recording_sid} is ready, saving to GCS...")
                        audio_result = await storage.save_audio(recording_url)
                        
                        if audio_result:
                            logger.info(f"Successfully saved audio for call ID: {call_id} to {audio_result}")
                            # Mark as saved in the call data
                            call_data["recording_saved"] = True
                            return  # Success, no need to continue retrying
                        else:
                            logger.error(f"Failed to save audio for call ID: {call_id}")
                    else:
                        logger.warning(f"Recording not completed yet for SID {recording_sid}, status: {recording.status}")
                
                # If we get here, all attempts failed
                logger.error(f"Failed to save recording after {max_attempts} attempts")
            except Exception as e:
                logger.error(f"Error saving audio: {str(e)}")
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
    voice_preference: str = Query("default")
):
    """Handle incoming Twilio voice calls."""
    start_time = time.time()
    logger.info(f"Received voice call with SID: {CallSid} from {From}")
    
    # Create a TwiML response
    response = VoiceResponse()
    
    # Create a unique identifier for this call
    call_id = CallSid or str(uuid.uuid4())
    
    # Initialize an OpenAI agent for this call if it doesn't exist
    if call_id not in active_calls:
        # Select voice configuration
        voice_config = VOICE_OPTIONS.get(voice_preference, VOICE_OPTIONS["default"])
        
        # Only initialize the agent after responding to the user
        # This avoids delay in the initial greeting
        active_calls[call_id] = {
            "caller_number": From,
            "voice_config": voice_config,
            "start_time": time.time(),
            "call_sid": CallSid,
            "last_activity": time.time()
        }
        
        # Use pre-enhanced greeting for faster response
        greeting_ssml = get_quick_response_ssml("greeting", voice_preference)
        
        # Say the greeting with enhanced voice
        say = Say(
            message=greeting_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # Initialize OpenAI and Storage in background to avoid initial delay
        background_tasks.add_task(initialize_call_resources, call_id, voice_config)
    else:
        # Call already exists - ensure agent is initialized
        if "agent" not in active_calls[call_id]:
            # Add a brief delay message while agent initializes
            voice_config = active_calls[call_id]["voice_config"]
            processing_ssml = get_quick_response_ssml("processing", voice_preference)
            
            say = Say(
                message=processing_ssml,
                voice=voice_config["voice"],
                language=voice_config["language"]
            )
            say.ssml = True
            response.append(say)
            
            # Initialize in background if not done already
            background_tasks.add_task(initialize_call_resources, call_id, voice_config)
    
    # Gather speech input from the user
    gather = Gather(
        input='speech',
        action=f'/twilio/process_speech?call_id={call_id}',
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
    response.redirect('/twilio/voice')
    
    processing_time = time.time() - start_time
    logger.info(f"Voice webhook processed in {processing_time:.3f} seconds")
    
    return Response(content=str(response), media_type="application/xml")

async def initialize_call_resources(call_id, voice_config):
    """Initialize AI agent and storage resources asynchronously"""
    try:
        # Only initialize if not already initialized
        if call_id in active_calls and "agent" not in active_calls[call_id]:
            logger.info(f"Initializing resources for call {call_id}")
            
            # First initialize storage
            storage = GCPStorage(call_id)
            active_calls[call_id]["storage"] = storage
            
            # Now initialize the agent
            active_calls[call_id]["agent"] = OpenAIAgent()
            
            logger.info(f"Finished initializing resources for call {call_id}")
    except Exception as e:
        logger.error(f"Error initializing call resources: {str(e)}")
        logger.error(traceback.format_exc())

@router.post("/process_speech", response_class=PlainTextResponse)
async def process_speech(
    request: Request,
    background_tasks: BackgroundTasks,
    call_id: str,
    SpeechResult: str = Form(None),
    CallSid: str = Form(None),
    Confidence: str = Form(None)
):
    """Process speech input from Twilio with optimized response time."""
    start_time = time.time()
    logger.info(f"Processing speech for call ID: {call_id}")
    
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
    voice_config = call_data["voice_config"]
    
    # Update activity timestamp
    call_data["last_activity"] = time.time()
    
    # Ensure call_sid is stored
    if CallSid and call_data.get("call_sid") != CallSid:
        call_data["call_sid"] = CallSid
        logger.info(f"Updated CallSid for call {call_id}: {CallSid}")
    
    # Record the call if it's not already being recorded
    if CallSid and "recording_sid" not in call_data:
        background_tasks.add_task(start_recording, CallSid, call_id)
    
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
        not_heard_ssml = get_quick_response_ssml("not_heard", voice_preference="default")
        say = Say(message=not_heard_ssml, voice=voice_config["voice"], language=voice_config["language"])
        say.ssml = True
        response.append(say)
        
        # Gather speech input again
        gather = Gather(
            input='speech',
            action=f'/twilio/process_speech?call_id={call_id}',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            enhanced=True
        )
        response.append(gather)
        response.redirect(f'/twilio/check_call_status?call_id={call_id}')
        return Response(content=str(response), media_type="application/xml")
    
    # Check for call termination commands - quick response path
    if SpeechResult and any(word in SpeechResult.lower() for word in ["goodbye", "end call", "hang up", "bye"]):
        # Say goodbye using pre-enhanced response
        goodbye_ssml = get_quick_response_ssml("goodbye", voice_preference="default")
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
        not_heard_ssml = get_quick_response_ssml("not_heard", voice_preference="default")
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
            action=f'/twilio/process_speech?call_id={call_id}',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            hints='help, goodbye, thank you, yes, no',
            enhanced=True
        )
        response.append(gather)
        
        # If still no input, check call status
        response.redirect(f'/twilio/check_call_status?call_id={call_id}')
        
        processing_time = time.time() - start_time
        logger.info(f"No speech processed in {processing_time:.3f} seconds")
        
        return Response(content=str(response), media_type="application/xml")
    
    # Check if agent is initialized
    if "agent" not in call_data:
        # Add a brief message while agent initializes
        processing_ssml = get_quick_response_ssml("processing", voice_preference="default")
        say = Say(
            message=processing_ssml,
            voice=voice_config["voice"],
            language=voice_config["language"]
        )
        say.ssml = True
        response.append(say)
        
        # Initialize in background if not done already
        background_tasks.add_task(initialize_call_resources, call_id, voice_config)
        
        # Gather speech input again after a moment
        gather = Gather(
            input='speech',
            action=f'/twilio/process_speech?call_id={call_id}',
            method='POST',
            speech_timeout='auto',
            language='en-US'
        )
        response.append(gather)
        
        # If no input, check call status
        response.redirect(f'/twilio/check_call_status?call_id={call_id}')
        
        return Response(content=str(response), media_type="application/xml")
    
    # Get the agent for this call
    agent = call_data["agent"]
    
    # Create a task to process the input with OpenAI with a timeout
    ai_response = None
    conversation_state = {}
    
    # Set a timeout for AI response - increase from 3 to 4 seconds
    # This provides a better balance between quality responses and latency
    AI_TIMEOUT = 4.0
    
    try:
        # Use asyncio.wait_for to limit the time spent waiting for AI
        ai_response, conversation_state = await asyncio.wait_for(
            agent.process_voice_input(SpeechResult),
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
    call_data["interactions"] = call_data.get("interactions", []) + [
        {"user": SpeechResult, "agent": ai_response, "confidence": Confidence}
    ]
    
    # Gather speech input again
    gather = Gather(
        input='speech',
        action=f'/twilio/process_speech?call_id={call_id}',
        method='POST',
        speech_timeout='auto',
        language='en-US',
        hints='help, goodbye, thank you, yes, no',
        enhanced=True
    )
    response.append(gather)
    
    # If no input received after timeout, check call status
    response.redirect(f'/twilio/check_call_status?call_id={call_id}')
    
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
        ai_response, conversation_state = await agent.process_voice_input(speech_input)
        
        # Update the last interaction with the actual AI response
        if "interactions" in call_data and call_data["interactions"]:
            # Find the last interaction with this user input
            for interaction in reversed(call_data["interactions"]):
                if interaction.get("user") == speech_input:
                    # Update it with the actual AI response
                    interaction["agent"] = ai_response
                    interaction["conversation_state"] = conversation_state
                    break
        
        logger.info(f"Background AI processing completed for call {call_id}")
        
    except Exception as e:
        logger.error(f"Error in background AI processing: {str(e)}")
        logger.error(traceback.format_exc())

async def start_recording(call_sid, call_id):
    """Start recording a call in the background"""
    try:
        if call_id not in active_calls:
            return
            
        recording = twilio_client.calls(call_sid).recordings.create()
        active_calls[call_id]["recording_sid"] = recording.sid
        logger.info(f"Started recording for call {call_sid} with recording SID {recording.sid}")
    except Exception as e:
        logger.error(f"Error starting call recording: {str(e)}")
        logger.error(traceback.format_exc())

@router.post("/check_call_status", response_class=PlainTextResponse)
async def check_call_status(request: Request, background_tasks: BackgroundTasks, call_id: str):
    """Check if the call is still active after timeout."""
    start_time = time.time()
    logger.info(f"Checking call status for ID: {call_id}")
    
    # Create a TwiML response
    response = VoiceResponse()
    
    # Check if the call exists
    if call_id in active_calls:
        call_data = active_calls[call_id]
        
        # If call has been inactive for too long, end it
        current_time = time.time()
        last_activity = call_data.get("last_activity", call_data.get("start_time", 0))
        
        if current_time - last_activity > 20:  # 20 seconds of inactivity (reduced from 30)
            logger.info(f"Call {call_id} inactive for too long, ending call")
            
            # Say goodbye
            voice_config = call_data["voice_config"]
            timeout_ssml = get_quick_response_ssml("timeout", voice_preference="default")
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
        anything_else_ssml = get_quick_response_ssml("anything_else", voice_preference="default")
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
            action=f'/twilio/process_speech?call_id={call_id}',
            method='POST',
            speech_timeout='auto',
            language='en-US',
            enhanced=True
        )
        response.append(gather)
        
        # If still no input, end the call
        response.redirect(f'/twilio/end_call?call_id={call_id}')
    else:
        # Call doesn't exist, redirect to end call
        response.redirect(f'/twilio/end_call?call_id={call_id}')
    
    processing_time = time.time() - start_time
    logger.info(f"Call status check processed in {processing_time:.3f} seconds")
    
    return Response(content=str(response), media_type="application/xml")

@router.post("/end_call", response_class=PlainTextResponse)
async def end_call(request: Request, background_tasks: BackgroundTasks, call_id: str):
    """End the call and save the conversation."""
    start_time = time.time()
    logger.info(f"Ending call with ID: {call_id}")
    
    # Create a TwiML response
    response = VoiceResponse()
    
    # Check if we have an agent for this call
    if call_id in active_calls:
        call_data = active_calls[call_id]
        voice_config = call_data["voice_config"]
        
        # Save all call data in background with force_save=True to ensure it saves
        background_tasks.add_task(background_save_call_data, call_id, call_data.get("call_sid"), True)
        
        # Say goodbye with enhanced voice - use quick response
        goodbye_ssml = get_quick_response_ssml("goodbye", voice_preference="default")
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
    CallStatus: str = Form(None)
):
    """Handle call status callbacks from Twilio."""
    logger.info(f"Call status callback for SID {CallSid}: {CallStatus}")
    
    # Find the call_id associated with this CallSid
    call_id = None
    for cid, data in active_calls.items():
        if data.get("call_sid") == CallSid:
            call_id = cid
            break
    
    if call_id and CallStatus in ["completed", "failed", "busy", "no-answer"]:
        logger.info(f"Call {CallSid} ended with status {CallStatus}. Saving data immediately.")
        
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