"""
OpenAI Agents SDK Voice Implementation

This module integrates the OpenAI Agents SDK with voice capabilities
to replace the direct use of OpenAI Assistant APIs.
"""

import logging
import time
import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from agents import Agent, function_tool, Runner
from agents.voice import VoicePipeline, SingleAgentVoiceWorkflow, AudioInput
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

from app.config import (
    OPENAI_API_KEY, 
    SERVICE_CONFIG, 
    DEFAULT_MODEL
)

logger = logging.getLogger(__name__)

class AgentsVoiceManager:
    """
    Manages voice interactions using the OpenAI Agents SDK.
    This replaces the previous OpenAIAgent implementation.
    """
    
    def __init__(self, service_type=None, caller_info=None):
        """
        Initialize the Agents Voice Manager.
        
        Args:
            service_type: The type of service (restaurant or hairdresser)
            caller_info: Optional caller information for personalization
        """
        self.service_type = service_type or "restaurant"  # Default to restaurant if not specified
        self.caller_info = caller_info or {}
        self.model = DEFAULT_MODEL
        self.conversation_state = {
            "intent": None,
            "sentiment": "neutral",
            "topics": [],
            "follow_up_questions": []
        }
        
        # Track timing for performance optimization
        self.setup_start_time = time.time()
        self.agent = self._create_agent()
        self.pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(self.agent))
        setup_duration = time.time() - self.setup_start_time
        logger.info(f"Initialized Agents Voice Manager for {self.service_type} service with model: {self.model} in {setup_duration:.2f} seconds")
        
    def _create_agent(self) -> Agent:
        """
        Create an Agent with the OpenAI Agents SDK.
        
        Returns:
            Agent: The configured agent
        """
        # Get service-specific configuration
        service_config = SERVICE_CONFIG.get(self.service_type, SERVICE_CONFIG["restaurant"])
        
        # Base instructions for voice assistant
        base_instructions = """
        You are a helpful voice assistant. Provide clear, concise responses 
        suitable for voice conversations. Keep your responses brief and to the point,
        as they will be read aloud to the caller. Avoid using visual elements or
        references that wouldn't make sense in a voice call.
        """
        
        # Add service-specific instructions
        service_instructions = service_config.get("assistant_instructions", "")
        
        # Combine all instructions
        full_instructions = base_instructions + service_instructions
        
        # Use the function_tool decorator directly on a method
        @function_tool
        def analyze_conversation():
            """Get the current conversation analysis with intent, sentiment, and topics"""
            return self.conversation_state
        
        # Return the configured agent
        return Agent(
            name=f"{service_config.get('name')} Voice Assistant",
            instructions=full_instructions,
            model="o3-mini",  # Change model to o3-mini
            tools=[analyze_conversation]
        )

    def _get_conversation_state(self) -> Dict[str, Any]:
        """Helper method to return the conversation state."""
        return self.conversation_state
    
    async def process_voice_input(self, audio_data: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Process voice input audio and generate a response using OpenAI Agents SDK.
        
        Args:
            audio_data: The raw audio data from the voice input
            
        Returns:
            Tuple[str, Dict[str, Any]]: The generated response text and conversation metadata
        """
        process_start = time.time()
        logger.info(f"Processing voice input for {self.service_type} service")
        
        try:
            # Create AudioInput from the raw audio data
            audio_input = AudioInput(buffer=audio_data)
            
            # Run the voice pipeline
            result = await self.pipeline.run(audio_input)
            
            # Get the text response
            response_text = ""
            audio_output = bytearray()
            
            # Process the streaming result to get both text and audio
            async for event in result.stream():
                if event.type == "voice_stream_event_text":
                    # This is the text that was generated
                    response_text = event.data
                elif event.type == "voice_stream_event_audio":
                    # This is the audio that was generated
                    # We collect it in case we need it later
                    audio_output.extend(event.data)
            
            # Update conversation analysis
            self._analyze_conversation(response_text)
            
            total_duration = time.time() - process_start
            logger.info(f"Generated response in {total_duration:.2f}s: {response_text}")
            
            return response_text, self.conversation_state
            
        except Exception as e:
            logger.error(f"Error processing voice input: {str(e)}")
            return "I'm sorry, I encountered an error processing your request.", {}
    
    async def process_text_input(self, text_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process text input and generate a response using OpenAI Agents SDK.
        This is used when we have already transcribed speech to text.
        
        Args:
            text_input: The transcribed text from voice input
            
        Returns:
            Tuple[str, Dict[str, Any]]: The generated response text and conversation metadata
        """
        process_start = time.time()
        logger.info(f"Processing text input for {self.service_type} service: {text_input}")
        
        try:
            # Run the agent with the text input
            result = await Runner.run(self.agent, input=text_input)
            response_text = str(result.final_output)
            
            # Update conversation analysis
            self._analyze_conversation(text_input)
            
            total_duration = time.time() - process_start
            logger.info(f"Generated response in {total_duration:.2f}s: {response_text}")
            
            return response_text, self.conversation_state
            
        except Exception as e:
            logger.error(f"Error processing text input: {str(e)}")
            return "I'm sorry, I encountered an error processing your request.", {}
    
    def _analyze_conversation(self, text_input: str):
        """
        Analyze the conversation and update the conversation state.
        
        Args:
            text_input: The user's input text
        """
        try:
            # Simple sentiment analysis based on keywords
            if any(word in text_input.lower() for word in ['happy', 'great', 'awesome', 'excellent', 'thank']):
                self.conversation_state["sentiment"] = "positive"
            elif any(word in text_input.lower() for word in ['bad', 'terrible', 'awful', 'angry', 'upset']):
                self.conversation_state["sentiment"] = "negative"
            
            # Extract potential topics based on service type
            words = text_input.lower().split()
            common_words = [
                'this', 'that', 'then', 'when', 'what', 'where', 'which', 'how', 
                'would', 'could', 'should', 'hello', 'thanks', 'thank', 'please'
            ]
            
            # Add service-specific stopwords
            if self.service_type == "restaurant":
                common_words.extend(['restaurant', 'table', 'booking', 'reservation'])
            elif self.service_type == "hairdresser":
                common_words.extend(['appointment', 'haircut', 'salon', 'hair', 'style'])
            
            nouns = [word for word in words if len(word) > 3 and word not in common_words]
            
            if nouns:
                # Update topics rather than replacing
                existing_topics = set(self.conversation_state.get("topics", []))
                new_topics = set(nouns[:3])
                combined_topics = existing_topics.union(new_topics)
                self.conversation_state["topics"] = list(combined_topics)[:5]  # Keep only top 5
            
            # Service-specific intent detection
            if self.service_type == "restaurant":
                if any(word in text_input.lower() for word in ['book', 'reserve', 'table', 'reservation']):
                    self.conversation_state["intent"] = "make_reservation"
                elif any(word in text_input.lower() for word in ['menu', 'food', 'dish', 'eat']):
                    self.conversation_state["intent"] = "inquire_menu"
                elif any(word in text_input.lower() for word in ['hour', 'open', 'time', 'when']):
                    self.conversation_state["intent"] = "ask_hours"
            elif self.service_type == "hairdresser":
                if any(word in text_input.lower() for word in ['book', 'schedule', 'appointment']):
                    self.conversation_state["intent"] = "book_appointment"
                elif any(word in text_input.lower() for word in ['service', 'haircut', 'style', 'color']):
                    self.conversation_state["intent"] = "inquire_services"
                elif any(word in text_input.lower() for word in ['price', 'cost', 'how much']):
                    self.conversation_state["intent"] = "ask_prices"
            
            # Generic intent detection as fallback
            if not self.conversation_state.get("intent"):
                if 'help' in text_input.lower() or 'assist' in text_input.lower():
                    self.conversation_state["intent"] = "seeking_help"
                elif any(q in text_input.lower() for q in ['how', 'what', 'why', 'when', 'where']):
                    self.conversation_state["intent"] = "asking_question"
                elif any(word in text_input.lower() for word in ['goodbye', 'bye', 'end', 'hang up']):
                    self.conversation_state["intent"] = "ending_conversation"
            
        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}")

    def get_transcript(self) -> List[Dict[str, Any]]:
        """
        Get the complete conversation transcript.
        Note: This implementation is limited as the Agents SDK doesn't 
        directly provide transcript access in the same way as Assistants API.
        
        Returns:
            List[Dict[str, Any]]: The conversation transcript
        """
        # This would need to be implemented differently, potentially by tracking
        # all interactions manually since the Agents SDK doesn't store conversation history 
        # in the same way as the Assistants API.
        return []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dict[str, Any]: Summary of the conversation
        """
        # This would need to be implemented differently, potentially by using
        # the agent to generate a summary of tracked interactions.
        return {
            "summary": "Unable to generate summary with Agents SDK",
            "message_count": 0,
            "topics": self.conversation_state.get("topics", []),
            "sentiment": self.conversation_state.get("sentiment", "neutral"),
            "service_type": self.service_type
        }

# Function tools for specific business logic
@function_tool
def check_restaurant_availability(date: str, time: str, party_size: int) -> str:
    """
    Check if a table is available at the restaurant.
    
    Args:
        date: The date for the reservation (YYYY-MM-DD)
        time: The time for the reservation (HH:MM)
        party_size: The number of people in the party
    
    Returns:
        str: Availability information
    """
    # In a real implementation, this would check a database or API
    # For now, this is a stub implementation
    logger.info(f"Checking availability for {party_size} people on {date} at {time}")
    
    # Simulate some basic logic
    if party_size > 8:
        return "I'm sorry, we can only accommodate parties up to 8 people in our regular dining area. For larger groups, please call our events team."
    
    # Mock some unavailable times
    unavailable_slots = [
        ("2025-04-20", "19:00"),
        ("2025-04-20", "19:30"),
        ("2025-04-20", "20:00"),
        ("2025-04-21", "20:00")
    ]
    
    if (date, time) in unavailable_slots:
        return f"I'm sorry, we don't have availability for {party_size} people on {date} at {time}. We do have openings 30 minutes earlier or later."
    
    return f"Yes, we have availability for {party_size} people on {date} at {time}. Would you like me to make a reservation?"

@function_tool
def book_restaurant_reservation(name: str, date: str, time: str, party_size: int, phone: str = "") -> str:
    """
    Book a reservation at the restaurant.
    
    Args:
        name: The name for the reservation
        date: The date for the reservation (YYYY-MM-DD)
        time: The time for the reservation (HH:MM)
        party_size: The number of people in the party
        phone: The contact phone number
    
    Returns:
        str: Confirmation message
    """
    # In a real implementation, this would update a database or API
    # For now, this is a stub implementation
    logger.info(f"Booking reservation for {name}, {party_size} people on {date} at {time}")
    
    # Generate a confirmation number
    confirmation = f"RC{date.replace('-', '')}{time.replace(':', '')}"
    
    return f"Great! I've booked your reservation for {party_size} people on {date} at {time} under the name {name}. Your confirmation number is {confirmation}."

@function_tool
def check_salon_availability(date: str, time: str, service: str = "haircut") -> str:
    """
    Check if a salon appointment is available.
    
    Args:
        date: The date for the appointment (YYYY-MM-DD)
        time: The time for the appointment (HH:MM)
        service: The type of service requested
    
    Returns:
        str: Availability information
    """
    # In a real implementation, this would check a database or API
    # For now, this is a stub implementation
    logger.info(f"Checking salon availability for {service} on {date} at {time}")
    
    # Mock some unavailable times
    unavailable_slots = [
        ("2025-04-20", "14:00"),
        ("2025-04-20", "15:00"),
        ("2025-04-21", "10:00")
    ]
    
    if (date, time) in unavailable_slots:
        return f"I'm sorry, we don't have availability for a {service} on {date} at {time}. We do have openings 1 hour earlier or later."
    
    # Service duration information
    durations = {
        "haircut": "45 minutes",
        "color": "2 hours",
        "styling": "30 minutes"
    }
    
    duration = durations.get(service.lower(), "45 minutes")
    
    return f"Yes, we have availability for a {service} on {date} at {time}. This service typically takes {duration}. Would you like me to book this appointment?"

@function_tool
def book_salon_appointment(name: str, date: str, time: str, service: str, phone: str = "") -> str:
    """
    Book an appointment at the salon.
    
    Args:
        name: The name for the appointment
        date: The date for the appointment (YYYY-MM-DD)
        time: The time for the appointment (HH:MM)
        service: The type of service requested
        phone: The contact phone number
    
    Returns:
        str: Confirmation message
    """
    # In a real implementation, this would update a database or API
    # For now, this is a stub implementation
    logger.info(f"Booking salon appointment for {name}, {service} on {date} at {time}")
    
    # Generate a confirmation number
    confirmation = f"SC{date.replace('-', '')}{time.replace(':', '')}"
    
    return f"Perfect! I've booked your {service} appointment on {date} at {time} under the name {name}. Your confirmation number is {confirmation}."