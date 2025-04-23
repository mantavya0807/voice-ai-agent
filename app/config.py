"""
Updated configuration for Twilio Media Control Platform (MCP)
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application Configuration
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
ADVANCED_FEATURES_ENABLED = os.getenv("ADVANCED_FEATURES_ENABLED", "True").lower() in ("true", "1", "t")

# Add Agents SDK Configuration
AGENTS_SDK_USE_TRACING = os.getenv("AGENTS_SDK_USE_TRACING", "False").lower() in ("true", "1", "t")
AGENTS_SDK_VOICE_MODEL = os.getenv("AGENTS_SDK_VOICE_MODEL", "whisper-1")
AGENTS_SDK_TTS_MODEL = os.getenv("AGENTS_SDK_TTS_MODEL", "tts-1")
AGENTS_SDK_TTS_VOICE = os.getenv("AGENTS_SDK_TTS_VOICE", "alloy")

# Restaurant Service Twilio Configuration
RESTAURANT_TWILIO_ACCOUNT_SID = os.getenv("RESTAURANT_TWILIO_ACCOUNT_SID")
RESTAURANT_TWILIO_AUTH_TOKEN = os.getenv("RESTAURANT_TWILIO_AUTH_TOKEN")
RESTAURANT_TWILIO_PHONE_NUMBER = os.getenv("RESTAURANT_TWILIO_PHONE_NUMBER")
RESTAURANT_DEFAULT_VOICE = os.getenv("RESTAURANT_DEFAULT_VOICE", "alloy")  # Using OpenAI voice names
RESTAURANT_DEFAULT_LANGUAGE = os.getenv("RESTAURANT_DEFAULT_LANGUAGE", "en-US")

# Hairdresser Service Twilio Configuration
HAIRDRESSER_TWILIO_ACCOUNT_SID = os.getenv("HAIRDRESSER_TWILIO_ACCOUNT_SID")
HAIRDRESSER_TWILIO_AUTH_TOKEN = os.getenv("HAIRDRESSER_TWILIO_AUTH_TOKEN")
HAIRDRESSER_TWILIO_PHONE_NUMBER = os.getenv("HAIRDRESSER_TWILIO_PHONE_NUMBER")
HAIRDRESSER_DEFAULT_VOICE = os.getenv("HAIRDRESSER_DEFAULT_VOICE", "nova")  # Using OpenAI voice names
HAIRDRESSER_DEFAULT_LANGUAGE = os.getenv("HAIRDRESSER_DEFAULT_LANGUAGE", "en-GB")

# MCP Configuration (New)
USE_MCP = os.getenv("USE_MCP", "True").lower() in ("true", "1", "t")
MCP_MEDIA_STREAM_URL = os.getenv("MCP_MEDIA_STREAM_URL")  # Base URL for media streams

# Audio Processing Configuration (New)
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "8000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "mulaw")  # Options: mulaw, pcm
AUDIO_CHUNK_DURATION_MS = int(os.getenv("AUDIO_CHUNK_DURATION_MS", "200"))  # Duration of audio chunks in ms
VAD_ENABLED = os.getenv("VAD_ENABLED", "True").lower() in ("true", "1", "t")  # Voice Activity Detection
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.3"))  # VAD threshold (0.0 to 1.0)
VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "1000"))  # Silence duration to consider end of speech

# Speech-to-Text Configuration (New)
STT_PROVIDER = os.getenv("STT_PROVIDER", "openai")  # Options: openai, google, azure
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")  # OpenAI model
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")  # Language code
STT_STREAMING = os.getenv("STT_STREAMING", "True").lower() in ("true", "1", "t")  # Use streaming API if available

# Text-to-Speech Configuration (New)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai")  # Options: openai, google, azure
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")  # OpenAI model
TTS_STREAMING = os.getenv("TTS_STREAMING", "True").lower() in ("true", "1", "t")  # Use streaming API if available

# Lookup dictionaries for service identification
SERVICE_PHONE_NUMBERS = {
    RESTAURANT_TWILIO_PHONE_NUMBER: "restaurant",
    HAIRDRESSER_TWILIO_PHONE_NUMBER: "hairdresser"
}

# Service-specific configuration (updated with MCP settings)
SERVICE_CONFIG = {
    "restaurant": {
        "name": "Bella Cucina Restaurant",
        "account_sid": RESTAURANT_TWILIO_ACCOUNT_SID,
        "auth_token": RESTAURANT_TWILIO_AUTH_TOKEN,
        "phone_number": RESTAURANT_TWILIO_PHONE_NUMBER,
        "voice": RESTAURANT_DEFAULT_VOICE,
        "language": RESTAURANT_DEFAULT_LANGUAGE,
        "greeting": "Thank you for calling Bella Cucina Restaurant. How may I help you with your reservation today?",
        "assistant_instructions": """
            You are a restaurant booking assistant for Bella Cucina, an Italian restaurant.
            Be friendly, professional, and efficient when handling reservations.
            
            When booking a reservation, make sure to collect:
            - Name
            - Date and time
            - Number of people
            - Contact phone number
            - Any special requests (high chairs, dietary restrictions, etc.)
            
            Our operating hours are:
            - Monday to Thursday: 5:00 PM - 10:00 PM
            - Friday and Saturday: 5:00 PM - 11:00 PM
            - Sunday: 4:00 PM - 9:00 PM
            
            We can accommodate groups up to 8 people in our regular dining area.
            For parties larger than 8, we require at least 48 hours notice.
            
            Keep your responses concise and suitable for a voice conversation.
        """,
        "tts_voice": RESTAURANT_DEFAULT_VOICE,  # Voice for TTS
        "tts_language": RESTAURANT_DEFAULT_LANGUAGE,
        "stt_language": "en-US",  # Language for STT
    },
    "hairdresser": {
        "name": "Style Studio Salon",
        "account_sid": HAIRDRESSER_TWILIO_ACCOUNT_SID,
        "auth_token": HAIRDRESSER_TWILIO_AUTH_TOKEN,
        "phone_number": HAIRDRESSER_TWILIO_PHONE_NUMBER,
        "voice": HAIRDRESSER_DEFAULT_VOICE,
        "language": HAIRDRESSER_DEFAULT_LANGUAGE,
        "greeting": "Welcome to Style Studio Salon. How may I assist you with your appointment today?",
        "assistant_instructions": """
            You are a professional appointment scheduler for Style Studio Salon.
            Be warm, helpful, and efficient when booking appointments.
            
            When booking an appointment, make sure to collect:
            - Client name
            - Service requested (haircut, color, styling, etc.)
            - Preferred date and time
            - Contact phone number
            - Stylist preference (if any)
            
            Our operating hours are:
            - Tuesday to Friday: 9:00 AM - 7:00 PM
            - Saturday: 9:00 AM - 5:00 PM
            - Closed Sunday and Monday
            
            Available services and approximate durations:
            - Men's haircut: 30 minutes
            - Women's haircut: 45-60 minutes
            - Color treatment: 2-3 hours
            - Blow-dry and styling: 30-45 minutes
            
            Keep your responses concise and suitable for a voice conversation.
        """,
        "tts_voice": HAIRDRESSER_DEFAULT_VOICE,  # Voice for TTS
        "tts_language": HAIRDRESSER_DEFAULT_LANGUAGE,
        "stt_language": "en-GB",  # Language for STT
    }
}

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
USE_BIGQUERY = os.getenv("USE_BIGQUERY", "False").lower() in ("true", "1", "t")

# Security Configuration
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "False").lower() in ("true", "1", "t")
API_KEY = os.getenv("API_KEY", "")

# Feature Flags
ENABLE_CALL_ANALYTICS = os.getenv("ENABLE_CALL_ANALYTICS", "True").lower() in ("true", "1", "t")
ENABLE_SENTIMENT_ANALYSIS = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "True").lower() in ("true", "1", "t")
ENABLE_FALLBACK_RESPONSES = os.getenv("ENABLE_FALLBACK_RESPONSES", "True").lower() in ("true", "1", "t")
MAX_CALL_DURATION_SECONDS = int(os.getenv("MAX_CALL_DURATION_SECONDS", "300"))  # 5 minutes by default
INACTIVITY_TIMEOUT_SECONDS = int(os.getenv("INACTIVITY_TIMEOUT_SECONDS", "30"))