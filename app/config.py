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

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "Polly.Joanna")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en-US")

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