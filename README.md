# Voice AI Agent

A high-performance voice AI agent built with OpenAI Agent SDK, Twilio for telephony, and Google Cloud Platform for cloud storage and deployment.

## Features

- **Natural Voice Conversations**: Uses OpenAI GPT-4o models for intelligent, context-aware responses
- **High Performance**: Optimized for minimal latency with non-blocking architecture
- **SSML Voice Enhancement**: Natural-sounding speech with proper pacing and pauses
- **Call Recording**: Automatically records and stores call audio
- **Conversation Transcript**: Stores complete conversation history with metadata
- **Cloud Storage**: Securely saves all data to Google Cloud Storage
- **Background Processing**: Handles resource-intensive tasks asynchronously
- **Production Ready**: Comprehensive logging, error handling, and timeout management

## Architecture

```
┌─────────────┐          ┌───────────────┐          ┌──────────────┐
│   Caller    │─────────▶│  Twilio PSTN  │─────────▶│   FastAPI    │
└─────────────┘          └───────────────┘          │   Backend    │
      ▲                                             │  (Cloud Run) │
      │                                             └──────────────┘
      │                                                    │
      │                         ┌──────────────┐           │
      │                         │ Google Cloud │◀──────────┘
      │                         │    Storage   │
      │                         └──────────────┘
      │                                │
      └────────────────────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- Twilio account with a phone number
- OpenAI API key with access to GPT-4o models
- Google Cloud Platform account with Storage access

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/voice-ai-agent.git
   cd voice-ai-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```
   # OpenAI
   OPENAI_API_KEY=your_openai_api_key

   # Twilio
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone_number

   # Google Cloud
   GOOGLE_APPLICATION_CREDENTIALS=path_to_your_gcp_credentials.json
   GCS_BUCKET_NAME=your_gcs_bucket_name

   # Application
   API_HOST=0.0.0.0
   API_PORT=8000
   DEBUG=False
   ```

5. Set up Google Cloud credentials:
   - Create a GCP project and enable Cloud Storage
   - Create a service account with Storage Object Admin permissions
   - Download the JSON key file and save it to your project
   - Update the `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to point to this file

### Running Locally

1. Start the FastAPI server:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

2. Expose your local server with ngrok:
   ```bash
   ngrok http 8000
   ```

3. Configure your Twilio phone number:
   - Go to the [Twilio Console](https://console.twilio.com/)
   - Select your phone number
   - Under "Voice & Fax", set:
     - A Call Comes In: Webhook, `https://your-ngrok-url/twilio/voice` (HTTP POST)
     - Status Callback URL: `https://your-ngrok-url/twilio/status_callback` (HTTP POST)

4. Test by calling your Twilio phone number

### Cloud Deployment

For production deployment on Google Cloud Run:

1. Build and push the Docker image:
   ```bash
   docker build -t gcr.io/your-project-id/voice-ai-agent .
   docker push gcr.io/your-project-id/voice-ai-agent
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy voice-ai-agent \
     --image gcr.io/your-project-id/voice-ai-agent \
     --platform managed \
     --region us-central1 \
     --memory 512Mi \
     --allow-unauthenticated \
     --set-env-vars "OPENAI_API_KEY=your_key,TWILIO_ACCOUNT_SID=your_sid,TWILIO_AUTH_TOKEN=your_token,TWILIO_PHONE_NUMBER=your_number,GCS_BUCKET_NAME=your_bucket"
   ```

3. Update your Twilio webhook URLs to point to your Cloud Run service URL

## Usage Guide

### Making Calls

Simply call your configured Twilio phone number. The AI agent will:
1. Answer with a greeting
2. Listen for your speech input
3. Process your request and respond naturally
4. Record the conversation and save the transcript

### Voice Customization

You can customize the voice by modifying the `VOICE_OPTIONS` in `app/voice/twilio_handler.py`:

```python
VOICE_OPTIONS = {
    "default": {
        "voice": "Polly.Joanna",  # Try different Amazon Polly voices
        "language": "en-US",
        "speed": "1.0"
    }
}
```

Available voices include:
- Polly.Joanna, Polly.Matthew (US English)
- Polly.Amy, Polly.Brian (British English)
- Polly.Celine, Polly.Mathieu (French)
- Many more at [Twilio's Amazon Polly documentation](https://www.twilio.com/docs/voice/twiml/say/amazon-polly)

### Response Speed Adjustment

To make the agent more responsive or more deliberate, adjust the AI timeout in `app/voice/twilio_handler.py`:

```python
# Faster responses (default)
AI_TIMEOUT = 3.0  # 3 seconds max wait

# More deliberate responses
AI_TIMEOUT = 5.0  # 5 seconds max wait
```

## Project Structure

```
voice-ai-agent/
├── app/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── openai_agent.py      # OpenAI integration
│   ├── storage/
│   │   ├── __init__.py
│   │   └── gcp.py               # Google Cloud Storage integration
│   ├── voice/
│   │   ├── __init__.py
│   │   └── twilio_handler.py    # Twilio voice handling
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── main.py                  # FastAPI application
│   └── utils.py                 # Utility functions
├── .env                         # Environment variables (create this)
├── .env.example                 # Example environment file
├── Dockerfile                   # For containerization
├── requirements.txt             # Python dependencies
└── README.md                    # This documentation
```

## Troubleshooting

### Call Connected but No Response
- Check OpenAI API key is valid
- Ensure `gpt-4o` model is available to your account
- Verify your environment variables are set correctly

### Slow Initial Response
- The agent initialization happens in background, initial greeting might be generic
- After first interaction, responses should be contextualized by the AI

### Error Saving Transcripts
- Verify GCP credentials are correctly set up
- Ensure the service account has Storage Object Admin permissions
- Check that your bucket exists and is accessible

