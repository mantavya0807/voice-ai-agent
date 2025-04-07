from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/test", response_class=PlainTextResponse)
async def test_voice(request: Request):
    """Super simple test endpoint for Twilio."""
    logger.info("TEST ENDPOINT CALLED - Received test voice webhook")
    
    # Create a very simple TwiML response
    response = VoiceResponse()
    response.say("This is a test response from the Voice AI Agent.")
    
    return Response(content=str(response), media_type="application/xml")