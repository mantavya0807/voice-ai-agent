import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def generate_call_id() -> str:
    """Generate a unique call ID."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def format_transcript(transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format the transcript for storage.
    
    Args:
        transcript: List of transcript messages
        
    Returns:
        List[Dict[str, Any]]: Formatted transcript
    """
    formatted = []
    for message in transcript:
        formatted.append({
            "timestamp": datetime.now().isoformat(),
            "role": message.get("role", "unknown"),
            "content": message.get("content", "")
        })
    return formatted

def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("voice_ai_agent.log")
        ]
    )