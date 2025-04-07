from fastapi import FastAPI, HTTPException, Request, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import uvicorn
import logging
import json
import os
import time
import asyncio
import signal
import traceback
from typing import Dict, Any, Optional
from datetime import datetime

from app.voice.twilio_handler import router as twilio_router, active_calls, background_save_call_data
from app.config import (
    API_HOST, API_PORT, LOG_LEVEL, API_KEY_REQUIRED, API_KEY,
    ENABLE_CALL_ANALYTICS, DEBUG
)
from app.utils import setup_logging

# Configure logging
setup_logging(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Voice AI Agent",
    description="A conversational AI voice agent powered by OpenAI Agent SDK and Twilio",
    version="2.0.0",
    debug=DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """
    Verify the API key if required.
    
    Args:
        api_key: The API key from the header
        
    Raises:
        HTTPException: If the API key is invalid
    """
    if API_KEY_REQUIRED and api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Include routers with conditional API key verification
if API_KEY_REQUIRED:
    app.include_router(
        twilio_router, 
        prefix="/twilio", 
        tags=["twilio"],
        dependencies=[Depends(verify_api_key)]
    )
else:
    app.include_router(twilio_router, prefix="/twilio", tags=["twilio"])

# Application metrics and monitoring
app_metrics = {
    "start_time": time.time(),
    "requests_total": 0,
    "errors_total": 0,
    "calls_total": 0,
    "processing_times": []
}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware to collect metrics about the application.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        The response from the next handler
    """
    start_time = time.time()
    
    # Increment request counter
    app_metrics["requests_total"] += 1
    
    try:
        response = await call_next(request)
        
        # Record processing time
        processing_time = time.time() - start_time
        app_metrics["processing_times"].append(processing_time)
        
        # Keep only the last 1000 processing times
        if len(app_metrics["processing_times"]) > 1000:
            app_metrics["processing_times"] = app_metrics["processing_times"][-1000:]
        
        # Track call-related metrics
        if "/twilio/voice" in request.url.path:
            app_metrics["calls_total"] += 1
            
        return response
    except Exception as e:
        # Track errors
        app_metrics["errors_total"] += 1
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.get("/")
async def root():
    return {
        "message": "Enhanced Voice AI Agent is running! Call your configured Twilio number to interact.",
        "version": "2.0.0",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "uptime": time.time() - app_metrics["start_time"]
    }

@app.get("/metrics", dependencies=[Depends(verify_api_key)] if API_KEY_REQUIRED else [])
async def get_metrics():
    """Get application metrics."""
    # Calculate some statistics
    avg_processing_time = sum(app_metrics["processing_times"]) / len(app_metrics["processing_times"]) if app_metrics["processing_times"] else 0
    
    return {
        "uptime_seconds": time.time() - app_metrics["start_time"],
        "requests_total": app_metrics["requests_total"],
        "errors_total": app_metrics["errors_total"],
        "error_rate": app_metrics["errors_total"] / app_metrics["requests_total"] if app_metrics["requests_total"] > 0 else 0,
        "calls_total": app_metrics["calls_total"],
        "average_processing_time_ms": avg_processing_time * 1000,
        "active_calls": len(active_calls)
    }

# Function to save all active calls on shutdown
async def save_all_active_calls():
    """Force save all active call data before shutdown"""
    logger.info(f"Saving data for {len(active_calls)} active calls before shutdown")
    
    save_tasks = []
    for call_id, call_data in active_calls.items():
        call_sid = call_data.get("call_sid")
        if call_sid:
            logger.info(f"Adding save task for call {call_id}")
            # Create task for each call
            save_tasks.append(background_save_call_data(call_id, call_sid, True))
    
    if save_tasks:
        # Wait for all save tasks to complete with a timeout
        try:
            await asyncio.wait_for(asyncio.gather(*save_tasks), timeout=10.0)
            logger.info("All call data saved successfully")
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for call data to save")
        except Exception as e:
            logger.error(f"Error saving call data: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        logger.info("No active calls to save")

@app.on_event("startup")
async def startup_event():
    logger.info("Enhanced Voice AI Agent is starting up")
    
    # Ensure required directories exist in GCS if call analytics are enabled
    if ENABLE_CALL_ANALYTICS:
        try:
            from google.cloud import storage
            from app.config import GOOGLE_APPLICATION_CREDENTIALS, GCS_BUCKET_NAME
            
            # Set GCP credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
            
            # Initialize storage client
            storage_client = storage.Client()
            
            # Check if bucket exists
            try:
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                if not bucket.exists():
                    logger.info(f"Creating bucket {GCS_BUCKET_NAME}")
                    storage_client.create_bucket(GCS_BUCKET_NAME)
                    bucket = storage_client.bucket(GCS_BUCKET_NAME)
            except Exception as e:
                logger.error(f"Error creating/accessing bucket: {str(e)}")
                logger.error(traceback.format_exc())
                
            # Create empty objects to ensure directories exist
            try:
                if not bucket.blob('transcripts/.keep').exists():
                    bucket.blob('transcripts/.keep').upload_from_string('')
                    logger.info("Created transcripts directory")
                    
                if not bucket.blob('audio/.keep').exists():
                    bucket.blob('audio/.keep').upload_from_string('')
                    logger.info("Created audio directory")
                    
                logger.info("GCP Storage directories verified")
            except Exception as e:
                logger.error(f"Error verifying directories: {str(e)}")
                logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"Error verifying GCP Storage directories: {str(e)}")
            logger.error(traceback.format_exc())

    # Register signal handlers for proper shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda signum, frame: asyncio.create_task(shutdown_signal_handler(signum)))
            logger.info(f"Registered signal handler for {sig.name}")
        except Exception as e:
            logger.error(f"Error registering signal handler: {str(e)}")
            logger.error(traceback.format_exc())

async def shutdown_signal_handler(signum):
    """Custom signal handler for graceful shutdown"""
    logger.info(f"Received shutdown signal {signum}")
    await save_all_active_calls()
    await shutdown_event()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Enhanced Voice AI Agent is shutting down")
    
    # Save all active calls before shutting down
    await save_all_active_calls()
    
    logger.info("Shutdown complete")

if __name__ == "__main__":
    logger.info(f"Starting Enhanced Voice AI Agent on {API_HOST}:{API_PORT}")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=DEBUG)