from fastapi import FastAPI, HTTPException, Request, Depends, Header, status, WebSocket, WebSocketDisconnect
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
from contextlib import asynccontextmanager

from app.voice.twilio_handler import router as twilio_router, active_calls, background_save_call_data
from app.voice.mcp_handler import router as mcp_router, active_streams  # Import the new MCP router
from app.config import (
    API_HOST, API_PORT, LOG_LEVEL, API_KEY_REQUIRED, API_KEY,
    ENABLE_CALL_ANALYTICS, DEBUG, USE_MCP,  # Add USE_MCP flag
    GOOGLE_APPLICATION_CREDENTIALS, GCS_BUCKET_NAME
)
from app.utils import setup_logging
from google.cloud import storage

# Configure logging
setup_logging(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Enhanced Voice AI Agent is starting up")
    app_metrics["start_time"] = time.time()
    
    # Log which mode we're using
    if USE_MCP:
        logger.info("Using Twilio Media Control Platform (MCP) for voice processing")
    else:
        logger.info("Using traditional TwiML for voice processing")

    # Ensure required directories exist in GCS if call analytics are enabled
    if ENABLE_CALL_ANALYTICS:
        try:
            if GOOGLE_APPLICATION_CREDENTIALS:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
            else:
                logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Skipping GCS verification.")
                yield

            storage_client = storage.Client()

            try:
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                if not bucket.exists():
                    logger.info(f"Creating bucket {GCS_BUCKET_NAME}")
                    storage_client.create_bucket(GCS_BUCKET_NAME)
                    bucket = storage_client.bucket(GCS_BUCKET_NAME)
            except Exception as e:
                logger.error(f"Error creating/accessing bucket: {str(e)}")
                logger.error(traceback.format_exc())

            try:
                if bucket.exists():
                    if not bucket.blob('transcripts/.keep').exists():
                        bucket.blob('transcripts/.keep').upload_from_string('')
                        logger.info("Created transcripts directory")

                    if not bucket.blob('audio/.keep').exists():
                        bucket.blob('audio/.keep').upload_from_string('')
                        logger.info("Created audio directory")

                    logger.info("GCP Storage directories verified")
                else:
                    logger.error(f"Bucket {GCS_BUCKET_NAME} does not exist. Cannot verify directories.")
            except Exception as e:
                logger.error(f"Error verifying directories: {str(e)}")
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Error verifying GCP Storage directories during startup: {str(e)}")
            logger.error(traceback.format_exc())

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_signal_handler(s)))
            logger.info(f"Registered signal handler for {sig.name}")
        except NotImplementedError:
            logger.warning(f"Cannot register signal handler for {sig.name} on this platform. Manual shutdown required.")
        except Exception as e:
            logger.error(f"Error registering signal handler: {str(e)}")
            logger.error(traceback.format_exc())

    yield

    # Shutdown logic
    logger.info("Enhanced Voice AI Agent is shutting down")
    await save_all_active_calls()
    
    # Clean up WebSocket connections
    for call_id, ws in list(active_streams.items()):
        try:
            await ws.close(1000)
            logger.info(f"Closed WebSocket for call {call_id}")
        except:
            pass
    
    logger.info("Shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Enhanced Voice AI Agent",
    description="A conversational AI voice agent powered by OpenAI Agent SDK and Twilio",
    version="2.0.0",
    debug=DEBUG,
    lifespan=lifespan
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
    if API_KEY_REQUIRED and api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Include routers with conditional API key verification
if API_KEY_REQUIRED:
    # Include the traditional TwiML router
    app.include_router(
        twilio_router,
        prefix="/twilio",
        tags=["twilio"],
        dependencies=[Depends(verify_api_key)]
    )
    
    # Include the new MCP router
    app.include_router(
        mcp_router,
        prefix="/twilio",
        tags=["twilio-mcp"],
        dependencies=[Depends(verify_api_key)]
    )
else:
    # Include routers without API key verification
    app.include_router(twilio_router, prefix="/twilio", tags=["twilio"])
    app.include_router(mcp_router, prefix="/twilio", tags=["twilio-mcp"])

# Application metrics and monitoring
app_metrics = {
    "start_time": time.time(),
    "requests_total": 0,
    "errors_total": 0,
    "calls_total": 0,
    "processing_times": [],
    "active_websockets": 0  # New metric for WebSocket connections
}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    app_metrics["requests_total"] += 1

    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        app_metrics["processing_times"].append(processing_time)

        if len(app_metrics["processing_times"]) > 1000:
            app_metrics["processing_times"] = app_metrics["processing_times"][-1000:]

        if "/twilio/voice" in request.url.path or "/twilio/mcp/incoming" in request.url.path:
            app_metrics["calls_total"] += 1

        return response
    except Exception as e:
        app_metrics["errors_total"] += 1
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting application with MCP {'enabled' if USE_MCP else 'disabled'}")
    # Register WebSocket event handlers
    app.add_event_handler("websocket_connect", websocket_connect)
    app.add_event_handler("websocket_disconnect", websocket_disconnect)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")

async def websocket_connect(websocket: WebSocket):
    app_metrics["active_websockets"] += 1
    logger.debug(f"WebSocket connected, active count: {app_metrics['active_websockets']}")

async def websocket_disconnect(websocket: WebSocket):
    app_metrics["active_websockets"] = max(0, app_metrics["active_websockets"] - 1)
    logger.debug(f"WebSocket disconnected, active count: {app_metrics['active_websockets']}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced Voice AI Agent is running! Call your configured Twilio number to interact.",
        "version": "2.0.0",
        "documentation": "/docs",
        "mode": "mcp" if USE_MCP else "twiml"
    }

@app.get("/health")
async def health_check():
    uptime = time.time() - app_metrics.get("start_time", time.time())
    return {
        "status": "healthy",
        "uptime": uptime
    }

@app.get("/metrics", dependencies=[Depends(verify_api_key)] if API_KEY_REQUIRED else [])
async def get_metrics():
    avg_processing_time = sum(app_metrics["processing_times"]) / len(app_metrics["processing_times"]) if app_metrics["processing_times"] else 0
    uptime = time.time() - app_metrics.get("start_time", time.time())

    return {
        "uptime_seconds": uptime,
        "requests_total": app_metrics["requests_total"],
        "errors_total": app_metrics["errors_total"],
        "error_rate": app_metrics["errors_total"] / app_metrics["requests_total"] if app_metrics["requests_total"] > 0 else 0,
        "calls_total": app_metrics["calls_total"],
        "average_processing_time_ms": avg_processing_time * 1000,
        "active_calls": len(active_calls),
        "active_websockets": app_metrics["active_websockets"],
        "active_streams": len(active_streams)
    }

async def save_all_active_calls():
    logger.info(f"Saving data for {len(active_calls)} active calls before shutdown")

    save_tasks = []
    active_calls_copy = list(active_calls.items())
    for call_id, call_data in active_calls_copy:
        call_sid = call_data.get("call_sid")
        if call_sid:
            logger.info(f"Adding save task for call {call_id}")
            save_tasks.append(background_save_call_data(call_id, call_sid, True))

    if save_tasks:
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

async def shutdown_signal_handler(signum):
    logger.info(f"Received shutdown signal {signum}, initiating graceful shutdown...")

if __name__ == "__main__":
    logger.info(f"Starting Enhanced Voice AI Agent on {API_HOST}:{API_PORT}")
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=DEBUG)