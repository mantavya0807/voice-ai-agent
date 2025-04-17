import logging
import json
import requests
import os
import base64
import time
import traceback
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import exceptions as gcp_exceptions
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import hashlib

from app.config import GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS, USE_BIGQUERY

logger = logging.getLogger(__name__)

class GCPStorage:
    def __init__(self, call_id: str):
        """
        Initialize the GCP Storage client.
        
        Args:
            call_id: Unique identifier for the call
        """
        # Set GCP credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
        
        self.storage_client = None
        self.bucket_name = GCS_BUCKET_NAME
        self.call_id = call_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log credential path for debugging
        logger.info(f"Using GCP credentials from: {GOOGLE_APPLICATION_CREDENTIALS}")
        logger.info(f"Using GCS bucket: {self.bucket_name}")
        
        try:
            self.storage_client = storage.Client()
            logger.info(f"Successfully initialized GCP Storage client")
        except Exception as e:
            logger.error(f"Error initializing GCP Storage client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Verify bucket exists
        self._verify_bucket()
        
        # Initialize BigQuery client if enabled
        self.bigquery_client = None
        if USE_BIGQUERY:
            try:
                self.bigquery_client = bigquery.Client()
                logger.info("Successfully initialized BigQuery client")
            except Exception as e:
                logger.error(f"Error initializing BigQuery client: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Initialized GCP Storage for call ID: {call_id}")
    
    def _verify_bucket(self):
        """Verify that the bucket exists and is accessible"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # Check if bucket exists first
            exists = False
            try:
                exists = bucket.exists()
                logger.info(f"Bucket existence check result: {exists}")
            except Exception as e:
                logger.error(f"Error checking if bucket exists: {str(e)}")
                logger.error(traceback.format_exc())
            
            if not exists:
                logger.error(f"Bucket {self.bucket_name} does not exist. Attempting to create it...")
                try:
                    # Try to create the bucket
                    new_bucket = self.storage_client.create_bucket(self.bucket_name)
                    logger.info(f"Successfully created bucket {self.bucket_name}")
                    bucket = new_bucket
                except Exception as e:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Ensure required directories exist
            self._ensure_directory_exists("transcripts")
            self._ensure_directory_exists("audio")
            
        except Exception as e:
            logger.error(f"Error verifying bucket: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _ensure_directory_exists(self, directory_name):
        """Ensure that a directory/prefix exists in the bucket"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(f"{directory_name}/.keep")
            
            # Check if directory marker exists
            exists = False
            try:
                exists = blob.exists()
                logger.info(f"Directory {directory_name} marker existence check: {exists}")
            except Exception as e:
                logger.error(f"Error checking if directory marker exists: {str(e)}")
                logger.error(traceback.format_exc())
            
            if not exists:
                blob.upload_from_string('', content_type='text/plain')
                logger.info(f"Created directory marker {directory_name}/.keep")
        except Exception as e:
            logger.error(f"Error ensuring directory {directory_name} exists: {str(e)}")
            logger.error(traceback.format_exc())
        
    async def save_transcript_with_metadata(
        self, 
        transcript: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Save the call transcript with metadata to GCS.
        
        Args:
            transcript: List of message dictionaries with role and content
            metadata: Additional metadata about the call
            
        Returns:
            Optional[str]: The blob name if successful, None otherwise
        """
        max_retries = 3
        retry_count = 0
        
        # Debug info
        logger.info(f"Saving transcript with {len(transcript)} messages for call {self.call_id}")
        
        while retry_count < max_retries:
            try:
                # Get the bucket
                bucket = self.storage_client.bucket(self.bucket_name)
                
                # Add timestamp to metadata
                metadata["saved_at"] = datetime.now().isoformat()
                
                # Create a blob for the transcript
                blob_name = f"transcripts/{self.call_id}_{self.timestamp}.json"
                blob = bucket.blob(blob_name)
                
                # Convert transcript to JSON
                transcript_data = {
                    "metadata": metadata,
                    "messages": transcript
                }
                
                # Print a sample of what we're saving
                sample_data = {
                    "metadata": metadata,
                    "message_count": len(transcript),
                    "sample_messages": transcript[:2] if len(transcript) > 2 else transcript
                }
                logger.info(f"Transcript sample data: {json.dumps(sample_data, indent=2)}")
                
                transcript_json = json.dumps(transcript_data, indent=2)
                
                # Upload the transcript
                logger.info(f"Uploading transcript to {blob_name}")
                blob.upload_from_string(
                    transcript_json,
                    content_type="application/json"
                )
                
                # Verify the upload was successful
                try:
                    if blob.exists():
                        logger.info(f"Successfully verified transcript upload to {blob_name}")
                    else:
                        logger.error(f"Verification failed: Transcript blob {blob_name} does not exist after upload")
                except Exception as e:
                    logger.error(f"Error verifying transcript upload: {str(e)}")
                    logger.error(traceback.format_exc())
                
                logger.info(f"Saved transcript to {blob_name}")
                return blob_name
                
            except Exception as e:
                logger.error(f"Error saving transcript (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                logger.error(traceback.format_exc())
                retry_count += 1
                time.sleep(1)  # Wait before retrying
            
        return None
    
    async def _save_to_bigquery(self, transcript_data: Dict[str, Any]) -> bool:
        """
        Save transcript data to BigQuery for analytics.
        
        Args:
            transcript_data: The transcript data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.bigquery_client:
                logger.warning("BigQuery client not initialized, skipping BigQuery save")
                return False
                
            dataset_id = "voice_ai"
            table_id = "call_transcripts"
            
            # Check if dataset exists, create it if not
            try:
                dataset_ref = self.bigquery_client.dataset(dataset_id)
                self.bigquery_client.get_dataset(dataset_ref)
                logger.info(f"BigQuery dataset {dataset_id} exists")
            except Exception as e:
                logger.info(f"BigQuery dataset {dataset_id} does not exist, creating it. Error: {str(e)}")
                # Dataset doesn't exist, create it
                dataset = bigquery.Dataset(self.bigquery_client.dataset(dataset_id))
                dataset.location = "US"  # Set the location
                self.bigquery_client.create_dataset(dataset)
                logger.info(f"Created BigQuery dataset {dataset_id}")
            
            # Check if table exists, create it if not
            table_ref = self.bigquery_client.dataset(dataset_id).table(table_id)
            try:
                self.bigquery_client.get_table(table_ref)
                logger.info(f"BigQuery table {table_id} exists")
            except Exception as e:
                logger.info(f"BigQuery table {table_id} does not exist, creating it. Error: {str(e)}")
                # Table doesn't exist, create it
                schema = [
                    bigquery.SchemaField("call_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("timestamp", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("message_timestamp", "STRING"),
                    bigquery.SchemaField("role", "STRING"),
                    bigquery.SchemaField("content", "STRING"),
                    bigquery.SchemaField("caller_number", "STRING"),
                    bigquery.SchemaField("sentiment", "STRING")
                ]
                table = bigquery.Table(table_ref, schema=schema)
                self.bigquery_client.create_table(table)
                logger.info(f"Created BigQuery table {table_id}")
            
            # Flatten the data for BigQuery
            rows = []
            
            metadata = transcript_data["metadata"]
            messages = transcript_data["messages"]
            
            for message in messages:
                row = {
                    "call_id": self.call_id,
                    "timestamp": self.timestamp,
                    "message_timestamp": message.get("created_at", self.timestamp),
                    "role": message.get("role", "unknown"),
                    "content": message.get("content", ""),
                    "caller_number": metadata.get("caller_number", ""),
                    "sentiment": metadata.get("sentiment", "unknown")
                }
                rows.append(row)
            
            # Insert rows
            errors = self.bigquery_client.insert_rows_json(table_ref, rows)
            
            if errors:
                logger.error(f"Errors inserting into BigQuery: {errors}")
                return False
                
            logger.info(f"Saved {len(rows)} messages to BigQuery")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to BigQuery: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    async def save_audio(self, audio_url: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Save the call audio to GCS.
        
        Args:
            audio_url: URL to the audio file
            metadata: Additional metadata about the call
        
        Returns:
            Optional[str]: The blob name if successful, None otherwise
        """
        max_retries = 3
        retry_count = 0
        
        logger.info(f"Attempting to save audio from URL: {audio_url}")
        
        while retry_count < max_retries:
            try:
                # Get the bucket
                bucket = self.storage_client.bucket(self.bucket_name)
                
                # Create a blob for the audio
                blob_name = f"audio/{self.call_id}_{self.timestamp}.mp3"
                blob = bucket.blob(blob_name)
                
                # Download the audio file
                headers = {}
                # Add authentication if needed for Twilio recordings
                if "twilio" in audio_url:
                    # Extract service type from metadata or default to "restaurant"
                    service_type = metadata.get("service_type", "restaurant")
                    
                    # Import the appropriate config based on service type
                    if service_type == "restaurant":
                        from app.config import RESTAURANT_TWILIO_ACCOUNT_SID as account_sid
                        from app.config import RESTAURANT_TWILIO_AUTH_TOKEN as auth_token
                    elif service_type == "hairdresser":
                        from app.config import HAIRDRESSER_TWILIO_ACCOUNT_SID as account_sid
                        from app.config import HAIRDRESSER_TWILIO_AUTH_TOKEN as auth_token
                    else:
                        # Fallback to restaurant
                        from app.config import RESTAURANT_TWILIO_ACCOUNT_SID as account_sid
                        from app.config import RESTAURANT_TWILIO_AUTH_TOKEN as auth_token
                    
                    auth_string = base64.b64encode(f"{account_sid}:{auth_token}".encode()).decode()
                    headers["Authorization"] = f"Basic {auth_string}"
                
                logger.info(f"Downloading audio from: {audio_url}")
                response = requests.get(audio_url, headers=headers)
                
                if response.status_code == 200:
                    logger.info(f"Successfully downloaded audio, content length: {len(response.content)} bytes")
                    # Upload the audio
                    logger.info(f"Uploading audio to {blob_name}")
                    blob.upload_from_string(
                        response.content,
                        content_type="audio/mpeg"
                    )
                    
                    # Set metadata for the audio file
                    metadata = {
                        "call_id": self.call_id,
                        "timestamp": self.timestamp,
                        "source": "twilio",
                        "content_type": "audio/mpeg"
                    }
                    blob.metadata = metadata
                    blob.patch()
                    
                    # Verify the upload was successful
                    try:
                        if blob.exists():
                            logger.info(f"Successfully verified audio upload to {blob_name}")
                        else:
                            logger.error(f"Verification failed: Audio blob {blob_name} does not exist after upload")
                    except Exception as e:
                        logger.error(f"Error verifying audio upload: {str(e)}")
                        logger.error(traceback.format_exc())
                    
                    logger.info(f"Saved audio to {blob_name}")
                    return blob_name
                else:
                    logger.error(f"Error downloading audio (HTTP {response.status_code}): {response.text}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Waiting before retry {retry_count + 1}...")
                        time.sleep(2)  # Wait before retrying
                    else:
                        logger.error(f"Failed to download audio after {max_retries} attempts")
                        return None
                    
            except gcp_exceptions.NotFound as e:
                logger.error(f"Bucket or blob not found: {str(e)}")
                # Create the directory if it doesn't exist
                self._ensure_directory_exists("audio")
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Error saving audio (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                logger.error(traceback.format_exc())
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Waiting before retry {retry_count + 1}...")
                    time.sleep(1)  # Wait before retrying
                else:
                    logger.error(f"Failed to save audio after {max_retries} attempts")
                    return None
                    
        return None
            
    async def get_previous_interactions(self, caller_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve previous interactions with a caller.
        
        Args:
            caller_id: The phone number or identifier of the caller
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of previous interactions
        """
        try:
            # Create a hash of the caller_id for privacy
            hashed_caller_id = hashlib.sha256(caller_id.encode()).hexdigest()
            logger.info(f"Looking up previous interactions for hashed caller ID: {hashed_caller_id[:8]}...")
            
            # Look up in BigQuery if enabled
            if USE_BIGQUERY and self.bigquery_client:
                logger.info("Using BigQuery for previous interactions lookup")
                return await self._get_interactions_from_bigquery(hashed_caller_id)
            
            # Otherwise, look in GCS
            logger.info("Using GCS for previous interactions lookup")
            bucket = self.storage_client.bucket(self.bucket_name)
            
            # List all transcript files
            blobs = list(bucket.list_blobs(prefix="transcripts/"))
            logger.info(f"Found {len(blobs)} transcript files in GCS")
            
            previous_interactions = []
            
            for blob in blobs:
                try:
                    # Download the transcript
                    content = blob.download_as_text()
                    transcript_data = json.loads(content)
                    
                    # Check if this transcript belongs to the caller
                    metadata = transcript_data.get("metadata", {})
                    transcript_caller_id = metadata.get("caller_number", "")
                    
                    # Hash the transcript caller ID for comparison
                    hashed_transcript_caller_id = hashlib.sha256(transcript_caller_id.encode()).hexdigest() if transcript_caller_id else ""
                    
                    if hashed_transcript_caller_id == hashed_caller_id:
                        # Extract timestamp from metadata or filename
                        timestamp = metadata.get("timestamp", blob.name.split("_")[1] if "_" in blob.name else "unknown")
                        
                        # Add this transcript to the list
                        summary = {
                            "call_id": metadata.get("call_id", "unknown"),
                            "timestamp": timestamp,
                            "transcript_url": f"gs://{self.bucket_name}/{blob.name}",
                            "message_count": len(transcript_data.get("messages", [])),
                        }
                        previous_interactions.append(summary)
                        logger.info(f"Found matching transcript: {blob.name}")
                        
                except Exception as e:
                    logger.error(f"Error processing transcript {blob.name}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Sort by timestamp (newest first)
            previous_interactions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            logger.info(f"Retrieved {len(previous_interactions)} previous interactions for caller")
            return previous_interactions[:5]  # Return the 5 most recent interactions
            
        except Exception as e:
            logger.error(f"Error retrieving previous interactions: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    async def _get_interactions_from_bigquery(self, hashed_caller_id: str) -> List[Dict[str, Any]]:
        """
        Get previous interactions from BigQuery.
        
        Args:
            hashed_caller_id: Hashed caller ID for privacy
            
        Returns:
            List[Dict[str, Any]]: List of previous interactions
        """
        try:
            query = f"""
            SELECT
                call_id,
                timestamp,
                COUNT(*) as message_count,
                ANY_VALUE(sentiment) as sentiment
            FROM
                `voice_ai.call_transcripts`
            WHERE
                caller_id_hash = @caller_id_hash
            GROUP BY
                call_id, timestamp
            ORDER BY
                timestamp DESC
            LIMIT 5
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("caller_id_hash", "STRING", hashed_caller_id)
                ]
            )
            
            query_job = self.bigquery_client.query(query, job_config=job_config)
            results = query_job.result()
            
            previous_interactions = []
            for row in results:
                interaction = {
                    "call_id": row.call_id,
                    "timestamp": row.timestamp,
                    "message_count": row.message_count,
                    "sentiment": row.sentiment
                }
                previous_interactions.append(interaction)
                
            return previous_interactions
            
        except Exception as e:
            logger.error(f"Error retrieving interactions from BigQuery: {str(e)}")
            logger.error(traceback.format_exc())
            return []
