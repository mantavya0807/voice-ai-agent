import logging
import time
import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from openai import OpenAI
from datetime import datetime

from app.config import OPENAI_API_KEY, ADVANCED_FEATURES_ENABLED

logger = logging.getLogger(__name__)

# Create a global OpenAI client to reuse across instances
global_openai_client = OpenAI(api_key=OPENAI_API_KEY)

class OpenAIAgent:
    def __init__(self, model="gpt-4o", caller_info=None):
        """
        Initialize the OpenAI Agent.
        
        Args:
            model: The model to use for the assistant
            caller_info: Optional caller information for personalization
        """
        self.model = model
        self.client = global_openai_client  # Use global client instead of creating a new one
        self.assistant = None
        self.thread = None
        self.caller_info = caller_info or {}
        self.conversation_state = {
            "intent": None,
            "sentiment": "neutral",
            "topics": [],
            "follow_up_questions": []
        }
        
        # Track timing for performance optimization
        self.setup_start_time = time.time()
        self.setup_assistant()
        setup_duration = time.time() - self.setup_start_time
        logger.info(f"Initialized OpenAI Agent with model: {model} in {setup_duration:.2f} seconds")

    def setup_assistant(self):
        """Create an OpenAI Assistant with enhanced capabilities."""
        try:
            # Build instructions based on caller info if available
            base_instructions = """
            You are a helpful voice assistant. Provide clear, concise responses 
            suitable for voice conversations. Keep your responses brief and to the point,
            as they will be read aloud to the caller. Avoid using visual elements or
            references that wouldn't make sense in a voice call.
            
            Guidelines:
            1. Keep responses under 3-4 sentences when possible
            2. Use natural, conversational language
            3. Avoid complex terminology unless specifically requested
            4. Be empathetic and personable
            5. If you don't know something, be honest about it
            6. IMPORTANT: Always respond in the context of the current conversation.
               Pay close attention to what the caller is asking and provide relevant answers.
            7. If the caller's question isn't clear, politely ask for clarification.
            8. If you detect a change in topic, adapt to the new topic immediately.
            9. If the speech recognition might be incorrect, try to infer what the caller
               might have actually meant based on context.
            """
            
            # Add personalization if caller information exists
            personalized_instructions = ""
            if self.caller_info:
                caller_name = self.caller_info.get("name", "the caller")
                personalized_instructions = f"""
                This caller's name is {caller_name}.
                """
                
                if "preferences" in self.caller_info:
                    prefs = self.caller_info["preferences"]
                    personalized_instructions += f"""
                    Caller preferences: {json.dumps(prefs)}
                    """
                    
                if "history" in self.caller_info:
                    personalized_instructions += """
                    Reference past interactions when relevant.
                    """
                    
                # Add previous conversations to instructions if available
                if "previous_conversations" in self.caller_info:
                    prev_convos = self.caller_info["previous_conversations"]
                    if prev_convos:
                        personalized_instructions += """
                        Here's a summary of the previous conversations with this caller:
                        """
                        
                        for i, convo in enumerate(prev_convos[:3]):  # Only include up to 3 recent conversations
                            date = convo.get("timestamp", "unknown date")
                            summary = convo.get("summary", "No summary available")
                            topics = ", ".join(convo.get("topics", []))
                            
                            personalized_instructions += f"""
                            Conversation {i+1} ({date}): {summary}
                            Topics: {topics}
                            """
            
            # Create an assistant - only create tools if advanced features are enabled
            tools = []
            if ADVANCED_FEATURES_ENABLED:
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_conversation",
                            "description": "Analyze the current conversation state",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "intent": {
                                        "type": "string",
                                        "description": "The inferred intent of the caller"
                                    },
                                    "sentiment": {
                                        "type": "string",
                                        "enum": ["positive", "neutral", "negative"],
                                        "description": "The sentiment of the caller"
                                    },
                                    "topics": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Key topics mentioned in the conversation"
                                    },
                                    "follow_up_questions": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Potential follow-up questions to ask"
                                    }
                                },
                                "required": ["intent", "sentiment", "topics"]
                            }
                        }
                    }
                ]
            
            # Create an assistant with a timeout
            create_start = time.time()
            self.assistant = self.client.beta.assistants.create(
                name="Enhanced Voice AI Agent",
                instructions=base_instructions + personalized_instructions,
                model=self.model,
                tools=tools if tools else None
            )
            create_duration = time.time() - create_start
            logger.info(f"Created assistant: {self.assistant.id} in {create_duration:.2f} seconds")
            
            # Create a thread for the conversation
            thread_start = time.time()
            self.thread = self.client.beta.threads.create()
            thread_duration = time.time() - thread_start
            logger.info(f"Created thread: {self.thread.id} in {thread_duration:.2f} seconds")
            
            # Add initial system message if caller info exists
            if self.caller_info:
                msg_start = time.time()
                self.client.beta.threads.messages.create(
                    thread_id=self.thread.id,
                    role="user",
                    content=f"This is a new call from {self.caller_info.get('name', 'a caller')}."
                )
                msg_duration = time.time() - msg_start
                logger.info(f"Added initial message in {msg_duration:.2f} seconds")
                
                # Run the assistant to process this context
                run_start = time.time()
                run = self.client.beta.threads.runs.create(
                    thread_id=self.thread.id,
                    assistant_id=self.assistant.id
                )
                
                # Wait for completion
                self._wait_for_run(run.id)
                run_duration = time.time() - run_start
                logger.info(f"Initial assistant run completed in {run_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error setting up assistant: {str(e)}")
            raise

    def _wait_for_run(self, run_id):
        """
        Wait for an assistant run to complete.
        
        Args:
            run_id: The ID of the run to wait for
            
        Returns:
            The completed run status
        """
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            try:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run_id
                )
                
                if run_status.status == 'completed':
                    return run_status
                elif run_status.status in ['failed', 'expired', 'cancelled']:
                    logger.error(f"Run failed with status: {run_status.status}")
                    if hasattr(run_status, 'last_error'):
                        logger.error(f"Error details: {run_status.last_error}")
                    raise Exception(f"Run {run_id} failed with status: {run_status.status}")
                
                # Check if requires action (function calling)
                elif run_status.status == 'requires_action':
                    self._handle_required_action(run_status)
                
                # Wait before polling again - use exponential backoff
                wait_time = 0.5 * (1.5 ** attempt)  # Start with 0.5s, then 0.75s, 1.125s, etc.
                time.sleep(wait_time)
                attempt += 1
                
            except Exception as e:
                logger.error(f"Error checking run status (attempt {attempt}/{max_attempts}): {str(e)}")
                attempt += 1
                time.sleep(0.5)
                
                # If this is the last attempt, re-raise the exception
                if attempt >= max_attempts:
                    raise
        
        raise Exception(f"Timed out waiting for run {run_id} to complete after {max_attempts} attempts")

    def _handle_required_action(self, run_status):
        """
        Handle actions required by the assistant during a run.
        
        Args:
            run_status: The current run status
        """
        try:
            if not hasattr(run_status, 'required_action'):
                return
                
            required_action = run_status.required_action
            if not hasattr(required_action, 'submit_tool_outputs'):
                return
                
            tool_calls = required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            
            for tool_call in tool_calls:
                if tool_call.function.name == "analyze_conversation":
                    # Process the analyze_conversation function call
                    try:
                        # Use the existing conversation state
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(self.conversation_state)
                        })
                    except Exception as e:
                        logger.error(f"Error handling analyze_conversation function: {str(e)}")
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": json.dumps({
                                "intent": "unknown",
                                "sentiment": "neutral",
                                "topics": [],
                                "follow_up_questions": []
                            })
                        })
            
            # Submit the tool outputs back to the assistant
            if tool_outputs:
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=run_status.id,
                    tool_outputs=tool_outputs
                )
                
        except Exception as e:
            logger.error(f"Error handling required action: {str(e)}")

    async def process_voice_input(self, text_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process voice input text and generate a response using OpenAI Assistant.
        
        Args:
            text_input: The transcribed text from voice input
            
        Returns:
            Tuple[str, Dict[str, Any]]: The generated response text and conversation metadata
        """
        process_start = time.time()
        logger.info(f"Processing voice input: {text_input}")
        
        # Clean up input text if needed
        cleaned_input = self._preprocess_input(text_input)
        
        try:
            # Add the user message to the thread
            add_msg_start = time.time()
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=cleaned_input
            )
            add_msg_duration = time.time() - add_msg_start
            logger.info(f"Added user message in {add_msg_duration:.2f}s")
            
            # Run the assistant on the thread
            run_start = time.time()
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                temperature=0.7  # Only use temperature, no max_tokens
            )
            
            # Wait for the run to complete
            run_status = self._wait_for_run(run.id)
            run_duration = time.time() - run_start
            logger.info(f"Assistant run completed in {run_duration:.2f}s")
            
            # Update the conversation state with analysis if available
            if ADVANCED_FEATURES_ENABLED:
                # Analyze the conversation
                self._analyze_conversation(cleaned_input)
            
            # Retrieve messages
            get_msg_start = time.time()
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            get_msg_duration = time.time() - get_msg_start
            logger.info(f"Retrieved messages in {get_msg_duration:.2f}s")
            
            # Get the last assistant message
            response_text = None
            for message in messages.data:
                if message.role == "assistant" and message.content:
                    for content_part in message.content:
                        if hasattr(content_part, 'text') and hasattr(content_part.text, 'value'):
                            response_text = content_part.text.value
                            break
                    if response_text:
                        break
            
            if not response_text:
                response_text = "I'm sorry, I couldn't generate a response."
                
            # Post-process response if needed
            response_text = self._postprocess_response(response_text)
            
            total_duration = time.time() - process_start
            logger.info(f"Generated response in {total_duration:.2f}s: {response_text}")
            return response_text, self.conversation_state
            
        except Exception as e:
            logger.error(f"Error processing voice input: {str(e)}")
            logger.error(traceback.format_exc())
            return "I'm sorry, I encountered an error processing your request.", {}

    def _preprocess_input(self, text_input: str) -> str:
        """
        Preprocess the input text to handle speech recognition issues.
        
        Args:
            text_input: The original input text
            
        Returns:
            str: The preprocessed text
        """
        if not text_input:
            return text_input
            
        # Convert to lowercase for easier processing
        text = text_input.strip()
        
        # Add preprocessing logic here if needed
        # For example, correct common speech recognition errors
        
        return text
        
    def _postprocess_response(self, response_text: str) -> str:
        """
        Postprocess the response text for better voice output.
        
        Args:
            response_text: The original response text
            
        Returns:
            str: The postprocessed text
        """
        if not response_text:
            return response_text
            
        # Trim whitespace
        text = response_text.strip()
        
        # Truncate overly long responses for voice
        if len(text) > 500:
            # Find the last sentence break within the first 500 chars
            last_period = text.rfind('.', 0, 500)
            if last_period > 0:
                text = text[:last_period+1]
                
        # Ensure the text ends with proper punctuation
        if not text[-1] in ['.', '!', '?']:
            text += '.'
            
        return text
    
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
            
            # Extract potential topics
            words = text_input.lower().split()
            nouns = [word for word in words if len(word) > 3 and word not in [
                'this', 'that', 'then', 'when', 'what', 'where', 'which', 'how', 
                'would', 'could', 'should', 'hello', 'thanks', 'thank', 'please'
            ]]
            
            if nouns:
                # Update topics rather than replacing
                existing_topics = set(self.conversation_state.get("topics", []))
                new_topics = set(nouns[:3])
                combined_topics = existing_topics.union(new_topics)
                self.conversation_state["topics"] = list(combined_topics)[:5]  # Keep only top 5
                
            # Attempt to infer intent
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
        
        Returns:
            List[Dict[str, Any]]: The conversation transcript
        """
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id
            )
            
            transcript = []
            for message in messages.data:
                content_text = ""
                if message.content:
                    for content_part in message.content:
                        if hasattr(content_part, 'text') and hasattr(content_part.text, 'value'):
                            content_text = content_part.text.value
                            break
                
                # Create ISO format timestamp if created_at is a unix timestamp
                timestamp = None
                if hasattr(message, 'created_at'):
                    try:
                        timestamp = datetime.fromtimestamp(message.created_at).isoformat()
                    except (TypeError, ValueError):
                        # If not a valid timestamp, use as is
                        timestamp = str(message.created_at)
                
                transcript.append({
                    "role": message.role,
                    "content": content_text,
                    "created_at": timestamp
                })
            
            # Reverse to get chronological order (oldest first)
            transcript.reverse()
            
            return transcript
        except Exception as e:
            logger.error(f"Error getting transcript: {str(e)}")
            return []
            
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dict[str, Any]: Summary of the conversation
        """
        try:
            transcript = self.get_transcript()
            
            # Create a summary prompt
            summary_content = "Please provide a brief summary of this conversation:\n\n"
            
            for message in transcript:
                role = "Customer" if message["role"] == "user" else "Assistant"
                summary_content += f"{role}: {message['content']}\n"
            
            # Use the API directly for this specific task
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that summarizes conversations."},
                    {"role": "user", "content": summary_content}
                ],
                max_tokens=150
            )
            
            summary = completion.choices[0].message.content
            
            return {
                "summary": summary,
                "message_count": len(transcript),
                "topics": self.conversation_state.get("topics", []),
                "sentiment": self.conversation_state.get("sentiment", "neutral")
            }
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return {
                "summary": "Unable to generate summary",
                "message_count": 0,
                "topics": [],
                "sentiment": "unknown"
            }