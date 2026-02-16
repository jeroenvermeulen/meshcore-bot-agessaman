#!/usr/bin/env python3
"""
AI command for the MeshCore Bot
Provides AI-powered responses using Google Gemini API
"""

import asyncio
from typing import Optional
from .base_command import BaseCommand
from ..models import MeshMessage


class AiCommand(BaseCommand):
    """Handles AI command with Gemini integration"""
    
    # Plugin metadata
    name = "ai"
    keywords = ['ai']
    description = "Ask a question to Gemini AI (max 125 chars, usage: ai <question>)"
    category = "entertainment"
    cooldown_seconds = 10  # 10 second cooldown per user to prevent API abuse
    requires_dm = False  # Works in both channels and DMs
    requires_internet = True  # Requires internet access for Gemini API
    
    # Constants
    MAX_QUESTION_LENGTH = 125 # 128 - 'ai ' prefix
    MAX_RESPONSE_LENGTH = 128
    TIMEOUT = 15  # seconds
    
    def __init__(self, bot):
        """Initialize the AI command.
        
        Args:
            bot: The bot instance.
        """
        super().__init__(bot)
        
        # Load configuration
        self.ai_enabled = self.get_config_value('Ai_Command', 'enabled', fallback=True, value_type='bool')
        self.gemini_api_key = bot.config.get('External_Data', 'gemini_api_key', fallback='')
        self.bot_name = bot.config.get('Bot', 'bot_name', fallback='MeshCore-Bot')
        
        # Initialize Gemini client (lazy loading)
        self._client = None
    
    def get_help_text(self) -> str:
        """Get help text for the AI command.
        
        Returns:
            str: The help text for this command.
        """
        return f"Usage: ai <question> - Ask a question to Gemini AI (max {self.MAX_QUESTION_LENGTH} chars)"
    
    def can_execute(self, message: MeshMessage) -> bool:
        """Check if this command can be executed.
        
        Args:
            message: The message triggering the command.
            
        Returns:
            bool: True if command can execute, False otherwise.
        """
        # Use base class for channel access, DM requirements, and cooldown
        if not super().can_execute(message):
            return False
        
        # Check if AI command is enabled
        if not self.ai_enabled:
            return False
        
        # Check if API key is configured
        if not self.gemini_api_key:
            self.logger.warning("Gemini API key not configured")
            return False
        
        return True
    
    def _get_gemini_client(self):
        """Get or create Gemini client (lazy loading).
        
        Returns:
            Client: The Gemini client instance.
        """
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.gemini_api_key)
                self.logger.info("Gemini client initialized successfully")
            except ImportError:
                self.logger.error("google-genai package not installed. Install with: pip install google-genai")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {e}")
                raise
        return self._client
    
    async def execute(self, message: MeshMessage) -> bool:
        """Execute the AI command.
        
        Args:
            message: The message triggering the command.
            
        Returns:
            bool: True if executed successfully, False otherwise.
        """
        content = message.content.strip()
        
        # Remove command prefix
        if content.startswith('!'):
            content = content[1:].strip()
        
        # Parse the command to extract question
        parts = content.split(maxsplit=1)
        if len(parts) < 2:
            await self.send_response(message, f"Usage: ai <question> (max {self.MAX_QUESTION_LENGTH} chars)")
            return True
        
        question = parts[1].strip()
        
        # Validate question length
        if len(question) > self.MAX_QUESTION_LENGTH:
            await self.send_response(message, f"Question too long! Max {self.MAX_QUESTION_LENGTH} chars.")
            return True
        
        if len(question) == 0:
            await self.send_response(message, "Please provide a question.")
            return True
        
        try:
            # Record execution for this user
            self.record_execution(message.sender_id)
            
            # Get AI response
            ai_response = await self.get_ai_response(question, message.sender_id)
            
            if ai_response is None:
                await self.send_response(message, "Sorry, couldn't get a response from AI. Try again later!")
                return True
            
            # Send the response
            await self.send_response(message, ai_response)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in AI command: {e}")
            await self.send_response(message, "Sorry, something went wrong with the AI!")
            return True

    async def get_ai_response(self, question: str, sender_id: str) -> Optional[str]:
        """Get AI response from Gemini API.

        Args:
            question: The user's question.
            sender_id: The sender's ID for the prompt.

        Returns:
            Optional[str]: The AI response, or None if it fails.
        """
        try:
            # Build the system prompt with stronger emphasis on character limit
            system_prompt = (
                f'You are a bot on MeshCore named "{self.bot_name}" which is publicly available. '
                f'Anyone can ask you a question of maximum {self.MAX_QUESTION_LENGTH} characters. '
                f'CRITICAL: Your answer MUST be maximum {self.MAX_RESPONSE_LENGTH} characters total. '
                f'Keep responses concise and complete sentences within {self.MAX_RESPONSE_LENGTH} characters. '
                f'Your answer is the end of the conversation, the user can\'t respond to your answer.'
            )

            # Build the user prompt
            user_prompt = f'The question from user "{sender_id}" is: "{question}"'

            self.logger.debug(f"Sending to Gemini - System: {system_prompt}")
            self.logger.debug(f"Sending to Gemini - User: {user_prompt}")

            # Get Gemini client
            client = self._get_gemini_client()

            # Call Gemini API (simplified - no executor needed)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config={
                    'system_instruction': system_prompt,
                    'max_output_tokens': 200,
                    'temperature': 0.7,
                }
            )

            # Extract text from response
            if response and hasattr(response, 'text'):
                ai_text = response.text.strip()
                self.logger.info(f"AI response ({len(ai_text)} chars): {ai_text}")
                return ai_text
            else:
                self.logger.warning("No text in Gemini response")
                return None

        except asyncio.TimeoutError:
            self.logger.error("Timeout getting response from Gemini API")
            return None
        except Exception as e:
            self.logger.error(f"Error getting response from Gemini API: {e}")
            return None
