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
        self.gemini_model = self.get_config_value('Ai_Command', 'gemini_model', fallback='gemini-2.5-flash', value_type='str')
        self.bot_name = bot.config.get('Bot', 'bot_name', fallback='MeshCore-Bot')

        # Load system prompt template (with default fallback)
        default_system_prompt = (
            'You are a bot on MeshCore named "{bot_name}" which is publicly available. '
            'CRITICAL: Your answer MUST be maximum {max_length} characters total. '
            'Keep responses concise and complete sentences within {max_length} characters. '
            'Your answer is the end of the conversation, the user can\'t respond to your answer.'
        )
        self.system_prompt_template = self.get_config_value(
            'Ai_Command', 'system_prompt',
            fallback=default_system_prompt,
            value_type='str'
        )

        # Load AI generation parameters
        self.max_output_tokens = self.get_config_value('Ai_Command', 'max_output_tokens', fallback=1000, value_type='int')
        self.temperature = self.get_config_value('Ai_Command', 'temperature', fallback=0.7, value_type='float')

        # Initialize Gemini client (lazy loading)
        self._client = None
    
    def get_help_text(self) -> str:
        """Get help text for the AI command.
        
        Returns:
            str: The help text for this command.
        """
        return f"Usage: ai <question> - Ask a question to Gemini AI"
    
    def can_execute(self, message: MeshMessage) -> bool:
        """Check if this command can be executed.

        Args:
            message: The message triggering the command.

        Returns:
            bool: True if command can execute, False otherwise.
        """
        # Use base class for channel access, DM requirements, and cooldown
        if not super().can_execute(message):
            # Log why the command can't execute for debugging
            channel_info = f"channel '{message.channel}'" if not message.is_dm else "DM"
            self.logger.debug(f"AI command can't execute in {channel_info} - failed base checks (channel access, DM requirements, or cooldown)")
            return False

        # Check if AI command is enabled
        if not self.ai_enabled:
            self.logger.debug("AI command is disabled in config")
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
                from google.genai.types import HttpOptions
                self._client = genai.Client(
                    api_key=self.gemini_api_key,
                    http_options=HttpOptions(
                        timeout=self.TIMEOUT * 1000  # 15 seconds = 15000 milliseconds
                    )
                )
                self.logger.info("Gemini client initialized successfully with timeout")
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
            await self.send_response(message, f"Usage: ai <question>")
            return True
        
        question = parts[1].strip()

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
            # Build the system prompt from template with placeholder substitution
            system_prompt = self.system_prompt_template.format(
                bot_name=self.bot_name,
                max_length=self.MAX_RESPONSE_LENGTH
            )

            # Build the user prompt
            user_prompt = f'The question from user "{sender_id}" is: "{question}"'

            self.logger.debug(f"Sending to Gemini - System: {system_prompt}")
            self.logger.debug(f"Sending to Gemini - User: {user_prompt}")

            # Get Gemini client
            client = self._get_gemini_client()

            # Run the synchronous API call in a thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=self.gemini_model,
                        contents=user_prompt,
                        config={
                            'system_instruction': system_prompt,
                            'max_output_tokens': self.max_output_tokens,
                            'temperature': self.temperature,
                        }
                    )
                ),
                timeout=self.TIMEOUT
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
