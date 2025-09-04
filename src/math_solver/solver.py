"""Core OpenAI client wrapper and response processing for math solving.

This module provides a clean interface to OpenAI's API with proper
error handling, response processing, and token tracking.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import base64
import mimetypes

from openai import OpenAI
from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseOutputMessage,
    ResponseOutputText
)

from config import config_manager
from .logging_system import ConversationLogger, ConversationEntry, conversation_logger
from .exceptions import MathSolverError, APIError, InputValidationError


class OpenAIClient:
    """Wrapper for OpenAI client with enhanced functionality."""
    
    def __init__(self, logger: Optional[ConversationLogger] = None):
        """Initialize OpenAI client.
        
        Args:
            logger: Optional conversation logger instance.
        """
        self.logger = logger or conversation_logger
        self.client = None
        self.system_instructions = ""
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client and load system instructions."""
        try:
            api_key = config_manager.get_api_key()
            self.client = OpenAI(api_key=api_key)
            self.system_instructions = config_manager.get_system_instructions()
        except Exception as e:
            raise APIError(f"Failed to initialize OpenAI client: {e}")
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for API submission.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded image string with proper MIME type prefix.
            
        Raises:
            InputValidationError: If image format is not supported or file is too large.
        """
        if not image_path.exists():
            raise InputValidationError(f"Image file not found: {image_path}")
        
        # Check file size
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config_manager.app_config.max_image_size_mb:
            raise InputValidationError(
                f"Image file too large: {file_size_mb:.1f}MB. "
                f"Maximum allowed: {config_manager.app_config.max_image_size_mb}MB"
            )
        
        # Check file format
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith('image/'):
            raise InputValidationError(f"Unsupported file type: {mime_type}")
        
        file_extension = image_path.suffix.lower().lstrip('.')
        if file_extension not in config_manager.app_config.supported_image_formats:
            raise InputValidationError(
                f"Unsupported image format: {file_extension}. "
                f"Supported formats: {config_manager.app_config.supported_image_formats}"
            )
        
        try:
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_image}"
        except Exception as e:
            raise InputValidationError(f"Failed to encode image: {e}")
    
    def _encode_multiple_images(self, image_paths: List[Path]) -> List[str]:
        """Encode multiple images to base64 for API submission.
        
        Args:
            image_paths: List of paths to image files.
            
        Returns:
            List of base64 encoded image strings.
            
        Raises:
            InputValidationError: If any image processing fails.
        """
        encoded_images = []
        
        for i, image_path in enumerate(image_paths):
            try:
                encoded_image = self._encode_image(image_path)
                encoded_images.append(encoded_image)
            except InputValidationError as e:
                raise InputValidationError(f"Failed to encode image {i+1}: {e}")
        
        return encoded_images
    
    def _extract_response_content(self, response: Any) -> Dict[str, Any]:
        """Extract structured content from OpenAI response.
        
        Args:
            response: OpenAI API response object.
            
        Returns:
            Dictionary containing extracted text, code, and metadata.
        """
        result = {
            'text_responses': [],
            'code_executions': [],
            'raw_output': []
        }
        
        try:
            for item in response.output:
                result['raw_output'].append(str(item))
                
                if isinstance(item, ResponseCodeInterpreterToolCall):
                    result['code_executions'].append({
                        'code': item.code,
                        'execution_id': getattr(item, 'id', None)
                    })
                    
                elif isinstance(item, ResponseOutputMessage):
                    for content_item in item.content:
                        if isinstance(content_item, ResponseOutputText):
                            result['text_responses'].append(content_item.text)
        
        except Exception as e:
            self.logger.error_logger.error(f"Error extracting response content: {e}")
            result['text_responses'].append(f"Error processing response: {e}")
        
        return result
    
    def solve_math_problem(
        self,
        text_input: Optional[str] = None,
        image_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        conversation_id: Optional[str] = None,
        ignore_history: bool = False
    ) -> Dict[str, Any]:
        """Solve a mathematics problem using OpenAI's API.
        
        Args:
            text_input: Text description of the math problem.
            image_paths: Optional path(s) to image(s) containing the problem.
                        Can be a single path or list of paths.
            conversation_id: Optional conversation ID for tracking.
            ignore_history: If True, don't include conversation history in context.
            
        Returns:
            Dictionary containing the solution, metadata, and logging information.
            
        Raises:
            MathSolverError: If solving fails or input is invalid.
        """
        if not text_input and not image_paths:
            raise InputValidationError("Either text_input or image_paths must be provided")
        
        # Handle conversation ID based on history preference
        if ignore_history:
            # Force a new conversation ID to avoid history context
            conversation_id = self.logger.generate_conversation_id()
        else:
            conversation_id = conversation_id or self.logger.generate_conversation_id()
        
        start_time = time.time()
        
        # Convert single image path to list for uniform processing
        if image_paths and isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        
        # Initialize conversation entry
        entry = ConversationEntry(
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            input_text=text_input,
            input_image_path=str(image_paths) if image_paths else None,
            model_used=config_manager.model_config.model
        )
        
        try:
            # Prepare the input prompt
            prompt_parts = []
            
            # Add history context if not ignored
            if not ignore_history:
                # Get recent conversation history for context (limit to last 3 conversations)
                recent_history = self.logger.get_conversation_history(limit=3)
                if recent_history:
                    prompt_parts.append("Previous conversation context (for reference only):")
                    for i, conv in enumerate(recent_history[-3:], 1):  # Last 3 conversations
                        if conv.get('input_text') and conv.get('output_text'):
                            prompt_parts.append(f"Previous Q{i}: {conv['input_text'][:200]}...")
                            prompt_parts.append(f"Previous A{i}: {conv['output_text'][:200]}...")
                    prompt_parts.append("---\nCurrent problem:")
            
            # Add current problem
            if text_input and image_paths:
                if len(image_paths) == 1:
                    prompt_parts.append(f"Math problem description: {text_input}")
                    prompt_parts.append("Please also analyze the provided image for additional context.")
                else:
                    prompt_parts.append(f"Math problem description: {text_input}")
                    prompt_parts.append(f"Please analyze all {len(image_paths)} provided images for additional context.")
            elif text_input:
                prompt_parts.append(text_input)
            else:
                if len(image_paths) == 1:
                    prompt_parts.append("Please solve the math problem shown in the image.")
                else:
                    prompt_parts.append(f"Please solve the math problem shown in the {len(image_paths)} images. Analyze all images together to understand the complete problem.")
            
            final_prompt = '\n\n'.join(prompt_parts)
            
            # Prepare API call parameters
            api_params = {
                'model': config_manager.model_config.model,
                'tools': [
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto"}
                    }
                ],
                'instructions': self.system_instructions,
                'input': [{
                    "type": "message",
                    "role": "user", 
                    "content": [{"type": "input_text", "text": final_prompt}]
                }]
            }
            
            # Add optional parameters if configured
            if config_manager.model_config.max_tokens:
                # Responses API expects max_output_tokens
                api_params['max_output_tokens'] = config_manager.model_config.max_tokens
            
            if config_manager.model_config.temperature is not None:
                api_params['temperature'] = config_manager.model_config.temperature
            
            # Handle image input(s)
            if image_paths:
                image_paths = [Path(path) for path in image_paths]
                encoded_images = self._encode_multiple_images(image_paths)
                
                # For OpenAI responses API, images should be included in the input content
                # Create a structured input with both text and images as a message
                message_content = [{"type": "input_text", "text": final_prompt}]
                
                # Add each image to the message content
                for encoded_image in encoded_images:
                    message_content.append({
                        "type": "input_image",
                        "image_url": encoded_image
                    })
                
                # Update the input parameter to use structured content as a message
                api_params['input'] = [{
                    "type": "message",
                    "role": "user",
                    "content": message_content
                }]
            
            # Debug: print api_params to see what's being passed
            print("API params being passed:", list(api_params.keys()))
            
            # Make API call
            response = self.client.responses.create(**api_params)
            
            # Extract response content
            content = self._extract_response_content(response)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    'prompt_tokens': getattr(response.usage, 'input_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'output_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }
            
            # Update conversation entry
            entry.output_text = self._format_output(content)
            entry.token_usage = token_usage
            entry.processing_time_seconds = processing_time
            entry.success = True
            
            # Log the conversation
            self.logger.log_conversation(entry)
            
            return {
                'conversation_id': conversation_id,
                'solution': content,
                'formatted_output': entry.output_text,
                'token_usage': token_usage,
                'processing_time': processing_time,
                'success': True,
                'images_processed': len(image_paths) if image_paths else 0,
                'history_ignored': ignore_history
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update conversation entry with error
            entry.success = False
            entry.error_message = str(e)
            entry.processing_time_seconds = processing_time
            
            # Log the error
            self.logger.log_error(conversation_id, e, {
                'text_input': text_input,
                'image_paths': str(image_paths) if image_paths else None,
                'ignore_history': ignore_history
            })
            self.logger.log_conversation(entry)
            
            if isinstance(e, (InputValidationError, APIError)):
                raise
            else:
                raise MathSolverError(f"Unexpected error during problem solving: {e}")
    
    def _format_output(self, content: Dict[str, Any]) -> str:
        """Format the extracted content into a readable output.
        
        Args:
            content: Extracted content dictionary.
            
        Returns:
            Formatted output string.
        """
        output_parts = []
        
        # Add text responses
        for text_response in content['text_responses']:
            output_parts.append("=== Assistant's Response ===")
            output_parts.append(text_response)
            output_parts.append("")
        
        # Add code executions
        for i, code_exec in enumerate(content['code_executions'], 1):
            output_parts.append(f"=== Code Execution {i} ===")
            output_parts.append(code_exec['code'])
            output_parts.append("")
        
        return '\n'.join(output_parts).strip()


class MathSolver:
    """High-level mathematics problem solver."""
    
    def __init__(self, logger: Optional[ConversationLogger] = None):
        """Initialize the math solver.
        
        Args:
            logger: Optional conversation logger instance.
        """
        self.client = OpenAIClient(logger)
        self.logger = logger or conversation_logger
    
    def solve(
        self,
        problem: str,
        image_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        ignore_history: bool = False,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Solve a mathematics problem.
        
        Args:
            problem: Text description of the math problem.
            image_paths: Optional path(s) to image(s) containing the problem.
                        Can be a single path or list of paths.
            ignore_history: If True, don't include conversation history in context.
            conversation_id: Optional conversation ID for tracking.
            
        Returns:
            Dictionary containing the solution and metadata.
        """
        return self.client.solve_math_problem(
            text_input=problem,
            image_paths=image_paths,
            conversation_id=conversation_id,
            ignore_history=ignore_history
        )
    
    def solve_from_images(
        self, 
        image_paths: Union[str, Path, List[Union[str, Path]]],
        ignore_history: bool = False,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Solve a mathematics problem from image(s) only.
        
        Args:
            image_paths: Path(s) to image(s) containing the math problem.
                        Can be a single path or list of paths.
            ignore_history: If True, don't include conversation history in context.
            conversation_id: Optional conversation ID for tracking.
            
        Returns:
            Dictionary containing the solution and metadata.
        """
        return self.client.solve_math_problem(
            text_input=None,
            image_paths=image_paths,
            conversation_id=conversation_id,
            ignore_history=ignore_history
        )
    
    def solve_with_context(
        self,
        problem: str,
        image_paths: Union[str, Path, List[Union[str, Path]]],
        additional_context: Optional[str] = None,
        ignore_history: bool = False,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Solve a mathematics problem with both text and image context.
        
        Args:
            problem: Text description of the math problem.
            image_paths: Path(s) to image(s) containing the problem.
                        Can be a single path or list of paths.
            additional_context: Optional additional context or instructions.
            ignore_history: If True, don't include conversation history in context.
            conversation_id: Optional conversation ID for tracking.
            
        Returns:
            Dictionary containing the solution and metadata.
        """
        full_problem = problem
        if additional_context:
            full_problem += f"\n\nAdditional context: {additional_context}"
        
        return self.client.solve_math_problem(
            text_input=full_problem,
            image_paths=image_paths,
            conversation_id=conversation_id,
            ignore_history=ignore_history
        )
    
    def get_usage_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for the specified period.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            Dictionary with usage statistics.
        """
        return self.logger.get_token_usage_stats(days)
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history.
        
        Args:
            limit: Maximum number of conversations to retrieve.
            
        Returns:
            List of recent conversations.
        """
        return self.logger.get_conversation_history(limit)
