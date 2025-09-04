"""Main application module for the Math Solver.

This module provides the primary interface for the math solver application
with comprehensive error handling and logging.
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from config import config_manager
from .solver import MathSolver
from .input_handlers import InputHandler
from .logging_system import conversation_logger
from .exceptions import (
    MathSolverError,
    APIError,
    InputValidationError,
    ConfigurationError,
    FileProcessingError
)


class MathSolverApp:
    """Main application class for the Math Solver."""
    
    def __init__(self):
        """Initialize the Math Solver application."""
        try:
            self.solver = MathSolver()
            self.input_handler = InputHandler()
            self.logger = conversation_logger
            
            # Ensure logs directory exists
            logs_dir = Path(config_manager.config_dir) / config_manager.app_config.logs_directory
            logs_dir.mkdir(exist_ok=True)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Math Solver application: {e}")
    
    def solve_problem(
        self,
        text_input: Optional[str] = None,
        image_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        conversation_id: Optional[str] = None,
        ignore_history: bool = False
    ) -> Dict[str, Any]:
        """Solve a mathematics problem with comprehensive error handling.
        
        Args:
            text_input: Optional text description of the math problem.
            image_paths: Optional path(s) to image(s) containing the problem.
                        Can be a single path or list of paths.
            conversation_id: Optional conversation ID for tracking.
            ignore_history: If True, don't include conversation history in context.
            
        Returns:
            Dictionary containing the solution and metadata.
            
        Raises:
            MathSolverError: If solving fails.
        """
        conversation_id = conversation_id or self.logger.generate_conversation_id()
        
        try:
            # Process and validate input
            processed_input = self.input_handler.process_input(
                text_input=text_input,
                image_paths=image_paths
            )
            
            # Create appropriate prompt
            prompt = self.input_handler.create_prompt(processed_input)
            
            # Solve the problem
            result = self.solver.solve(
                problem=prompt,
                image_paths=image_paths,
                ignore_history=ignore_history,
                conversation_id=conversation_id
            )
            
            # Add input information to result
            result['input_information'] = processed_input
            result['conversation_id'] = conversation_id
            
            return result
            
        except (InputValidationError, FileProcessingError) as e:
            self.logger.log_error(conversation_id, e)
            raise
        except APIError as e:
            self.logger.log_error(conversation_id, e)
            raise MathSolverError(f"API error: {e}")
        except Exception as e:
            self.logger.log_error(conversation_id, e, {
                'text_input': text_input,
                'image_paths': str(image_paths) if image_paths else None,
                'ignore_history': ignore_history,
                'traceback': traceback.format_exc()
            })
            raise MathSolverError(f"Unexpected error: {e}")
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get application usage statistics.
        
        Args:
            days: Number of days to analyze.
            
        Returns:
            Dictionary with usage statistics.
        """
        try:
            return self.solver.get_usage_statistics(days)
        except Exception as e:
            self.logger.error_logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history.
        
        Args:
            limit: Maximum number of conversations to retrieve.
            
        Returns:
            List of recent conversations.
        """
        try:
            return self.solver.get_recent_conversations(limit)
        except Exception as e:
            self.logger.error_logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the application configuration.
        
        Returns:
            Dictionary with validation results.
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Test API key
            config_manager.get_api_key()
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"API key validation failed: {e}")
        
        try:
            # Test system instructions
            instructions = config_manager.get_system_instructions()
            if not instructions:
                validation_result['warnings'].append("System instructions file is empty")
        except Exception as e:
            validation_result['warnings'].append(f"System instructions issue: {e}")
        
        # Check logs directory
        logs_dir = Path(config_manager.config_dir) / config_manager.app_config.logs_directory
        if not logs_dir.exists():
            validation_result['warnings'].append(f"Logs directory does not exist: {logs_dir}")
        
        return validation_result


def create_app() -> MathSolverApp:
    """Factory function to create and configure the Math Solver application.
    
    Returns:
        Configured MathSolverApp instance.
        
    Raises:
        ConfigurationError: If application cannot be properly configured.
    """
    try:
        app = MathSolverApp()
        
        # Validate configuration
        validation = app.validate_configuration()
        if not validation['valid']:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(validation['errors'])}"
            )
        
        # Log warnings if any
        for warning in validation['warnings']:
            app.logger.error_logger.warning(warning)
        
        return app
        
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        else:
            raise ConfigurationError(f"Failed to create application: {e}")


# Global application instance
app = None


def get_app() -> MathSolverApp:
    """Get the global application instance, creating it if necessary.
    
    Returns:
        MathSolverApp instance.
    """
    global app
    if app is None:
        app = create_app()
    return app
