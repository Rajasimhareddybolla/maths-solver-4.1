"""
Math Solver Package

A professional production-level math solver using OpenAI's API with support
for text and image inputs, comprehensive logging, and robust error handling.
"""

from .api import solve_math_problem, solve_from_images, get_usage_statistics
from .app import MathSolverApp
from .solver import MathSolver, OpenAIClient
from .exceptions import MathSolverError, InputValidationError, APIError
from .logging_system import conversation_logger
from .batch import process_math_problems_batch, batch_processor, BatchProcessor, BatchResult

__version__ = "4.1.0"
__all__ = [
    "solve_math_problem",
    "solve_from_images", 
    "get_usage_statistics",
    "MathSolverApp",
    "MathSolver",
    "OpenAIClient",
    "MathSolverError",
    "InputValidationError", 
    "APIError",
    "conversation_logger",
    "process_math_problems_batch",
    "batch_processor",
    "BatchProcessor",
    "BatchResult"
]

from .app import MathSolverApp, create_app, get_app
from .solver import MathSolver, OpenAIClient
from .input_handlers import InputHandler, TextInputProcessor, ImageInputProcessor
from .logging_system import ConversationLogger, ConversationEntry
from .exceptions import (
    MathSolverError,
    APIError,
    InputValidationError,
    ConfigurationError,
    LoggingError,
    FileProcessingError
)

# Main API functions - Primary entry points
from .api import (
    solve_math_problem,
    solve_from_images,
    solve_with_context,
    get_conversation_history,
    get_usage_statistics,
    validate_configuration,
    solve,  # Alias for solve_math_problem
    solve_images,  # Alias for solve_from_images
    get_history,  # Alias for get_conversation_history
    get_stats  # Alias for get_usage_statistics
)

__version__ = "1.0.0"
__author__ = "Math Solver Team"

__all__ = [
    # Primary API functions (recommended entry points)
    'solve_math_problem',
    'solve_from_images', 
    'solve_with_context',
    'get_conversation_history',
    'get_usage_statistics',
    'validate_configuration',
    
    # Convenience aliases
    'solve',
    'solve_images',
    'get_history',
    'get_stats',
    
    # Main application
    'MathSolverApp',
    'create_app',
    'get_app',
    
    # Core components
    'MathSolver',
    'OpenAIClient',
    'InputHandler',
    'TextInputProcessor',
    'ImageInputProcessor',
    'ConversationLogger',
    'ConversationEntry',
    
    # Exceptions
    'MathSolverError',
    'APIError',
    'InputValidationError',
    'ConfigurationError',
    'LoggingError',
    'FileProcessingError',
    
    # Metadata
    '__version__',
    '__author__'
]
