"""Custom exceptions for the Math Solver application.

This module defines custom exception classes for better error handling
and debugging throughout the application.
"""


class MathSolverError(Exception):
    """Base exception class for Math Solver application."""
    
    def __init__(self, message: str, details: dict = None):
        """Initialize the exception.
        
        Args:
            message: Error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class APIError(MathSolverError):
    """Exception raised for OpenAI API related errors."""
    
    def __init__(self, message: str, status_code: int = None, details: dict = None):
        """Initialize the API error.
        
        Args:
            message: Error message.
            status_code: HTTP status code if applicable.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.status_code = status_code


class InputValidationError(MathSolverError):
    """Exception raised for input validation failures."""
    
    def __init__(self, message: str, field_name: str = None, details: dict = None):
        """Initialize the validation error.
        
        Args:
            message: Error message.
            field_name: Name of the field that failed validation.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.field_name = field_name


class ConfigurationError(MathSolverError):
    """Exception raised for configuration related errors."""
    pass


class LoggingError(MathSolverError):
    """Exception raised for logging system errors."""
    pass


class FileProcessingError(MathSolverError):
    """Exception raised for file processing errors."""
    
    def __init__(self, message: str, file_path: str = None, details: dict = None):
        """Initialize the file processing error.
        
        Args:
            message: Error message.
            file_path: Path to the file that caused the error.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, details)
        self.file_path = file_path
