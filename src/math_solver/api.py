"""Main functional entry point for the Math Solver.

This module provides the primary Python API for using the math solver
with options for history management and context control.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from .app import get_app
from .exceptions import MathSolverError


def solve_math_problem(
    problem: Optional[str] = None,
    image_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    ignore_history: bool = False,
    conversation_id: Optional[str] = None,
    additional_context: Optional[str] = None
) -> Dict[str, Any]:
    """Solve a mathematics problem using OpenAI's API.
    
    This is the main functional entry point for the math solver.
    
    Args:
        problem: Text description of the math problem.
        image_paths: Optional path(s) to image(s) containing the problem.
                    Can be a single path or list of paths.
        ignore_history: If True, don't include conversation history in the context.
                       This creates a fresh conversation without previous context.
        conversation_id: Optional conversation ID for tracking. If ignore_history
                        is True, this will create a new conversation.
        additional_context: Optional additional context or instructions to include.
        
    Returns:
        Dictionary containing:
        - conversation_id: Unique identifier for this conversation
        - solution: Structured solution content
        - formatted_output: Human-readable solution text
        - token_usage: Token consumption details
        - processing_time: Time taken to solve the problem
        - success: Whether the solution was successful
        - images_processed: Number of images processed
        - input_information: Details about the processed input
        
    Raises:
        MathSolverError: If the problem cannot be solved due to various errors.
        
    Examples:
        # Simple text problem
        result = solve_math_problem("Solve 3x + 11 = 14")
        
        # Problem with image
        result = solve_math_problem(
            "What is shown in this graph?",
            image_paths="graph.png"
        )
        
        # Multiple images without history
        result = solve_math_problem(
            "Compare these equations",
            image_paths=["eq1.png", "eq2.png"],
            ignore_history=True
        )
        
        # Get the solution
        print(result['formatted_output'])
    """
    try:
        app = get_app()
        
        # Validate inputs
        if not problem and not image_paths:
            raise MathSolverError("Either 'problem' or 'image_paths' must be provided")
        
        # Build the full problem text
        full_problem = problem
        if additional_context:
            if full_problem:
                full_problem += f"\n\nAdditional context: {additional_context}"
            else:
                full_problem = additional_context
        
        # Handle history management
        if ignore_history:
            # Create a fresh conversation without history context
            conversation_id = None  # Let the system generate a new one
        
        # Solve the problem
        result = app.solve_problem(
            text_input=full_problem,
            image_paths=image_paths,
            conversation_id=conversation_id
        )
        
        # Add metadata about history usage
        result['history_ignored'] = ignore_history
        
        return result
        
    except Exception as e:
        if isinstance(e, MathSolverError):
            raise
        else:
            raise MathSolverError(f"Failed to solve math problem: {e}")


def solve_from_images(
    image_paths: Union[str, Path, List[Union[str, Path]]],
    ignore_history: bool = False,
    conversation_id: Optional[str] = None,
    additional_context: Optional[str] = None
) -> Dict[str, Any]:
    """Solve a mathematics problem from image(s) only.
    
    Args:
        image_paths: Path(s) to image(s) containing the math problem.
        ignore_history: If True, don't include conversation history.
        conversation_id: Optional conversation ID for tracking.
        additional_context: Optional additional context or instructions.
        
    Returns:
        Dictionary containing the solution and metadata.
    """
    return solve_math_problem(
        problem=None,
        image_paths=image_paths,
        ignore_history=ignore_history,
        conversation_id=conversation_id,
        additional_context=additional_context
    )


def solve_with_context(
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
        additional_context: Optional additional context or instructions.
        ignore_history: If True, don't include conversation history.
        conversation_id: Optional conversation ID for tracking.
        
    Returns:
        Dictionary containing the solution and metadata.
    """
    return solve_math_problem(
        problem=problem,
        image_paths=image_paths,
        ignore_history=ignore_history,
        conversation_id=conversation_id,
        additional_context=additional_context
    )


def get_conversation_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent conversation history.
    
    Args:
        limit: Maximum number of conversations to retrieve.
        
    Returns:
        List of conversation dictionaries with metadata.
    """
    try:
        app = get_app()
        return app.get_conversation_history(limit)
    except Exception as e:
        raise MathSolverError(f"Failed to get conversation history: {e}")


def get_usage_statistics(days: int = 30) -> Dict[str, Any]:
    """Get usage statistics for the specified period.
    
    Args:
        days: Number of days to analyze.
        
    Returns:
        Dictionary with usage statistics including token counts and success rates.
    """
    try:
        app = get_app()
        return app.get_statistics(days)
    except Exception as e:
        raise MathSolverError(f"Failed to get usage statistics: {e}")


def validate_configuration() -> Dict[str, Any]:
    """Validate the math solver configuration.
    
    Returns:
        Dictionary with validation results including any errors or warnings.
    """
    try:
        app = get_app()
        return app.validate_configuration()
    except Exception as e:
        raise MathSolverError(f"Failed to validate configuration: {e}")


# Convenience aliases for backward compatibility and ease of use
solve = solve_math_problem
solve_images = solve_from_images
get_history = get_conversation_history
get_stats = get_usage_statistics


# Main function for direct script execution
def main():
    """Main function for direct script execution."""
    import sys
    from .cli import main as cli_main
    return cli_main()


if __name__ == '__main__':
    main()
