"""Command-line interface for the Math Solver application.

This module provides a user-friendly CLI for interacting with the math solver,
including support for text input, image processing, and various output formats.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from math_solver import get_app, MathSolverError


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Production-level Mathematics Problem Solver using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a text problem
  python -m math_solver "Solve 3x + 11 = 14"
  
  # Solve from a single image
  python -m math_solver --image problem.png
  
  # Solve from multiple images
  python -m math_solver --image problem1.png --image problem2.png
  python -m math_solver --images problem1.png problem2.png problem3.png
  
  # Solve with both text and images (fresh context)
  python -m math_solver "Find the derivative" --image graph.jpg --image equation.png --ignore-history
  
  # Get usage statistics
  python -m math_solver --stats
  
  # View recent conversations
  python -m math_solver --history 5
        """
    )
    
    # Input options
    parser.add_argument(
        'problem',
        nargs='?',
        help='Text description of the math problem to solve'
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        action='append',
        help='Path to an image containing the math problem (can be used multiple times for multiple images)'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Multiple image paths containing the math problem'
    )
    
    # Remove the no-ocr option since we're not using OCR anymore
    
    # Output options
    parser.add_argument(
        '--output-format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed information'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except the solution'
    )
    
    # Utility options
    parser.add_argument(
        '--stats',
        type=int,
        nargs='?',
        const=30,
        help='Show usage statistics for the last N days (default: 30)'
    )
    
    parser.add_argument(
        '--history',
        type=int,
        nargs='?',
        const=10,
        help='Show the last N conversations (default: 10)'
    )
    
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration and exit'
    )
    
    parser.add_argument(
        '--conversation-id',
        type=str,
        help='Specify a custom conversation ID for tracking'
    )
    
    parser.add_argument(
        '--ignore-history',
        action='store_true',
        help='Ignore conversation history and start fresh context'
    )
    
    return parser


def format_output(result: dict, output_format: str, verbose: bool = False) -> str:
    """Format the solver result for display.
    
    Args:
        result: Result dictionary from the solver.
        output_format: Output format ('text' or 'json').
        verbose: Whether to include verbose information.
        
    Returns:
        Formatted output string.
    """
    if output_format == 'json':
        if verbose:
            return json.dumps(result, indent=2, ensure_ascii=False)
        else:
            # Return only essential information for JSON output
            essential = {
                'conversation_id': result.get('conversation_id'),
                'solution': result.get('formatted_output', ''),
                'success': result.get('success', True),
                'token_usage': result.get('token_usage', {}),
                'processing_time': result.get('processing_time', 0)
            }
            return json.dumps(essential, indent=2, ensure_ascii=False)
    
    # Text format
    output_lines = []
    
    if not verbose:
        # Simple output - just the solution
        output_lines.append(result.get('formatted_output', 'No solution available'))
    else:
        # Verbose output with metadata
        output_lines.append("=== Math Problem Solution ===")
        output_lines.append("")
        
        if result.get('input_information', {}).get('text_input'):
            output_lines.append(f"Problem: {result['input_information']['text_input']}")
            output_lines.append("")
        
        if result.get('input_information', {}).get('image_paths'):
            image_paths = result['input_information']['image_paths']
            if len(image_paths) == 1:
                output_lines.append(f"Image: {image_paths[0]}")
            else:
                output_lines.append(f"Images ({len(image_paths)}):")
                for i, path in enumerate(image_paths, 1):
                    output_lines.append(f"  {i}. {path}")
            output_lines.append("")
        
        output_lines.append("Solution:")
        output_lines.append(result.get('formatted_output', 'No solution available'))
        output_lines.append("")
        
        # Add metadata
        output_lines.append("=== Metadata ===")
        output_lines.append(f"Conversation ID: {result.get('conversation_id', 'N/A')}")
        output_lines.append(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
        
        token_usage = result.get('token_usage', {})
        if token_usage:
            output_lines.append(f"Token Usage: {token_usage.get('total_tokens', 0)} total")
            output_lines.append(f"  - Prompt: {token_usage.get('prompt_tokens', 0)}")
            output_lines.append(f"  - Completion: {token_usage.get('completion_tokens', 0)}")
    
    return '\n'.join(output_lines)


def format_statistics(stats: dict) -> str:
    """Format usage statistics for display.
    
    Args:
        stats: Statistics dictionary.
        
    Returns:
        Formatted statistics string.
    """
    lines = [
        "=== Usage Statistics ===",
        f"Total Conversations: {stats.get('total_conversations', 0)}",
        f"Successful: {stats.get('successful_conversations', 0)}",
        f"Failed: {stats.get('failed_conversations', 0)}",
        f"Total Tokens Used: {stats.get('total_tokens', 0):,}",
        f"Average Tokens per Conversation: {stats.get('average_tokens_per_conversation', 0):.1f}",
        "",
        "Token Breakdown:",
        f"  - Prompt Tokens: {stats.get('total_prompt_tokens', 0):,}",
        f"  - Completion Tokens: {stats.get('total_completion_tokens', 0):,}"
    ]
    return '\n'.join(lines)


def format_history(conversations: list) -> str:
    """Format conversation history for display.
    
    Args:
        conversations: List of conversation dictionaries.
        
    Returns:
        Formatted history string.
    """
    if not conversations:
        return "No conversation history found."
    
    lines = ["=== Recent Conversations ===", ""]
    
    for i, conv in enumerate(reversed(conversations), 1):
        lines.append(f"{i}. {conv.get('timestamp', 'Unknown time')}")
        lines.append(f"   ID: {conv.get('conversation_id', 'N/A')}")
        
        if conv.get('input_text'):
            input_preview = conv['input_text'][:100]
            if len(conv['input_text']) > 100:
                input_preview += "..."
            lines.append(f"   Input: {input_preview}")
        
        if conv.get('input_image_path'):
            # Handle both single and multiple image paths
            image_info = conv['input_image_path']
            if image_info.startswith('[') and image_info.endswith(']'):
                # Multiple images (stored as string representation of list)
                lines.append(f"   Images: {image_info}")
            else:
                # Single image
                lines.append(f"   Image: {image_info}")
        
        lines.append(f"   Success: {conv.get('success', True)}")
        lines.append(f"   Tokens: {conv.get('token_usage', {}).get('total_tokens', 0)}")
        lines.append("")
    
    return '\n'.join(lines)


def main() -> int:
    """Main entry point for the CLI application.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        app = get_app()
        
        # Handle utility commands
        if args.validate_config:
            validation = app.validate_configuration()
            if validation['valid']:
                print("✓ Configuration is valid")
                if validation['warnings']:
                    print("\nWarnings:")
                    for warning in validation['warnings']:
                        print(f"  - {warning}")
                return 0
            else:
                print("✗ Configuration validation failed:")
                for error in validation['errors']:
                    print(f"  - {error}")
                return 1
        
        if args.stats is not None:
            stats = app.get_statistics(args.stats)
            print(format_statistics(stats))
            return 0
        
        if args.history is not None:
            history = app.get_conversation_history(args.history)
            print(format_history(history))
            return 0
        
        # Main problem solving
        if not args.problem and not args.image and not args.images:
            parser.print_help()
            return 1
        
        # Combine image inputs
        image_paths = []
        if args.image:
            image_paths.extend(args.image)
        if args.images:
            image_paths.extend(args.images)
        
        # Solve the problem
        result = app.solve_problem(
            text_input=args.problem,
            image_paths=image_paths if image_paths else None,
            conversation_id=args.conversation_id,
            ignore_history=args.ignore_history
        )
        
        # Format and display output
        if not args.quiet:
            formatted_output = format_output(result, args.output_format, args.verbose)
            print(formatted_output)
        else:
            # Quiet mode - just the solution
            print(result.get('formatted_output', 'No solution available'))
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
    except MathSolverError as e:
        if not args.quiet:
            print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        if not args.quiet:
            print(f"Unexpected error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
