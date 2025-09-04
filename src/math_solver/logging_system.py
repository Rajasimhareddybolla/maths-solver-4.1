"""Comprehensive logging system for the Math Solver application.

This module provides structured logging with conversation tracking,
token usage monitoring, and performance metrics.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict

from config import config_manager


@dataclass
class ConversationEntry:
    """Represents a single conversation entry in the logs."""
    
    conversation_id: str
    timestamp: str
    input_text: Optional[str] = None
    input_image_path: Optional[str] = None
    output_text: str = ""
    token_usage: Dict[str, int] = None
    processing_time_seconds: float = 0.0
    model_used: str = ""
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.token_usage is None:
            self.token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }


class ConversationLogger:
    """Handles conversation logging with structured data storage."""
    
    def __init__(self, logs_dir: Optional[Path] = None):
        """Initialize the conversation logger.
        
        Args:
            logs_dir: Directory to store log files. Defaults to configured logs directory.
        """
        self.logs_dir = logs_dir or Path(config_manager.config_dir) / config_manager.app_config.logs_directory
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create separate log files for different purposes
        self.conversation_log_file = self.logs_dir / "conversations.jsonl"
        self.error_log_file = self.logs_dir / "errors.log"
        self.performance_log_file = self.logs_dir / "performance.log"
        
        self._setup_loggers()
    
    def _setup_loggers(self) -> None:
        """Set up different loggers for various purposes."""
        # Error logger
        self.error_logger = logging.getLogger("math_solver.errors")
        self.error_logger.setLevel(getattr(logging, config_manager.logging_config.log_level))
        
        error_handler = RotatingFileHandler(
            self.error_log_file,
            maxBytes=config_manager.logging_config.max_log_size_mb * 1024 * 1024,
            backupCount=config_manager.logging_config.backup_count
        )
        error_formatter = logging.Formatter(config_manager.logging_config.log_format)
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Performance logger
        self.performance_logger = logging.getLogger("math_solver.performance")
        self.performance_logger.setLevel(logging.INFO)
        
        perf_handler = RotatingFileHandler(
            self.performance_log_file,
            maxBytes=config_manager.logging_config.max_log_size_mb * 1024 * 1024,
            backupCount=config_manager.logging_config.backup_count
        )
        perf_formatter = logging.Formatter(config_manager.logging_config.log_format)
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)
        
        # Console logger for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        
        # Add console handler to both loggers
        self.error_logger.addHandler(console_handler)
        self.performance_logger.addHandler(console_handler)
    
    def generate_conversation_id(self) -> str:
        """Generate a unique conversation ID.
        
        Returns:
            Unique conversation identifier.
        """
        return str(uuid.uuid4())
    
    def log_conversation(self, entry: ConversationEntry) -> None:
        """Log a conversation entry to the JSONL file.
        
        Args:
            entry: Conversation entry to log.
        """
        try:
            # Convert to dictionary and ensure JSON serializable
            entry_dict = asdict(entry)
            
            # Write to JSONL file (one JSON object per line)
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry_dict, ensure_ascii=False) + '\n')
            
            # Log performance metrics
            self.performance_logger.info(
                f"Conversation {entry.conversation_id}: "
                f"tokens={entry.token_usage['total_tokens']}, "
                f"time={entry.processing_time_seconds:.2f}s, "
                f"success={entry.success}"
            )
            
        except Exception as e:
            self.error_logger.error(f"Failed to log conversation: {e}")
    
    def log_error(self, conversation_id: str, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log an error with context.
        
        Args:
            conversation_id: ID of the conversation where error occurred.
            error: The exception that occurred.
            context: Additional context information.
        """
        context = context or {}
        self.error_logger.error(
            f"Conversation {conversation_id}: {type(error).__name__}: {error}",
            extra={'context': context}
        )
    
    def get_conversation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent conversation history.
        
        Args:
            limit: Maximum number of conversations to retrieve.
            
        Returns:
            List of conversation dictionaries.
        """
        conversations = []
        
        if not self.conversation_log_file.exists():
            return conversations
        
        try:
            with open(self.conversation_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Get the last N lines
            for line in lines[-limit:]:
                if line.strip():
                    conversations.append(json.loads(line.strip()))
            
        except Exception as e:
            self.error_logger.error(f"Failed to read conversation history: {e}")
        
        return conversations
    
    def get_token_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get token usage statistics for the specified period.
        
        Args:
            days: Number of days to look back.
            
        Returns:
            Dictionary with token usage statistics.
        """
        stats = {
            'total_conversations': 0,
            'total_tokens': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'average_tokens_per_conversation': 0,
            'successful_conversations': 0,
            'failed_conversations': 0
        }
        
        if not self.conversation_log_file.exists():
            return stats
        
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        try:
            with open(self.conversation_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        
                        # Parse timestamp
                        entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                        if entry_time < cutoff_date:
                            continue
                        
                        stats['total_conversations'] += 1
                        if entry.get('success', True):
                            stats['successful_conversations'] += 1
                        else:
                            stats['failed_conversations'] += 1
                        
                        token_usage = entry.get('token_usage', {})
                        stats['total_tokens'] += token_usage.get('total_tokens', 0)
                        stats['total_prompt_tokens'] += token_usage.get('prompt_tokens', 0)
                        stats['total_completion_tokens'] += token_usage.get('completion_tokens', 0)
            
            if stats['total_conversations'] > 0:
                stats['average_tokens_per_conversation'] = (
                    stats['total_tokens'] / stats['total_conversations']
                )
        
        except Exception as e:
            self.error_logger.error(f"Failed to calculate token usage stats: {e}")
        
        return stats


# Global logger instance
conversation_logger = ConversationLogger()
