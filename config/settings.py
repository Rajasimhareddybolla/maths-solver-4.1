"""Configuration management for the Math Solver application.

This module handles all configuration settings including API keys,
model parameters, and application settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for OpenAI model parameters."""
    
    model: str = "gpt-4.1"
    max_tokens: Optional[int] = None
    temperature: float = 0.1
    timeout: int = 60


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    
    api_key_file: str = ".openai_api_key"
    system_instructions_file: str = "system_instructions.txt"
    logs_directory: str = "logs"
    max_image_size_mb: int = 20
    supported_image_formats: tuple = ("png", "jpg", "jpeg", "gif", "webp")


class ConfigManager:
    """Manages application configuration and environment variables."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Optional path to configuration directory.
                       Defaults to current working directory.
        """
        self.config_dir = config_dir or Path.cwd()
        self.model_config = ModelConfig()
        self.logging_config = LoggingConfig()
        self.app_config = AppConfig()
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from files and environment variables."""
        # Load from config file if it exists
        config_file = self.config_dir / "config.json"
        if config_file.exists():
            self._load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_file(self, config_file: Path) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update model config
            if 'model' in config_data:
                for key, value in config_data['model'].items():
                    if hasattr(self.model_config, key):
                        setattr(self.model_config, key, value)
            
            # Update logging config
            if 'logging' in config_data:
                for key, value in config_data['logging'].items():
                    if hasattr(self.logging_config, key):
                        setattr(self.logging_config, key, value)
            
            # Update app config
            if 'app' in config_data:
                for key, value in config_data['app'].items():
                    if hasattr(self.app_config, key):
                        setattr(self.app_config, key, value)
                        
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load config file: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Model configuration
        if os.getenv('OPENAI_MODEL'):
            self.model_config.model = os.getenv('OPENAI_MODEL')
        if os.getenv('OPENAI_MAX_TOKENS'):
            self.model_config.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS'))
        if os.getenv('OPENAI_TEMPERATURE'):
            self.model_config.temperature = float(os.getenv('OPENAI_TEMPERATURE'))
        
        # App configuration
        if os.getenv('API_KEY_FILE'):
            self.app_config.api_key_file = os.getenv('API_KEY_FILE')
        if os.getenv('LOGS_DIRECTORY'):
            self.app_config.logs_directory = os.getenv('LOGS_DIRECTORY')
    
    def get_api_key(self) -> str:
        """Get OpenAI API key from file or environment variable.
        
        Returns:
            API key string.
            
        Raises:
            FileNotFoundError: If API key file is not found.
            ValueError: If API key is empty or invalid.
        """
        # Try environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key.strip()
        
        # Try file
        api_key_path = self.config_dir / self.app_config.api_key_file
        if not api_key_path.exists():
            raise FileNotFoundError(
                f"API key file not found: {api_key_path}. "
                "Please create the file or set OPENAI_API_KEY environment variable."
            )
        
        try:
            with open(api_key_path, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
        except Exception as e:
            raise ValueError(f"Error reading API key file: {e}")
        
        if not api_key:
            raise ValueError("API key is empty")
        
        return api_key
    
    def get_system_instructions(self) -> str:
        """Get system instructions from file.
        
        Returns:
            System instructions string.
            
        Raises:
            FileNotFoundError: If instructions file is not found.
        """
        instructions_path = self.config_dir / self.app_config.system_instructions_file
        if not instructions_path.exists():
            raise FileNotFoundError(
                f"System instructions file not found: {instructions_path}"
            )
        
        try:
            with open(instructions_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Error reading system instructions: {e}")
    
    def save_config(self, config_file: Optional[Path] = None) -> None:
        """Save current configuration to file.
        
        Args:
            config_file: Optional path to save configuration.
                        Defaults to config.json in config directory.
        """
        if config_file is None:
            config_file = self.config_dir / "config.json"
        
        config_data = {
            'model': asdict(self.model_config),
            'logging': asdict(self.logging_config),
            'app': asdict(self.app_config)
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of all configuration.
        """
        return {
            'model': asdict(self.model_config),
            'logging': asdict(self.logging_config),
            'app': asdict(self.app_config)
        }


# Global configuration instance
config_manager = ConfigManager()
