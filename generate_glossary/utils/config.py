"""
Configuration management for the glossary generation system.

This module handles loading and managing configuration settings,
separate from security concerns which are in the security module.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .logger import setup_logger

logger = setup_logger("config")

class Config:
    """Central configuration management."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config_file = config_file or self._find_config_file()
        self.config = self._load_config()
        
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations."""
        search_paths = [
            Path.cwd() / "config.json",
            Path.cwd() / ".glossary" / "config.json",
            Path.home() / ".glossary" / "config.json",
            Path(__file__).parent.parent / "config.json"
        ]
        
        for path in search_paths:
            if path.exists():
                logger.info(f"Found config file at: {path}")
                return path
                
        return None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config = self._get_defaults()
        
        # Load from file if exists
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                    logger.info(f"Loaded config from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Override with environment variables
        config.update(self._load_env_vars())
        
        return config
        
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            # Processing settings
            "batch_size": 10,
            "max_retries": 3,
            "timeout": 300,
            
            # File paths
            "data_dir": "generate_glossary/data",
            "output_dir": "data/final",
            "checkpoint_dir": ".checkpoints",
            
            # LLM settings
            "default_provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            
            # Web mining settings
            "use_firecrawl": True,
            "max_urls_per_concept": 5,
            "search_timeout": 30,
            
            # Validation settings
            "min_confidence": 0.7,
            "require_consensus": True,
            "consensus_threshold": 0.6,
            
            # Level-specific settings
            "levels": {
                "0": {
                    "min_sources": 3,
                    "frequency_threshold": 0.1
                },
                "1": {
                    "min_sources": 2,
                    "frequency_threshold": 0.05
                },
                "2": {
                    "min_sources": 1,
                    "frequency_threshold": 0.03
                },
                "3": {
                    "min_sources": 1,
                    "frequency_threshold": 0.02
                }
            }
        }
        
    def _load_env_vars(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            "GLOSSARY_BATCH_SIZE": ("batch_size", int),
            "GLOSSARY_MAX_RETRIES": ("max_retries", int),
            "GLOSSARY_TIMEOUT": ("timeout", int),
            "GLOSSARY_DATA_DIR": ("data_dir", str),
            "GLOSSARY_OUTPUT_DIR": ("output_dir", str),
            "GLOSSARY_DEFAULT_PROVIDER": ("default_provider", str),
            "GLOSSARY_TEMPERATURE": ("temperature", float),
            "GLOSSARY_USE_FIRECRAWL": ("use_firecrawl", lambda x: x.lower() == 'true')
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    env_config[config_key] = converter(os.environ[env_var])
                except Exception as e:
                    logger.warning(f"Failed to parse {env_var}: {e}")
                    
        return env_config
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def get_level_config(self, level: int) -> Dict[str, Any]:
        """
        Get configuration for a specific level.
        
        Args:
            level: Hierarchy level (0, 1, 2, 3)
            
        Returns:
            Level-specific configuration
        """
        base_config = self.config.copy()
        level_config = self.get(f"levels.{level}", {})
        base_config.update(level_config)
        return base_config
        
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to (uses default if not provided)
        """
        save_path = path or self.config_file
        if not save_path:
            save_path = Path.cwd() / "config.json"
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Saved configuration to {save_path}")


# Global configuration instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def load_config(config_file: Optional[Path] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configuration instance
    """
    global _config
    _config = Config(config_file)
    return _config