"""Logging utilities for the generate_glossary package."""

import os
import logging
from pathlib import Path

def setup_logger(name: str, level: int = None) -> logging.Logger:
    """Set up a logger with the given name and level.
    
    Args:
        name: The name of the logger.
        level: The logging level (default: from LOGLEVEL env var or INFO).
        
    Returns:
        logging.Logger: The configured logger.
    """
    # Get log level from environment or use default
    if level is None:
        level_name = os.getenv('LOGLEVEL', 'INFO').upper()
        level = getattr(logging, level_name, logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler with the same log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create file handler which logs even debug messages
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatters and add them to the handlers
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger 