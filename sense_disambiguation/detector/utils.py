"""
Utility functions shared across the sense disambiguation package.
"""

import logging
import json
import numpy as np
from typing import Any, Dict, List, Union

# Custom logging level definition
# This needs to match the value in cli.py
PROGRESS = 15  # Between DEBUG (10) and INFO (20)

def setup_custom_logging():
    """
    Set up custom logging levels and methods.
    This should be called by any module that wants to use custom logging methods.
    """
    # Add the PROGRESS level if it doesn't exist
    if not hasattr(logging, 'PROGRESS'):
        # Define the level name
        logging.addLevelName(PROGRESS, "PROGRESS")
        
        # Add the progress method to the Logger class if not already present
        if not hasattr(logging.Logger, 'progress'):
            def progress(self, message, *args, **kwargs):
                """Log a message with severity 'PROGRESS'."""
                if self.isEnabledFor(PROGRESS):
                    self._log(PROGRESS, message, args, **kwargs)
            
            # Add the method to the Logger class
            logging.Logger.progress = progress

def get_progress_logger(name: str) -> logging.Logger:
    """
    Get a logger with the progress method added.
    
    Args:
        name: Name for the logger
        
    Returns:
        Logger instance with progress method
    """
    # Ensure custom logging is set up
    setup_custom_logging()
    
    # Return the logger
    return logging.getLogger(name)

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Any object that might contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle potential NaN/Inf values gracefully
        if np.isnan(obj):
            return None # Or use a string like 'NaN'
        elif np.isinf(obj):
            return None # Or use a string like 'Infinity'
        else:
            return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj 