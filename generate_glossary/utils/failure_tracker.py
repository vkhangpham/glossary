"""Simple failure tracking utility for saving failures to JSON files."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default failure log directory
FAILURE_LOG_DIR = Path("data/failures")


def save_failure(
    module: str,
    function: str,
    error_type: str,
    error_message: str,
    context: Optional[Dict[str, Any]] = None,
    failure_dir: Optional[Path] = None,
) -> None:
    """
    Save failure information to a JSON file.
    
    Args:
        module: Module where failure occurred
        function: Function where failure occurred
        error_type: Type of error (e.g., exception class name)
        error_message: Error message
        context: Optional context data
        failure_dir: Directory to save failures (defaults to data/failures)
    """
    try:
        # Use provided directory or default
        log_dir = failure_dir or FAILURE_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on date
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"failures_{date_str}.json"
        
        # Create failure record
        failure_record = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "function": function,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        # Load existing failures or create new list
        failures = []
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    content = f.read().strip()
                    if content:
                        failures = json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read existing failures: {e}")
                failures = []
        
        # Append new failure
        failures.append(failure_record)
        
        # Save to file
        with open(log_file, "w") as f:
            json.dump(failures, f, indent=2, default=str)
            
        logger.debug(f"Saved failure to {log_file}")
        
    except Exception as e:
        # Don't let failure tracking cause crashes
        logger.warning(f"Could not save failure: {e}")


def load_failures(
    date: Optional[str] = None,
    failure_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Load failure records from JSON file.
    
    Args:
        date: Date string (YYYY-MM-DD) or None for today
        failure_dir: Directory containing failure logs
        
    Returns:
        List of failure records
    """
    try:
        # Use provided directory or default
        log_dir = failure_dir or FAILURE_LOG_DIR
        
        # Use provided date or today
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        log_file = log_dir / f"failures_{date}.json"
        
        if not log_file.exists():
            return []
            
        with open(log_file, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            return []
            
    except Exception as e:
        logger.warning(f"Could not load failures: {e}")
        return []