"""Enhanced logging utilities with correlation ID support for the generate_glossary package."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional


# Simple correlation ID tracking
import threading
import uuid

_thread_local = threading.local()

def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from thread-local storage."""
    return getattr(_thread_local, 'correlation_id', None)

def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current thread."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _thread_local.correlation_id = correlation_id
    return correlation_id


class CorrelationIdFormatter(logging.Formatter):
    """Formatter that includes correlation ID in log messages."""
    
    def format(self, record):
        # Add correlation ID to the record if available
        correlation_id = get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id[:8]  # Use short version for readability
        else:
            record.correlation_id = "--------"
        
        return super().format(record)


def get_logger(name: str, level: int = None) -> logging.Logger:
    """Get a logger with enhanced functionality."""
    return create_enhanced_logger(name, level)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
):
    """
    Log a message with context and correlation ID.
    
    Args:
        logger: The logger to use
        level: Logging level
        message: The log message
        context: Additional context to include
        correlation_id: Override correlation ID
    """
    if correlation_id:
        original_id = get_correlation_id()
        set_correlation_id(correlation_id)
    
    extra = {"correlation_id": get_correlation_id()}
    if context:
        extra.update(context)
    
    logger.log(level, message, extra=extra)
    
    if correlation_id:
        # Restore original correlation ID
        if original_id:
            set_correlation_id(original_id)


def log_error_with_context(
    logger: logging.Logger,
    message: str,
    exception: Optional[Exception] = None,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
):
    """
    Log an error with context and correlation ID.
    
    Args:
        logger: The logger to use
        message: The error message
        exception: The exception that occurred
        context: Additional context to include
        correlation_id: Override correlation ID
    """
    error_context = {"error_type": type(exception).__name__ if exception else "Unknown"}
    if context:
        error_context.update(context)
    
    log_message = message
    if exception:
        log_message += f": {exception}"
    
    log_with_context(
        logger,
        logging.ERROR,
        log_message,
        error_context,
        correlation_id
    )


def log_processing_step(
    logger: logging.Logger,
    step: str,
    status: str = "started",
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
):
    """
    Log a processing step with context and correlation ID.
    
    Args:
        logger: The logger to use
        step: The processing step name
        status: The step status (started, completed, failed)
        context: Additional context to include
        correlation_id: Override correlation ID
    """
    step_context = {"processing_step": step, "step_status": status}
    if context:
        step_context.update(context)
    
    level = logging.INFO if status in ["started", "completed"] else logging.ERROR
    
    log_with_context(
        logger,
        level,
        f"Processing step '{step}' {status}",
        step_context,
        correlation_id
    )


def create_enhanced_logger(name: str, level: int = None) -> logging.Logger:
    """
    Create a logger with enhanced correlation ID support.
    
    Args:
        name: The name of the logger
        level: The logging level (default: from LOGLEVEL env var or INFO)
        
    Returns:
        Enhanced logger with correlation ID support
    """
    # Get log level from environment or use default
    if level is None:
        level_name = os.getenv('LOGLEVEL', 'INFO').upper()
        level = getattr(logging, level_name, logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, it's already been configured
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    logger.propagate = False
    
    # Create console handler with correlation ID formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create file handler
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    
    # Create enhanced formatters with correlation ID support
    console_format = CorrelationIdFormatter(
        '%(asctime)s [%(correlation_id)s] %(levelname)s - %(message)s'
    )
    file_format = CorrelationIdFormatter(
        '%(asctime)s [%(correlation_id)s] %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger