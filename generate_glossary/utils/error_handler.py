"""Simplified error handling utilities for standardized error management."""

import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from .failure_tracker import save_failure as save_failure_record
from .logger import get_logger, get_correlation_id, set_correlation_id

logger = get_logger(__name__)


# Custom Exception Classes
class ProcessingError(Exception):
    """Error during data processing operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class ValidationError(Exception):
    """Error during data validation."""
    
    def __init__(self, message: str, invalid_data: Any = None):
        super().__init__(message)
        self.invalid_data = invalid_data


class ConfigurationError(Exception):
    """Error in configuration or setup."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message)
        self.config_key = config_key


class ExternalServiceError(Exception):
    """Error from external service (API, database, etc.)."""
    
    def __init__(self, message: str, service: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.service = service
        self.status_code = status_code


# Correlation ID Management
def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracking."""
    return str(uuid.uuid4())


# Standardized Error Handler
def handle_error(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    operation: Optional[str] = None,
    reraise: bool = False,
    save_failure: bool = True
) -> None:
    """
    Handle an error with standardized logging and tracking.
    
    Args:
        exception: The exception to handle
        context: Additional context about the error
        operation: The operation that failed
        reraise: Whether to re-raise the exception after handling
        save_failure: Whether to save to failure tracker
    """
    correlation_id = get_correlation_id() or set_correlation_id()
    
    error_context = {
        "operation": operation or "unknown",
        "correlation_id": correlation_id,
        **(context or {})
    }
    
    # Log error
    logger.error(
        f"[{correlation_id}] Error in {operation or 'operation'}: {exception}",
        extra={
            "correlation_id": correlation_id,
            "error_context": error_context
        }
    )
    
    # Save to failure tracker
    if save_failure:
        # Use format_exception for reliable traceback capture
        traceback_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        
        try:
            save_failure_record(
                module="generate_glossary.utils.error_handler",
                function=operation or "unknown",
                error_type=type(exception).__name__,
                error_message=str(exception),
                context={
                    "error_context": error_context,
                    "correlation_id": correlation_id,
                    "traceback": traceback_str
                }
            )
        except Exception as save_error:
            # Don't let failure tracking mask the original exception
            logger.warning(f"Failed to save failure record: {save_error}")
    
    if reraise:
        raise exception


# Context Manager
@contextmanager
def processing_context(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    generate_new_id: bool = True
):
    """
    Context manager for processing with automatic correlation ID and error handling.
    
    Args:
        operation: Name of the operation being performed
        context: Additional context for the operation
        generate_new_id: Whether to generate a new correlation ID
    """
    # Set up correlation ID
    if generate_new_id:
        correlation_id = set_correlation_id()
    else:
        correlation_id = get_correlation_id() or set_correlation_id()
    
    operation_context = {
        "operation": operation,
        "correlation_id": correlation_id,
        **(context or {})
    }
    
    logger.info(
        f"[{correlation_id}] Starting {operation}",
        extra=operation_context
    )
    
    start_time = datetime.now()
    
    try:
        yield correlation_id
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"[{correlation_id}] Completed {operation} in {duration:.2f}s",
            extra={**operation_context, "duration": duration}
        )
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        handle_error(
            e,
            context={**operation_context, "duration": duration},
            operation=operation,
            reraise=True
        )