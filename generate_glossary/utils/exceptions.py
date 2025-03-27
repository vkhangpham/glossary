"""
Custom exceptions for LLM-related errors.
This module provides a hierarchy of exceptions for handling various error cases in the LLM system.
"""

from typing import Optional, Dict, Any

class LLMError(Exception):
    """Base exception class for all LLM-related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class LLMConfigError(LLMError):
    """
    Raised when there are configuration-related errors.
    Examples:
    - Missing API keys
    - Invalid model names
    - Invalid configuration parameters
    """
    pass

class LLMAPIError(LLMError):
    """
    Raised when there are API-related errors.
    Examples:
    - Network timeouts
    - Rate limit exceeded
    - Service unavailable
    - Invalid API responses
    """
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.status_code = status_code
        self.response = response

class LLMValidationError(LLMError):
    """
    Raised when there are data validation errors.
    Examples:
    - Invalid input format
    - Invalid response format
    - Schema validation failures
    """
    def __init__(
        self,
        message: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.validation_errors = validation_errors

class LLMRetryError(LLMError):
    """
    Raised when maximum retry attempts are exhausted.
    Examples:
    - Too many failed API calls
    - Persistent rate limiting
    - Service degradation
    """
    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.attempts = attempts
        self.last_error = last_error

class LLMProviderError(LLMError):
    """
    Raised when there are provider-specific errors.
    Examples:
    - Provider-specific API errors
    - Model-specific limitations
    - Provider service issues
    """
    def __init__(
        self,
        message: str,
        provider: str,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.provider = provider
        self.model = model 