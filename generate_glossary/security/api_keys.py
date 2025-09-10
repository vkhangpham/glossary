"""
Secure configuration management for API keys and sensitive data.

This module provides:
- Safe API key loading with validation
- Key masking for logs and error messages  
- Environment variable validation
- Secure credential management
"""

import os
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from generate_glossary.utils.logger import get_logger

logger = get_logger("secure_config")

@dataclass
class APIKeyInfo:
    """Information about an API key"""
    name: str
    env_var: str
    required: bool
    min_length: int = 8
    pattern: Optional[str] = None  # Regex pattern for validation
    masked_value: str = ""

class SecureConfigError(Exception):
    """Raised when there are configuration security issues"""
    pass

class APIKeyManager:
    """Manages API keys with security best practices"""
    
    # Known API key patterns for validation
    KEY_PATTERNS = {
        "OPENAI_API_KEY": r"^sk-[A-Za-z0-9_-]{48,}$",
        "GOOGLE_API_KEY": r"^[A-Za-z0-9_-]{32,}$",
        "GEMINI_API_KEY": r"^[A-Za-z0-9_-]{32,}$", 
        "TAVILY_API_KEY": r"^tvly-[A-Za-z0-9_-]{32,}$",
        "RAPIDAPI_KEY": r"^[A-Za-z0-9_-]{32,}$"
    }
    
    # Known API key prefixes for masking detection
    KEY_PREFIXES = ["sk-", "tvly-", "gsk_", "AIza"]
    
    def __init__(self):
        self.loaded_keys: Dict[str, APIKeyInfo] = {}
        self.failed_keys: List[str] = []
        
    def mask_key(self, key: str) -> str:
        """
        Mask an API key for safe logging
        
        Args:
            key: The API key to mask
            
        Returns:
            Masked version showing only first 4 and last 4 characters
        """
        if not key or len(key) < 8:
            return "[INVALID_KEY]"
            
        if len(key) <= 12:
            # For shorter keys, show less
            return f"{key[:2]}***{key[-2:]}"
        else:
            # Standard masking for longer keys
            return f"{key[:4]}***{key[-4:]}"
            
    def detect_and_mask_keys(self, text: str) -> str:
        """
        Detect and mask potential API keys in text
        
        Args:
            text: Text that may contain API keys
            
        Returns:
            Text with API keys masked
        """
        if not text:
            return text
            
        masked_text = text
        
        # Look for known API key patterns
        for pattern in self.KEY_PATTERNS.values():
            matches = re.finditer(pattern, text)
            for match in matches:
                key = match.group(0)
                masked_key = self.mask_key(key)
                masked_text = masked_text.replace(key, masked_key)
                
        # Look for common API key prefixes
        for prefix in self.KEY_PREFIXES:
            # Match prefix followed by alphanumeric characters
            pattern = rf"{re.escape(prefix)}[A-Za-z0-9_-]+(?=\s|$|[\"']|[,}}\]])"
            matches = re.finditer(pattern, masked_text)
            for match in matches:
                key = match.group(0)
                if len(key) > len(prefix) + 4:  # Only mask if reasonable length
                    masked_key = self.mask_key(key)
                    masked_text = masked_text.replace(key, masked_key)
                    
        return masked_text
        
    def validate_key_format(self, key: str, env_var: str) -> bool:
        """
        Validate API key format using known patterns
        
        Args:
            key: The API key to validate
            env_var: Environment variable name for context
            
        Returns:
            True if key format appears valid
        """
        if not key:
            return False
            
        # Check against known patterns
        pattern = self.KEY_PATTERNS.get(env_var)
        if pattern:
            if re.match(pattern, key):
                logger.debug(f"API key {env_var} matches expected pattern")
                return True
            else:
                logger.warning(f"API key {env_var} does not match expected pattern")
                return False
                
        # Generic validation for unknown key types
        if len(key) < 8:
            logger.warning(f"API key {env_var} appears too short (< 8 characters)")
            return False
            
        # Check for reasonable character set
        if not re.match(r'^[A-Za-z0-9_-]+$', key):
            logger.warning(f"API key {env_var} contains unexpected characters")
            return False
            
        return True
        
    def load_api_key(
        self, 
        env_var: str, 
        required: bool = True,
        min_length: int = 8
    ) -> Optional[str]:
        """
        Safely load an API key from environment variables
        
        Args:
            env_var: Environment variable name
            required: Whether this key is required
            min_length: Minimum expected key length
            
        Returns:
            The API key if found and valid, None otherwise
            
        Raises:
            SecureConfigError: If required key is missing or invalid
        """
        key = os.environ.get(env_var)
        
        # Store key info for tracking
        key_info = APIKeyInfo(
            name=env_var,
            env_var=env_var,
            required=required,
            min_length=min_length
        )
        
        if not key:
            if required:
                self.failed_keys.append(env_var)
                raise SecureConfigError(f"Required API key {env_var} not set in environment")
            else:
                logger.info(f"Optional API key {env_var} not provided")
                key_info.masked_value = "[NOT_PROVIDED]"
                self.loaded_keys[env_var] = key_info
                return None
                
        # Validate key format
        if not self.validate_key_format(key, env_var):
            if required:
                self.failed_keys.append(env_var)
                raise SecureConfigError(f"API key {env_var} appears to have invalid format")
            else:
                logger.warning(f"Optional API key {env_var} has questionable format")
                
        # Store masked version for logging
        key_info.masked_value = self.mask_key(key)
        self.loaded_keys[env_var] = key_info
        
        logger.info(f"Successfully loaded API key {env_var}: {key_info.masked_value}")
        return key
        
    def get_key_status(self) -> Dict[str, Any]:
        """
        Get status of all loaded keys
        
        Returns:
            Dictionary with key loading status information
        """
        return {
            "loaded_keys": {
                name: {
                    "masked_value": info.masked_value,
                    "required": info.required
                }
                for name, info in self.loaded_keys.items()
            },
            "failed_keys": self.failed_keys,
            "total_loaded": len([k for k in self.loaded_keys.values() if k.masked_value != "[NOT_PROVIDED]"]),
            "total_failed": len(self.failed_keys)
        }
        
    def validate_environment(self, required_keys: List[str]) -> bool:
        """
        Validate that all required API keys are properly configured
        
        Args:
            required_keys: List of required environment variable names
            
        Returns:
            True if all required keys are valid
        """
        all_valid = True
        
        for env_var in required_keys:
            try:
                self.load_api_key(env_var, required=True)
            except SecureConfigError as e:
                logger.error(f"Key validation failed: {e}")
                all_valid = False
                
        return all_valid

# Global instance
key_manager = APIKeyManager()

def mask_sensitive_data(text: str) -> str:
    """
    Mask sensitive data in text (convenience function)
    
    Args:
        text: Text that may contain sensitive information
        
    Returns:
        Text with sensitive data masked
    """
    return key_manager.detect_and_mask_keys(text)

def safe_load_api_key(env_var: str, required: bool = True) -> Optional[str]:
    """
    Safely load API key (convenience function)
    
    Args:
        env_var: Environment variable name
        required: Whether the key is required
        
    Returns:
        The API key or None if not available
    """
    return key_manager.load_api_key(env_var, required)

def validate_api_keys() -> bool:
    """
    Validate all standard API keys used by the system
    
    Returns:
        True if all required keys are valid
    """
    required_keys = ["OPENAI_API_KEY"]  # Minimum requirement
    optional_keys = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "RAPIDAPI_KEY"]
    
    # Load required keys
    all_valid = key_manager.validate_environment(required_keys)
    
    # Load optional keys
    for env_var in optional_keys:
        try:
            key_manager.load_api_key(env_var, required=False)
        except SecureConfigError:
            pass  # Optional keys don't fail validation
            
    # Log summary
    status = key_manager.get_key_status()
    logger.info(f"API Key validation complete: {status['total_loaded']} loaded, {status['total_failed']} failed")
    
    if not all_valid:
        logger.error("Some required API keys are missing or invalid")
    
    return all_valid

class SecureLogger:
    """Logger wrapper that automatically masks sensitive data"""
    
    def __init__(self, logger_instance):
        self.logger = logger_instance
        
    def _mask_message(self, message: str) -> str:
        """Mask sensitive data in log message"""
        return mask_sensitive_data(str(message))
        
    def debug(self, message: str, *args, **kwargs):
        """Debug log with masking"""
        self.logger.debug(self._mask_message(message), *args, **kwargs)
        
    def info(self, message: str, *args, **kwargs):
        """Info log with masking"""
        self.logger.info(self._mask_message(message), *args, **kwargs)
        
    def warning(self, message: str, *args, **kwargs):
        """Warning log with masking"""
        self.logger.warning(self._mask_message(message), *args, **kwargs)
        
    def error(self, message: str, *args, **kwargs):
        """Error log with masking"""
        self.logger.error(self._mask_message(message), *args, **kwargs)
        
    def exception(self, message: str, *args, **kwargs):
        """Exception log with masking"""
        self.logger.exception(self._mask_message(message), *args, **kwargs)

def create_secure_logger(name: str):
    """
    Create a logger that automatically masks sensitive data
    
    Args:
        name: Logger name
        
    Returns:
        SecureLogger instance
    """
    base_logger = get_logger(name)
    return SecureLogger(base_logger)