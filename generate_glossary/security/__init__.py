"""
Security utilities for API key management and data sanitization.
"""

from .api_keys import (
    APIKeyManager,
    APIKeyInfo,
    SecureConfigError
)

__all__ = [
    # API Key Management
    'APIKeyManager',
    'APIKeyInfo',
    'SecureConfigError',
    'mask_key',
    'detect_and_mask_keys',
    'validate_api_key',
    'load_api_key',
    'get_api_key_manager'
]

# Convenience functions
def mask_key(key: str) -> str:
    """
    Mask an API key for safe logging.
    
    Args:
        key: The API key to mask
        
    Returns:
        Masked version showing only first 4 and last 4 characters
    """
    manager = get_api_key_manager()
    return manager.mask_key(key)

def validate_api_key(key_name: str, key_value: str) -> bool:
    """
    Validate an API key against known patterns.
    
    Args:
        key_name: Name of the API key (e.g., 'OPENAI_API_KEY')
        key_value: The key value to validate
        
    Returns:
        True if valid, False otherwise
    """
    manager = get_api_key_manager()
    return manager.validate_key(key_name, key_value)

def load_api_key(key_name: str, required: bool = True) -> str:
    """
    Load an API key from environment.
    
    Args:
        key_name: Name of the API key to load
        required: Whether the key is required
        
    Returns:
        The API key value
        
    Raises:
        SecureConfigError: If required key is missing
    """
    manager = get_api_key_manager()
    return manager.load_key(key_name, required)

def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _manager
    try:
        return _manager
    except NameError:
        from .api_keys import APIKeyManager
        _manager = APIKeyManager()
        return _manager