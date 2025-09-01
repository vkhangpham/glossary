"""
Public API for concept validation.

Simple, functional interface for validating technical concepts.
"""

# Main validation function
from .main import validate_terms

# Essential query functions
from .api import (
    get_valid_terms,
    get_invalid_terms,
    save_validation_results,
    load_validation_results
)

# Version
__version__ = "2.0.0"

# Public API
__all__ = [
    "validate_terms",
    "get_valid_terms",
    "get_invalid_terms",
    "save_validation_results",
    "load_validation_results"
]

# Backward compatibility
validate = validate_terms 