"""
Core utility modules for the generate_glossary package.

This package contains essential utilities:
- logger: Logging setup
"""

from .logger import get_logger

__all__ = [
    'get_logger',
]


def __getattr__(name: str):
    """Provide informative error messages for removed Config APIs."""
    if name in ('Config', 'get_config', 'load_config'):
        raise ImportError(
            f"'{name}' has been moved from generate_glossary.utils to generate_glossary.config. "
            f"Please update your import: from generate_glossary.config import {name}"
        )
    raise ImportError(f"module '{__name__}' has no attribute '{name}'")