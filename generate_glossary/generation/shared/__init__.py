"""
Shared utilities for generation modules.

This package contains minimal, focused utilities that are actually reused
across multiple generation steps. No over-engineering, just simple functions
that solve real problems.
"""

from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    clear_checkpoint,
    process_with_checkpoint
)

__all__ = [
    'save_checkpoint',
    'load_checkpoint', 
    'clear_checkpoint',
    'process_with_checkpoint'
]