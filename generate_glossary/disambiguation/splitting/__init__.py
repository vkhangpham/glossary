"""Sense splitting functionality for ambiguous terms."""

from .generator import (
    generate_split_proposals,
    create_tag_generator,
    with_llm_function,
    generate_splits  # Legacy function
)
from .validator import (
    validate_split_proposals,
    create_llm_validator,
    validate_splits  # Legacy function
)
from .applicator import (
    apply_splits_to_hierarchy,
    apply_to_hierarchy  # Legacy function
)

__all__ = [
    "generate_split_proposals",
    "create_tag_generator",
    "with_llm_function",
    "validate_split_proposals",
    "create_llm_validator",
    "apply_splits_to_hierarchy",
    # Legacy functions
    "generate_splits",
    "validate_splits",
    "apply_to_hierarchy",
]