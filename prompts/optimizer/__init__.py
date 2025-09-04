"""
GEPA-based prompt optimization for the glossary project.
Provides adapters and utilities for optimizing prompts using evolutionary algorithms.
"""

from .concept_extraction_adapter import ConceptExtractionAdapter
from .optimizer import optimize_prompt, compare_prompts

__all__ = ["ConceptExtractionAdapter", "optimize_prompt", "compare_prompts"]
