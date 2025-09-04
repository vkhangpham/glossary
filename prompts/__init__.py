"""
Centralized prompt management system for the glossary project.

This package provides:
- Centralized prompt storage and versioning
- Pure functional API for prompt access
- Optional GEPA optimization capabilities
"""

from .registry import get_prompt, list_prompts, get_prompt_versions, register_prompt

__all__ = ["get_prompt", "list_prompts", "get_prompt_versions", "register_prompt"]