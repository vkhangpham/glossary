"""
Term disambiguation module for academic glossary generation.

This module identifies and resolves ambiguous terms in the glossary
using semantic clustering and multi-signal validation.
"""

from .api import (
    disambiguate_terms,
    detect_ambiguous,
    split_senses
)

from . import embedding_disambiguator
from . import hierarchy_disambiguator
from . import global_disambiguator

from .sense_splitter import (
    generate_splits,
    validate_splits,
    apply_to_hierarchy
)

__all__ = [
    # Public API
    "disambiguate_terms",
    "detect_ambiguous",
    "split_senses",
    
    # Detection modules
    "embedding_disambiguator",
    "hierarchy_disambiguator",
    "global_disambiguator",
    
    # Splitting functions  
    "generate_splits",
    "validate_splits",
    "apply_to_hierarchy"
]