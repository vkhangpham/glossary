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

from .embedding_disambiguator import detect_ambiguous_by_embeddings
from .hierarchy_disambiguator import detect_ambiguous_by_hierarchy
from .global_disambiguator import detect_ambiguous_by_global_clustering

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
    
    # Detection functions
    "detect_ambiguous_by_embeddings",
    "detect_ambiguous_by_hierarchy",
    "detect_ambiguous_by_global_clustering",
    
    # Splitting functions  
    "generate_splits",
    "validate_splits",
    "apply_to_hierarchy"
]