"""
Sense disambiguation module for academic glossary generation.

This module identifies and resolves ambiguous terms in the glossary
using semantic clustering and multi-signal validation.
"""

from .main import (
    detect_ambiguous_terms,
    split_ambiguous_terms,
    run_disambiguation_pipeline
)

from .detector import (
    detect_with_embeddings,
    detect_with_hierarchy,
    detect_with_global_clustering
)

from .splitter import (
    generate_sense_splits,
    validate_splits,
    apply_splits_to_hierarchy
)

__all__ = [
    # Main API
    "detect_ambiguous_terms",
    "split_ambiguous_terms", 
    "run_disambiguation_pipeline",
    
    # Detection functions
    "detect_with_embeddings",
    "detect_with_hierarchy",
    "detect_with_global_clustering",
    
    # Splitting functions
    "generate_sense_splits",
    "validate_splits",
    "apply_splits_to_hierarchy"
]