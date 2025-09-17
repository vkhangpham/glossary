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

# Import detection modules from new structure
from . import detectors
from .detectors.embedding import detect_embedding_ambiguity, detect as embedding_detect
from .detectors.hierarchy import detect_hierarchy_ambiguity, detect as hierarchy_detect
from .detectors.global_clustering import detect_global_ambiguity, detect as global_detect

# Import splitting functions from new structure
from .splitting.generator import generate_split_proposals, generate_splits
from .splitting.validator import validate_split_proposals, validate_splits
from .splitting.applicator import apply_splits_to_hierarchy, apply_to_hierarchy

__all__ = [
    # Public API
    "disambiguate_terms",
    "detect_ambiguous",
    "split_senses",

    # Detection modules (new structure)
    "detectors",
    "detect_embedding_ambiguity",
    "detect_hierarchy_ambiguity",
    "detect_global_ambiguity",

    # Detection functions with aliases
    "embedding_detect",
    "hierarchy_detect",
    "global_detect",

    # Splitting functions (new functional API)
    "generate_split_proposals",
    "validate_split_proposals",
    "apply_splits_to_hierarchy",

    # Legacy splitting functions
    "generate_splits",
    "validate_splits",
    "apply_to_hierarchy"
]