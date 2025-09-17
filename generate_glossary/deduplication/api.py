# generate_glossary/deduplication/api.py
"""
Deprecated API wrapper for deduplication graph queries.

This module is a legacy wrapper that continues to work with graphs created by
the new functional core (build_deduplication_graph from core.py). All functions
are fully compatible with both legacy and functional graphs.

For new code, prefer importing directly from:
- deduplication.graph.query for graph query functions
- deduplication.core for functional graph building
"""
import warnings
warnings.warn(
    "deduplication.api is deprecated; use 'from deduplication.core import build_deduplication_graph' "
    "for graph building and 'from deduplication.graph.query import *' for querying. "
    "This wrapper remains functional for backward compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

# All query functions work with both legacy and functional graphs
from .graph.query import (
    get_canonical_terms,
    get_terms_with_variations,
    get_variations_for_term,
    get_duplicate_pairs,
    get_duplicate_clusters,
    is_duplicate_pair,
    get_deduplicated_terms,
    get_all_variations,
    query_graph,
)

__all__ = [name for name in dir() if not name.startswith('_')]