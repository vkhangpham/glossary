"""
Deduplication module for glossary generation.

Graph-based deduplication where the graph is everything.
Terms are nodes, duplicates are connected components.
"""

from .main import build_graph, main
from .api import (
    get_canonical_terms,
    get_terms_with_variations,
    get_variations_for_term,
    get_duplicate_pairs,
    get_duplicate_clusters,
    is_duplicate_pair,
    get_deduplicated_terms,
    get_all_variations,
    query_graph
)
from .graph_io import save_graph, load_graph
from .utils import (
    normalize_text,
    get_term_variations,
    is_compound_term,
)

__all__ = [
    # Main graph building
    'build_graph',
    'main',
    
    # API functions for querying
    'get_canonical_terms',
    'get_terms_with_variations',
    'get_variations_for_term',
    'get_duplicate_pairs',
    'get_duplicate_clusters',
    'is_duplicate_pair',
    'get_deduplicated_terms',
    'get_all_variations',
    'query_graph',
    
    # Graph I/O
    'save_graph',
    'load_graph',
    
    # Utilities
    'normalize_text',
    'get_term_variations',
    'is_compound_term',
] 