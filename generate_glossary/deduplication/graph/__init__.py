"""Graph operation modules for deduplication graph management.

This package contains modules for graph construction, persistence, and querying:
- builder: Graph construction and validation functionality
- io: Graph serialization and deserialization
- query: Graph querying and duplicate detection API
"""

from .builder import *
from .io import *
from .query import *

__all__ = [
    # Graph building
    "create_deduplication_graph",
    "add_terms_as_nodes",
    "get_graph_stats",
    "remove_weak_edges",
    "validate_graph",

    # Graph I/O
    "save_graph",
    "load_graph",

    # Graph querying
    "get_canonical_terms",
    "get_terms_with_variations",
    "get_variations_for_term",
    "get_duplicate_pairs",
    "get_duplicate_clusters",
    "is_duplicate_pair",
    "get_deduplicated_terms",
    "get_all_variations",
    "query_graph",
]