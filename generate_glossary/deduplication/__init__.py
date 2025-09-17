"""
Deduplication module for glossary generation.

This module provides both a functional API and legacy compatibility for graph-based
deduplication where the graph is everything. Terms are nodes, duplicates are connected components.

## Functional API (Recommended)

The functional API provides immutable operations, comprehensive error handling, and
parallel processing capabilities:

```python
from generate_glossary.deduplication import (
    build_deduplication_graph,
    DeduplicationConfig,
    RuleConfig,
    WebConfig,
    LLMConfig
)

# Configure and build graph functionally
config = DeduplicationConfig(
    rule_config=RuleConfig(),
    web_config=WebConfig(),
    llm_config=LLMConfig()
)

result = build_deduplication_graph(terms, config)
if is_success(result):
    graph = get_value(result)
```

## Legacy API (Deprecated)

Legacy functions are maintained for backward compatibility but emit deprecation warnings.
Consider migrating to the functional API for better error handling and type safety.

## Migration Guide

- `build_graph()` → `build_deduplication_graph()` with `DeduplicationConfig`
- Direct graph mutation → Immutable operations with `Result` types
- Exception handling → Functional error handling with `Success`/`Failure`
"""

import warnings
from typing import Any, Dict, List, Optional

# Functional Core API
from .core import (
    # Main functional orchestration
    build_deduplication_graph,
    build_level_graph,
    create_all_edges,
    add_edges_to_graph,
    remove_weak_edges_functional,

    # Result types and error handling
    Success,
    Failure,
    Result,
    EdgeCreationResult,
    GraphBuildResult,

    # Helper functions
    is_success,
    is_failure,
    get_value,
    get_error,

    # Functional composition utilities
    compose_edge_creators,
    with_error_handling,
    parallel_edge_creation,
    filter_edges,
    combine_edge_results,
)

# Configuration System
from .types import (
    # Configuration classes
    DeduplicationConfig,
    RuleConfig,
    WebConfig,
    LLMConfig,

    # Data structures
    Edge,
    EdgeBatch,
    WebResource,
    TermWebContent,

    # Constants
    EDGE_TYPES,
    METHODS,
    SUPPORTED_PROVIDERS,
)

# Pure Edge Creation Functions
from .edges.rule_based import create_rule_edges
from .edges.web_based import create_web_edges

def create_llm_edges(*args, **kwargs):
    """Lazy import wrapper for create_llm_edges to reduce import-time overhead."""
    from .edges.llm_based import create_llm_edges as _create
    return _create(*args, **kwargs)

# Legacy API - maintained for backward compatibility
from .main import main

def build_graph(*args, **kwargs):
    """Legacy build_graph function with deprecation warning.

    DEPRECATED: Use build_deduplication_graph() with DeduplicationConfig instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "build_graph() is deprecated. Use build_deduplication_graph() with DeduplicationConfig.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .main import build_graph as _build_graph
    return _build_graph(*args, **kwargs)
from .graph.query import (
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
from .graph.io import save_graph, load_graph
from .utils import (
    normalize_text,
    get_term_variations,
    is_compound_term,
)


def build_graph_legacy(*args, **kwargs) -> Any:
    """Legacy wrapper for build_graph with deprecation warning.

    DEPRECATED: Use build_deduplication_graph() with DeduplicationConfig instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "build_graph_legacy() is deprecated. Use build_deduplication_graph() "
        "with DeduplicationConfig for better error handling and type safety.",
        DeprecationWarning,
        stacklevel=2
    )
    return build_graph(*args, **kwargs)


__all__ = [
    # Functional Core API
    'build_deduplication_graph',
    'build_level_graph',
    'create_all_edges',
    'add_edges_to_graph',
    'remove_weak_edges_functional',

    # Result types and error handling
    'Success',
    'Failure',
    'Result',
    'EdgeCreationResult',
    'GraphBuildResult',
    'is_success',
    'is_failure',
    'get_value',
    'get_error',

    # Configuration System
    'DeduplicationConfig',
    'RuleConfig',
    'WebConfig',
    'LLMConfig',
    'Edge',
    'EdgeBatch',
    'WebResource',
    'TermWebContent',
    'EDGE_TYPES',
    'METHODS',
    'SUPPORTED_PROVIDERS',

    # Pure Edge Creation Functions
    'create_rule_edges',
    'create_web_edges',
    'create_llm_edges',

    # Functional Composition
    'compose_edge_creators',
    'with_error_handling',
    'parallel_edge_creation',
    'filter_edges',
    'combine_edge_results',

    # Legacy API (Deprecated)
    'build_graph',
    'build_graph_legacy',
    'main',

    # Legacy querying API
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