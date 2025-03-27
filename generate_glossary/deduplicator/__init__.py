"""
Deduplicator package for technical concepts.

This package provides functionality for deduplicating technical concepts
using different modes: rule-based, web-based, and LLM-based.
"""

from .deduplication_modes import (
    deduplicate_rule_based,
    deduplicate_web_based,
    deduplicate_llm_based,
)

from .graph_dedup import (
    deduplicate_graph_based,
)

from .dedup_utils import (
    normalize_text,
    get_term_variations,
    is_compound_term,
)

__all__ = [
    'deduplicate_rule_based',
    'deduplicate_web_based',
    'deduplicate_llm_based',
    'deduplicate_graph_based',
    'normalize_text',
    'get_term_variations',
    'is_compound_term',
    'batch_normalize_terms',
    'batch_get_variations',
    'are_wiki_pages_similar',
] 