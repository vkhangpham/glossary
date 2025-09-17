"""Edge creation modules for deduplication graph construction.

This package contains modules for creating different types of edges in the deduplication graph:
- rule_based: Deterministic rules for creating edges
- web_based: Web content similarity and domain-specific edges
- llm_based: LLM-powered semantic similarity edges
"""

from .rule_based import *
from .web_based import *
from .llm_based import *

__all__ = [
    # Rule-based edge creation
    "add_rule_based_edges",
    "add_acronym_edges",
    "add_synonym_edges",
    "create_rule_edges",

    # Web-based edge creation
    "add_web_based_edges",
    "add_domain_specific_edges",
    "add_content_similarity_edges",
    "create_web_edges",

    # LLM-based edge creation
    "add_llm_based_edges",
    "create_llm_edges",
]