"""
Simplified web mining interface with Firecrawl v2.0 integration.

This module provides a clean, unified interface for web content extraction 
using Firecrawl v2.0 features including batch scraping, smart crawling,
enhanced caching, and summary format optimization.
"""

# Import the new unified API
from .mining import mine_concepts, initialize_firecrawl, ConceptDefinition, WebResource

# Backward compatibility aliases
mine_concepts_with_firecrawl = mine_concepts

__all__ = [
    "mine_concepts",
    "mine_concepts_with_firecrawl",
    "initialize_firecrawl",
    "ConceptDefinition",
    "WebResource"
]