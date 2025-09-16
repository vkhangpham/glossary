"""
Unified web mining interface with Firecrawl v2.2.0 integration.

This module provides a clean, unified interface for web content extraction
using Firecrawl v2.2.0 features including batch scraping (500% performance improvement),
smart crawling with natural language prompts, enhanced caching, summary format
optimization, actions for dynamic content interaction, and new v2.2.0 capabilities.

Key Features:
- Batch scraping for 500% performance improvement
- Smart crawling with natural language prompts
- Enhanced caching with maxAge parameter
- Summary format for optimized content extraction
- Actions support for dynamic content interaction
- Research category filtering for academic content
- JSON schema extraction for structured data
- Comprehensive error handling and logging
- Queue status monitoring and predictive management
- 15x faster Map endpoint for URL discovery
- Enhanced webhooks with signature verification

Usage:
    from generate_glossary.mining import mine_concepts

    results = mine_concepts(
        ["machine learning", "neural networks"],
        use_batch_scrape=True,
        use_summary=True,
        max_age=172800000,  # 2 days cache
        max_pages=5,  # v2.2.0: Limit PDF parsing
        enable_queue_monitoring=True  # v2.2.0: Queue monitoring
    )
"""

# Import everything from mining module
from .mining import *

# Backward compatibility aliases
mine_concepts_with_firecrawl = mine_concepts

# Set __all__ to include all exports from mining module
__all__ = __import__('generate_glossary.mining.mining', fromlist=['__all__']).__all__ + [
    "mine_concepts_with_firecrawl"  # Add our backward compatibility alias
]