"""
Simplified web mining interface with Firecrawl v2.0 integration.

This module provides a clean, unified interface for web content extraction 
using Firecrawl v2.0 features including batch scraping, smart crawling,
enhanced caching, and summary format optimization.

Key Features:
- Unified mine_concepts() function with comprehensive v2.0 feature support
- Batch scraping for 500% performance improvement over sequential scraping
- Smart crawling with natural language prompts for academic content extraction
- Enhanced caching with maxAge parameter for faster repeated requests
- Summary format for optimized content extraction and reduced token usage
- Research category filtering for academic-focused content
- Clean, simple API replacing the old complex multi-file structure
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