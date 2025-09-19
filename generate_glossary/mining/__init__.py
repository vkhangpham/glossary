"""
Simplified mining module with direct Firecrawl v2.2.0 integration.

This module provides clean, focused web content extraction using Firecrawl v2.2.0
features including batch scraping, smart crawling, enhanced caching, and summary
format optimization.

Key Features:
- Direct Firecrawl SDK usage without custom abstractions
- Batch scraping for improved performance
- Academic content filtering and processing
- Simple configuration management
- Clean CLI interface

Usage:
    from generate_glossary.mining import main

    # Use via CLI
    uv run mine-web terms.txt -o results.json

    # Or programmatically
    from generate_glossary.mining.config import load_config, get_firecrawl_client
    from generate_glossary.mining.main import mine_concepts_simple

    config = load_config()
    client = get_firecrawl_client(config.firecrawl)
    results = mine_concepts_simple(["machine learning"], client, config.mining)
"""

# Core functionality
from .main import main, mine_concepts_simple

# Configuration management
from .config import (
    load_config,
    get_firecrawl_client,
    MiningModuleConfig,
    FirecrawlConfig,
    MiningConfig,
    OutputConfig,
    ConfigError,
)

# Essential utilities
from .utils import (
    load_terms_from_file,
    save_mining_results,
    aggregate_mining_results,
    format_concept_results,
    filter_academic_urls,
    validate_url,
    is_academic_domain,
)

# Firecrawl client helpers
from .core.firecrawl_client import (
    create_client,
    create_async_client,
    search_academic_concepts,
    batch_scrape_urls,
    scrape_single_url,
    get_queue_status,
)

__all__ = [
    # Main entry points
    "main",
    "mine_concepts_simple",

    # Configuration
    "load_config",
    "get_firecrawl_client",
    "MiningModuleConfig",
    "FirecrawlConfig",
    "MiningConfig",
    "OutputConfig",
    "ConfigError",

    # Essential utilities
    "load_terms_from_file",
    "save_mining_results",
    "aggregate_mining_results",
    "format_concept_results",
    "filter_academic_urls",
    "validate_url",
    "is_academic_domain",

    # Firecrawl client helpers
    "create_client",
    "create_async_client",
    "search_academic_concepts",
    "batch_scrape_urls",
    "scrape_single_url",
    "get_queue_status",
]
