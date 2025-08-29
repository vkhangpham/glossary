"""
Web mining utilities for extracting content from the web.

This module provides a unified interface for web content extraction
using various providers including Firecrawl.
"""

from .runner import run_web_mining
from .firecrawl import (
    mine_concepts_with_firecrawl,
    initialize_firecrawl,
    extract_definitions_firecrawl,
    search_concept_firecrawl,
    ConceptDefinition,
    WebResource
)

__all__ = [
    # Main interface
    'run_web_mining',
    'mine_web_content',
    
    # Firecrawl specific
    'mine_concepts_with_firecrawl',
    'initialize_firecrawl',
    'extract_definitions_firecrawl',
    'search_concept_firecrawl',
    
    # Data models
    'ConceptDefinition',
    'WebResource'
]

def mine_web_content(terms, output_file, use_firecrawl=True, **kwargs):
    """
    Unified interface for mining web content.
    
    Args:
        terms: List of terms or path to terms file
        output_file: Output file path
        use_firecrawl: Whether to use Firecrawl (default: True)
        **kwargs: Additional arguments passed to the miner
        
    Returns:
        MiningResult object with extracted content
    """
    return run_web_mining(
        input_file=terms if isinstance(terms, str) else None,
        output_file=output_file,
        terms=terms if isinstance(terms, list) else None,
        use_firecrawl=use_firecrawl,
        **kwargs
    )