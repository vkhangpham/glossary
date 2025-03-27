"""Package for generating and managing technical glossaries."""

from .utils import (
    setup_logger,
)

__version__ = "0.1.0"

__all__ = [
    # Data models
    'WebContentTrust',
    'ContentVerification',
    'WebContent',
    'MiningResult',
    
    # Web verification functions
    'verify_content',
    'verify_batch',
    'get_domain_trust_level',
    
    # Web mining functions
    'get_domain',
    'process_urls',
    'mine_urls',
    
    # Other utilities
    'analyze_content',
    'setup_logger',
] 