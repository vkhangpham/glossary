"""
Core utility modules for the generate_glossary package.

This package now contains only essential utilities:
- logger: Logging setup
- llm: LLM interface  
- config: Configuration management

Other functionality has been moved to:
- metadata: File discovery and metadata collection
- processing: Checkpointing and resilient processing
- mining: Web content mining
- security: API key management
"""

from .logger import setup_logger
from .llm import (
    LLMClient,
    completion,
    async_completion,
    structured_completion_consensus,
)
from .config import (
    Config,
    get_config,
    load_config
)

__all__ = [
    # Logger
    'setup_logger',
    
    # LLM
    'LLMClient',
    'completion',
    'async_completion',
    'structured_completion_consensus',
    
    # Config
    'Config',
    'get_config',
    'load_config'
]