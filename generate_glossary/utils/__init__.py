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
    get_llm_client,
    structured_completion,
    text_completion,
    infer_structured,
    infer_text,
    get_random_llm_config
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
    'get_llm_client',
    'structured_completion',
    'text_completion',
    'infer_structured',
    'infer_text',
    'get_random_llm_config',
    
    # Config
    'Config',
    'get_config',
    'load_config'
]