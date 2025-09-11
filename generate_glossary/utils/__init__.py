"""
Core utility modules for the generate_glossary package.

This package contains essential utilities:
- logger: Logging setup
- llm: DSPy-based LLM interface (now a top-level module) with optimized prompts and async support
- config: Configuration management

The LLM interface has been moved to a top-level module but is re-exported here for backward compatibility.
The interface uses the DSPy framework, providing:
- Optimized prompt engineering with signature-based optimization
- Structured completion with enhanced DSPy integration
- Async completion support with event loop safety
- Advanced configuration and context management
- Async utilities for event loop conflict resolution

Other functionality has been moved to:
- metadata: File discovery and metadata collection
- processing: Checkpointing and resilient processing
- mining: Web content mining
- security: API key management
"""

from .logger import get_logger
from generate_glossary.llm import (
    completion,
    structured_completion,
    async_completion,
    async_structured_completion,
    configure_dspy_global,
    configure_dspy_cache,
    with_lm_context,
    run_async_safely,
    # Testing and validation functions
    test_optimized_prompt_integration,
    validate_prompt_format,
    benchmark_integration,
    configure_for_optimized_prompts,
    get_optimized_prompt_stats,
    enhance_prompt_for_dspy,
)
from .config import (
    Config,
    get_config,
    load_config
)

__all__ = [
    # Logger
    'get_logger',
    # LLM (DSPy-based) - Core functions
    'completion', 'structured_completion', 'async_completion', 'async_structured_completion',
    'configure_dspy_global', 'configure_dspy_cache', 'with_lm_context', 'run_async_safely',
    # LLM (DSPy-based) - Testing and validation
    'test_optimized_prompt_integration', 'validate_prompt_format', 'benchmark_integration',
    'configure_for_optimized_prompts', 'get_optimized_prompt_stats', 'enhance_prompt_for_dspy',
    # Config
    'Config', 'get_config', 'load_config',
]