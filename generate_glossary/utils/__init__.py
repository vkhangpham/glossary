"""
Core utility modules for the generate_glossary package.

This package contains essential utilities:
- logger: Logging setup  
- llm: DSPy-based LLM interface (now a top-level module)
- config: Configuration management
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
    'get_logger',
    'completion', 'structured_completion', 'async_completion', 'async_structured_completion',
    'configure_dspy_global', 'configure_dspy_cache', 'with_lm_context', 'run_async_safely',
    'test_optimized_prompt_integration', 'validate_prompt_format', 'benchmark_integration',
    'configure_for_optimized_prompts', 'get_optimized_prompt_stats', 'enhance_prompt_for_dspy',
    'Config', 'get_config', 'load_config',
]