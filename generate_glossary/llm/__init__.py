"""
Enhanced DSPy-based LLM interface module with native signature optimization.

This module provides a comprehensive DSPy-based interface for LLM interactions with enhanced
support for optimized prompts through signature-based integration. It combines traditional
text-based prompt optimization with DSPy's native signature and predictor patterns for
improved performance and structured reasoning.

Key Features:
- **DSPy-Native Signature Integration**: Automatic conversion of optimized prompts to DSPy signatures
- **Chain of Thought Support**: Enhanced reasoning capabilities through ChainOfThought predictors
- **Structured Output Integration**: Seamless structured output with optimized prompts
- **Fallback Compatibility**: Graceful degradation to text-based approaches when needed
- **Advanced Configuration**: Tier-based model selection with caching control
- **Testing & Validation**: Comprehensive testing utilities for prompt integration
- **Performance Analytics**: Statistical tracking of signature vs text-based approaches

Enhanced DSPy Integration (2024-2025):
- **Signature Metadata Integration**: Direct use of optimization-time signature metadata
- **Declarative Programming**: Separation of task logic from prompt formatting
- **Modular Composition**: Support for ChainOfThought and TypedPredictor patterns
- **Metric-Driven Development**: Comprehensive validation and benchmarking
- **DSPy Compliance Validation**: Automated checks against DSPy best practices
- **Field Extraction**: Intelligent mapping of message content to signature input fields
- **Format Enhancement**: Tools for improving prompt optimization output for DSPy compatibility

Core Functions:
- completion(): Text-only completions with signature-based optimization
- structured_completion(): Structured output with enhanced DSPy integration
- async_completion(): Async completions with signature support
- async_structured_completion(): Async structured output with optimization

Async Utilities:
- run_async_safely(): Safely run async functions from sync context
- is_async_context(): Check if running in an async context

Configuration Functions:
- configure_dspy_global(): Global DSPy configuration setup
- configure_dspy_cache(): Cache configuration with cleanup
- configure_for_optimized_prompts(): Optimize DSPy for signature-based prompts
- configure_for_signature_based_workflows(): DSPy 2024-2025 signature optimization
- with_lm_context(): Context manager for temporary LM switching

Testing & Validation Functions:
- test_optimized_prompt_integration(): Test integration with existing optimized prompts
- validate_prompt_format(): Validate prompt compatibility with DSPy signatures
- benchmark_integration(): Performance comparison of integration approaches
- enhance_prompt_for_dspy(): Enhance prompts for better DSPy compatibility
- validate_dspy_compliance(): Check DSPy 2024-2025 best practices compliance

Analytics Functions:
- get_optimized_prompt_stats(): Usage statistics and performance metrics

Migration Guide:
Existing code continues to work unchanged. Enhanced features activate automatically
when signature metadata is available. For optimal DSPy 2024-2025 compliance:

1. **Use Signature Metadata**: Leverage optimization-time metadata when available
2. **Configure for Signatures**: configure_for_signature_based_workflows()
3. **Validate Compliance**: validate_dspy_compliance("your_use_case")
4. **Monitor Performance**: get_optimized_prompt_stats() for metadata vs text inference usage
5. **Test Integration**: test_optimized_prompt_integration("your_use_case", messages)

Backward Compatibility:
- Text-based prompts continue working with automatic signature inference
- Graceful fallback when signature metadata is unavailable
- Progressive enhancement based on available optimization artifacts
"""

# Import all public functions from the specialized modules
from .completions import (
    # Core completion functions
    completion, structured_completion, async_completion, async_structured_completion,
    
    # Testing and validation functions
    test_optimized_prompt_integration, validate_prompt_format, benchmark_integration,
    
    # Configuration and statistics functions
    configure_for_optimized_prompts, get_optimized_prompt_stats,
)

from .dspy_config import (
    # DSPy configuration functions
    configure_dspy_global, configure_dspy_cache, get_dspy_context,
    configure_for_optimization, configure_for_signature_based_workflows, cleanup_cache,
)

from .helpers import (
    # Context management and async utilities
    with_lm_context, run_async_safely, is_async_context,
)

from .signatures import (
    # Enhancement utilities
    enhance_prompt_for_dspy, validate_dspy_compliance,
)

# For backward compatibility with tests that import private functions,
# also re-export private functions that are accessed by tests
from .helpers import (
    _resolve_model, _resolve_temperature, _get_lm, _join_messages,
    _safe_format_template, _extract_prompt_variables, _apply_optimized_prompts,
    _normalize_model_name, _is_gpt5_family, _is_likely_openai_model, _strip_role_markers,
)

from .signatures import (
    _parse_signature_from_prompt, _create_dynamic_signature,
    _extract_content_for_signature, _parse_structured_output,
)

from .completions import (
    _try_load_optimized_prompt, _apply_optimized_signature, _create_optimized_predictor,
)

# Maintain the same __all__ list as the original module for public API compatibility
__all__ = [
    # Core completion functions
    "completion", "structured_completion", "async_completion", "async_structured_completion",
    
    # DSPy configuration functions
    "configure_dspy_global", "configure_dspy_cache", "with_lm_context", "get_dspy_context",
    "configure_for_optimization", "configure_for_signature_based_workflows", "cleanup_cache",
    
    # Enhanced DSPy integration functions
    "test_optimized_prompt_integration", "validate_prompt_format", "benchmark_integration",
    "configure_for_optimized_prompts", "get_optimized_prompt_stats", "enhance_prompt_for_dspy",
    "validate_dspy_compliance",
    
    # Async utilities
    "run_async_safely", "is_async_context"
]