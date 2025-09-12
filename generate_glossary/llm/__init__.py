"""
Enhanced DSPy-based LLM interface module with native signature optimization.

This module provides a comprehensive DSPy-based interface for LLM interactions with enhanced
support for optimized prompts through signature-based integration. It combines traditional
text-based prompt optimization with DSPy's native signature and predictor patterns for
improved performance and structured reasoning.
"""

from .completions import (
    completion, structured_completion, async_completion, async_structured_completion,
    test_optimized_prompt_integration, validate_prompt_format, benchmark_integration,
    configure_for_optimized_prompts, get_optimized_prompt_stats,
)

from .dspy_config import (
    configure_dspy_global, configure_dspy_cache, get_dspy_context,
    configure_for_optimization, configure_for_signature_based_workflows, cleanup_cache,
)

from .helpers import (
    with_lm_context, run_async_safely, is_async_context,
)

from .signatures import (
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

__all__ = [
    "completion", "structured_completion", "async_completion", "async_structured_completion",
    "configure_dspy_global", "configure_dspy_cache", "with_lm_context", "get_dspy_context",
    "configure_for_optimization", "configure_for_signature_based_workflows", "cleanup_cache",
    "test_optimized_prompt_integration", "validate_prompt_format", "benchmark_integration",
    "configure_for_optimized_prompts", "get_optimized_prompt_stats", "enhance_prompt_for_dspy",
    "validate_dspy_compliance",
    "run_async_safely", "is_async_context"
]