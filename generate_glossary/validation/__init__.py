"""
Public API for concept validation.

Simple, functional interface for validating technical concepts.
"""
import warnings
from typing import Dict, List, Any, Optional, Union
from enum import Enum


# Legacy ValidationModes enum for backward compatibility
class ValidationModes(Enum):
    """Legacy validation modes enum for backward compatibility."""
    RULE = "rule"
    WEB = "web"
    LLM = "llm"


# Functional validation core (new)
from .core import (
    validate_terms_functional,
    ValidationResult,
    filter_valid_terms,
    filter_invalid_terms,
    get_validation_summary,
    # Functional composition utilities
    compose_validators,
    parallel_validate,
    combine_results,
    create_validation_pipeline,
    with_timeout,
    with_retry,
    conditional_validate,
    filter_results,
    transform_results,
    # Cache functions
    validate_terms_with_cache,
    with_cache_support
)

# Configuration system
from .config import (
    ValidationConfig,
    RuleValidationConfig,
    WebValidationConfig,
    LLMValidationConfig,
    # Predefined profiles
    STRICT_PROFILE,
    PERMISSIVE_PROFILE,
    ACADEMIC_PROFILE,
    TECHNICAL_PROFILE,
    FAST_PROFILE,
    COMPREHENSIVE_PROFILE,
    PROFILE_REGISTRY,
    # Profile management functions
    get_profile,
    list_profiles,
    get_profile_summary,
    # Configuration factory functions
    create_rule_config,
    create_web_config,
    create_llm_config,
    create_validation_config,
    create_profile,
    # Higher-order functions
    with_config,
    merge_configs,
    override_config,
    # Utility functions
    get_default_config,
    get_recommended_profile,
    validate_config_compatibility
)

# Cache functions from cache module
from .cache import (
    CacheState,
    load_cache_from_disk,
    save_cache_to_disk,
    cache_get_validation,
    cache_set_validation,
    cache_get_rejected,
    filter_cached_terms_functional,
    with_cache,
    with_cache_state
)

# Essential query functions
from .api import (
    get_valid_terms,
    get_invalid_terms,
    save_validation_results,
    load_validation_results
)



def validate_terms(
    terms: Union[str, List[str]],
    modes: Optional[List[str]] = None,
    web_content: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    llm_provider: str = "gemini",
    config: Optional[Dict[str, Any]] = None,
    existing_results: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    use_cache: bool = True,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Legacy validation function with deprecation warning.

    ⚠️  DEPRECATED: This function is deprecated and will be removed in a future version.
    Use validate_terms_functional() instead for better performance and features.

    Migration example:
        # Old way:
        results = validate_terms(terms, modes=['rule', 'web'])

        # New way:
        from generate_glossary.validation import validate_terms_functional, ACADEMIC_PROFILE, override_config
        config = override_config(ACADEMIC_PROFILE, modes=('rule', 'web'))
        results = validate_terms_functional(terms, config)

    Args:
        terms: Terms to validate
        modes: List of validation modes ('rule', 'web', 'llm')
        web_content: Web content for web validation mode
        llm_provider: LLM provider name
        config: Legacy configuration dictionary
        existing_results: Existing validation results to merge with
        show_progress: Whether to show progress bars
        use_cache: Whether to use caching
        *args: Legacy positional arguments
        **kwargs: Legacy keyword arguments

    Returns:
        Legacy format validation results
    """
    warnings.warn(
        "validate_terms() is deprecated and will be removed in v4.0.0. "
        "Use validate_terms_functional() with configuration profiles instead. "
        "See migration guide: https://docs.example.com/validation-migration",
        DeprecationWarning,
        stacklevel=2
    )

    # Build legacy config kwargs from function arguments
    legacy_kwargs = {}
    if modes is not None:
        legacy_kwargs["modes"] = modes
    if llm_provider != "gemini":
        legacy_kwargs["llm_provider"] = llm_provider
    if not show_progress:
        legacy_kwargs["show_progress"] = False
    if not use_cache:
        legacy_kwargs["use_cache"] = False

    # Merge with provided config
    if config:
        legacy_kwargs.update(config)

    # Add any remaining kwargs
    legacy_kwargs.update(kwargs)

    # Translate legacy config to functional config
    functional_config = get_validation_config_legacy(**legacy_kwargs)

    # Convert existing results from legacy format if provided
    existing_functional_results = None
    if existing_results:
        from .main import convert_legacy_to_functional_results
        existing_functional_results = convert_legacy_to_functional_results(existing_results)

    # Use functional validation
    if use_cache:
        from .cache import load_cache_from_disk, save_cache_to_disk
        cache_state = load_cache_from_disk()
        validation_results, updated_cache_state = validate_terms_with_cache(
            terms, functional_config, web_content, existing_functional_results, cache_state, auto_save=True
        )
    else:
        validation_results = validate_terms_functional(
            terms, functional_config, web_content, existing_functional_results
        )
        updated_cache_state = None

    # Convert results back to legacy format
    from .main import convert_functional_to_legacy_results
    return convert_functional_to_legacy_results(validation_results)


def get_validation_config_legacy(**kwargs) -> ValidationConfig:
    """
    Legacy configuration function with deprecation warning.

    ⚠️  DEPRECATED: This function is deprecated and will be removed in a future version.
    Use the functional configuration system instead.

    Migration example:
        # Old way:
        config = get_validation_config_legacy(modes=['rule', 'web'])

        # New way:
        from generate_glossary.validation import ACADEMIC_PROFILE, override_config
        config = override_config(ACADEMIC_PROFILE, modes=('rule', 'web'))

    Args:
        **kwargs: Legacy configuration arguments
            - modes: List of validation modes ('rule', 'web', 'llm')
            - min_confidence: Minimum confidence threshold (float)
            - min_score: Minimum score threshold (float)
            - min_relevance_score: Minimum relevance score threshold (float)
            - parallel: Enable parallel processing (bool)
            - show_progress: Show progress bars (bool)
            - llm_provider: LLM provider name (str)
            - use_cache: Enable caching (bool)

    Returns:
        ValidationConfig object constructed from legacy arguments
    """
    warnings.warn(
        "get_validation_config_legacy() is deprecated and will be removed in v4.0.0. "
        "Use the functional configuration system with profiles instead. "
        "See migration guide: https://docs.example.com/validation-migration",
        DeprecationWarning,
        stacklevel=2
    )

    # Extract and translate legacy kwargs to functional config
    from .config import ACADEMIC_PROFILE, override_config
    from types import MappingProxyType

    # Convert modes from list to tuple, handling ValidationModes enums
    modes = kwargs.get("modes", ["rule"])
    if isinstance(modes, list):
        # Convert ValidationModes enum values to strings
        string_modes = []
        for mode in modes:
            if isinstance(mode, ValidationModes):
                string_modes.append(mode.value)
            else:
                string_modes.append(mode)
        modes = tuple(string_modes)

    # Build translation dict
    config_overrides = {}

    if "modes" in kwargs:
        config_overrides["modes"] = modes
    if "min_confidence" in kwargs:
        config_overrides["min_confidence"] = kwargs["min_confidence"]
    if "min_score" in kwargs:
        config_overrides["min_score"] = kwargs["min_score"]
    if "min_relevance_score" in kwargs:
        config_overrides["min_relevance_score"] = kwargs["min_relevance_score"]
    if "parallel" in kwargs:
        config_overrides["parallel"] = kwargs["parallel"]
    if "use_cache" in kwargs:
        config_overrides["use_cache"] = kwargs["use_cache"]
    if "llm_provider" in kwargs:
        # LLM provider is part of llm_config, need to override it
        from .config import create_llm_config
        if "llm_config" not in config_overrides:
            config_overrides["llm_config"] = create_llm_config(provider=kwargs["llm_provider"])

    # Handle legacy confidence_weights if provided
    if "confidence_weights" in kwargs:
        weights = kwargs["confidence_weights"]
        if isinstance(weights, dict):
            config_overrides["confidence_weights"] = MappingProxyType(weights)

    return override_config(ACADEMIC_PROFILE, **config_overrides)


# Public API
__all__ = [
    # Legacy API (backward compatibility with deprecation warnings)
    "validate_terms",
    "ValidationModes",
    "get_validation_config_legacy",
    "get_valid_terms",
    "get_invalid_terms",
    "save_validation_results",
    "load_validation_results",

    # Functional validation core
    "validate_terms_functional",
    "ValidationResult",
    "filter_valid_terms",
    "filter_invalid_terms",
    "get_validation_summary",

    # Functional composition utilities
    "compose_validators",
    "parallel_validate",
    "combine_results",
    "create_validation_pipeline",
    "with_timeout",
    "with_retry",
    "conditional_validate",
    "filter_results",
    "transform_results",

    # Cache functions
    "CacheState",
    "load_cache_from_disk",
    "save_cache_to_disk",
    "validate_terms_with_cache",
    "with_cache_support",
    "cache_get_validation",
    "cache_set_validation",
    "cache_get_rejected",
    "filter_cached_terms_functional",
    "with_cache",
    "with_cache_state",

    # Configuration system
    "ValidationConfig",
    "RuleValidationConfig",
    "WebValidationConfig",
    "LLMValidationConfig",

    # Predefined profiles
    "STRICT_PROFILE",
    "PERMISSIVE_PROFILE",
    "ACADEMIC_PROFILE",
    "TECHNICAL_PROFILE",
    "FAST_PROFILE",
    "COMPREHENSIVE_PROFILE",
    "PROFILE_REGISTRY",

    # Profile management functions
    "get_profile",
    "list_profiles",
    "get_profile_summary",

    # Configuration factory functions
    "create_rule_config",
    "create_web_config",
    "create_llm_config",
    "create_validation_config",
    "create_profile",

    # Higher-order functions
    "with_config",
    "merge_configs",
    "override_config",

    # Utility functions
    "get_default_config",
    "get_recommended_profile",
    "validate_config_compatibility"
]

# Backward compatibility aliases
validate = validate_terms  # Keep for existing code 