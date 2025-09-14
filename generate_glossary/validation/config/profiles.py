"""
Predefined validation profiles and higher-order configuration functions.

This module provides predefined validation configurations optimized for different
use cases and functional composition utilities for configuration management.
"""

import logging
from typing import Dict, Any, Callable, List, Union, Optional
from types import MappingProxyType
from functools import partial

from .defaults import (
    ValidationConfig,
    RuleValidationConfig,
    WebValidationConfig,
    LLMValidationConfig,
    DEFAULT_BLACKLIST_TERMS
)
from ..core import ValidationConfig as CoreValidationConfig

# Predefined validation profiles
STRICT_PROFILE = ValidationConfig(
    modes=("rule", "web", "llm"),
    confidence_weights=MappingProxyType({
        "rule": 0.2,
        "web": 0.5,
        "llm": 0.3
    }),
    min_confidence=0.8,
    min_score=0.7,
    min_relevance_score=0.7,
    parallel=True,
    use_cache=True,
    rule_config=RuleValidationConfig(
        max_workers=4,
        blacklist_terms=DEFAULT_BLACKLIST_TERMS,
        min_term_length=3,
        max_term_length=80,
        show_progress=True
    ),
    web_config=WebValidationConfig(
        min_score=0.7,
        min_relevance_score=0.7,
        min_relevant_sources=2,
        high_quality_content_threshold=0.8,
        high_quality_relevance_threshold=0.8,
        max_workers=4,
        show_progress=True
    ),
    llm_config=LLMValidationConfig(
        provider="gemini",
        batch_size=5,
        max_workers=2,
        tier="standard",
        max_tokens=150,
        batch_max_tokens=750,
        show_progress=True
    )
)

PERMISSIVE_PROFILE = ValidationConfig(
    modes=("rule",),
    confidence_weights=MappingProxyType({
        "rule": 1.0
    }),
    min_confidence=0.3,
    min_score=0.3,
    min_relevance_score=0.3,
    parallel=False,
    use_cache=True,
    rule_config=RuleValidationConfig(
        max_workers=8,
        blacklist_terms=frozenset(),  # No blacklist
        min_term_length=1,
        max_term_length=150,
        show_progress=True
    ),
    web_config=WebValidationConfig(
        min_score=0.3,
        min_relevance_score=0.3,
        min_relevant_sources=1,
        high_quality_content_threshold=0.5,
        high_quality_relevance_threshold=0.5,
        max_workers=8,
        show_progress=True
    ),
    llm_config=LLMValidationConfig(
        provider="gemini",
        batch_size=20,
        max_workers=8,
        tier="budget",
        max_tokens=50,
        batch_max_tokens=300,
        show_progress=True
    )
)

ACADEMIC_PROFILE = ValidationConfig(
    modes=("rule", "web"),
    confidence_weights=MappingProxyType({
        "rule": 0.4,
        "web": 0.6
    }),
    min_confidence=0.6,
    min_score=0.6,
    min_relevance_score=0.6,
    parallel=True,
    use_cache=True,
    rule_config=RuleValidationConfig(
        max_workers=4,
        blacklist_terms=DEFAULT_BLACKLIST_TERMS | frozenset({
            'tutorial', 'guide', 'howto', 'faq', 'help', 'support'
        }),
        min_term_length=2,
        max_term_length=100,
        show_progress=True
    ),
    web_config=WebValidationConfig(
        min_score=0.6,
        min_relevance_score=0.6,
        min_relevant_sources=1,
        high_quality_content_threshold=0.75,
        high_quality_relevance_threshold=0.75,
        max_workers=4,
        show_progress=True
    ),
    llm_config=LLMValidationConfig(
        provider="gemini",
        batch_size=10,
        max_workers=4,
        tier="budget",
        max_tokens=100,
        batch_max_tokens=500,
        show_progress=True
    )
)

TECHNICAL_PROFILE = ValidationConfig(
    modes=("rule", "web", "llm"),
    confidence_weights=MappingProxyType({
        "rule": 0.3,
        "web": 0.4,
        "llm": 0.3
    }),
    min_confidence=0.7,
    min_score=0.65,
    min_relevance_score=0.65,
    parallel=True,
    use_cache=True,
    rule_config=RuleValidationConfig(
        max_workers=4,
        blacklist_terms=DEFAULT_BLACKLIST_TERMS | frozenset({
            'blog', 'post', 'article', 'news', 'update'
        }),
        min_term_length=2,
        max_term_length=120,
        show_progress=True
    ),
    web_config=WebValidationConfig(
        min_score=0.65,
        min_relevance_score=0.65,
        min_relevant_sources=1,
        high_quality_content_threshold=0.75,
        high_quality_relevance_threshold=0.75,
        max_workers=4,
        show_progress=True
    ),
    llm_config=LLMValidationConfig(
        provider="gemini",
        batch_size=8,
        max_workers=4,
        tier="standard",
        max_tokens=120,
        batch_max_tokens=600,
        show_progress=True
    )
)

FAST_PROFILE = ValidationConfig(
    modes=("rule",),
    confidence_weights=MappingProxyType({
        "rule": 1.0
    }),
    min_confidence=0.5,
    min_score=0.5,
    min_relevance_score=0.5,
    parallel=True,
    use_cache=True,
    rule_config=RuleValidationConfig(
        max_workers=8,
        blacklist_terms=DEFAULT_BLACKLIST_TERMS,
        min_term_length=2,
        max_term_length=100,
        show_progress=False  # No progress for speed
    ),
    web_config=WebValidationConfig(
        min_score=0.5,
        min_relevance_score=0.5,
        min_relevant_sources=1,
        high_quality_content_threshold=0.7,
        high_quality_relevance_threshold=0.7,
        max_workers=8,
        show_progress=False
    ),
    llm_config=LLMValidationConfig(
        provider="gemini",
        batch_size=50,  # Large batches for speed
        max_workers=8,
        tier="budget",
        max_tokens=50,
        batch_max_tokens=250,
        show_progress=False
    )
)

COMPREHENSIVE_PROFILE = ValidationConfig(
    modes=("rule", "web", "llm"),
    confidence_weights=MappingProxyType({
        "rule": 0.25,
        "web": 0.5,
        "llm": 0.25
    }),
    min_confidence=0.75,
    min_score=0.7,
    min_relevance_score=0.7,
    parallel=True,
    use_cache=True,
    rule_config=RuleValidationConfig(
        max_workers=4,
        blacklist_terms=DEFAULT_BLACKLIST_TERMS,
        min_term_length=3,
        max_term_length=100,
        show_progress=True
    ),
    web_config=WebValidationConfig(
        min_score=0.7,
        min_relevance_score=0.7,
        min_relevant_sources=3,  # Require multiple sources
        high_quality_content_threshold=0.8,
        high_quality_relevance_threshold=0.8,
        max_workers=4,
        show_progress=True
    ),
    llm_config=LLMValidationConfig(
        provider="gemini",
        batch_size=5,  # Small batches for quality
        max_workers=2,
        tier="premium",
        max_tokens=200,
        batch_max_tokens=1000,
        show_progress=True
    )
)

# Profile registry for easy access
PROFILE_REGISTRY: Dict[str, ValidationConfig] = {
    "strict": STRICT_PROFILE,
    "permissive": PERMISSIVE_PROFILE,
    "academic": ACADEMIC_PROFILE,
    "technical": TECHNICAL_PROFILE,
    "fast": FAST_PROFILE,
    "comprehensive": COMPREHENSIVE_PROFILE
}


def get_profile(name: str) -> ValidationConfig:
    """
    Retrieve predefined validation profile by name.

    Args:
        name: Profile name (strict, permissive, academic, technical, fast, comprehensive)

    Returns:
        Validation configuration for the profile

    Raises:
        ValueError: If profile name is not found
    """
    if name not in PROFILE_REGISTRY:
        available = ', '.join(PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{name}'. Available profiles: {available}")
    return PROFILE_REGISTRY[name]


def list_profiles() -> List[str]:
    """Get list of available profile names."""
    return list(PROFILE_REGISTRY.keys())


def create_rule_config(**overrides) -> RuleValidationConfig:
    """
    Create rule validation config with overrides.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        Configured RuleValidationConfig
    """
    defaults = {
        'max_workers': 4,
        'blacklist_terms': DEFAULT_BLACKLIST_TERMS,
        'min_term_length': 2,
        'max_term_length': 100,
        'show_progress': True
    }
    defaults.update(overrides)
    return RuleValidationConfig(**defaults)


def create_web_config(**overrides) -> WebValidationConfig:
    """
    Create web validation config with overrides.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        Configured WebValidationConfig
    """
    defaults = {
        'min_score': 0.5,
        'min_relevance_score': 0.5,
        'min_relevant_sources': 1,
        'high_quality_content_threshold': 0.7,
        'high_quality_relevance_threshold': 0.7,
        'max_workers': 4,
        'show_progress': True
    }
    defaults.update(overrides)
    return WebValidationConfig(**defaults)


def create_llm_config(**overrides) -> LLMValidationConfig:
    """
    Create LLM validation config with overrides.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        Configured LLMValidationConfig
    """
    defaults = {
        'provider': 'gemini',
        'batch_size': 10,
        'max_workers': 4,
        'tier': 'budget',
        'max_tokens': 100,
        'batch_max_tokens': 500,
        'show_progress': True
    }
    defaults.update(overrides)
    return LLMValidationConfig(**defaults)


def create_validation_config(**overrides) -> ValidationConfig:
    """
    Create full validation config with overrides.

    Args:
        **overrides: Fields to override from defaults

    Returns:
        Configured ValidationConfig
    """
    defaults = {
        'modes': ("rule",),
        'confidence_weights': None,  # Will be set in __post_init__
        'min_confidence': 0.5,
        'min_score': 0.5,
        'min_relevance_score': 0.5,
        'parallel': True,
        'use_cache': True,
        'rule_config': None,  # Will be set in __post_init__
        'web_config': None,   # Will be set in __post_init__
        'llm_config': None    # Will be set in __post_init__
    }
    defaults.update(overrides)
    return ValidationConfig(**defaults)


def to_core_config(config: ValidationConfig) -> CoreValidationConfig:
    """
    Adapter to convert new ValidationConfig to core ValidationConfig schema.

    Maps fields between the new composite configuration schema and the
    core validation schema expected by validate_terms_functional.

    Threshold Resolution:
    Web-specific thresholds from config.web_config take precedence over
    top-level thresholds to maintain validator-specific configuration.

    Args:
        config: New ValidationConfig with per-validator configs

    Returns:
        CoreValidationConfig compatible with validation/core module
    """
    # Derive show_progress from any of the validator configs
    show_progress = (
        config.rule_config.show_progress or
        config.web_config.show_progress or
        config.llm_config.show_progress
    )

    # Use web config thresholds if present, falling back to top-level
    core_min_score = config.web_config.min_score if "web" in config.modes else config.min_score
    core_min_relevance_score = config.web_config.min_relevance_score if "web" in config.modes else config.min_relevance_score

    return CoreValidationConfig(
        modes=config.modes,
        confidence_weights=config.confidence_weights,
        min_confidence=config.min_confidence,
        min_score=core_min_score,
        min_relevance_score=core_min_relevance_score,
        parallel=config.parallel,
        show_progress=show_progress,
        llm_provider=config.llm_config.provider,
        use_cache=config.use_cache,
        max_workers_rule=config.rule_config.max_workers,
        max_workers_web=config.web_config.max_workers
    )


def with_config(validator_fn: Callable, config: Union[ValidationConfig, RuleValidationConfig, WebValidationConfig, LLMValidationConfig], llm_fn: Optional[Callable] = None) -> Callable:
    """
    Partially apply configuration to a validator function.

    This higher-order function creates a new function with the configuration
    pre-applied, following functional composition patterns.

    Args:
        validator_fn: Validator function to configure
        config: Configuration to apply
        llm_fn: LLM function for LLM validation (optional, will auto-import if None)

    Returns:
        Configured validator function

    Examples:
        >>> from generate_glossary.validation.rule_validator import rule_validate
        >>> rule_config = create_rule_config(max_workers=8)
        >>> fast_rule_validator = with_config(rule_validate, rule_config)
        >>> results = fast_rule_validator(terms)

        >>> from generate_glossary.validation.llm_validator import llm_validate
        >>> from generate_glossary.llm import completion
        >>> llm_config = create_llm_config(provider="gemini")
        >>> llm_validator = with_config(llm_validate, llm_config, llm_fn=completion)
    """
    if isinstance(config, RuleValidationConfig):
        return partial(
            validator_fn,
            max_workers=config.max_workers,
            blacklist_terms=config.blacklist_terms,
            min_term_length=config.min_term_length,
            max_term_length=config.max_term_length,
            show_progress=config.show_progress
        )
    elif isinstance(config, WebValidationConfig):
        return partial(
            validator_fn,
            min_score=config.min_score,
            min_relevance_score=config.min_relevance_score,
            min_relevant_sources=config.min_relevant_sources,
            high_quality_content_threshold=config.high_quality_content_threshold,
            high_quality_relevance_threshold=config.high_quality_relevance_threshold,
            max_workers=config.max_workers,
            show_progress=config.show_progress
        )
    elif isinstance(config, LLMValidationConfig):
        if llm_fn is None:
            try:
                from generate_glossary.llm import completion as llm_fn
            except Exception:
                pass  # keep None, caller must pass llm_fn later
        return partial(
            validator_fn,
            llm_fn=llm_fn,
            provider=config.provider,
            batch_size=config.batch_size,
            max_workers=config.max_workers,
            validation_prompt=config.validation_prompt,
            batch_prompt=config.batch_prompt,
            tier=config.tier,
            max_tokens=config.max_tokens,
            batch_max_tokens=config.batch_max_tokens,
            show_progress=config.show_progress
        )
    elif isinstance(config, ValidationConfig):
        # For ValidationConfig, build per-validator configured callables
        # and implement internal pipeline for maximum control
        from ..rule_validator import rule_validate
        from ..web_validator import web_validate
        from ..llm_validator import llm_validate
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from collections import defaultdict

        # Build configured validator callables
        validator_callables = {}

        if "rule" in config.modes:
            validator_callables["rule"] = with_config(rule_validate, config.rule_config)

        if "web" in config.modes:
            validator_callables["web"] = with_config(web_validate, config.web_config)

        if "llm" in config.modes:
            if llm_fn is None:
                try:
                    from generate_glossary.llm import completion as llm_fn
                except Exception:
                    pass  # Will fail later if LLM mode used without llm_fn
            validator_callables["llm"] = with_config(llm_validate, config.llm_config, llm_fn=llm_fn)

        def internal_pipeline(terms, web_content=None, existing_results=None, cache_state=None):
            """Internal pipeline that executes configured validators."""
            # Normalize terms input
            if isinstance(terms, str):
                terms_list = [terms]
            else:
                terms_list = list(terms)

            if not terms_list:
                return existing_results or {}

            # Execute validators in parallel if requested
            validation_results_by_mode = {}

            if config.parallel and len(validator_callables) > 1:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=min(len(validator_callables), 4)) as executor:
                    future_to_mode = {}

                    for mode, validator_fn in validator_callables.items():
                        if mode == "web" and web_content is not None:
                            future = executor.submit(validator_fn, terms_list, web_content)
                        elif mode != "web":
                            future = executor.submit(validator_fn, terms_list)
                        else:
                            # Skip web validation if no web_content provided
                            continue
                        future_to_mode[future] = mode

                    # Collect results
                    for future in as_completed(future_to_mode):
                        mode = future_to_mode[future]
                        try:
                            validation_results_by_mode[mode] = future.result()
                        except Exception as e:
                            logger = logging.getLogger(__name__)
                            logger.exception(f"Validation failed for mode {mode}: {e}")
                            validation_results_by_mode[mode] = {}
            else:
                # Sequential execution
                for mode, validator_fn in validator_callables.items():
                    try:
                        if mode == "web" and web_content is not None:
                            validation_results_by_mode[mode] = validator_fn(terms_list, web_content)
                        elif mode != "web":
                            validation_results_by_mode[mode] = validator_fn(terms_list)
                        # Skip web validation if no web_content provided
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.exception(f"Validation failed for mode {mode}: {e}")
                        validation_results_by_mode[mode] = {}

            # Combine results using similar logic as core.create_validation_result
            combined_results = {}
            for term in terms_list:
                # Extract per-mode results for this term
                term_results = {}
                for mode, mode_data in validation_results_by_mode.items():
                    if term in mode_data:
                        term_results[mode] = mode_data[term]

                # Calculate combined confidence using weighted approach
                total_weight = 0.0
                weighted_confidence_sum = 0.0
                weighted_score_sum = 0.0

                for mode, result in term_results.items():
                    if mode in config.confidence_weights:
                        weight = config.confidence_weights[mode]
                        confidence = result.get('confidence', 0.0)
                        score = result.get('score', confidence)

                        weighted_confidence_sum += weight * confidence
                        weighted_score_sum += weight * score
                        total_weight += weight

                combined_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0
                combined_score = weighted_score_sum / total_weight if total_weight > 0 else 0.0

                # Extract relevance score (primarily from web results)
                relevance_score = None
                for mode, result in term_results.items():
                    if 'relevance_score' in result:
                        relevance_score = result['relevance_score']
                        break
                    elif mode == 'web':
                        val = result.get('details', {}).get('avg_relevance_score')
                        if val is not None:
                            relevance_score = float(val)
                            break

                # Determine validity based on thresholds
                # Use web config thresholds if web mode active, otherwise top-level
                effective_min_score = config.web_config.min_score if "web" in config.modes else config.min_score
                effective_min_relevance_score = config.web_config.min_relevance_score if "web" in config.modes else config.min_relevance_score

                is_valid = (
                    combined_confidence >= config.min_confidence and
                    combined_score >= effective_min_score and
                    (relevance_score is None or relevance_score >= effective_min_relevance_score)
                )

                combined_results[term] = {
                    'term': term,
                    'is_valid': is_valid,
                    'confidence': combined_confidence,
                    'score': combined_score,
                    'relevance_score': relevance_score,
                    'mode_results': term_results
                }

            # Merge with existing results if provided
            if existing_results:
                final_results = dict(existing_results)
                final_results.update(combined_results)
                return final_results

            return combined_results

        return internal_pipeline
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")


def merge_configs(*configs: ValidationConfig, precedence: str = "last") -> ValidationConfig:
    """
    Merge multiple ValidationConfig objects with specified precedence.

    Args:
        *configs: ValidationConfig objects to merge
        precedence: Merge precedence ("first" or "last")

    Returns:
        Merged ValidationConfig

    Raises:
        ValueError: If no configs provided or invalid precedence
    """
    if not configs:
        raise ValueError("At least one config must be provided")

    if precedence not in ("first", "last"):
        raise ValueError("precedence must be 'first' or 'last'")

    # Start with the base config based on precedence
    base_config = configs[0] if precedence == "first" else configs[-1]
    merge_order = configs[1:] if precedence == "first" else reversed(configs[:-1])

    # Build merged configuration
    merged_data = {
        'modes': base_config.modes,
        'confidence_weights': dict(base_config.confidence_weights),
        'min_confidence': base_config.min_confidence,
        'min_score': base_config.min_score,
        'min_relevance_score': base_config.min_relevance_score,
        'parallel': base_config.parallel,
        'use_cache': base_config.use_cache,
        'rule_config': base_config.rule_config,
        'web_config': base_config.web_config,
        'llm_config': base_config.llm_config
    }

    # Apply overrides from other configs
    for config in merge_order:
        merged_data['modes'] = config.modes
        merged_data['confidence_weights'].update(config.confidence_weights)
        merged_data.update({
            'min_confidence': config.min_confidence,
            'min_score': config.min_score,
            'min_relevance_score': config.min_relevance_score,
            'parallel': config.parallel,
            'use_cache': config.use_cache,
            'rule_config': config.rule_config,
            'web_config': config.web_config,
            'llm_config': config.llm_config
        })

    return ValidationConfig(**merged_data)


def override_config(base_config: ValidationConfig, **overrides) -> ValidationConfig:
    """
    Create new ValidationConfig with field overrides.

    Args:
        base_config: Base configuration to override
        **overrides: Fields to override

    Returns:
        New ValidationConfig with overrides applied

    Examples:
        >>> academic_fast = override_config(ACADEMIC_PROFILE, parallel=False, use_cache=False)
        >>> strict_with_llm = override_config(STRICT_PROFILE, modes=("rule", "web", "llm"))
    """
    config_data = {
        'modes': base_config.modes,
        'confidence_weights': dict(base_config.confidence_weights),
        'min_confidence': base_config.min_confidence,
        'min_score': base_config.min_score,
        'min_relevance_score': base_config.min_relevance_score,
        'parallel': base_config.parallel,
        'use_cache': base_config.use_cache,
        'rule_config': base_config.rule_config,
        'web_config': base_config.web_config,
        'llm_config': base_config.llm_config
    }
    config_data.update(overrides)
    return ValidationConfig(**config_data)


def create_profile(name: str, base_profile: Optional[str] = None, **overrides) -> ValidationConfig:
    """
    Create custom validation profile based on existing profile or defaults.

    Args:
        name: Name for the custom profile (for documentation)
        base_profile: Name of base profile to extend (optional)
        **overrides: Configuration overrides

    Returns:
        New ValidationConfig with custom configuration

    Examples:
        >>> custom_academic = create_profile("custom_academic", "academic", min_confidence=0.8)
        >>> minimal = create_profile("minimal", modes=("rule",), parallel=False)
    """
    if base_profile:
        base_config = get_profile(base_profile)
        return override_config(base_config, **overrides)
    else:
        return create_validation_config(**overrides)


def get_profile_summary(config: ValidationConfig) -> Dict[str, Any]:
    """
    Get a summary of a validation configuration.

    Args:
        config: ValidationConfig to summarize

    Returns:
        Dictionary with configuration summary
    """
    # Determine which thresholds will be used by core validation
    # (matches to_core_config logic for consistency)
    effective_min_score = config.web_config.min_score if "web" in config.modes else config.min_score
    effective_min_relevance_score = config.web_config.min_relevance_score if "web" in config.modes else config.min_relevance_score

    return {
        "modes": list(config.modes),
        "confidence_weights": dict(config.confidence_weights),
        "thresholds": {
            "min_confidence": config.min_confidence,
            "min_score": effective_min_score,
            "min_relevance_score": effective_min_relevance_score,
            "_note": "Effective thresholds shown - web config values override top-level when web mode active"
        },
        "execution": {
            "parallel": config.parallel,
            "use_cache": config.use_cache
        },
        "rule_config": {
            "max_workers": config.rule_config.max_workers,
            "min_term_length": config.rule_config.min_term_length,
            "max_term_length": config.rule_config.max_term_length,
            "blacklist_size": len(config.rule_config.blacklist_terms)
        },
        "web_config": {
            "max_workers": config.web_config.max_workers,
            "min_relevant_sources": config.web_config.min_relevant_sources,
            "min_score": config.web_config.min_score,
            "min_relevance_score": config.web_config.min_relevance_score,
            "quality_thresholds": {
                "content": config.web_config.high_quality_content_threshold,
                "relevance": config.web_config.high_quality_relevance_threshold
            }
        },
        "llm_config": {
            "provider": config.llm_config.provider,
            "batch_size": config.llm_config.batch_size,
            "tier": config.llm_config.tier,
            "max_workers": config.llm_config.max_workers
        }
    }