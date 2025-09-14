"""
Functional validation configuration system.

This module provides a comprehensive configuration system for validation
that supports immutable configurations, predefined profiles, and functional
composition patterns.

Public API:
- Configuration classes: ValidationConfig, FunctionalValidationConfig (alias), RuleValidationConfig, WebValidationConfig, LLMValidationConfig
- Predefined profiles: STRICT_PROFILE, ACADEMIC_PROFILE, TECHNICAL_PROFILE, etc.
- Factory functions: create_rule_config, create_web_config, create_llm_config, create_validation_config
- Profile functions: get_profile, list_profiles, create_profile, get_profile_summary
- Higher-order functions: with_config, merge_configs, override_config

Note: FunctionalValidationConfig is an alias for ValidationConfig to avoid naming
collision with validation.core.ValidationConfig. New code should prefer the alias
for clarity when importing from both modules.
"""

from typing import Dict, Any

__author__ = "Glossary Generation System"
__compatibility__ = "Python 3.8+"

# Core configuration classes
from .defaults import (
    ValidationConfig,
    RuleValidationConfig,
    WebValidationConfig,
    LLMValidationConfig,
    DEFAULT_BLACKLIST_TERMS,
    DEFAULT_VALIDATION_PROMPT,
    BATCH_VALIDATION_PROMPT
)

# Export alias to avoid naming collision with core.ValidationConfig
FunctionalValidationConfig = ValidationConfig

# Predefined profiles
from .profiles import (
    STRICT_PROFILE,
    PERMISSIVE_PROFILE,
    ACADEMIC_PROFILE,
    TECHNICAL_PROFILE,
    FAST_PROFILE,
    COMPREHENSIVE_PROFILE,
    PROFILE_REGISTRY
)

# Profile management functions
from .profiles import (
    get_profile,
    list_profiles,
    get_profile_summary
)

# Configuration factory functions
from .profiles import (
    create_rule_config,
    create_web_config,
    create_llm_config,
    create_validation_config,
    create_profile
)

# Higher-order configuration functions
from .profiles import (
    with_config,
    merge_configs,
    override_config
)

# Public API exports
__all__ = [
    # Metadata
    "__author__",
    "__compatibility__",

    # Core configuration classes
    "ValidationConfig",
    "FunctionalValidationConfig",  # Alias to avoid naming collision
    "RuleValidationConfig",
    "WebValidationConfig",
    "LLMValidationConfig",

    # Constants
    "DEFAULT_BLACKLIST_TERMS",
    "DEFAULT_VALIDATION_PROMPT",
    "BATCH_VALIDATION_PROMPT",

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
    "override_config"
]


def get_default_config() -> ValidationConfig:
    """
    Get the default validation configuration.

    This is equivalent to ACADEMIC_PROFILE, which provides a balanced
    configuration suitable for most academic term validation tasks.

    Returns:
        Default ValidationConfig
    """
    return ACADEMIC_PROFILE


def get_recommended_profile(use_case: str) -> ValidationConfig:
    """
    Get recommended profile for a specific use case.

    Args:
        use_case: Use case description
            - "speed": Fast processing, minimal validation
            - "quality": High quality, comprehensive validation
            - "academic": Academic term validation (default)
            - "technical": Technical term validation
            - "strict": Maximum validation rigor
            - "permissive": Minimal validation constraints

    Returns:
        ValidationConfig for the use case

    Raises:
        ValueError: If use case is not recognized
    """
    use_case_mapping = {
        "speed": FAST_PROFILE,
        "quality": COMPREHENSIVE_PROFILE,
        "academic": ACADEMIC_PROFILE,
        "technical": TECHNICAL_PROFILE,
        "strict": STRICT_PROFILE,
        "permissive": PERMISSIVE_PROFILE
    }

    if use_case not in use_case_mapping:
        available = ', '.join(use_case_mapping.keys())
        raise ValueError(f"Unknown use case '{use_case}'. Available: {available}")

    return use_case_mapping[use_case]


def validate_config_compatibility(config: ValidationConfig) -> Dict[str, Any]:
    """
    Validate configuration compatibility and provide warnings.

    Args:
        config: ValidationConfig to validate

    Returns:
        Dictionary with validation results and warnings
    """
    warnings = []
    errors = []

    # Check mode compatibility
    if "web" in config.modes and config.web_config.min_relevant_sources > 5:
        warnings.append("High min_relevant_sources may cause many terms to fail validation")

    if "llm" in config.modes and config.llm_config.batch_size > 50:
        warnings.append("Large LLM batch_size may hit API rate limits")

    if len(config.modes) > 1 and not config.parallel:
        warnings.append("Sequential processing with multiple modes will be slow")

    # Check threshold compatibility
    if config.min_confidence > 0.9:
        warnings.append("Very high min_confidence may reject valid terms")

    if config.min_score < 0.2:
        warnings.append("Very low min_score may accept invalid terms")

    # Check weight consistency
    total_weight = sum(config.confidence_weights[mode] for mode in config.modes)
    if total_weight < 0.5:
        warnings.append("Low total confidence weights may underweight validation results")
    elif total_weight > 1.5:
        warnings.append("High total confidence weights may overweight validation results")

    # Check resource allocation
    total_workers = 0
    if "rule" in config.modes:
        total_workers += config.rule_config.max_workers
    if "web" in config.modes:
        total_workers += config.web_config.max_workers
    if "llm" in config.modes:
        total_workers += config.llm_config.max_workers

    if total_workers > 20:
        warnings.append("High total max_workers may overwhelm system resources")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_workers": total_workers,
        "estimated_speed": "fast" if len(config.modes) == 1 and config.parallel else "moderate" if config.parallel else "slow"
    }