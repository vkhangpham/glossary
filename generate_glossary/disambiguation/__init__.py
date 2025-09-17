"""
Term disambiguation module for academic glossary generation.

This module provides a comprehensive functional API for identifying and resolving
ambiguous terms in academic glossaries using semantic clustering, hierarchy analysis,
and multi-signal validation.

## Functional API (Recommended)

The new functional API provides immutable data structures, pure functions with
dependency injection, error isolation, and comprehensive configuration:

```python
from generate_glossary.disambiguation import (
    # Core functional pipeline
    create_detection_pipeline,
    parallel_detect,
    combine_detection_results,
    get_detection_summary,

    # Configuration system
    DisambiguationConfig,
    ACADEMIC_PROFILE,
    get_profile,

    # Pure detection functions
    detect_embedding_ambiguity,
    detect_hierarchy_ambiguity,
    detect_global_ambiguity,

    # Pure splitting functions
    generate_split_proposals,
    validate_split_proposals,
    apply_splits_to_hierarchy,

    # Result types and helpers
    Success,
    Failure,
    is_success,
    get_value
)

# Create detection pipeline with configuration
results = create_detection_pipeline(
    terms=["machine learning", "AI", "neural networks"],
    web_content=content,
    hierarchy=hierarchy,
    config=ACADEMIC_PROFILE
)

summary = get_detection_summary(results)
```

## Legacy API (Deprecated)

The legacy API is maintained for backward compatibility but will be removed
in a future version. Use the functional API for new code.

## Migration Guide

- `detect_ambiguous()` → `create_detection_pipeline()` + `parallel_detect()`
- `split_senses()` → `generate_split_proposals()` + `validate_split_proposals()`
- Configuration moved to `DisambiguationConfig` and predefined profiles
- Results now use `Success`/`Failure` types for better error handling
"""

import warnings
from typing import Dict, List, Any, Optional

# Legacy API imports (with deprecation path)
from .api import (
    disambiguate_terms as _disambiguate_terms_impl,
    detect_ambiguous as _detect_ambiguous_impl,
    split_senses as _split_senses_impl
)

# Functional core imports
from .core import (
    # Main orchestration functions
    create_detection_pipeline,
    parallel_detect,
    sequential_detect,
    combine_detection_results,

    # Safety and utility functions
    safe_detect,
    filter_ambiguous_terms,
    get_detection_summary,

    # Result types and helpers
    Success,
    Failure,
    Result,
    is_success,
    is_failure,
    get_value,
    get_error,

    # Functional composition utilities
    compose_detectors,
    with_timeout,
    with_retry,

    # Pipeline builders
    create_embedding_pipeline,
    create_hierarchy_pipeline,
    create_global_pipeline,
    create_hybrid_pipeline,

    # Context management
    detection_context
)

# Configuration system imports
from .config import (
    # Configuration classes
    DisambiguationConfig,
    EmbeddingConfig,
    HierarchyConfig,
    GlobalConfig,
    LevelConfig,
    DetectionResult,
    SplitProposal,

    # Predefined profiles
    ACADEMIC_PROFILE,
    FAST_PROFILE,
    COMPREHENSIVE_PROFILE,
    CONSERVATIVE_PROFILE,

    # Profile management functions
    get_profile,
    list_profiles,
    create_custom_profile,
    merge_profiles,
    create_level_configs
)

# Detection modules and pure functions
from . import detectors
from .detectors.embedding import (
    detect_embedding_ambiguity,
    detect as embedding_detect,
    with_embedding_model,
    create_embedding_model
)
from .detectors.hierarchy import (
    detect_hierarchy_ambiguity,
    detect as hierarchy_detect
)
from .detectors.global_clustering import (
    detect_global_ambiguity,
    detect as global_detect,
    with_global_embedding_model
)

# Splitting functions with LLM injection
from .splitting.generator import (
    generate_split_proposals,
    generate_splits,
    create_tag_generator,
    with_llm_function
)
from .splitting.validator import (
    validate_split_proposals,
    validate_splits,
    create_llm_validator
)
from .splitting.applicator import (
    apply_splits_to_hierarchy,
    apply_to_hierarchy
)


# Legacy compatibility functions with deprecation warnings
def disambiguate_terms(*args, **kwargs) -> Dict[str, Any]:
    """
    Legacy function for running complete disambiguation pipeline.

    DEPRECATED: Use create_detection_pipeline(), generate_split_proposals(),
    validate_split_proposals(), or main.run_disambiguation_pipeline() instead.

    The new functional API provides better error handling, configuration,
    and composability. See module documentation for migration examples.
    """
    warnings.warn(
        "disambiguate_terms() is deprecated. Use create_detection_pipeline(), "
        "generate_split_proposals(), validate_split_proposals(), or "
        "main.run_disambiguation_pipeline() from the functional API instead. "
        "See module documentation for migration examples.",
        DeprecationWarning,
        stacklevel=2
    )
    return _disambiguate_terms_impl(*args, **kwargs)


def detect_ambiguous(*args, **kwargs) -> Dict[str, Any]:
    """
    Legacy function for detecting ambiguous terms.

    DEPRECATED: Use create_detection_pipeline() + parallel_detect() instead.

    The new functional API provides better error handling, configuration,
    and composability. See module documentation for migration examples.
    """
    warnings.warn(
        "detect_ambiguous() is deprecated. Use create_detection_pipeline() + "
        "parallel_detect() from the functional API instead. "
        "See module documentation for migration examples.",
        DeprecationWarning,
        stacklevel=2
    )
    return _detect_ambiguous_impl(*args, **kwargs)


def split_senses(*args, **kwargs) -> Dict[str, Any]:
    """
    Legacy function for splitting ambiguous term senses.

    DEPRECATED: Use generate_split_proposals() + validate_split_proposals() instead.

    The new functional API provides better error handling, immutable data structures,
    and dependency injection. See module documentation for migration examples.
    """
    warnings.warn(
        "split_senses() is deprecated. Use generate_split_proposals() + "
        "validate_split_proposals() from the functional API instead. "
        "See module documentation for migration examples.",
        DeprecationWarning,
        stacklevel=2
    )
    return _split_senses_impl(*args, **kwargs)


__all__ = [
    # Legacy API (deprecated)
    "disambiguate_terms",
    "detect_ambiguous",
    "split_senses",

    # Functional core - main orchestration
    "create_detection_pipeline",
    "parallel_detect",
    "sequential_detect",
    "combine_detection_results",

    # Functional core - safety and utilities
    "safe_detect",
    "filter_ambiguous_terms",
    "get_detection_summary",

    # Result types and helpers
    "Success",
    "Failure",
    "Result",
    "is_success",
    "is_failure",
    "get_value",
    "get_error",

    # Functional composition utilities
    "compose_detectors",
    "with_timeout",
    "with_retry",

    # Pipeline builders
    "create_embedding_pipeline",
    "create_hierarchy_pipeline",
    "create_global_pipeline",
    "create_hybrid_pipeline",

    # Context management
    "detection_context",

    # Configuration classes
    "DisambiguationConfig",
    "EmbeddingConfig",
    "HierarchyConfig",
    "GlobalConfig",
    "LevelConfig",
    "DetectionResult",
    "SplitProposal",

    # Predefined profiles
    "ACADEMIC_PROFILE",
    "FAST_PROFILE",
    "COMPREHENSIVE_PROFILE",
    "CONSERVATIVE_PROFILE",

    # Profile management functions
    "get_profile",
    "list_profiles",
    "create_custom_profile",
    "merge_profiles",
    "create_level_configs",

    # Detection modules
    "detectors",

    # Pure detection functions
    "detect_embedding_ambiguity",
    "detect_hierarchy_ambiguity",
    "detect_global_ambiguity",

    # Detection functions with aliases
    "embedding_detect",
    "hierarchy_detect",
    "global_detect",

    # Dependency injection for detection
    "with_embedding_model",
    "create_embedding_model",
    "with_global_embedding_model",

    # Pure splitting functions
    "generate_split_proposals",
    "validate_split_proposals",
    "apply_splits_to_hierarchy",

    # LLM injection for splitting
    "create_tag_generator",
    "with_llm_function",
    "create_llm_validator",

    # Legacy splitting functions (deprecated)
    "generate_splits",
    "validate_splits",
    "apply_to_hierarchy"
]