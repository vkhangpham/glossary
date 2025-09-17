"""
Configuration module for the disambiguation system.

This module provides access to configuration classes and predefined profiles
for the disambiguation system.

## Configuration Design Philosophy

The disambiguation system uses method-specific configuration objects rather than
flat configuration fields. This design provides several benefits:

- **Type Safety**: Each method's parameters are grouped in their respective config objects
- **Modularity**: Configuration can be easily extended without affecting other methods
- **Encapsulation**: Related parameters are kept together (e.g., EmbeddingConfig contains
  model_name, clustering_algorithm, eps, min_samples, min_resources)
- **Backward Compatibility**: New parameters can be added to method configs without
  breaking existing code

### Configuration Structure

- `DisambiguationConfig`: Main configuration containing method-specific configs
- `EmbeddingConfig`: Settings for embedding-based detection
- `HierarchyConfig`: Settings for hierarchy-based detection
- `GlobalConfig`: Settings for global clustering detection
- `LevelConfig`: Level-specific parameters for different hierarchy levels

This structure differs from a flat configuration with top-level fields like
`embedding_model`, `clustering_algorithm`, etc., which would create a less
maintainable and type-safe design.
"""

from ..types import (
    DisambiguationConfig,
    EmbeddingConfig,
    HierarchyConfig,
    GlobalConfig,
    LevelConfig,
    DetectionResult,
    SplitProposal
)

from .profiles import (
    ACADEMIC_PROFILE,
    FAST_PROFILE,
    COMPREHENSIVE_PROFILE,
    CONSERVATIVE_PROFILE,
    get_profile,
    list_profiles,
    create_custom_profile,
    merge_profiles,
    create_level_configs
)

__all__ = [
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
    "create_level_configs"
]