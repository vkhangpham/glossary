"""
Configuration module for the disambiguation system.

This module provides access to configuration classes and predefined profiles
for the disambiguation system.
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