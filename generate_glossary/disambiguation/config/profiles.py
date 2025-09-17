"""
Predefined disambiguation profiles and configuration utilities.

This module provides standard configuration profiles for different disambiguation
use cases and utility functions for profile management.
"""

from typing import Dict, List, Any, Mapping
import types
from ..types import (
    DisambiguationConfig,
    EmbeddingConfig,
    HierarchyConfig,
    GlobalConfig,
    LevelConfig
)
from ..utils import LEVEL_PARAMS


def create_level_configs() -> Mapping[int, LevelConfig]:
    """Create immutable LevelConfig objects from LEVEL_PARAMS."""
    level_configs_dict = {
        level: LevelConfig(
            eps=params["eps"],
            min_samples=params["min_samples"],
            description=params["description"],
            separation_threshold=params["separation_threshold"],
            examples=params["examples"]
        )
        for level, params in LEVEL_PARAMS.items()
    }
    return types.MappingProxyType(level_configs_dict)


# Predefined Profiles
#
# Note: These profiles use method-specific configuration objects (EmbeddingConfig,
# HierarchyConfig, GlobalConfig) rather than flat fields on DisambiguationConfig.
# This design provides better type safety and modularity.

ACADEMIC_PROFILE = DisambiguationConfig(
    methods=("embedding", "hierarchy", "global"),
    min_confidence=0.5,
    level_configs=create_level_configs(),
    embedding_config=EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        clustering_algorithm="dbscan",
        eps=0.45,
        min_samples=2,
        min_resources=5
    ),
    hierarchy_config=HierarchyConfig(
        min_parent_overlap=0.3,
        max_parent_similarity=0.7,
        enable_web_enhancement=True
    ),
    global_config=GlobalConfig(
        model_name="all-MiniLM-L6-v2",
        eps=0.3,
        min_samples=3,
        min_resources=5,
        max_resources_per_term=10
    ),
    parallel_processing=True,
    use_cache=True
)


FAST_PROFILE = DisambiguationConfig(
    methods=("embedding",),
    min_confidence=0.4,
    level_configs=create_level_configs(),
    embedding_config=EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        clustering_algorithm="dbscan",
        eps=0.5,
        min_samples=2,
        min_resources=3
    ),
    hierarchy_config=HierarchyConfig(),
    global_config=GlobalConfig(),
    parallel_processing=True,
    use_cache=True
)


COMPREHENSIVE_PROFILE = DisambiguationConfig(
    methods=("embedding", "hierarchy", "global"),
    min_confidence=0.6,
    level_configs=create_level_configs(),
    embedding_config=EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        clustering_algorithm="dbscan",
        eps=0.4,
        min_samples=3,
        min_resources=8
    ),
    hierarchy_config=HierarchyConfig(
        min_parent_overlap=0.2,
        max_parent_similarity=0.8,
        enable_web_enhancement=True
    ),
    global_config=GlobalConfig(
        model_name="all-MiniLM-L6-v2",
        eps=0.25,
        min_samples=4,
        min_resources=8,
        max_resources_per_term=15
    ),
    parallel_processing=True,
    use_cache=True
)


CONSERVATIVE_PROFILE = DisambiguationConfig(
    methods=("embedding", "hierarchy", "global"),
    min_confidence=0.7,
    level_configs=create_level_configs(),
    embedding_config=EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        clustering_algorithm="dbscan",
        eps=0.3,
        min_samples=4,
        min_resources=8
    ),
    hierarchy_config=HierarchyConfig(
        min_parent_overlap=0.4,
        max_parent_similarity=0.6,
        enable_web_enhancement=True
    ),
    global_config=GlobalConfig(
        model_name="all-MiniLM-L6-v2",
        eps=0.2,
        min_samples=5,
        min_resources=8,
        max_resources_per_term=8
    ),
    parallel_processing=True,
    use_cache=True
)


# Profile Registry
PROFILES = {
    "academic": ACADEMIC_PROFILE,
    "fast": FAST_PROFILE,
    "comprehensive": COMPREHENSIVE_PROFILE,
    "conservative": CONSERVATIVE_PROFILE
}


def get_profile(name: str) -> DisambiguationConfig:
    """
    Retrieve a predefined profile by name.

    Args:
        name: Profile name ("academic", "fast", "comprehensive", "conservative")

    Returns:
        DisambiguationConfig object for the specified profile

    Raises:
        KeyError: If the profile name is not found
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(f"Profile '{name}' not found. Available profiles: {available}")

    return PROFILES[name]


def list_profiles() -> List[str]:
    """
    List available profile names.

    Returns:
        List of available profile names
    """
    return list(PROFILES.keys())


def create_custom_profile(**overrides) -> DisambiguationConfig:
    """
    Create a custom profile based on the academic profile with overrides.

    Args:
        **overrides: Configuration values to override in the base profile

    Returns:
        DisambiguationConfig with specified overrides
    """
    # Start with academic profile as base
    base_config = ACADEMIC_PROFILE

    # Create new config with overrides
    config_dict = {
        "methods": overrides.get("methods", base_config.methods),
        "min_confidence": overrides.get("min_confidence", base_config.min_confidence),
        "level_configs": overrides.get("level_configs", base_config.level_configs),
        "embedding_config": overrides.get("embedding_config", base_config.embedding_config),
        "hierarchy_config": overrides.get("hierarchy_config", base_config.hierarchy_config),
        "global_config": overrides.get("global_config", base_config.global_config),
        "parallel_processing": overrides.get("parallel_processing", base_config.parallel_processing),
        "use_cache": overrides.get("use_cache", base_config.use_cache)
    }

    return DisambiguationConfig(**config_dict)


def merge_profiles(base_profile: DisambiguationConfig, **overrides) -> DisambiguationConfig:
    """
    Merge a base profile with configuration overrides.

    Args:
        base_profile: Base DisambiguationConfig to start from
        **overrides: Configuration values to override

    Returns:
        New DisambiguationConfig with merged configuration
    """
    config_dict = {
        "methods": overrides.get("methods", base_profile.methods),
        "min_confidence": overrides.get("min_confidence", base_profile.min_confidence),
        "level_configs": overrides.get("level_configs", base_profile.level_configs),
        "embedding_config": overrides.get("embedding_config", base_profile.embedding_config),
        "hierarchy_config": overrides.get("hierarchy_config", base_profile.hierarchy_config),
        "global_config": overrides.get("global_config", base_profile.global_config),
        "parallel_processing": overrides.get("parallel_processing", base_profile.parallel_processing),
        "use_cache": overrides.get("use_cache", base_profile.use_cache)
    }

    return DisambiguationConfig(**config_dict)