"""
Immutable data structures for the disambiguation system.

This module provides comprehensive type definitions for detection results,
configuration objects, and split proposals used throughout the disambiguation pipeline.
All data structures are immutable (@frozen dataclasses) to support functional programming.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Mapping, Callable
import types


@dataclass(frozen=True)
class DetectionResult:
    """Immutable result structure for ambiguity detection methods."""

    term: str
    level: int
    method: str  # "embedding", "hierarchy", "global"
    confidence: float  # 0.0-1.0
    evidence: Mapping[str, Any]
    clusters: Optional[Tuple[Mapping[str, Any], ...]] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: types.MappingProxyType({}))

    def __post_init__(self):
        # Convert evidence to immutable mapping if it's a dict
        if isinstance(self.evidence, dict):
            object.__setattr__(self, 'evidence', types.MappingProxyType(self.evidence))

        # Convert clusters to immutable tuple of mappings if needed
        if self.clusters is not None and isinstance(self.clusters, list):
            immutable_clusters = tuple(
                types.MappingProxyType(cluster) if isinstance(cluster, dict) else cluster
                for cluster in self.clusters
            )
            object.__setattr__(self, 'clusters', immutable_clusters)

        # Convert metadata to immutable mapping if it's a dict
        if isinstance(self.metadata, dict):
            object.__setattr__(self, 'metadata', types.MappingProxyType(self.metadata))


@dataclass(frozen=True)
class SplitProposal:
    """Structure for sense splitting proposals."""

    original_term: str
    level: int  # hierarchy level (0-3)
    proposed_senses: Tuple[Mapping[str, Any], ...]
    confidence: float
    evidence: Mapping[str, Any]
    validation_status: Optional[str] = None  # "approved", "rejected", "pending"

    def __post_init__(self):
        # Convert proposed_senses to immutable tuple of mappings if it's a list
        if isinstance(self.proposed_senses, list):
            immutable_senses = tuple(
                types.MappingProxyType(sense) if isinstance(sense, dict) else sense
                for sense in self.proposed_senses
            )
            object.__setattr__(self, 'proposed_senses', immutable_senses)

        # Convert evidence to immutable mapping if it's a dict
        if isinstance(self.evidence, dict):
            object.__setattr__(self, 'evidence', types.MappingProxyType(self.evidence))


@dataclass(frozen=True)
class LevelConfig:
    """Level-specific configuration parameters."""

    eps: float
    min_samples: int
    description: str
    separation_threshold: float
    examples: str


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding-based disambiguation detection."""

    model_name: str = "all-MiniLM-L6-v2"
    clustering_algorithm: str = "dbscan"
    eps: float = 0.45
    min_samples: int = 2
    min_resources: int = 5


@dataclass(frozen=True)
class HierarchyConfig:
    """Configuration for hierarchy-based disambiguation detection."""

    min_parent_overlap: float = 0.3
    max_parent_similarity: float = 0.7
    enable_web_enhancement: bool = True
    max_web_resources_for_keywords: int = 5


@dataclass(frozen=True)
class GlobalConfig:
    """Configuration for global clustering-based disambiguation detection."""

    model_name: str = "all-MiniLM-L6-v2"
    eps: float = 0.3
    min_samples: int = 3
    min_resources: int = 5
    max_resources_per_term: int = 10
    min_total_resources: int = 50


@dataclass(frozen=True)
class DisambiguationConfig:
    """Main configuration for the disambiguation system.

    Note: Method-specific settings are encapsulated in EmbeddingConfig,
    HierarchyConfig, and GlobalConfig objects rather than as top-level fields.
    This design provides better modularity and type safety.
    """

    methods: Tuple[str, ...] = ("embedding", "hierarchy", "global")
    min_confidence: float = 0.5
    level_configs: Mapping[int, LevelConfig] = field(default_factory=lambda: types.MappingProxyType({}))
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hierarchy_config: HierarchyConfig = field(default_factory=HierarchyConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    parallel_processing: bool = True
    use_cache: bool = True

    def __post_init__(self):
        # Convert level_configs to immutable mapping if it's a dict
        if isinstance(self.level_configs, dict):
            object.__setattr__(self, 'level_configs', types.MappingProxyType(self.level_configs))


@dataclass(frozen=True)
class SplittingConfig:
    """Configuration for sense splitting operations."""

    use_llm: bool = True
    llm_provider: str = "gemini"
    min_cluster_size: int = 2
    min_separation_score: float = 0.5
    max_sample_resources: int = 3
    create_backup: bool = True
    tag_generation_max_tokens: int = 20
    validation_max_tokens: int = 100


# Type aliases for splitting functions
LLMFunction = Callable[[List[Dict[str, Any]]], str]
TagGeneratorFunction = Callable[[str, List[Dict], int], str]
ValidationFunction = Callable[[str, List[Dict], int], Tuple[bool, str]]
