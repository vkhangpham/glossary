"""
Immutable data structures for the disambiguation system.

This module provides comprehensive type definitions for detection results,
configuration objects, and split proposals used throughout the disambiguation pipeline.
All data structures are immutable (@frozen dataclasses) to support functional programming.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass(frozen=True)
class DetectionResult:
    """Immutable result structure for ambiguity detection methods."""

    term: str
    method: str  # "embedding", "hierarchy", "global"
    confidence: float  # 0.0-1.0
    evidence: Dict[str, Any]
    clusters: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitProposal:
    """Structure for sense splitting proposals."""

    original_term: str
    level: int  # hierarchy level (0-3)
    proposed_senses: List[Dict[str, Any]]
    confidence: float
    evidence: Dict[str, Any]
    validation_status: Optional[str] = None  # "approved", "rejected", "pending"


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


@dataclass(frozen=True)
class GlobalConfig:
    """Configuration for global clustering-based disambiguation detection."""

    model_name: str = "all-MiniLM-L6-v2"
    eps: float = 0.3
    min_samples: int = 3
    min_resources: int = 5
    max_resources_per_term: int = 10


@dataclass(frozen=True)
class DisambiguationConfig:
    """Main configuration for the disambiguation system."""

    methods: Tuple[str, ...] = ("embedding", "hierarchy", "global")
    min_confidence: float = 0.5
    level_configs: Dict[int, LevelConfig] = field(default_factory=dict)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hierarchy_config: HierarchyConfig = field(default_factory=HierarchyConfig)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    parallel_processing: bool = True
    use_cache: bool = True