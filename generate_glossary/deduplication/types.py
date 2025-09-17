"""
Immutable data structures for the deduplication system.

This module provides frozen dataclasses for type-safe configuration and edge representation,
establishing a foundation for functional programming in the deduplication pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, FrozenSet, Union, Mapping, List, TypedDict
from types import MappingProxyType
# frozenset is a built-in type


class WebResource(TypedDict):
    """Type definition for individual web resource data."""
    url: str
    title: Optional[str]
    content: Optional[str]
    relevance_score: Optional[float]
    domain: Optional[str]


class TermWebContent(TypedDict):
    """Type definition for term's web content structure."""
    results: List[WebResource]
    metadata: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class Edge:
    """Immutable representation of a deduplication edge."""

    source: str                                    # Source term
    target: str                                    # Target term
    weight: float                                  # Similarity/confidence score (0.0-1.0)
    edge_type: str                                # Edge type (see EDGE_TYPES)
    method: str                                   # Creation method (see METHODS)
    metadata: Mapping[str, Any] = field(default_factory=dict)  # Additional edge-specific data

    def __post_init__(self):
        """Validate edge attributes."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")

        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"Invalid edge_type: {self.edge_type}. Must be one of {EDGE_TYPES}")

        if self.method not in METHODS:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {METHODS}")
        
        # Make metadata immutable
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

@dataclass(frozen=True)
class EdgeBatch:
    """Container for edge creation results with both successful edges and errors."""
    
    edges: List[Edge]           # Successfully created edges
    errors: List[str]           # Error messages from failed edge creation attempts


@dataclass(frozen=True)
class RuleConfig:
    """Configuration for rule-based deduplication."""

    min_similarity: float = 0.85                          # Minimum text similarity threshold
    synonym_patterns: Dict[str, str] = field(default_factory=dict)  # Synonym mapping patterns
    blacklist_terms: FrozenSet[str] = frozenset()         # Terms to exclude
    enable_acronym_detection: bool = True                  # Enable acronym edge creation
    enable_synonym_detection: bool = True                  # Enable synonym edge creation

    def __post_init__(self):
        """Validate rule configuration."""
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be between 0.0 and 1.0, got {self.min_similarity}")

def create_default_domain_patterns() -> Dict[str, float]:
    """Create default domain patterns for web configuration."""
    return {
        "edu": 1.2,
        "org": 1.1,
        "gov": 1.1,
        "ieee": 1.3,
        "acm": 1.3,
        "arxiv": 1.2,
        "scholar": 1.2
    }


@dataclass(frozen=True)
class WebConfig:
    """Configuration for web-based deduplication."""

    min_url_overlap: int = 2                               # Minimum overlapping URLs
    min_relevance_score: float = 0.3                      # Minimum relevance threshold
    domain_patterns: Dict[str, float] = field(default_factory=create_default_domain_patterns)  # Domain weight patterns
    min_content_similarity: float = 0.6                   # Content similarity threshold
    enable_domain_specific: bool = True                    # Enable domain-specific edges
    enable_content_similarity: bool = True                 # Enable content similarity edges

    def __post_init__(self):
        """Validate web configuration."""
        if self.min_url_overlap < 1:
            raise ValueError(f"min_url_overlap must be >= 1, got {self.min_url_overlap}")

        if not 0.0 <= self.min_relevance_score <= 1.0:
            raise ValueError(f"min_relevance_score must be between 0.0 and 1.0, got {self.min_relevance_score}")

        if not 0.0 <= self.min_content_similarity <= 1.0:
            raise ValueError(f"min_content_similarity must be between 0.0 and 1.0, got {self.min_content_similarity}")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM-based deduplication."""

    provider: str = "gemini"                               # LLM provider
    batch_size: int = 5                                    # Terms per LLM call
    confidence_threshold: float = 0.8                      # Minimum confidence for edge creation
    min_url_overlap: int = 1                               # Minimum URL overlap to consider
    max_url_overlap: int = 2                               # Maximum URL overlap (below web threshold)
    level: Optional[int] = None                            # Hierarchy level for prompt optimization
    use_case: Optional[str] = None                         # Specific use case for prompt selection
    temperature: float = 0.3                               # LLM temperature setting

    def __post_init__(self):
        """Validate LLM configuration."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")

        if self.min_url_overlap < 0:
            raise ValueError(f"min_url_overlap must be >= 0, got {self.min_url_overlap}")

        if self.max_url_overlap < self.min_url_overlap:
            raise ValueError(f"max_url_overlap must be >= min_url_overlap, got {self.max_url_overlap} < {self.min_url_overlap}")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")

        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Invalid provider: {self.provider}. Must be one of {SUPPORTED_PROVIDERS}")


@dataclass(frozen=True)
class DeduplicationConfig:
    """Main configuration for the deduplication system."""

    rule_config: RuleConfig = field(default_factory=RuleConfig)      # Rule-based configuration
    web_config: WebConfig = field(default_factory=WebConfig)         # Web-based configuration
    llm_config: Optional[LLMConfig] = None                           # Optional LLM configuration
    remove_weak_edges: bool = True                                   # Whether to remove weak edges
    weak_edge_threshold: float = 0.3                                 # Threshold for weak edge removal
    parallel_processing: bool = True                                 # Enable parallel edge creation

    def __post_init__(self):
        """Validate deduplication configuration."""
        if not 0.0 <= self.weak_edge_threshold <= 1.0:
            raise ValueError(f"weak_edge_threshold must be between 0.0 and 1.0, got {self.weak_edge_threshold}")


# Constants for validation
EDGE_TYPES = {
    "text_similarity",
    "compound_term",
    "acronym",
    "synonym",
    "web_overlap",
    "domain_specific",
    "content_similarity",
    "llm_web_analysis"
}

METHODS = {
    "rule_based",
    "web_based",
    "llm_based"
}

SUPPORTED_PROVIDERS = {
    "gemini",
    "openai",
    "claude"
}


def create_default_synonym_patterns() -> Dict[str, str]:
    """Create default synonym patterns for rule configuration."""
    return {
        r"\bML\b": "machine learning",
        r"\bAI\b": "artificial intelligence",
        r"\bNLP\b": "natural language processing",
        r"\bCV\b": "computer vision",
        r"\bDL\b": "deep learning"
    }