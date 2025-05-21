"""
Detector sub-modules for sense disambiguation.

This package contains various detector implementations for identifying
potentially ambiguous terms in the academic glossary.
"""

# Check if these modules contain the embedding cache
try:
    # Look for EmbeddingCache in resource_cluster.py
    from .resource_cluster import EmbeddingCache, global_embedding_cache
except (ImportError, AttributeError):
    # Define empty placeholders if not found
    class EmbeddingCache:
        pass
    global_embedding_cache = None

from .parent_context import ParentContextDetector
from .resource_cluster import ResourceClusterDetector, HDBSCAN_AVAILABLE
from .radial_polysemy import RadialPolysemyDetector
from .hybrid import HybridAmbiguityDetector
from .utils import convert_numpy_types

__all__ = [
    "EmbeddingCache",
    "global_embedding_cache",
    "ParentContextDetector",
    "ResourceClusterDetector",
    "HDBSCAN_AVAILABLE",
    "HybridAmbiguityDetector",
    "RadialPolysemyDetector",
    "convert_numpy_types"
] 