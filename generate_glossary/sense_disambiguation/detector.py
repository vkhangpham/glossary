"""
Legacy module that re-exports detector classes from the new sub-modules.

This file is maintained for backward compatibility and forwards imports
to the individual detector modules.
"""

# Re-export all detector classes from their respective modules
from generate_glossary.sense_disambiguation.detector.embedding_cache import EmbeddingCache, global_embedding_cache, global_vector_store
from generate_glossary.sense_disambiguation.detector.parent_context import ParentContextDetector
from generate_glossary.sense_disambiguation.detector.resource_cluster import ResourceClusterDetector, HDBSCAN_AVAILABLE
from generate_glossary.sense_disambiguation.detector.radial_polysemy import RadialPolysemyDetector
from generate_glossary.sense_disambiguation.detector.hybrid import HybridAmbiguityDetector
from generate_glossary.sense_disambiguation.detector.utils import convert_numpy_types

# For backward compatibility
__all__ = [
    "EmbeddingCache",
    "global_embedding_cache",
    "global_vector_store",
    "ParentContextDetector",
    "ResourceClusterDetector",
    "HDBSCAN_AVAILABLE",
    "HybridAmbiguityDetector",
    "RadialPolysemyDetector",
    "convert_numpy_types"
]