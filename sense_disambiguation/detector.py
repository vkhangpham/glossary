"""
Legacy module that re-exports detector classes from the new sub-modules.

This file is maintained for backward compatibility and forwards imports
to the individual detector modules.
"""

# Define __all__ first to specify what will be exported
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

# Use lazy imports to avoid loading heavy dependencies when just showing help
import sys as _sys
import importlib as _importlib

# Define placeholder variables
EmbeddingCache = None
global_embedding_cache = None
global_vector_store = None
ParentContextDetector = None
ResourceClusterDetector = None
HDBSCAN_AVAILABLE = False
HybridAmbiguityDetector = None
RadialPolysemyDetector = None
convert_numpy_types = None

def __getattr__(name):
    """
    Lazily import modules only when their attributes are accessed.
    This avoids importing heavy dependencies when just showing help text.
    """
    if name in __all__:
        # Import the relevant modules only when needed
        if name in ["EmbeddingCache", "global_embedding_cache", "global_vector_store"]:
            try:
                module = _importlib.import_module("sense_disambiguation.detector.embedding_cache")
                globals()["EmbeddingCache"] = module.EmbeddingCache
                globals()["global_embedding_cache"] = module.global_embedding_cache
                globals()["global_vector_store"] = module.global_vector_store
            except ImportError:
                # Try to import from resource_cluster instead
                module = _importlib.import_module("sense_disambiguation.detector.resource_cluster")
                globals()["EmbeddingCache"] = module.EmbeddingCache
                globals()["global_embedding_cache"] = module.global_embedding_cache
                globals()["global_vector_store"] = module.global_vector_store
        
        elif name == "ParentContextDetector":
            module = _importlib.import_module("sense_disambiguation.detector.parent_context")
            globals()["ParentContextDetector"] = module.ParentContextDetector
        
        elif name in ["ResourceClusterDetector", "HDBSCAN_AVAILABLE"]:
            module = _importlib.import_module("sense_disambiguation.detector.resource_cluster")
            globals()["ResourceClusterDetector"] = module.ResourceClusterDetector
            globals()["HDBSCAN_AVAILABLE"] = module.HDBSCAN_AVAILABLE
        
        elif name == "RadialPolysemyDetector":
            module = _importlib.import_module("sense_disambiguation.detector.radial_polysemy")
            globals()["RadialPolysemyDetector"] = module.RadialPolysemyDetector
        
        elif name == "HybridAmbiguityDetector":
            module = _importlib.import_module("sense_disambiguation.detector.hybrid")
            globals()["HybridAmbiguityDetector"] = module.HybridAmbiguityDetector
        
        elif name == "convert_numpy_types":
            module = _importlib.import_module("sense_disambiguation.detector.utils")
            globals()["convert_numpy_types"] = module.convert_numpy_types
        
        return globals()[name]
    
    raise AttributeError(f"module {__name__} has no attribute {name}")