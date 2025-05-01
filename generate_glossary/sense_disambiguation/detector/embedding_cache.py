"""
Caching functionality for embeddings to avoid redundant calculations.
"""

import os
import hashlib
import logging
from collections import OrderedDict
import numpy as np
from typing import Optional, Dict, Union

# Import the PersistentVectorStore
from ..vector_store import PersistentVectorStore

class EmbeddingCache:
    """A cache for embeddings to avoid redundant calculations."""
    
    def __init__(self, max_size: int = 100000):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to store in cache (default: 100000)
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _get_hash(self, text: str) -> str:
        """Generate a hash for a text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text from cache if available.
        
        Args:
            text: The text to get embedding for
            
        Returns:
            Cached embedding vector or None if not in cache
        """
        text_hash = self._get_hash(text)
        if text_hash in self.cache:
            self.hits += 1
            # Move to end to mark as recently used
            self.cache.move_to_end(text_hash)
            return self.cache[text_hash]
        self.misses += 1
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: The text the embedding is for
            embedding: The embedding vector to cache
        """
        text_hash = self._get_hash(text)
        self.cache[text_hash] = embedding
        self.cache.move_to_end(text_hash)
        
        # Remove oldest items if cache exceeds max size
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

# Global embedding cache shared across detector instances
global_embedding_cache = EmbeddingCache()

# Global persistent vector store
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
vector_store_dir = os.path.join(repo_root, "data", "vector_store")
global_vector_store = PersistentVectorStore(persist_dir=vector_store_dir)
logging.info(f"Initialized global vector store at {vector_store_dir}") 