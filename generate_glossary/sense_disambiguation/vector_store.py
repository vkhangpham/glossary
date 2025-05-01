import faiss
import numpy as np
import pickle
import os
from pathlib import Path
import datetime
import hashlib
import logging
from typing import Dict, Any, Optional, List, Union

class PersistentVectorStore:
    """A persistent vector store that saves embeddings to disk using FAISS."""
    
    def __init__(self, persist_dir: str = "./vector_store"):
        """
        Initialize the persistent vector store. The dimension is inferred from the first added embedding.
        
        Args:
            persist_dir: Directory to store the index and metadata
        """
        self.dimension = None # Will be inferred later
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        
        self.index_path = self.persist_dir / "embeddings.faiss"
        self.metadata_path = self.persist_dir / "metadata.pickle"
        
        # Maps from hash to metadata (includes original text and any auxiliary data)
        self.metadata = {}
        
        # Track cache statistics
        self.hits = 0
        self.misses = 0
        
        self.index = None # Index will be created on first put/put_batch or loaded
        
        # Load existing index if available
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.dimension = self.index.d # Set dimension from loaded index
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logging.info(f"Loaded vector store with {self.index.ntotal} embeddings (dim={self.dimension}) from {self.persist_dir}")
            except Exception as e:
                logging.warning(f"Error loading existing vector store from {self.persist_dir}: {e}. A new store will be created on first write.")
                # Reset index and metadata in case of partial load failure
                self.index = None
                self.metadata = {}
        else:
            logging.info(f"No existing vector store found at {self.persist_dir}. It will be created on first write.")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text from store if available.
        
        Args:
            text: The text to get embedding for
            
        Returns:
            Embedding vector or None if not in store
        """
        text_hash = self._get_hash(text)
        if text_hash in self.metadata:
            self.hits += 1
            # Retrieve vector by ID
            idx = self.metadata[text_hash]["idx"]
            vector = self._get_vector_by_idx(idx)
            return vector
        self.misses += 1
        return None
    
    def _get_hash(self, text: str) -> str:
        """Generate a hash for a text string."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_vector_by_idx(self, idx: int) -> np.ndarray:
        """Retrieve a vector by its index in the FAISS store."""
        if self.index is None or self.dimension is None:
            logging.warning("_get_vector_by_idx called before index was initialized.")
            return None
            
        # Create a single-vector array to receive the reconstructed vector
        vector = np.zeros((1, self.dimension), dtype=np.float32)
        
        # Reconstruct the vector from the index
        if hasattr(self.index, 'reconstruct'):
            try:
                self.index.reconstruct(int(idx), vector[0])
                return vector[0]
            except RuntimeError as e:
                logging.error(f"Failed to reconstruct vector at index {idx}: {e}")
                return None
        else:
            # For index types that don't support direct reconstruction
            logging.warning("Index type does not support vector reconstruction")
            return None
    
    def put(self, text: str, embedding: np.ndarray, auxiliary_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Store embedding in vector store.
        
        Args:
            text: The text the embedding is for
            embedding: The embedding vector to store
            auxiliary_data: Optional additional metadata
        """
        # Check if index exists, create if not
        if self.index is None:
            self.dimension = embedding.shape[0]
            self.index = faiss.IndexFlatL2(self.dimension)
            logging.info(f"Created new FAISS index with dimension {self.dimension}")
            
        # Validate embedding dimension
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}")
            
        text_hash = self._get_hash(text)
        
        # If already exists, just update metadata
        if text_hash in self.metadata:
            if auxiliary_data:
                self.metadata[text_hash].update(auxiliary_data)
            return
        
        # Add vector to index
        embedding_reshaped = np.array([embedding]).astype('float32')
        logging.debug(f"[PersistentVectorStore] Adding vector with shape {embedding_reshaped.shape} to index with dimension {self.index.d}")
        idx = self.index.ntotal  # Get current count before adding
        self.index.add(embedding_reshaped)
        
        # Store metadata
        self.metadata[text_hash] = {
            "text": text,
            "idx": idx,
            "added_at": datetime.datetime.now().isoformat()
        }
        
        if auxiliary_data:
            self.metadata[text_hash].update(auxiliary_data)
    
    def put_batch(self, texts: List[str], embeddings: np.ndarray, auxiliary_data_list: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Store multiple embeddings in vector store.
        
        Args:
            texts: List of texts
            embeddings: Array of embedding vectors
            auxiliary_data_list: Optional list of additional metadata
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of embeddings ({len(embeddings)})" )
        
        if not texts: # Handle empty batch
            return
        
        # Check if index exists, create if not using the first embedding in the batch
        if self.index is None:
            self.dimension = embeddings[0].shape[0]
            self.index = faiss.IndexFlatL2(self.dimension)
            logging.info(f"Created new FAISS index with dimension {self.dimension}")
            
        # Validate embedding dimensions for the batch
        if embeddings.shape[1] != self.dimension:
             raise ValueError(f"Embedding dimension mismatch in batch: expected {self.dimension}, got {embeddings.shape[1]}")
        
        if auxiliary_data_list and len(texts) != len(auxiliary_data_list):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of auxiliary data items ({len(auxiliary_data_list)})" )
        
        # Collect vectors to add (only those not already in store)
        new_vectors = []
        new_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_hash(text)
            if text_hash not in self.metadata:
                new_vectors.append(embeddings[i])
                new_indices.append(i)
        
        # Add new vectors in batch
        if new_vectors:
            vectors_array = np.array(new_vectors).astype('float32')
            starting_idx = self.index.ntotal
            logging.info(f"[PersistentVectorStore] Adding {len(new_vectors)} new vectors to the index (current total: {starting_idx})...")
            self.index.add(vectors_array)
            logging.info(f"[PersistentVectorStore] Finished adding vectors. Index total: {self.index.ntotal}")
            
            # Update metadata for new vectors
            for j, i in enumerate(new_indices):
                text = texts[i]
                text_hash = self._get_hash(text)
                
                self.metadata[text_hash] = {
                    "text": text,
                    "idx": starting_idx + j,
                    "added_at": datetime.datetime.now().isoformat()
                }
                
                if auxiliary_data_list:
                    self.metadata[text_hash].update(auxiliary_data_list[i])
    
    def save(self) -> None:
        """Save the index and metadata to disk."""
        if self.index is None or self.index.ntotal == 0:
            logging.info("[PersistentVectorStore] No index data to save.")
            return
            
        try:
            logging.info(f"[PersistentVectorStore] Saving FAISS index with {self.index.ntotal} vectors to {self.index_path}...")
            faiss.write_index(self.index, str(self.index_path))
            logging.info(f"[PersistentVectorStore] FAISS index saved.")
            
            logging.info(f"[PersistentVectorStore] Saving metadata ({len(self.metadata)} entries) to {self.metadata_path}...")
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logging.info(f"[PersistentVectorStore] Metadata saved.")
            
            logging.info(f"Saved vector store with {self.index.ntotal} vectors to {self.persist_dir}")
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The vector to search with
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata)
        """
        if self.index is None:
            logging.warning("Search called before index was initialized or populated.")
            return np.array([]), np.array([]), []
            
        # Ensure vector is in correct shape and type
        query_vector = np.array([query_vector]).astype('float32')
        
        # Validate query dimension
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: index is {self.dimension}, query is {query_vector.shape[1]}")
        
        # Perform search
        k_actual = min(k, self.index.ntotal) # Ensure k is not larger than the number of items in index
        if k_actual <= 0:
            return np.array([]), np.array([]), []
        
        distances, indices = self.index.search(query_vector, k_actual)
        
        # Collect metadata for results
        metadata_results = []
        for idx in indices[0]:
            # Find metadata entry with this index
            meta = None
            for hash_key, data in self.metadata.items():
                if data.get('idx') == idx:
                    meta = data
                    break
            metadata_results.append(meta)
        
        return distances[0], indices[0], metadata_results
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the vector store."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "size_mb": os.path.getsize(self.index_path) / (1024 * 1024) if self.index_path.exists() else 0,
            "metadata_size_mb": os.path.getsize(self.metadata_path) / (1024 * 1024) if self.metadata_path.exists() else 0,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = {}
        self.hits = 0
        self.misses = 0
        # Remove files if they exist
        if self.index_path.exists():
            os.remove(self.index_path)
        if self.metadata_path.exists():
            os.remove(self.metadata_path) 