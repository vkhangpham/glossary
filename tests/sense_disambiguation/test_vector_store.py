#!/usr/bin/env python3
"""
Test script for the PersistentVectorStore implementation.
This script demonstrates the basic functionality of the vector store
and verifies that it's working correctly.
"""

import os
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from generate_glossary.sense_disambiguation.vector_store import PersistentVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_vector_store():
    """Run a simple test of the vector store functionality."""
    # Create a test directory
    test_dir = Path("./test_vector_store")
    test_dir.mkdir(exist_ok=True)
    logging.info(f"Created test directory at {test_dir}")
    
    try:
        # Initialize the vector store
        logging.info("Initializing vector store...")
        vector_store = PersistentVectorStore(persist_dir=str(test_dir))
        
        # Load a sentence transformer model
        logging.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample texts for testing
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Vector embeddings represent text in high-dimensional space.",
            "FAISS is a library for efficient similarity search.",
            "The quick brown fox jumps over the lazy dog."  # Duplicate to test caching
        ]
        
        # Embed and store the texts
        logging.info("Embedding and storing texts...")
        for text in test_texts:
            # Check if already in store
            embedding = vector_store.get(text)
            if embedding is None:
                # Not in store, encode and store it
                embedding = model.encode(text)
                vector_store.put(text, embedding, {"length": len(text)})
                logging.info(f"Added: '{text[:30]}...' ({len(text)} chars)")
            else:
                logging.info(f"Found in store: '{text[:30]}...'")
        
        # Save to disk
        logging.info("Saving vector store to disk...")
        vector_store.save()
        
        # Verify stats
        store_stats = vector_store.get_stats()
        logging.info(f"Vector store stats: {store_stats}")
        
        # Try search functionality
        logging.info("Testing search functionality...")
        query = "Embedding vectors for machine learning"
        query_embedding = model.encode(query)
        
        distances, indices, metadata = vector_store.search(query_embedding, k=3)
        
        logging.info("Search results for: '%s'", query)
        for i, (dist, meta) in enumerate(zip(distances, metadata)):
            if meta:
                logging.info(f"Result {i+1}: '{meta['text'][:50]}...' (Distance: {dist:.4f})")
        
        # Reinitialize from saved data
        logging.info("Reinitializing vector store from disk...")
        new_store = PersistentVectorStore(persist_dir=str(test_dir))
        
        # Verify loaded correctly
        new_stats = new_store.get_stats()
        logging.info(f"Reloaded vector store stats: {new_stats}")
        
        assert new_stats["total_vectors"] == store_stats["total_vectors"], "Vector count doesn't match after reload!"
        
        # Test a retrieval from loaded store
        test_text = test_texts[1]
        embedding = new_store.get(test_text)
        assert embedding is not None, f"Failed to retrieve embedding for '{test_text}'"
        logging.info(f"Successfully retrieved embedding for '{test_text[:30]}...'")
        
        logging.info("All tests passed successfully!")
        
    finally:
        # Clean up (uncomment to preserve test directory)
        import shutil
        shutil.rmtree(test_dir)
        logging.info(f"Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    test_vector_store() 