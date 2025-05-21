#!/usr/bin/env python

import json
import os
import numpy as np
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Set, Optional
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directories (matching the paths used in other modules)
DATA_DIR = 'data'
FINAL_DIR = os.path.join(DATA_DIR, 'final')
CACHE_DIR = os.path.join(DATA_DIR, 'vector_store')
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, 'term_embeddings.pkl')

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDINGS_AVAILABLE = False
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence_transformers package not found. Embedding-based ranking will be disabled.")
    logger.warning("Install it with: pip install sentence-transformers")

# For more advanced vector DB (optional)
VECTOR_DB_AVAILABLE = False
try:
    import faiss
    VECTOR_DB_AVAILABLE = True
    logger.info("FAISS vector database available for faster similarity search")
except ImportError:
    logger.info("FAISS not available. Using standard embedding cache instead.")
    logger.info("Install it with: pip install faiss-cpu")

# Global embedding model
embedding_model = None
# Global embedding cache
embedding_cache = {}
# Embedding metadata for tracking changes
embedding_metadata = {
    "model_name": "all-MiniLM-L6-v2",
    "creation_time": None,
    "term_count": 0,
    "hierarchy_hash": None
}

def get_embedding_model():
    """Load or get the embedding model."""
    global embedding_model
    if embedding_model is None and EMBEDDINGS_AVAILABLE:
        try:
            # Use a smaller, faster model for efficiency
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
    return embedding_model

def compute_hierarchy_hash(hierarchy: Dict[str, Any]) -> str:
    """Compute a simple hash of the hierarchy to detect changes."""
    import hashlib
    # Use term count, relationship count, and a sample of terms as a fingerprint
    term_count = len(hierarchy["terms"])
    relationship_count = len(hierarchy["relationships"]["parent_child"])
    # Sample 10 random terms and their parents (or fewer if there are fewer terms)
    import random
    sample_size = min(10, term_count)
    sampled_terms = random.sample(list(hierarchy["terms"].keys()), sample_size)
    sample_data = []
    for term in sampled_terms:
        sample_data.append(f"{term}:{','.join(hierarchy['terms'][term]['parents'])}")
    
    fingerprint = f"{term_count}:{relationship_count}:{':'.join(sample_data)}"
    return hashlib.md5(fingerprint.encode()).hexdigest()

def save_embeddings_cache():
    """Save the embeddings cache to disk."""
    global embedding_cache, embedding_metadata
    
    if not embedding_cache:
        logger.info("No embeddings to save.")
        return False
    
    try:
        # Update metadata
        embedding_metadata["creation_time"] = time.time()
        embedding_metadata["term_count"] = len(embedding_cache)
        
        # Create a dictionary with both embeddings and metadata
        cache_data = {
            "metadata": embedding_metadata,
            "embeddings": embedding_cache
        }
        
        # Save to file
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved {len(embedding_cache)} term embeddings to {EMBEDDINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving embeddings cache: {e}")
        return False

def load_embeddings_cache(hierarchy: Dict[str, Any] = None) -> bool:
    """Load the embeddings cache from disk."""
    global embedding_cache, embedding_metadata
    
    if not os.path.exists(EMBEDDINGS_FILE):
        logger.info(f"Embeddings cache file not found: {EMBEDDINGS_FILE}")
        return False
    
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if the cache format is valid
        if not isinstance(cache_data, dict) or "embeddings" not in cache_data:
            logger.warning("Invalid cache format. Creating a new cache.")
            embedding_cache = {}
            return False
        
        # Extract embeddings and metadata
        embedding_cache = cache_data.get("embeddings", {})
        stored_metadata = cache_data.get("metadata", {})
        
        # Update our metadata with stored values
        for key, value in stored_metadata.items():
            embedding_metadata[key] = value
        
        # Check if the hierarchy has changed (if provided)
        if hierarchy is not None:
            current_hash = compute_hierarchy_hash(hierarchy)
            stored_hash = embedding_metadata.get("hierarchy_hash")
            
            if stored_hash and stored_hash != current_hash:
                logger.warning("Hierarchy has changed since cache was created. Some embeddings may be outdated.")
        
        logger.info(f"Loaded {len(embedding_cache)} term embeddings from cache")
        return True
    except Exception as e:
        logger.error(f"Error loading embeddings cache: {e}")
        embedding_cache = {}
        return False

def compute_text_embedding(text: str) -> Optional[np.ndarray]:
    """Compute embedding for a text string."""
    if not EMBEDDINGS_AVAILABLE:
        return None
    
    model = get_embedding_model()
    if model is None:
        return None
    
    try:
        # Truncate text if it's too long to avoid memory issues
        max_words = 1000
        if len(text.split()) > max_words:
            text = " ".join(text.split()[:max_words])
        
        # Compute embedding
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error computing embedding: {e}")
        return None

def load_resources_for_level(level: int) -> Dict[str, List[Dict[str, Any]]]:
    """Load resources for a specific level."""
    resources_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
    
    if not os.path.exists(resources_file):
        logger.warning(f"Resources file not found: {resources_file}")
        return {}
    
    try:
        with open(resources_file, 'r', encoding='utf-8') as f:
            resources = json.load(f)
        logger.info(f"Loaded resources for {len(resources)} terms from level {level}")
        return resources
    except Exception as e:
        logger.error(f"Error loading resources file {resources_file}: {e}")
        return {}

def extract_term_content(resources: Dict[str, List[Dict[str, Any]]], term: str) -> str:
    """Extract processed content from resources for a term."""
    if term not in resources:
        return ""
    
    # Concatenate processed content from all resources
    content = []
    for resource in resources[term]:
        if "processed_content" in resource:
            content.append(resource["processed_content"])
    
    return " ".join(content)

def compute_term_embeddings(hierarchy: Dict[str, Any], verbose: bool = False, force_recompute: bool = False) -> Dict[str, Optional[np.ndarray]]:
    """Compute embeddings for terms based on their resources.
    
    Args:
        hierarchy: The hierarchy containing terms
        verbose: Whether to print verbose output
        force_recompute: Whether to force recomputation of all embeddings
        
    Returns:
        Dictionary mapping terms to their embeddings
    """
    global embedding_cache, embedding_metadata
    
    if not EMBEDDINGS_AVAILABLE:
        if verbose:
            logger.info("Embedding computation skipped - sentence_transformers not available")
        return {}
    
    # Try to load existing cache if not forcing recomputation
    if not force_recompute and not embedding_cache:
        load_embeddings_cache(hierarchy)
    
    # Update hierarchy hash
    if hierarchy:
        embedding_metadata["hierarchy_hash"] = compute_hierarchy_hash(hierarchy)
    
    # Set of terms that need embedding computation
    terms_to_compute = set()
    
    # Load resources for all levels
    level_resources = {}
    for level in range(4):  # Levels 0, 1, 2, 3
        level_resources[level] = load_resources_for_level(level)
    
    start_time = time.time()
    compute_count = 0
    cache_hit_count = 0
    
    # Check which terms need computation
    for term, term_data in hierarchy["terms"].items():
        # Skip if already in cache and not forcing recomputation
        if not force_recompute and term in embedding_cache:
            cache_hit_count += 1
            continue
            
        terms_to_compute.add(term)
    
    if verbose:
        logger.info(f"Cache hits: {cache_hit_count}, Terms to compute: {len(terms_to_compute)}")
    
    # Compute embeddings for terms that need it
    for term in terms_to_compute:
        term_data = hierarchy["terms"][term]
        level = term_data["level"]
        
        # Get content for the term
        content = extract_term_content(level_resources.get(level, {}), term)
        
        # If no content, try using sources as fallback
        if not content and term_data["sources"]:
            content = " ".join(term_data["sources"])
        
        # Skip if no content available
        if not content:
            embedding_cache[term] = None
            continue
        
        # Compute embedding
        embedding = compute_text_embedding(content)
        embedding_cache[term] = embedding
        compute_count += 1
        
        # Log progress for large computations
        if verbose and compute_count % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Computed {compute_count}/{len(terms_to_compute)} embeddings ({elapsed:.2f}s elapsed)")
    
    # Save updated cache
    if compute_count > 0:
        save_embeddings_cache()
    
    if verbose:
        computed_count = sum(1 for emb in embedding_cache.values() if emb is not None)
        logger.info(f"Using embeddings for {computed_count}/{len(embedding_cache)} terms "
                   f"({compute_count} newly computed, {cache_hit_count} from cache)")
    
    return embedding_cache

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def score_parent_relationships(hierarchy: Dict[str, Any], verbose: bool = False, force_recompute_embeddings: bool = False) -> Dict[str, Dict[str, float]]:
    """Score parent-child relationships based on various metrics.
    
    Returns a dictionary mapping terms to their parents with scores.
    Higher scores indicate stronger/more appropriate relationships.
    """
    # Initialize parent scores 
    parent_scores = {}
    
    # Count term co-occurrence with parents in sources
    term_parent_cooccurrence = defaultdict(Counter)
    
    # Count parent frequencies across the entire hierarchy
    parent_frequency = Counter()
    
    # Count parent usage per level
    level_parent_counts = defaultdict(Counter)
    
    # Get all parent-child relationships
    relationships = hierarchy["relationships"]["parent_child"]
    
    # Count parent frequency across all terms
    for parent, child, _ in relationships:
        parent_frequency[parent] += 1
        level = hierarchy["terms"][child]["level"]
        level_parent_counts[level][parent] += 1
    
    # Compute term embeddings if available
    term_embeddings = compute_term_embeddings(
        hierarchy, 
        verbose=verbose,
        force_recompute=force_recompute_embeddings
    ) if EMBEDDINGS_AVAILABLE else {}
    
    # Analyze parent-term co-occurrence in sources
    for term, term_data in hierarchy["terms"].items():
        # Skip terms with no parents
        if not term_data["parents"]:
            continue
            
        # Initialize scores for this term
        if term not in parent_scores:
            parent_scores[term] = {}
        
        # Extract terms from sources to detect co-occurrence
        source_text = " ".join(term_data["sources"]).lower()
        
        # Score each parent for this term
        for parent in term_data["parents"]:
            # Start with a base score of 1
            score = 1.0
            
            # Factor 1: Co-occurrence in sources (highest weight)
            if parent.lower() in source_text:
                score += 5.0
            
            # Factor 2: Parent frequency in this level (common parents in a level are generally good)
            parent_in_level_count = level_parent_counts[term_data["level"]][parent]
            # Convert level to string for dictionary access
            level_key = str(term_data["level"])
            if level_key in hierarchy["levels"]:
                level_size = len(hierarchy["levels"][level_key])
                if level_size > 0:
                    level_frequency_score = min(3.0, (parent_in_level_count / level_size) * 10)
                    score += level_frequency_score
            
            # Factor 3: Level gap penalty (parents should ideally be one level above)
            if parent in hierarchy["terms"]:
                parent_level = hierarchy["terms"][parent]["level"]
                level_gap = term_data["level"] - parent_level
                # Ideal: gap of 1 (direct parent-child)
                if level_gap == 1:
                    score += 2.0
                elif level_gap > 1:
                    score -= (level_gap - 1) * 0.5  # Small penalty for larger gaps
            
            # Factor 4: Semantic similarity using embeddings
            if term in term_embeddings and parent in term_embeddings:
                term_embedding = term_embeddings[term]
                parent_embedding = term_embeddings[parent]
                
                if term_embedding is not None and parent_embedding is not None:
                    similarity = cosine_similarity(term_embedding, parent_embedding)
                    # Scale similarity to a reasonable score (0-4)
                    embedding_score = similarity * 4.0
                    score += embedding_score
            
            # Store the final score
            parent_scores[term][parent] = score
    
    if verbose:
        logger.info(f"Scored parent relationships for {len(parent_scores)} terms")
    
    return parent_scores

def rank_parents(hierarchy: Dict[str, Any], verbose: bool = False, force_recompute_embeddings: bool = False) -> Dict[str, List[Tuple[str, float]]]:
    """Rank parents for each term by their score.
    
    Returns a dictionary mapping terms to ordered lists of (parent, score) tuples.
    """
    # Get parent scores
    parent_scores = score_parent_relationships(
        hierarchy, 
        verbose=verbose,
        force_recompute_embeddings=force_recompute_embeddings
    )
    
    # Rank parents by score
    ranked_parents = {}
    for term, scores in parent_scores.items():
        ranked_parents[term] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if verbose:
        logger.info(f"Ranked parents for {len(ranked_parents)} terms")
    
    return ranked_parents

def apply_parent_ranking(hierarchy: Dict[str, Any], 
                        min_score: float = 2.0,
                        max_parents: int = 3, 
                        verbose: bool = False,
                        force_recompute_embeddings: bool = False) -> Dict[str, Any]:
    """Apply parent ranking to the hierarchy.
    
    Creates a new hierarchy with parent relationships filtered based on ranking.
    
    Args:
        hierarchy: The original hierarchy
        min_score: Minimum score for a parent to be included
        max_parents: Maximum number of parents to keep per term
        verbose: Whether to print information about the process
        force_recompute_embeddings: Whether to force recomputation of all embeddings
        
    Returns:
        A new hierarchy with filtered parent relationships
    """
    # Rank parents
    ranked_parents = rank_parents(
        hierarchy, 
        verbose=verbose,
        force_recompute_embeddings=force_recompute_embeddings
    )
    
    # Create a copy of the hierarchy to modify
    filtered_hierarchy = {
        "levels": hierarchy["levels"].copy(),
        "relationships": {
            "parent_child": [],
            "variations": hierarchy["relationships"]["variations"].copy()
        },
        "terms": {},
        "stats": hierarchy["stats"].copy()
    }
    
    # Keep track of removed and kept relationships
    removed_relationships = 0
    kept_relationships = 0
    
    # Update terms with filtered parents
    for term, term_data in hierarchy["terms"].items():
        # Create a copy of the term data
        filtered_term_data = term_data.copy()
        
        # Filter parents if this term has ranked parents
        if term in ranked_parents:
            # Get the top parents that meet the minimum score
            top_parents = [
                parent for parent, score in ranked_parents[term][:max_parents] 
                if score >= min_score
            ]
            
            # Update parents list
            filtered_term_data["parents"] = top_parents
        
        # Add the term to the filtered hierarchy
        filtered_hierarchy["terms"][term] = filtered_term_data
    
    # Rebuild parent-child relationships
    for term, term_data in filtered_hierarchy["terms"].items():
        for parent in term_data["parents"]:
            # Add relationship if parent exists in hierarchy
            if parent in filtered_hierarchy["terms"]:
                level = term_data["level"]
                filtered_hierarchy["relationships"]["parent_child"].append((parent, term, level))
                kept_relationships += 1
                
                # Add child to parent's children list
                if "children" not in filtered_hierarchy["terms"][parent]:
                    filtered_hierarchy["terms"][parent]["children"] = []
                if term not in filtered_hierarchy["terms"][parent]["children"]:
                    filtered_hierarchy["terms"][parent]["children"].append(term)
    
    # Update statistics
    removed_relationships = len(hierarchy["relationships"]["parent_child"]) - kept_relationships
    filtered_hierarchy["stats"]["total_relationships"] = kept_relationships
    
    if verbose:
        logger.info(f"Applied parent ranking:")
        logger.info(f"  - Removed {removed_relationships} relationships")
        logger.info(f"  - Kept {kept_relationships} relationships")
    
    return filtered_hierarchy

def save_ranked_hierarchy(hierarchy: Dict[str, Any], 
                         output_file: str = None,
                         min_score: float = 2.0,
                         max_parents: int = 3,
                         verbose: bool = False,
                         force_recompute_embeddings: bool = False) -> Dict[str, Any]:
    """Apply parent ranking and save the result to a file.
    
    Args:
        hierarchy: The original hierarchy
        output_file: Output file path
        min_score: Minimum score for a parent to be included
        max_parents: Maximum number of parents to keep per term
        verbose: Whether to print information about the process
        force_recompute_embeddings: Whether to force recomputation of all embeddings
        
    Returns:
        The filtered hierarchy
    """
    filtered_hierarchy = apply_parent_ranking(
        hierarchy, 
        min_score=min_score, 
        max_parents=max_parents, 
        verbose=verbose,
        force_recompute_embeddings=force_recompute_embeddings
    )
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_hierarchy, f, indent=2, ensure_ascii=False)
        
        if verbose:
            logger.info(f"Saved ranked hierarchy to {output_file}")
    
    return filtered_hierarchy

def analyze_term_parents(hierarchy: Dict[str, Any], terms_to_analyze: List[str], verbose: bool = False) -> None:
    """Analyze parent rankings for specific terms."""
    # Get parent scores
    parent_scores = score_parent_relationships(hierarchy, verbose=verbose)
    
    # Go through requested terms
    for term in terms_to_analyze:
        if term not in hierarchy["terms"]:
            logger.warning(f"Term '{term}' not found in hierarchy")
            continue
            
        term_data = hierarchy["terms"][term]
        logger.info(f"\nAnalysis for term: '{term}' (level {term_data['level']})")
        logger.info(f"Current parents: {term_data['parents']}")
        
        if term in parent_scores:
            logger.info("Parent scores:")
            for parent, score in sorted(parent_scores[term].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {parent}: {score:.2f}")
        else:
            logger.info("No parent scores computed for this term")

def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rank and filter parent relationships in the hierarchy')
    parser.add_argument('-i', '--input', type=str, default='data/final/hierarchy.json',
                        help='Input hierarchy file path (default: data/final/hierarchy.json)')
    parser.add_argument('-o', '--output', type=str, default='data/final/ranked_hierarchy.json',
                        help='Output hierarchy file path (default: data/final/ranked_hierarchy.json)')
    parser.add_argument('-s', '--min-score', type=float, default=2.0,
                        help='Minimum score for a parent to be included (default: 2.0)')
    parser.add_argument('-p', '--max-parents', type=int, default=3,
                        help='Maximum number of parents to keep per term (default: 3)')
    parser.add_argument('-a', '--analyze', type=str, nargs='*',
                        help='Analyze specific terms (e.g., "data science" "artificial intelligence")')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recomputation of all embeddings')
    parser.add_argument('--cache-info', action='store_true',
                        help='Show information about the embedding cache')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Show cache info if requested
    if args.cache_info:
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                
                metadata = cache_data.get("metadata", {})
                embeddings = cache_data.get("embeddings", {})
                
                logger.info(f"Embedding cache information:")
                logger.info(f"  - Cache file: {EMBEDDINGS_FILE}")
                logger.info(f"  - Model name: {metadata.get('model_name', 'unknown')}")
                logger.info(f"  - Creation time: {time.ctime(metadata.get('creation_time', 0))}")
                logger.info(f"  - Term count: {len(embeddings)}")
                
                # Count terms with valid embeddings
                valid_count = sum(1 for emb in embeddings.values() if emb is not None)
                logger.info(f"  - Terms with valid embeddings: {valid_count}")
                
                # Show embedding dimensions if available
                for term, embedding in embeddings.items():
                    if embedding is not None:
                        logger.info(f"  - Embedding dimensions: {embedding.shape}")
                        break
                
                if args.force_recompute:
                    logger.info("Will force recomputation of all embeddings.")
            except Exception as e:
                logger.error(f"Error reading cache file: {e}")
        else:
            logger.info(f"No embedding cache file found at {EMBEDDINGS_FILE}")
        
        if not args.analyze and args.input == 'data/final/hierarchy.json' and args.output == 'data/final/ranked_hierarchy.json':
            return
    
    # Load hierarchy
    with open(args.input, 'r', encoding='utf-8') as f:
        hierarchy = json.load(f)
    
    if args.verbose:
        logger.info(f"Loaded hierarchy from {args.input}")
    
    # Analyze specific terms if requested
    if args.analyze:
        analyze_term_parents(hierarchy, args.analyze, args.verbose)
        return
    
    # Apply parent ranking and save the result
    save_ranked_hierarchy(
        hierarchy, 
        args.output, 
        min_score=args.min_score, 
        max_parents=args.max_parents, 
        verbose=args.verbose,
        force_recompute_embeddings=args.force_recompute
    )
    
    if args.verbose:
        logger.info("Done!")

if __name__ == '__main__':
    main() 