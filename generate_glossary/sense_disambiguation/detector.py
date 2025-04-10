import json
import glob
import os
from collections import defaultdict, OrderedDict
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, List, Any, Tuple, Literal, Union
import datetime
import re
import hashlib

# Import the PersistentVectorStore
from .vector_store import PersistentVectorStore

# Import HDBSCAN if available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN package not found. Only DBSCAN clustering will be available. "
                  "To use HDBSCAN, install with: pip install hdbscan")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
vector_store_dir = os.path.join(repo_root, "data", "vector_store")
global_vector_store = PersistentVectorStore(persist_dir=vector_store_dir)
logging.info(f"Initialized global vector store at {vector_store_dir}")

class ParentContextDetector:
    """Detects potentially ambiguous terms by analyzing parent contexts in the hierarchy."""

    def __init__(self, hierarchy_file_path: str, final_term_files_pattern: str, level: Optional[int] = None):
        """Initializes the detector.

        Args:
            hierarchy_file_path: Path to the hierarchy.json file.
            final_term_files_pattern: Glob pattern for lv*_final.txt files.
            level: Optional hierarchy level (0-3) for targeted analysis.
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.level = level
        self.hierarchy_data = None
        self.canonical_terms = set()
        self.term_details = None
        self._ancestor_cache = {}
        self._loaded = False # Flag to prevent multiple loads

    def _load_data(self):
        """Loads the hierarchy data and canonical term lists."""
        if self._loaded:
            return True

        # Load hierarchy
        logging.info(f"[ParentContextDetector] Loading hierarchy from {self.hierarchy_file_path}...")
        try:
            with open(self.hierarchy_file_path, 'r') as f:
                self.hierarchy_data = json.load(f)
            self.term_details = self.hierarchy_data.get('terms', {})
            if not self.term_details:
                logging.warning("[ParentContextDetector] Hierarchy file loaded, but 'terms' dictionary is missing or empty.")
                return False
            logging.info(f"[ParentContextDetector] Loaded {len(self.term_details)} terms from hierarchy.")
        except FileNotFoundError:
            logging.error(f"[ParentContextDetector] Hierarchy file not found: {self.hierarchy_file_path}")
            return False
        except json.JSONDecodeError:
            logging.error(f"[ParentContextDetector] Error decoding JSON from {self.hierarchy_file_path}")
            return False

        # Load canonical terms
        logging.info(f"[ParentContextDetector] Loading canonical terms from pattern: {self.final_term_files_pattern}")
        final_term_files = glob.glob(self.final_term_files_pattern)
        if not final_term_files:
            logging.warning(f"[ParentContextDetector] No files found matching pattern: {self.final_term_files_pattern}")

        for file_path in final_term_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            self.canonical_terms.add(term)
            except FileNotFoundError:
                logging.warning(f"[ParentContextDetector] Final term file not found during glob: {file_path}")
            except Exception as e:
                logging.error(f"[ParentContextDetector] Error reading final term file {file_path}: {e}")

        logging.info(f"[ParentContextDetector] Loaded {len(self.canonical_terms)} unique canonical terms.")
        if not self.canonical_terms:
             logging.warning("[ParentContextDetector] No canonical terms were loaded. Ambiguity detection based on canonical terms will not yield results.")

        self._loaded = True
        return True

    def _get_ancestors(self, term: str, visited: set) -> tuple[str | None, str | None]:
        """Finds the Level 0 and Level 1 ancestors for a given term.

        Uses caching and prevents infinite loops.

        Args:
            term: The term string to find ancestors for.
            visited: A set of terms already visited in the current traversal path.

        Returns:
            A tuple (l0_ancestor, l1_ancestor). Returns (None, None) if not found or error.
        """
        if term in self._ancestor_cache:
            return self._ancestor_cache[term]

        if term not in self.term_details:
            # logging.debug(f"Term '{term}' not found in hierarchy details.")
            return None, None

        if term in visited:
            logging.warning(f"[ParentContextDetector] Cycle detected involving term '{term}'. Stopping traversal.")
            return None, None

        visited.add(term)

        term_data = self.term_details[term]
        level = term_data.get('level')
        parents = term_data.get('parents', [])

        l0_ancestor = None
        l1_ancestor = None

        if level == 0:
            l0_ancestor = term
        elif level == 1:
            l1_ancestor = term
            # Try to find L0 parent
            for parent in parents:
                parent_l0, _ = self._get_ancestors(parent, visited.copy())
                if parent_l0:
                    l0_ancestor = parent_l0
                    break # Assume first L0 parent found is the one

        elif level is not None and level > 1:
            # Look through parents
            for parent in parents:
                parent_l0, parent_l1 = self._get_ancestors(parent, visited.copy())
                if parent_l0 and not l0_ancestor:
                    l0_ancestor = parent_l0
                if parent_l1 and not l1_ancestor:
                    l1_ancestor = parent_l1
                # If we found both, we can potentially stop early
                if l0_ancestor and l1_ancestor:
                    break

        visited.remove(term)
        self._ancestor_cache[term] = (l0_ancestor, l1_ancestor)
        return l0_ancestor, l1_ancestor

    def detect_ambiguous_terms(self) -> list[str]:
        """Performs the ambiguity detection based on parent contexts.

        Returns:
            A list of canonical term strings identified as potentially ambiguous.
        """
        if not self._load_data():
            logging.error("[ParentContextDetector] Failed to load necessary data. Aborting detection.")
            return []

        if not self.term_details or not self.canonical_terms:
            logging.warning("[ParentContextDetector] Missing term details or canonical terms. Cannot perform detection.")
            return []

        # Track all ambiguous terms before level filtering
        all_ambiguous_terms = []
        self._ancestor_cache = {} # Reset cache for each run

        level_info = f" for level {self.level}" if self.level is not None else ""
        logging.info(f"[ParentContextDetector] Starting ambiguity detection{level_info}...")
        processed_count = 0
        for term, term_data in self.term_details.items():
            processed_count += 1
            if processed_count % 5000 == 0: # Log less frequently maybe
                logging.info(f"[ParentContextDetector] Processed {processed_count}/{len(self.term_details)} terms...")

            # 1. Check if canonical
            if term not in self.canonical_terms:
                continue

            # 2. Check for multiple parents
            parents = term_data.get('parents', [])
            if len(parents) <= 1:
                continue

            # 3. Collect parent contexts
            parent_contexts = set()
            for parent_term in parents:
                # Pass an empty set for 'visited' for each top-level parent lookup
                l0_anc, l1_anc = self._get_ancestors(parent_term, set())
                # We only care about contexts where at least L0 or L1 is found
                if l0_anc or l1_anc:
                    parent_contexts.add((l0_anc, l1_anc))

            # 4. Analyze contexts
            if len(parent_contexts) <= 1:
                continue # All parents map to the same context or couldn't be traced

            unique_l0 = {ctx[0] for ctx in parent_contexts if ctx[0] is not None}
            # Consider L1 differences only if L0 is consistent (or absent but L1 exists)
            unique_l1 = {ctx[1] for ctx in parent_contexts if ctx[1] is not None}

            is_ambiguous = False
            if len(unique_l0) > 1:
                is_ambiguous = True
                logging.debug(f"[ParentContextDetector] Term '{term}' flagged: multiple L0 ancestors {unique_l0}")
            elif len(unique_l0) <= 1 and len(unique_l1) > 1:
                 # Check if the L0s are actually the same or if one context lacked an L0
                 # If all contexts have the same single L0 (or all lack L0), then L1 diff matters
                 l0_values = [ctx[0] for ctx in parent_contexts]
                 # Check if all non-None L0 values are the same
                 non_none_l0 = [l0 for l0 in l0_values if l0 is not None]
                 if len(set(non_none_l0)) <= 1:
                     is_ambiguous = True
                     logging.debug(f"[ParentContextDetector] Term '{term}' flagged: single/no L0, multiple L1 ancestors {unique_l1}")

            if is_ambiguous:
                all_ambiguous_terms.append(term)

        # Now filter by level if needed
        ambiguous_terms = []
        if self.level is not None:
            # Filter to include only terms from the specified level
            for term in all_ambiguous_terms:
                term_level = self.term_details.get(term, {}).get('level')
                if term_level == self.level:
                    ambiguous_terms.append(term)
            
            logging.info(f"[ParentContextDetector] Found {len(all_ambiguous_terms)} potentially ambiguous terms across all levels")
            logging.info(f"[ParentContextDetector] After level filtering, {len(ambiguous_terms)} terms remain at level {self.level}")
        else:
            # If no level specified, use all ambiguous terms
            ambiguous_terms = all_ambiguous_terms

        level_info = f" for level {self.level}" if self.level is not None else ""
        logging.info(f"[ParentContextDetector] Ambiguity detection complete{level_info}. Found {len(ambiguous_terms)} potentially ambiguous terms.")
        return ambiguous_terms

class ResourceClusterDetector:
    """Detects potentially ambiguous terms by clustering their associated resource content."""

    def __init__(self, hierarchy_file_path: str, final_term_files_pattern: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 min_resources: int = 5,
                 clustering_algorithm: Literal['dbscan', 'hdbscan'] = 'dbscan',
                 dbscan_eps: float = 0.4, dbscan_min_samples: int = 2,
                 hdbscan_min_cluster_size: int = 2, hdbscan_min_samples: int = 2,
                 level: Optional[int] = None,  # Used only for filtering results, not parameter adjustment
                 use_embedding_cache: bool = True,
                 output_dir: Optional[str] = None):
        """Initializes the detector.

        Args:
            hierarchy_file_path: Path to the hierarchy.json file.
            final_term_files_pattern: Glob pattern for lv*_final.txt files.
            model_name: Name of the sentence-transformer model to use.
            min_resources: Minimum number of resource snippets required for a term to be analyzed.
            clustering_algorithm: Which clustering algorithm to use ('dbscan' or 'hdbscan').
                HDBSCAN is often better at finding clusters of varying density but requires
                the hdbscan package to be installed.
            dbscan_eps: The maximum distance between two samples for one to be considered as
                in the neighborhood of the other (DBSCAN parameter).
            dbscan_min_samples: The number of samples in a neighborhood for a point
                to be considered as a core point (DBSCAN parameter).
            hdbscan_min_cluster_size: The minimum size of clusters (HDBSCAN parameter).
            hdbscan_min_samples: Similar to dbscan_min_samples but for HDBSCAN.
            level: Optional hierarchy level (0-3) for filtering results. 
                   Does not affect clustering parameters.
            use_embedding_cache: Whether to use the global embedding cache (default: True)
            output_dir: Directory to save results. If None, will use a default directory.
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.model_name = model_name
        self.min_resources = min_resources
        self.level = level
        self.use_embedding_cache = use_embedding_cache
        
        # Validate clustering algorithm
        if clustering_algorithm not in ['dbscan', 'hdbscan']:
            logging.warning(f"Unknown clustering algorithm '{clustering_algorithm}'. Defaulting to 'dbscan'.")
            self.clustering_algorithm = 'dbscan'
        else:
            self.clustering_algorithm = clustering_algorithm
            
        # Check if HDBSCAN is requested but not available
        if self.clustering_algorithm == 'hdbscan' and not HDBSCAN_AVAILABLE:
            logging.warning("HDBSCAN requested but not available. Falling back to DBSCAN.")
            self.clustering_algorithm = 'dbscan'
        
        # Use fixed settings for clustering parameters regardless of level
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        
        # Log clustering parameters
        logging.info(f"Using {self.clustering_algorithm.upper()} clustering parameters:")
        if self.clustering_algorithm == 'dbscan':
            logging.info(f"  eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples}")
        else:
            logging.info(f"  min_cluster_size={self.hdbscan_min_cluster_size}, min_samples={self.hdbscan_min_samples}")
        
        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.output_dir = os.path.join(repo_root, "data", "ambiguity_detection_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.hierarchy_data = None
        self.canonical_terms = set()
        self.term_details = None
        self._loaded = False
        
        # Storage for detailed results
        self.cluster_results = {}  # Maps term -> list of cluster labels for each resource
        self.cluster_metrics = {}  # Maps term -> metrics about its clustering
        
        # Cache for all results before level filtering - used for reusing calculations
        self._cached_all_ambiguous_terms = None
        self._cached_all_cluster_results = None
        self._cached_all_cluster_metrics = None
        self._clustering_complete = False
        
        try:
            logging.info(f"Loading sentence transformer model '{model_name}'...")
            self.embedding_model = SentenceTransformer(model_name)
            logging.info(f"Model '{model_name}' loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model '{model_name}': {e}")
            logging.error("Please ensure 'sentence-transformers' and its dependencies (like torch or tensorflow) are installed.")
            logging.error("You can install with: pip install -U sentence-transformers")
            self.embedding_model = None

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings, using cache when possible.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Array of embedding vectors
        """
        if not self.use_embedding_cache:
            # Skip cache entirely if disabled
            return self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Use cache and vector store for each text
        embeddings = []
        texts_to_encode = []
        indices_to_update = []
        
        # Check caches for each text
        for i, text in enumerate(texts):
            # First check memory cache for fastest access
            cached_embedding = global_embedding_cache.get(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                continue
                
            # Then check persistent vector store
            stored_embedding = global_vector_store.get(text)
            if stored_embedding is not None:
                # If found in vector store, also add to memory cache for faster access next time
                global_embedding_cache.put(text, stored_embedding)
                embeddings.append(stored_embedding)
                continue
                
            # Not found in either cache
            texts_to_encode.append(text)
            indices_to_update.append(i)
        
        # If some texts weren't in cache, encode them
        if texts_to_encode:
            new_embeddings = self.embedding_model.encode(texts_to_encode, show_progress_bar=False)
            
            # Store in both caches
            for j, text in enumerate(texts_to_encode):
                global_embedding_cache.put(text, new_embeddings[j])
                global_vector_store.put(text, new_embeddings[j])
            
            # Merge with cached embeddings
            all_embeddings = np.zeros((len(texts), new_embeddings.shape[1]), dtype=new_embeddings.dtype)
            for i, embedding in enumerate(embeddings):
                all_embeddings[i] = embedding
            for j, idx in enumerate(indices_to_update):
                all_embeddings[idx] = new_embeddings[j]
                
            # Periodically save vector store to disk (every 100 new texts)
            if len(texts_to_encode) >= 100:
                global_vector_store.save()
                
            return all_embeddings
        else:
            # All embeddings were cached
            return np.array(embeddings)

    def _load_data(self):
        """Loads the hierarchy data and canonical term lists."""
        # This is identical to ParentContextDetector._load_data
        # Consider refactoring into a shared function or base class if complexity grows
        if self._loaded:
            return True

        logging.info(f"[ResourceClusterDetector] Loading hierarchy from {self.hierarchy_file_path}...")
        try:
            with open(self.hierarchy_file_path, 'r') as f:
                self.hierarchy_data = json.load(f)
            self.term_details = self.hierarchy_data.get('terms', {})
            if not self.term_details:
                logging.warning("[ResourceClusterDetector] Hierarchy file loaded, but 'terms' dictionary is missing or empty.")
                return False
            logging.info(f"[ResourceClusterDetector] Loaded {len(self.term_details)} terms from hierarchy.")
        except FileNotFoundError:
            logging.error(f"[ResourceClusterDetector] Hierarchy file not found: {self.hierarchy_file_path}")
            return False
        except json.JSONDecodeError:
            logging.error(f"[ResourceClusterDetector] Error decoding JSON from {self.hierarchy_file_path}")
            return False

        logging.info(f"[ResourceClusterDetector] Loading canonical terms from pattern: {self.final_term_files_pattern}")
        final_term_files = glob.glob(self.final_term_files_pattern)
        if not final_term_files:
            logging.warning(f"[ResourceClusterDetector] No files found matching pattern: {self.final_term_files_pattern}")

        for file_path in final_term_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            self.canonical_terms.add(term)
            except FileNotFoundError:
                logging.warning(f"[ResourceClusterDetector] Final term file not found during glob: {file_path}")
            except Exception as e:
                logging.error(f"[ResourceClusterDetector] Error reading final term file {file_path}: {e}")

        logging.info(f"[ResourceClusterDetector] Loaded {len(self.canonical_terms)} unique canonical terms.")
        if not self.canonical_terms:
             logging.warning("[ResourceClusterDetector] No canonical terms were loaded. Detection might not yield results.")

        self._loaded = True
        return True

    def _extract_informative_content(self, content, max_length=100000):
        """
        Extract the most informative portion of the content for embedding.
        This prefers the middle section of text which often contains the core information.
        
        Args:
            content: The content string or list to process
            max_length: Maximum character length to extract
            
        Returns:
            Processed string ready for embedding
        """
        # Handle different content types
        if isinstance(content, list):
            # Join list elements into a string
            content = ' '.join([str(item) for item in content if item])
        elif not isinstance(content, str):
            # Convert other types to string
            content = str(content)
            
        # Clean the content
        content = content.strip()
        if not content:
            return ""
            
        # If content is short enough, return it all
        if len(content) <= max_length:
            return content
            
        # Strategy: Take portions from beginning, middle and end
        # Beginning often has definitions, middle has core information, end has conclusions
        beginning_size = max_length // 4
        end_size = max_length // 4
        middle_size = max_length - beginning_size - end_size
        
        # Extract the parts
        beginning = content[:beginning_size]
        
        # Middle section starts at 1/3 and ends at 2/3 of the document
        middle_start = max(beginning_size, len(content) // 3)
        middle_end = min(len(content) - end_size, 2 * len(content) // 3)
        middle_available = middle_end - middle_start
        
        if middle_available <= 0:
            # Content too short for our strategy, just take beginning and end
            return beginning + "..." + content[-end_size:]
            
        # Take a section from the middle
        middle_extract_start = middle_start + (middle_available - middle_size) // 2
        middle = content[middle_extract_start:middle_extract_start + middle_size]
        
        # Extract end portion
        end = content[-end_size:] if end_size > 0 else ""
        
        # Combine with indicators
        return beginning + "..." + middle + "..." + end

    def get_cluster_results(self):
        """
        Returns the detailed clustering results.
        Should be called after detect_ambiguous_terms().
        
        Returns:
            Dictionary mapping terms to their resource cluster labels.
        """
        return self.cluster_results
        
    def get_cluster_metrics(self):
        """
        Returns metrics about the clustering process for each term.
        Should be called after detect_ambiguous_terms().
        
        Returns:
            Dictionary mapping terms to their clustering metrics.
        """
        return self.cluster_metrics

    def save_detailed_results(self, filename: Optional[str] = None) -> str:
        """
        Saves the detailed clustering results to a JSON file.
        Should be called after detect_ambiguous_terms().
        
        Args:
            filename: Optional custom filename. If not provided, generates a default name.
            
        Returns:
            Path to the saved file.
        """
        if not filename:
            filename = f"cluster_results.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare the data structure
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": {
                "eps": self.dbscan_eps,
                "min_samples": self.dbscan_min_samples,
                "min_resources": self.min_resources,
                "model_name": self.model_name
            },
            "ambiguous_terms": list(self.cluster_results.keys()),
            "total_ambiguous_terms": len(self.cluster_results),
            "cluster_results": self.cluster_results,
            "metrics": self.cluster_metrics
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            logging.info(f"[ResourceClusterDetector] Saved detailed cluster results to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[ResourceClusterDetector] Error saving results to {filepath}: {e}")
            return None

    def save_comprehensive_cluster_details(self, filename: Optional[str] = None) -> str:
        """
        Saves comprehensive cluster details to a JSON file, including all resources 
        and their content for each cluster. This is a more detailed version of 
        save_detailed_results that includes full resource information for each cluster.
        
        Should be called after detect_ambiguous_terms().
        
        Args:
            filename: Optional custom filename. If not provided, generates a default name.
            
        Returns:
            Path to the saved file.
        """
        if not filename:
            filename = f"comprehensive_cluster_details.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Initialize the data structure with metadata
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": {
                "eps": self.dbscan_eps,
                "min_samples": self.dbscan_min_samples,
                "min_resources": self.min_resources,
                "model_name": self.model_name,
                "clustering_algorithm": self.clustering_algorithm
            },
            "term_clusters": {}
        }
        
        # For each term with clusters
        for term, cluster_labels in self.cluster_results.items():
            if term not in self.term_details:
                continue
                
            term_data = self.term_details[term]
            resources = term_data.get('resources', [])
            
            # Skip if mismatched resource and label counts
            if len(resources) != len(cluster_labels):
                logging.warning(f"[ResourceClusterDetector] Mismatch between resource count ({len(resources)}) "
                                f"and cluster label count ({len(cluster_labels)}) for term '{term}'. Skipping.")
                continue
            
            # Group resources by cluster
            clusters = defaultdict(list)
            
            for i, resource in enumerate(resources):
                cluster_id = int(cluster_labels[i])
                
                # Create a clean resource entry with important fields only
                resource_entry = {
                    "url": resource.get("url", ""),
                    "title": resource.get("title", ""),
                    "processed_content": resource.get("processed_content", ""),
                    "cluster_id": cluster_id
                }
                
                clusters[cluster_id].append(resource_entry)
            
            # Get metrics for this term if available
            term_metrics = self.cluster_metrics.get(term, {})
            
            # Create term entry with all details
            term_entry = {
                "level": term_data.get('level'),
                "cluster_count": len([c for c in clusters if c != -1]),  # Don't count noise cluster
                "metrics": term_metrics,
                "clusters": {str(cluster_id): cluster_resources for cluster_id, cluster_resources in clusters.items()},
                "parent_terms": term_data.get('parent_ids', []),
                "variations": term_data.get('variations', [])
            }
            
            output_data["term_clusters"][term] = term_entry
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            logging.info(f"[ResourceClusterDetector] Saved comprehensive cluster details to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[ResourceClusterDetector] Error saving comprehensive details to {filepath}: {e}")
            return None

    def _cluster_embeddings(self, embeddings: np.ndarray, term: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster embeddings using the selected algorithm.
        
        Args:
            embeddings: Embedding vectors to cluster
            term: The term being processed (for logging)
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        metrics = {}
        
        if self.clustering_algorithm == 'dbscan':
            logging.debug(f"Clustering '{term}' with DBSCAN: eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples}")
            clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='cosine')
            cluster_labels = clusterer.fit_predict(embeddings)
            
            metrics = {
                "algorithm": "dbscan",
                "eps": self.dbscan_eps,
                "min_samples": self.dbscan_min_samples
            }
            
        elif self.clustering_algorithm == 'hdbscan':
            logging.debug(f"Clustering '{term}' with HDBSCAN: min_cluster_size={self.hdbscan_min_cluster_size}, min_samples={self.hdbscan_min_samples}")
            # HDBSCAN works best with euclidean distance and standardized data
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric='euclidean',
                cluster_selection_method='eom'  # Excess of Mass - often gives better results
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Add HDBSCAN-specific metrics
            metrics = {
                "algorithm": "hdbscan",
                "min_cluster_size": self.hdbscan_min_cluster_size,
                "min_samples": self.hdbscan_min_samples,
                "probabilities": clusterer.probabilities_.tolist() if hasattr(clusterer, 'probabilities_') else [],
                "outlier_scores": clusterer.outlier_scores_.tolist() if hasattr(clusterer, 'outlier_scores_') else []
            }
        
        return cluster_labels, metrics

    def detect_ambiguous_terms(self) -> list[str]:
        """
        Performs ambiguity detection based on resource content clustering.
        
        Returns:
            List of terms identified as potentially ambiguous.
        """
        if not self.embedding_model:
             logging.error("[ResourceClusterDetector] Sentence transformer model not loaded. Aborting.")
             return []
        if not self._load_data():
            logging.error("[ResourceClusterDetector] Failed to load necessary data. Aborting detection.")
            return []

        if not self.term_details or not self.canonical_terms:
            logging.warning("[ResourceClusterDetector] Missing term details or canonical terms. Cannot perform detection.")
            return []

        # Check if we can use cached results
        if self._clustering_complete and self._cached_all_ambiguous_terms is not None:
            logging.info("[ResourceClusterDetector] Using cached results from previous clustering")
            
            # If level is set, filter the cached results
            if self.level is not None:
                # Filter to only the terms at this level
                ambiguous_terms = [term for term in self._cached_all_ambiguous_terms 
                                if self.term_details.get(term, {}).get('level') == self.level]
                
                # Filter metrics and results to match
                self.cluster_results = {term: results for term, results in self._cached_all_cluster_results.items()
                                      if self.term_details.get(term, {}).get('level') == self.level}
                self.cluster_metrics = {term: metrics for term, metrics in self._cached_all_cluster_metrics.items()
                                       if self.term_details.get(term, {}).get('level') == self.level}
                
                logging.info(f"[ResourceClusterDetector] After filtering: {len(ambiguous_terms)} terms at level {self.level}")
                return ambiguous_terms
            else:
                # Return all cached terms
                self.cluster_results = self._cached_all_cluster_results
                self.cluster_metrics = self._cached_all_cluster_metrics
                return self._cached_all_ambiguous_terms

        # No cache available, perform full clustering
        # Reset storage for this run
        self.cluster_results = {}
        self.cluster_metrics = {}
        
        # Track terms that have valid clustering results but will be filtered by level
        all_ambiguous_terms = []
        
        level_info = f" for level {self.level}" if self.level is not None else ""
        logging.info(f"[ResourceClusterDetector] Starting ambiguity detection{level_info} using {self.clustering_algorithm.upper()} clustering...")
        processed_count = 0
        # Track statistics for debugging
        terms_with_sufficient_resources = 0
        terms_processed_fully = 0
        max_cluster_count = 0
        max_cluster_term = ""

        for term, term_data in self.term_details.items():
            processed_count += 1
            if processed_count % 1000 == 0: # Log less frequently for this potentially slower process
                logging.info(f"[ResourceClusterDetector] Processing {processed_count}/{len(self.term_details)}: '{term}'") # Log current term
                if self.use_embedding_cache:
                    cache_stats = global_embedding_cache.get_stats()
                    logging.info(f"[ResourceClusterDetector] Embedding cache stats: {cache_stats['size']}/{cache_stats['max_size']} entries, {cache_stats['hit_rate']:.2%} hit rate")

            # 1. Check if canonical
            if term not in self.canonical_terms:
                continue

            # Store the term's level for filtering later
            term_level = term_data.get('level')

            # 2. Check for sufficient resources
            resources = term_data.get('resources', [])
            initial_resource_count = len(resources)
            # Improved content extraction
            content_snippets = []
            for res_idx, res in enumerate(resources):
                if not res.get('processed_content'):
                    continue
                    
                content = res.get('processed_content')
                # Process and extract informative content
                processed_content = self._extract_informative_content(content)
                if len(processed_content.strip()) > 10:
                    content_snippets.append((res_idx, processed_content))
            
            valid_snippet_count = len(content_snippets)
            
            # Skip terms with too few resources - do this before level checking so we have accurate stats
            if valid_snippet_count < self.min_resources:
                continue
            
            terms_with_sufficient_resources += 1
                
            # Extract just the text snippets for embedding
            texts = [snippet[1] for snippet in content_snippets]
            
            try:
                # Apply sentence embeddings to all snippet texts
                embeddings = self.encode_texts(texts)
                if embeddings.shape[0] != len(texts):
                    logging.warning(f"[ResourceClusterDetector] Embedding count mismatch for '{term}': expected {len(texts)}, got {embeddings.shape[0]}")
                    continue
                    
                # Perform clustering on the embeddings
                cluster_labels, metrics = self._cluster_embeddings(embeddings, term)
                
                # Check if clustering was successful and found multiple clusters
                if cluster_labels is None:
                    continue
                    
                terms_processed_fully += 1
                
                # Skip terms that didn't form multiple clusters (all noise points)
                unique_clusters = set(cluster_labels)
                if -1 in unique_clusters:
                    unique_clusters.remove(-1)  # Don't count noise points
                
                cluster_count = len(unique_clusters)
                
                if cluster_count > max_cluster_count:
                    max_cluster_count = cluster_count
                    max_cluster_term = term
                    
                if cluster_count < 2:
                    continue
                    
                # Log metrics for this term
                metrics.update({
                    "num_resources": initial_resource_count,
                    "valid_snippets": valid_snippet_count,
                    "num_clusters": cluster_count,
                    "noise_points": sum(1 for label in cluster_labels if label == -1),
                    "cluster_sizes": {str(i): sum(1 for label in cluster_labels if label == i) 
                                     for i in unique_clusters},
                    "level": term_level,  # Include the term's level in metrics
                    "algorithm": self.clustering_algorithm
                })
                
                if self.clustering_algorithm == 'dbscan':
                    metrics.update({
                        "eps": self.dbscan_eps,
                        "min_samples": self.dbscan_min_samples
                    })
                else:  # hdbscan
                    metrics.update({
                        "min_cluster_size": self.hdbscan_min_cluster_size,
                        "min_samples": self.hdbscan_min_samples,
                    })
                
                # Store results for this term
                self.cluster_results[term] = [int(label) for label in cluster_labels]
                self.cluster_metrics[term] = metrics
                
                # Add to all ambiguous terms regardless of level
                all_ambiguous_terms.append(term)
                
            except Exception as e:
                logging.error(f"[ResourceClusterDetector] Error processing term '{term}': {e}")
                continue
        
        # Cache all results for future use
        self._cached_all_ambiguous_terms = all_ambiguous_terms.copy()
        self._cached_all_cluster_results = self.cluster_results.copy()
        self._cached_all_cluster_metrics = self.cluster_metrics.copy()
        self._clustering_complete = True
        
        # Now filter the results by level if needed
        ambiguous_terms = []
        if self.level is not None:
            # Filter results to only include terms from the specified level
            for term in all_ambiguous_terms:
                term_level = self.term_details.get(term, {}).get('level')
                if term_level == self.level:
                    ambiguous_terms.append(term)
            
            # Filter metrics and results as well to keep them consistent
            self.cluster_metrics = {term: metrics for term, metrics in self.cluster_metrics.items() 
                                   if self.term_details.get(term, {}).get('level') == self.level}
            self.cluster_results = {term: results for term, results in self.cluster_results.items()
                                  if self.term_details.get(term, {}).get('level') == self.level}
            
            logging.info(f"[ResourceClusterDetector] Found {len(all_ambiguous_terms)} potentially ambiguous terms across all levels")
            logging.info(f"[ResourceClusterDetector] After level filtering, {len(ambiguous_terms)} terms remain at level {self.level}")
        else:
            # If no level specified, use all ambiguous terms
            ambiguous_terms = all_ambiguous_terms
            
        level_info = f" for level {self.level}" if self.level is not None else ""
        logging.info(f"[ResourceClusterDetector] Ambiguity detection complete{level_info} using {self.clustering_algorithm.upper()}. Found {len(ambiguous_terms)} potentially ambiguous terms.")
        logging.info(f"[ResourceClusterDetector] Statistics{level_info}: {terms_with_sufficient_resources}/{len(self.canonical_terms)} canonical terms had sufficient resources")
        logging.info(f"[ResourceClusterDetector] Statistics{level_info}: {terms_processed_fully} terms were processed completely")
        logging.info(f"[ResourceClusterDetector] Statistics{level_info}: Maximum cluster count was {max_cluster_count} for term '{max_cluster_term}'")
        
        # Log embedding cache statistics if used
        if self.use_embedding_cache:
            cache_stats = global_embedding_cache.get_stats()
            logging.info(f"[ResourceClusterDetector] Memory cache stats: size={cache_stats['size']}, hit_rate={cache_stats['hit_rate']:.2%}")
            logging.info(f"[ResourceClusterDetector] Cache hits: {cache_stats['hits']}, misses: {cache_stats['misses']}")
            
            # Save and log vector store stats
            global_vector_store.save()
            store_stats = global_vector_store.get_stats()
            logging.info(f"[ResourceClusterDetector] Vector store stats: {store_stats['total_vectors']} vectors, {store_stats['size_mb']:.2f}MB")
            logging.info(f"[ResourceClusterDetector] Vector store hit rate: {store_stats['hit_rate']:.2%}")
        
        # Generate detailed insights for debugging
        logging.info(f"[ResourceClusterDetector] Statistics: {terms_with_sufficient_resources}/{len(self.canonical_terms)} canonical terms had sufficient resources.")
        logging.info(f"[ResourceClusterDetector] Statistics: {terms_processed_fully}/{terms_with_sufficient_resources} terms with sufficient resources were processed fully.")
        logging.info(f"[ResourceClusterDetector] Statistics: {len(all_ambiguous_terms)}/{terms_processed_fully} processed terms were found ambiguous.")
        if max_cluster_count > 0:
            logging.info(f"[ResourceClusterDetector] Maximum cluster count: {max_cluster_count} clusters for term '{max_cluster_term}'")
        
        # Save comprehensive cluster details including all resources with content
        try:
            comprehensive_output_path = self.save_comprehensive_cluster_details()
            logging.info(f"[ResourceClusterDetector] Saved comprehensive cluster details to {comprehensive_output_path}")
        except Exception as e:
            logging.error(f"[ResourceClusterDetector] Error saving comprehensive cluster details: {e}")
        
        return ambiguous_terms

class HybridAmbiguityDetector:
    """
    Combines multiple ambiguity detection approaches for higher confidence scoring.
    
    This detector runs multiple detection strategies and combines their results,
    assigning confidence scores based on agreement between detectors.
    """
    
    def __init__(self, hierarchy_file_path: str, final_term_files_pattern: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 output_dir: Optional[str] = None,
                 min_resources: int = 5,
                 level: Optional[int] = None):
        """
        Initialize the hybrid detector with configurations for all sub-detectors.
        
        Args:
            hierarchy_file_path: Path to the hierarchy.json file
            final_term_files_pattern: Glob pattern for canonical term files
            model_name: Name of the embedding model to use
            output_dir: Directory to save results
            min_resources: Minimum resources needed for ResourceClusterDetector
            level: Optional hierarchy level (0-3) for targeted analysis
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.model_name = model_name
        self.min_resources = min_resources
        self.level = level
        
        if output_dir:
            self.output_dir = output_dir
        else:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.output_dir = os.path.join(repo_root, "data", "ambiguity_detection_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the sub-detectors (but don't run them yet)
        self.context_detector = ParentContextDetector(
            hierarchy_file_path=hierarchy_file_path,
            final_term_files_pattern=final_term_files_pattern,
            level=level
        )
        
        # Initialize DBSCAN-based detector
        self.dbscan_detector = ResourceClusterDetector(
            hierarchy_file_path=hierarchy_file_path,
            final_term_files_pattern=final_term_files_pattern,
            model_name=model_name,
            min_resources=min_resources,
            clustering_algorithm='dbscan',
            level=level,
            output_dir=output_dir
        )
        
        # Initialize HDBSCAN-based detector if available
        if HDBSCAN_AVAILABLE:
            self.hdbscan_detector = ResourceClusterDetector(
                hierarchy_file_path=hierarchy_file_path,
                final_term_files_pattern=final_term_files_pattern,
                model_name=model_name,
                min_resources=min_resources,
                clustering_algorithm='hdbscan',
                level=level,
                output_dir=output_dir
            )
        else:
            self.hdbscan_detector = None
            logging.warning("HDBSCAN not available. Hybrid detector will not use HDBSCAN clustering.")
        
        # Results storage
        self.results = {}
        self.confidence_scores = {}
        
        # Cache for full analysis results
        self._all_level_performed = False
        self._all_level_results = {}
        self._cached_context_terms = None
        self._cached_dbscan_terms = None
        self._cached_dbscan_metrics = None
        self._cached_hdbscan_terms = None
        self._cached_hdbscan_metrics = None
    
    def detect_ambiguous_terms(self) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple detection strategies and combine results with confidence scoring.
        
        Returns:
            Dictionary mapping terms to detailed results with confidence scores
        """
        level_info = f" for level {self.level}" if self.level is not None else ""
        logging.info(f"[HybridAmbiguityDetector] Starting hybrid ambiguity detection{level_info}...")
        
        # Check if we have already run all detectors (without level constraint)
        if self.level is not None and self._all_level_performed:
            logging.info("[HybridAmbiguityDetector] Using cached results and filtering by level")
            
            # Get cached terms for current level
            if self._cached_context_terms is not None:
                self.context_detector.level = self.level
                context_terms = set(self.context_detector.detect_ambiguous_terms())
            else:
                context_terms = set()
                
            # Get cached resource cluster terms for current level
            if self._cached_dbscan_terms is not None:
                self.dbscan_detector.level = self.level
                dbscan_terms = set(self.dbscan_detector.detect_ambiguous_terms())
                dbscan_metrics = self.dbscan_detector.get_cluster_metrics()
            else:
                dbscan_terms = set()
                dbscan_metrics = {}
                
            # Get HDBSCAN terms if available
            hdbscan_terms = set()
            hdbscan_metrics = {}
            if self.hdbscan_detector and self._cached_hdbscan_terms is not None:
                self.hdbscan_detector.level = self.level
                hdbscan_terms = set(self.hdbscan_detector.detect_ambiguous_terms())
                hdbscan_metrics = self.hdbscan_detector.get_cluster_metrics()
        else:
            # Run all detectors
            # 1. Run the ParentContextDetector
            logging.info("[HybridAmbiguityDetector] Running ParentContextDetector...")
            context_terms = set(self.context_detector.detect_ambiguous_terms())
            self._cached_context_terms = context_terms.copy()
            logging.info(f"[HybridAmbiguityDetector] ParentContextDetector found {len(context_terms)} ambiguous terms")
            
            # 2. Run the DBSCAN-based ResourceClusterDetector
            logging.info("[HybridAmbiguityDetector] Running ResourceClusterDetector with DBSCAN...")
            dbscan_terms = set(self.dbscan_detector.detect_ambiguous_terms())
            dbscan_metrics = self.dbscan_detector.get_cluster_metrics()
            self._cached_dbscan_terms = dbscan_terms.copy()
            self._cached_dbscan_metrics = dbscan_metrics.copy()
            logging.info(f"[HybridAmbiguityDetector] DBSCAN detector found {len(dbscan_terms)} ambiguous terms")
            
            # 3. Run the HDBSCAN-based ResourceClusterDetector if available
            hdbscan_terms = set()
            hdbscan_metrics = {}
            if self.hdbscan_detector:
                logging.info("[HybridAmbiguityDetector] Running ResourceClusterDetector with HDBSCAN...")
                hdbscan_terms = set(self.hdbscan_detector.detect_ambiguous_terms())
                hdbscan_metrics = self.hdbscan_detector.get_cluster_metrics()
                self._cached_hdbscan_terms = hdbscan_terms.copy()
                self._cached_hdbscan_metrics = hdbscan_metrics.copy()
                logging.info(f"[HybridAmbiguityDetector] HDBSCAN detector found {len(hdbscan_terms)} ambiguous terms")
                
            # Mark that we have run all detectors
            self._all_level_performed = True
        
        # 4. Combine all unique terms
        all_terms = context_terms.union(dbscan_terms).union(hdbscan_terms)
        logging.info(f"[HybridAmbiguityDetector] Total unique ambiguous terms across all detectors: {len(all_terms)}")
        
        # 5. Calculate confidence scores and organize results
        results = {}
        for term in all_terms:
            # Initialize result structure
            result = {
                "term": term,
                "detected_by": {
                    "parent_context": term in context_terms,
                    "dbscan": term in dbscan_terms,
                    "hdbscan": term in hdbscan_terms
                },
                "detection_count": sum([
                    1 if term in context_terms else 0,
                    1 if term in dbscan_terms else 0,
                    1 if term in hdbscan_terms else 0
                ]),
                "metrics": {}
            }
            
            # Add available metrics
            if term in dbscan_metrics:
                result["metrics"]["dbscan"] = dbscan_metrics[term]
            if term in hdbscan_metrics:
                result["metrics"]["hdbscan"] = hdbscan_metrics[term]
            
            # Calculate confidence score (0.0-1.0)
            # Base score based on how many detectors found the term
            detector_count = 3 if self.hdbscan_detector else 2
            base_confidence = result["detection_count"] / detector_count
            
            # Boost confidence based on resource cluster separation when available
            boost = 0.0
            if term in dbscan_metrics and "average_separation" in dbscan_metrics[term]:
                sep_score = min(1.0, dbscan_metrics[term]["average_separation"] * 2)
                boost = max(boost, sep_score * 0.3)  # Boost up to 0.2 based on separation
            
            if term in hdbscan_metrics and "average_separation" in hdbscan_metrics[term]:
                sep_score = min(1.0, hdbscan_metrics[term]["average_separation"] * 2)
                boost = max(boost, sep_score * 0.2)  # HDBSCAN gets slightly lower weight
                
            # Final confidence score
            result["confidence_score"] = min(1.0, base_confidence + boost)
            
            # Assign confidence level
            if result["confidence_score"] >= 0.8:
                result["confidence_level"] = "high"
            elif result["confidence_score"] >= 0.5:
                result["confidence_level"] = "medium"
            else:
                result["confidence_level"] = "low"
                
            results[term] = result
        
        # Cache results for this level
        if self.level is not None:
            self._all_level_results[self.level] = results.copy()
            
        self.results = results
        logging.info("[HybridAmbiguityDetector] Hybrid ambiguity detection complete.")
        return results
    
    def get_results_by_confidence(self, min_confidence: float = 0.0) -> Dict[str, list]:
        """
        Get results organized by confidence level with optional filtering.
        
        Args:
            min_confidence: Minimum confidence score to include (0.0-1.0)
            
        Returns:
            Dictionary with high/medium/low confidence lists
        """
        if not self.results:
            logging.warning("[HybridAmbiguityDetector] No results available. Run detect_ambiguous_terms() first.")
            return {"high": [], "medium": [], "low": []}
            
        filtered_results = {term: data for term, data in self.results.items() 
                           if data["confidence_score"] >= min_confidence}
        
        by_confidence = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        for term, data in filtered_results.items():
            by_confidence[data["confidence_level"]].append(term)
            
        # Sort lists alphabetically
        for level in by_confidence:
            by_confidence[level] = sorted(by_confidence[level])
            
        return by_confidence
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save detection results to a JSON file.
        
        Args:
            filename: Optional custom filename. If not provided, generates a default name.
            
        Returns:
            Path to the saved file.
        """
        if not self.results:
            logging.warning("[HybridAmbiguityDetector] No results available. Run detect_ambiguous_terms() first.")
            return None
            
        if not filename:
            level_str = f"_level{self.level}" if self.level is not None else ""
            filename = f"hybrid_detection_results{level_str}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare the data structure
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": {
                "model_name": self.model_name,
                "min_resources": self.min_resources,
                "level": self.level,
            },
            "total_ambiguous_terms": len(self.results),
            "results_by_confidence": self.get_results_by_confidence(),
            "detailed_results": self.results
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            logging.info(f"[HybridAmbiguityDetector] Saved results to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[HybridAmbiguityDetector] Error saving results to {filepath}: {e}")
            return None


# --- Example Usage ---
if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_hierarchy_path = os.path.join(repo_root, "data", "hierarchy.json")
    default_final_terms_pattern = os.path.join(repo_root, "data", "final", "lv*", "lv*_final.txt")
    output_dir = os.path.join(repo_root, "data", "ambiguity_detection_results")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    if not os.path.exists(default_hierarchy_path):
         print(f"Default hierarchy file not found at: {default_hierarchy_path}")
    else:
        print(f"Using hierarchy file: {default_hierarchy_path}")
        print(f"Using final terms pattern: {default_final_terms_pattern}")

        # --- Run Parent Context Detector (Once) ---
        print("\n--- Running Parent Context Detector ---")
        parent_detector = ParentContextDetector(default_hierarchy_path, default_final_terms_pattern)
        potentially_ambiguous_parent = sorted(list(parent_detector.detect_ambiguous_terms()))
        parent_output_file = os.path.join(output_dir, "parent_context_ambiguous.txt")
        print(f"Parent Context Detector: Found {len(potentially_ambiguous_parent)} terms. Saved to {parent_output_file}")
        
        # Save results to file
        with open(parent_output_file, 'w') as f:
            for term in potentially_ambiguous_parent:
                f.write(f"{term}\n")

        # --- Run Resource Cluster Detector with different algorithms ---
        for algorithm in ['dbscan', 'hdbscan']:
            # Check if HDBSCAN is available
            if algorithm == 'hdbscan' and not HDBSCAN_AVAILABLE:
                print("\nSkipping HDBSCAN as it's not installed. Install with: pip install hdbscan")
                continue
                
            print(f"\n--- Running Resource Cluster Detector with {algorithm.upper()} ---")
            
            # Process each level separately
            for level in range(4):  # Levels 0-3
                print(f"\nProcessing Level {level} with {algorithm.upper()}...")
                
                # Initialize detector with level-specific settings and selected algorithm
                resource_detector = ResourceClusterDetector(
                    hierarchy_file_path=default_hierarchy_path,
                    final_term_files_pattern=default_final_terms_pattern,
                    model_name='all-MiniLM-L6-v2',
                    min_resources=5,
                    clustering_algorithm=algorithm,
                    level=level,
                    output_dir=output_dir
                )
                
                # Run detection
                potentially_ambiguous_resource = sorted(list(resource_detector.detect_ambiguous_terms()))
                count = len(potentially_ambiguous_resource)
                print(f"Found {count} potentially ambiguous terms at level {level}.")

                # Save basic term list
                resource_output_file = os.path.join(output_dir, f"resource_cluster_ambiguous_{algorithm}_level{level}.txt")
                print(f"Saving basic term list to {resource_output_file}")
                with open(resource_output_file, 'w') as f:
                    for term in potentially_ambiguous_resource:
                        f.write(f"{term}\n")
                
                # Save detailed results including cluster assignments and metrics
                detailed_output_path = resource_detector.save_detailed_results(
                    f"cluster_results_{algorithm}_level{level}.json"
                )
                print(f"Saved detailed cluster results to {detailed_output_path}")
                
            print(f"\n--- Completed {algorithm.upper()} Resource Cluster Detection for All Levels ---")
            
        print("\n--- All Resource Cluster Detection Methods Completed ---")
        
    # --- Add Hybrid Detector after the individual detectors ---
    print("\n--- Running Hybrid Ambiguity Detector ---")
    
    # Process each level separately as before
    for level in range(4):  # Levels 0-3
        print(f"\nProcessing Level {level} with Hybrid Detector...")
        
        # Initialize hybrid detector
        hybrid_detector = HybridAmbiguityDetector(
            hierarchy_file_path=default_hierarchy_path,
            final_term_files_pattern=default_final_terms_pattern,
            model_name='all-MiniLM-L6-v2',
            min_resources=5,
            level=level,
            output_dir=output_dir
        )
        
        # Run detection
        results = hybrid_detector.detect_ambiguous_terms()
        
        # Get results by confidence
        confidence_results = hybrid_detector.get_results_by_confidence(min_confidence=0.5)
        high_confidence = len(confidence_results["high"])
        medium_confidence = len(confidence_results["medium"])
        print(f"Found {len(results)} potentially ambiguous terms at level {level}.")
        print(f"  High confidence: {high_confidence}")
        print(f"  Medium confidence: {medium_confidence}")
        print(f"  Total reliable (confidence ≥ 0.5): {high_confidence + medium_confidence}")
        
        # Save detailed results
        result_path = hybrid_detector.save_results()
        print(f"Saved hybrid detection results to {result_path}")
        
        # Also save term lists by confidence
        for confidence_level, terms in confidence_results.items():
            if terms:  # Only save non-empty lists
                confidence_file = os.path.join(output_dir, f"hybrid_ambiguous_{confidence_level}_level{level}.txt")
                with open(confidence_file, 'w') as f:
                    for term in terms:
                        f.write(f"{term}\n")
                print(f"Saved {confidence_level} confidence terms to {confidence_file}")
                
    print("\n--- Completed Hybrid Ambiguity Detection for All Levels ---") 