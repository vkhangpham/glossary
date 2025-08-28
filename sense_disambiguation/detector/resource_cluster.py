"""
Resource clustering-based detector for ambiguity detection.

This module implements the ResourceClusterDetector class that analyzes
resources associated with terms to identify semantic clustering that
suggests potential ambiguity.
"""

import json
import glob
import os
import re
import logging
import datetime
import hashlib
import pickle
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from typing import Optional, Dict, List, Any, Tuple, Literal, Union
from sentence_transformers import SentenceTransformer
import warnings

# Import base detector classes
from .base import EvidenceBuilder, get_detector_version

# Add NLTK for improved tokenization
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Download necessary NLTK data if not already available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK package not found. Using basic tokenization instead. "
                  "For improved results, install NLTK: pip install nltk")

# Try to import EmbeddingCache, or define a minimal implementation if not available
try:
    from .embedding_cache import EmbeddingCache, global_embedding_cache, global_vector_store
except ImportError:
    # Define a minimal EmbeddingCache implementation
    logging.warning("embedding_cache module not found, using minimal implementation")
    
    class EmbeddingCache:
        """A minimal implementation of EmbeddingCache when the module is not available."""
        
        def __init__(self):
            self.memory_cache = {}
            self.hits = 0
            self.misses = 0
        
        def get(self, text):
            """Get embedding for text from cache if available."""
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in self.memory_cache:
                self.hits += 1
                return self.memory_cache[text_hash]
            self.misses += 1
            return None
        
        def put(self, text, embedding, save_to_disk=False):
            """Store embedding in cache."""
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            self.memory_cache[text_hash] = embedding
        
        def save(self):
            """Save cache to disk (no-op in minimal implementation)."""
            pass
        
        def get_stats(self):
            """Get cache statistics."""
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total": total,
                "hit_rate": hit_rate,
                "memory_cache_size": len(self.memory_cache)
            }
    
    # Try to import and initialize the vector store directly
    try:
        from sense_disambiguation.vector_store import PersistentVectorStore
        # Create a global vector store instance with updated path
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        vector_store_dir = os.path.join(repo_root, "sense_disambiguation/data/vector_store")
        os.makedirs(vector_store_dir, exist_ok=True)
        global_vector_store = PersistentVectorStore(persist_dir=vector_store_dir)
        logging.info(f"Initialized global vector store at {vector_store_dir}")
    except ImportError:
        logging.warning("Could not import PersistentVectorStore, vector storage will be disabled")
        global_vector_store = None
    
    # Create global embedding cache instance
    global_embedding_cache = EmbeddingCache()

from .utils import convert_numpy_types

# Import HDBSCAN if available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN package not found. Only DBSCAN clustering will be available. "
                  "To use HDBSCAN, install with: pip install hdbscan")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                 output_dir: Optional[str] = None,
                 remove_stopwords: bool = False):
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
            remove_stopwords: Whether to remove stopwords during tokenization
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.model_name = model_name
        self.min_resources = min_resources
        self.level = level
        self.use_embedding_cache = use_embedding_cache
        self.remove_stopwords = remove_stopwords
        
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
        
        # Initialize stopwords if using NLTK
        if NLTK_AVAILABLE and self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()
        
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
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.output_dir = os.path.join(repo_root, "sense_disambiguation/data", "ambiguity_detection_results")
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

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens using improved tokenization.
        
        Args:
            text: Text string to tokenize
            
        Returns:
            List of token strings
        """
        if not text:
            return []
            
        # Convert to lowercase for consistent processing
        text = text.lower()
        
        if NLTK_AVAILABLE:
            # Use NLTK's more advanced tokenization
            tokens = word_tokenize(text)
            
            # Remove stopwords if requested
            if self.remove_stopwords:
                tokens = [token for token in tokens if token not in self.stopwords]
                
            # Filter to only include alphanumeric tokens
            tokens = [token for token in tokens if token.isalnum()]
        else:
            # Fallback to basic regex tokenization
            tokens = re.findall(r'\b\w+\b', text)
            
        return tokens

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
            
        # Use NLTK to extract sentences if available
        if NLTK_AVAILABLE:
            try:
                # Split into sentences for more natural text chunking
                sentences = sent_tokenize(content)
                
                # Strategy: Take sentences from beginning, middle and end
                beginning_size = max_length // 4
                end_size = max_length // 4
                middle_size = max_length - beginning_size - end_size
                
                beginning_sentences = []
                beginning_chars = 0
                for sent in sentences:
                    if beginning_chars + len(sent) + 1 <= beginning_size:
                        beginning_sentences.append(sent)
                        beginning_chars += len(sent) + 1  # +1 for space
                    else:
                        break
                        
                end_sentences = []
                end_chars = 0
                for sent in reversed(sentences):
                    if end_chars + len(sent) + 1 <= end_size:
                        end_sentences.insert(0, sent)
                        end_chars += len(sent) + 1
                    else:
                        break
                
                # Find middle section
                middle_start_idx = len(beginning_sentences)
                middle_end_idx = len(sentences) - len(end_sentences)
                
                # Adjust if we have room for only a few sentences
                if middle_start_idx >= middle_end_idx:
                    # Just combine beginning and end if no room for middle
                    return ' '.join(beginning_sentences) + "..." + ' '.join(end_sentences)
                
                middle_range = middle_end_idx - middle_start_idx
                middle_sentences = []
                middle_chars = 0
                
                # Take sentences from the middle section
                middle_center = middle_start_idx + middle_range // 2
                middle_left = middle_center
                middle_right = middle_center + 1
                
                while middle_chars < middle_size and (middle_left >= middle_start_idx or middle_right < middle_end_idx):
                    # Try to take from left
                    if middle_left >= middle_start_idx:
                        sent = sentences[middle_left]
                        if middle_chars + len(sent) + 1 <= middle_size:
                            middle_sentences.insert(0, sent)
                            middle_chars += len(sent) + 1
                        middle_left -= 1
                        
                    # Try to take from right
                    if middle_right < middle_end_idx:
                        sent = sentences[middle_right]
                        if middle_chars + len(sent) + 1 <= middle_size:
                            middle_sentences.append(sent)
                            middle_chars += len(sent) + 1
                        middle_right += 1
                
                # Combine the sections
                return ' '.join(beginning_sentences) + "..." + ' '.join(middle_sentences) + "..." + ' '.join(end_sentences)
                
            except Exception as e:
                logging.debug(f"Error using NLTK for sentence extraction: {e}. Falling back to character-based extraction.")
                # Fall back to the original method if NLTK processing fails
                pass
                
        # Original character-based method (fallback)
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
        
        DEPRECATED: This method will be removed in a future version.
        The recommended approach is to get evidence blocks via detect() and store them in a unified context file.
        
        Args:
            filename: Optional custom filename. If not provided, generates a default name.
            
        Returns:
            Path to the saved file.
        """
        warnings.warn(
            "ResourceClusterDetector.save_detailed_results() is deprecated; direct file writes will be removed in future",
            DeprecationWarning, 
            stacklevel=2
        )
        
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            level_str = f"_level{self.level}" if self.level is not None else ""
            filename = f"cluster_results_eps{self.dbscan_eps}_minsamples{self.dbscan_min_samples}{level_str}_{timestamp}.json"
            
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

        # Convert numpy types before saving
        output_data_serializable = convert_numpy_types(output_data)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data_serializable, f, indent=2) # Use the serializable version
            logging.info(f"[ResourceClusterDetector] Saved detailed cluster results to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[ResourceClusterDetector] Error saving results to {filepath}: {e}")
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
        
        DEPRECATED: Use detect() instead which returns EvidenceBuilder objects.
        This method is maintained for backward compatibility.
        
        Returns:
            List of terms identified as potentially ambiguous.
        """
        warnings.warn(
            "ResourceClusterDetector.detect_ambiguous_terms() is deprecated; use detect() instead",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            # Improved content extraction - include title
            content_snippets_data = [] # Store tuples of (original_index, text_for_embedding)
            for res_idx, res in enumerate(resources):
                content = res.get('processed_content')
                title = res.get('title', '') # Get title, default to empty string

                if not content:
                    logging.debug(f"Skipping resource {res_idx} for term '{term}' due to missing processed_content.")
                    continue

                # Process and extract informative content from the main content
                processed_snippet = self._extract_informative_content(content)

                # Combine title and snippet for embedding input, ensuring title presence
                text_for_embedding = f"{title}. {processed_snippet}" if title else processed_snippet

                if len(text_for_embedding.strip()) > 10: # Check combined length
                    content_snippets_data.append((res_idx, text_for_embedding))
                else:
                     logging.debug(f"Skipping resource {res_idx} for term '{term}' due to short combined content.")


            valid_snippet_count = len(content_snippets_data)

            # Skip terms with too few resources - do this before level checking so we have accurate stats
            if valid_snippet_count < self.min_resources:
                continue

            terms_with_sufficient_resources += 1

            # Extract just the text snippets for embedding
            texts = [snippet[1] for snippet in content_snippets_data]
            # Keep track of original resource indices corresponding to the embeddings
            original_indices = [snippet[0] for snippet in content_snippets_data]

            try:
                # Apply sentence embeddings to all snippet texts
                embeddings = self.encode_texts(texts)
                logging.debug(f"[ResourceClusterDetector] Term '{term}': Encoded {len(texts)} snippets, embeddings shape: {embeddings.shape}")
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
                    
                # Calculate silhouette score if we have more than 1 cluster
                silhouette_avg = None
                # Ensure we have embeddings and labels only for non-noise points for silhouette calc
                non_noise_mask = cluster_labels != -1
                embeddings_for_silhouette = embeddings[non_noise_mask]
                labels_for_silhouette = cluster_labels[non_noise_mask]

                if cluster_count >= 2 and len(labels_for_silhouette) >= 2: # Need at least 2 points and 2 clusters
                    try:
                        silhouette_avg = silhouette_score(embeddings_for_silhouette, labels_for_silhouette, metric='cosine')
                        logging.debug(f"Term '{term}': Calculated silhouette score: {silhouette_avg:.4f}")
                    except ValueError as e:
                        logging.warning(f"Term '{term}': Could not calculate silhouette score (ValueError: {e}). Skipping.")
                        silhouette_avg = None # Set to None explicitly
                    except Exception as e:
                        logging.error(f"Term '{term}': Unexpected error calculating silhouette score: {e}")
                        silhouette_avg = None
                else:
                     logging.debug(f"Term '{term}': Not calculating silhouette score (clusters={cluster_count}, non-noise points={len(labels_for_silhouette)}).")


                # Log metrics for this term
                metrics.update({
                    "num_resources": initial_resource_count,
                    "valid_snippets": valid_snippet_count,
                    "num_clusters": cluster_count,
                    "noise_points": sum(1 for label in cluster_labels if label == -1),
                    "cluster_sizes": {str(i): sum(1 for label in cluster_labels if label == i)
                                     for i in unique_clusters},
                    "level": term_level,  # Include the term's level in metrics
                    "algorithm": self.clustering_algorithm,
                    "silhouette_score": silhouette_avg # Add silhouette score to metrics
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
                
                # Store metrics for this term
                # Need to map cluster labels back to the original full resource list if some were skipped
                full_cluster_labels = [-2] * initial_resource_count # Use -2 for resources skipped before embedding
                for i, orig_idx in enumerate(original_indices):
                    full_cluster_labels[orig_idx] = int(cluster_labels[i])

                # Always store the clustering results and metrics for the term
                self.cluster_results[term] = full_cluster_labels # Store labels aligned with original resources
                self.cluster_metrics[term] = metrics
                
                # But only add terms with 2+ clusters to the ambiguous terms list
                # This keeps the ambiguity detection logic the same while allowing low confidence
                # terms to be included in self.cluster_results for comprehensive output
                if cluster_count >= 2:
                    # Add to all ambiguous terms regardless of level
                    all_ambiguous_terms.append(term)
                
            except Exception as e:
                logging.exception(f"[ResourceClusterDetector] Error processing term '{term}'")
                continue
        
        # Cache all results for future use
        self._cached_all_ambiguous_terms = all_ambiguous_terms.copy()
        self._cached_all_cluster_results = self.cluster_results.copy()
        self._cached_all_cluster_metrics = self.cluster_metrics.copy()
        self._clustering_complete = True
        
        # Calculate TF-IDF confidence scores for all processed terms
        processed_terms = list(self.cluster_results.keys())
        tfidf_confidence_scores = self._calculate_tfidf_confidence(processed_terms)
        
        # Add TF-IDF confidence to metrics for all terms that have scores
        for term, score in tfidf_confidence_scores.items():
            if term in self.cluster_metrics:
                self.cluster_metrics[term]["tfidf_confidence"] = score
                
                # If TF-IDF confidence is high (>0.6) for a term with exactly 1 cluster,
                # consider adding it to ambiguous terms for further analysis
                if (score > 0.6 and 
                    self.cluster_metrics[term].get("num_clusters", 0) == 1 and
                    term not in all_ambiguous_terms):
                    
                    logging.info(f"[ResourceClusterDetector] Adding term '{term}' to ambiguous candidates based on high TF-IDF confidence ({score:.2f})")
                    all_ambiguous_terms.append(term)
                
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
        
        # Save comprehensive cluster details including all resources with content
        try:
            comprehensive_output_path = self.save_comprehensive_cluster_details(include_low_confidence=True)
            logging.info(f"[ResourceClusterDetector] Saved comprehensive cluster details to {comprehensive_output_path}")
        except Exception as e:
            logging.error(f"[ResourceClusterDetector] Error saving comprehensive cluster details: {e}")
        
        return ambiguous_terms 

    def save_comprehensive_cluster_details(self, filename: Optional[str] = None, include_low_confidence: bool = True) -> str:
        """
        Saves comprehensive cluster details to a JSON file, including all resources 
        and their content for each cluster. This is a more detailed version of 
        save_detailed_results that includes full resource information for each cluster.
        
        DEPRECATED: This method will be removed in a future version.
        The recommended approach is to get evidence blocks via detect() and store them in a unified context file.
        
        Should be called after detect_ambiguous_terms().
        
        Args:
            filename: Optional custom filename. If not provided, generates a default name.
            include_low_confidence: If True, also include terms with low confidence clusters
                                  (those that didn't meet ambiguity criteria)
            
        Returns:
            Path to the saved file.
        """
        warnings.warn(
            "ResourceClusterDetector.save_comprehensive_cluster_details() is deprecated; direct file writes will be removed in future",
            DeprecationWarning, 
            stacklevel=2
        )
        
        if not filename:
            # Use a stable default name based on the algorithm
            # Don't include timestamp or parameters to avoid creating multiple files
            filename = f"comprehensive_cluster_details_{self.clustering_algorithm}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Initialize the data structure with metadata
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "parameters": {
                "eps": self.dbscan_eps,
                "min_samples": self.dbscan_min_samples,
                "min_resources": self.min_resources,
                "model_name": self.model_name,
                "clustering_algorithm": self.clustering_algorithm,
                "include_low_confidence": include_low_confidence
            },
            "term_clusters": {}
        }
        
        # Get all terms with cluster results, potentially including those that weren't classified as ambiguous
        if include_low_confidence:
            # Use all terms that have cluster results, not just those deemed ambiguous
            terms_to_process = list(self.cluster_results.keys())
            logging.info(f"[ResourceClusterDetector] Including {len(terms_to_process)} terms with clustering results (including low confidence).")
        else:
            # Only include terms that were detected as ambiguous (standard behavior)
            terms_to_process = self._cached_all_ambiguous_terms if self._cached_all_ambiguous_terms else list(self.cluster_results.keys())
            logging.info(f"[ResourceClusterDetector] Including only {len(terms_to_process)} terms detected as ambiguous.")
        
        # For each term with clusters
        for term in terms_to_process:
            if term not in self.cluster_results or term not in self.term_details:
                continue
                
            cluster_labels = self.cluster_results[term]
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
                
                clusters[str(cluster_id)].append(resource_entry)
            
            # Get metrics for this term if available
            term_metrics = self.cluster_metrics.get(term, {})
            
            # Add confidence flags
            unique_clusters = set(int(c) for c in clusters.keys() if c != "-1") # Don't count noise cluster
            term_metrics["is_low_confidence"] = unique_clusters is None or len(unique_clusters) < 2
            
            # Create term entry with all details
            term_entry = {
                "level": term_data.get('level'),
                "cluster_count": len([c for c in clusters.keys() if c != "-1"]),  # Don't count noise cluster
                "metrics": term_metrics,
                "clusters": clusters,
                "parent_terms": term_data.get('parent_ids', []),
                "variations": term_data.get('variations', [])
            }
            
            output_data["term_clusters"][term] = term_entry
        
        # Convert numpy types before saving
        output_data_serializable = convert_numpy_types(output_data)

        try:
            with open(filepath, 'w') as f:
                json.dump(output_data_serializable, f, indent=2)
            logging.info(f"[ResourceClusterDetector] Saved comprehensive cluster details to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"[ResourceClusterDetector] Error saving comprehensive details to {filepath}: {e}")
            return None

    def _calculate_tfidf_confidence(self, all_terms: list[str]) -> Dict[str, float]:
        """
        Calculate a confidence score based on TF-IDF analysis of the entire corpus.
        
        This method:
        1. Builds a corpus from all term resources
        2. Calculates TF-IDF scores for each term
        3. Assigns higher confidence to short terms (1-2 words) with high frequency
        
        The method also caches the corpus and TF-IDF matrix to disk for future use.
        
        Args:
            all_terms: List of terms to analyze
            
        Returns:
            Dictionary mapping terms to their TF-IDF confidence scores (0.0-1.0)
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            import pickle
            import os
            
            # Set up paths for cached data
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            cache_dir = os.path.join(repo_root, "sense_disambiguation/data", "corpus_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            corpus_path = os.path.join(cache_dir, "corpus.pkl")
            term_indices_path = os.path.join(cache_dir, "term_indices.pkl")
            vectorizer_path = os.path.join(cache_dir, "vectorizer.pkl")
            tfidf_matrix_path = os.path.join(cache_dir, "tfidf_matrix.npz")
            hierarchy_hash_path = os.path.join(cache_dir, "hierarchy_hash.txt")
            
            logging.info("[ResourceClusterDetector] Calculating TF-IDF confidence scores for terms")
            
            # Calculate a hash of the hierarchy file to check compatibility
            current_hierarchy_hash = ""
            try:
                with open(self.hierarchy_file_path, 'rb') as f:
                    hierarchy_content = f.read()
                    current_hierarchy_hash = hashlib.md5(hierarchy_content).hexdigest()
            except Exception as e:
                logging.warning(f"[ResourceClusterDetector] Could not calculate hierarchy hash: {e}")
            
            # Try to load cached data if available
            cached_data_exists = (
                os.path.exists(corpus_path) and
                os.path.exists(term_indices_path) and
                os.path.exists(vectorizer_path) and
                os.path.exists(tfidf_matrix_path) and
                os.path.exists(hierarchy_hash_path)
            )
            
            corpus = []
            term_indices = {}
            vectorizer = None
            tfidf_matrix = None
            
            if cached_data_exists and current_hierarchy_hash:
                try:
                    # Check if cached data is compatible with current hierarchy
                    with open(hierarchy_hash_path, 'r') as f:
                        cached_hash = f.read().strip()
                    
                    if cached_hash != current_hierarchy_hash:
                        logging.info("[ResourceClusterDetector] Hierarchy has changed since cache was created. Rebuilding corpus.")
                        cached_data_exists = False
                    else:
                        logging.info("[ResourceClusterDetector] Loading cached corpus and TF-IDF data")
                        
                        with open(corpus_path, 'rb') as f:
                            corpus = pickle.load(f)
                            
                        with open(term_indices_path, 'rb') as f:
                            term_indices = pickle.load(f)
                            
                        with open(vectorizer_path, 'rb') as f:
                            vectorizer = pickle.load(f)
                        
                        from scipy.sparse import load_npz
                        tfidf_matrix = load_npz(tfidf_matrix_path)
                        
                        logging.info(f"[ResourceClusterDetector] Loaded cached corpus with {len(corpus)} documents")
                    
                except Exception as e:
                    logging.warning(f"[ResourceClusterDetector] Error loading cached corpus: {e}. Building new corpus.")
                    cached_data_exists = False
            
            # Build corpus if not loaded from cache
            if not cached_data_exists:
                logging.info("[ResourceClusterDetector] Building new corpus from resources")
                
                # 1. Collect all processed content into a corpus
                corpus = []
                term_indices = {}  # Maps term to its indices in the corpus
                
                for term_idx, term in enumerate(all_terms):
                    if term not in self.term_details:
                        continue
                        
                    term_indices[term] = []
                    term_data = self.term_details[term]
                    resources = term_data.get('resources', [])
                    
                    for res in resources:
                        content = res.get('processed_content')
                        if isinstance(content, str) and content:
                            corpus.append(content)
                            term_indices[term].append(len(corpus) - 1)
                
                if not corpus:
                    logging.warning("[ResourceClusterDetector] No content found for TF-IDF analysis")
                    return {}
                    
                # 2. Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_df=0.85,  # Ignore terms that appear in >85% of documents
                    min_df=5,     # Ignore terms that appear in fewer than 5 documents
                    max_features=10000,
                    stop_words='english'
                )
                
                # 3. Calculate TF-IDF matrix
                tfidf_matrix = vectorizer.fit_transform(corpus)
                
                # 4. Save corpus and TF-IDF data for future use
                try:
                    logging.info("[ResourceClusterDetector] Saving corpus and TF-IDF data to cache")
                    
                    with open(corpus_path, 'wb') as f:
                        pickle.dump(corpus, f)
                        
                    with open(term_indices_path, 'wb') as f:
                        pickle.dump(term_indices, f)
                        
                    with open(vectorizer_path, 'wb') as f:
                        pickle.dump(vectorizer, f)
                    
                    from scipy.sparse import save_npz
                    save_npz(tfidf_matrix_path, tfidf_matrix)
                    
                    # Save the hierarchy hash for compatibility checking
                    if current_hierarchy_hash:
                        with open(hierarchy_hash_path, 'w') as f:
                            f.write(current_hierarchy_hash)
                    
                    logging.info(f"[ResourceClusterDetector] Saved corpus with {len(corpus)} documents to {cache_dir}")
                    
                except Exception as e:
                    logging.warning(f"[ResourceClusterDetector] Error saving corpus to cache: {e}")
            
            # Ensure we have all the required data
            if not corpus or not term_indices or not vectorizer or tfidf_matrix is None:
                logging.error("[ResourceClusterDetector] Failed to prepare corpus data for TF-IDF analysis")
                return {}
                
            # Get feature names from the vectorizer
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate TF-IDF confidence scores for each term
            tfidf_confidence = {}
            
            # 4. Process each term to calculate its confidence score
            for term, doc_indices in term_indices.items():
                if not doc_indices:
                    tfidf_confidence[term] = 0.0
                    continue
                    
                # Get the word count for the term
                word_count = len(term.split('_'))
                
                # Extract the term's documents from the TF-IDF matrix
                term_docs = tfidf_matrix[doc_indices]
                
                # Get the TF-IDF scores for the term itself
                term_words = term.replace('_', ' ').split()
                term_word_indices = []
                
                for word in term_words:
                    if word in vectorizer.vocabulary_:
                        term_word_indices.append(vectorizer.vocabulary_[word])
                
                if not term_word_indices:
                    tfidf_confidence[term] = 0.0
                    continue
                
                # Calculate average TF-IDF for term words across all its documents
                term_tfidf_values = []
                for idx in term_word_indices:
                    col_values = term_docs[:, idx].toarray().flatten()
                    non_zero_values = col_values[col_values > 0]
                    if len(non_zero_values) > 0:
                        term_tfidf_values.append(np.mean(non_zero_values))
                
                if not term_tfidf_values:
                    tfidf_confidence[term] = 0.0
                    continue
                
                avg_tfidf = np.mean(term_tfidf_values)
                
                # Calculate document frequency (percentage of term's documents containing the term)
                doc_freq = sum(1 for idx in term_word_indices 
                              for doc_idx in range(len(doc_indices)) 
                              if term_docs[doc_idx, idx] > 0) / (len(doc_indices) * len(term_word_indices))
                
                # Boost score for short terms (1-2 words)
                word_count_factor = 1.0 if word_count <= 2 else (3.0 / word_count)
                
                # Calculate final confidence (normalized to 0-1 range)
                # Higher for short terms with high TF-IDF and high document frequency
                raw_confidence = avg_tfidf * doc_freq * word_count_factor
                
                # Normalize and store
                tfidf_confidence[term] = min(1.0, raw_confidence * 2.0)  # Scale up but cap at 1.0
                
            logging.info(f"[ResourceClusterDetector] Calculated TF-IDF confidence for {len(tfidf_confidence)} terms")
            return tfidf_confidence
            
        except Exception as e:
            logging.error(f"[ResourceClusterDetector] Error calculating TF-IDF confidence: {e}")
            return {} 

    def detect(self) -> List[EvidenceBuilder]:
        """
        Detects ambiguous terms based on resource clustering and returns evidence builders.
        
        This is the new preferred API that returns structured evidence blocks for the splitter.
        
        Returns:
            List of EvidenceBuilder objects for detected ambiguous terms.
        """
        # First run the original detection logic to populate self.cluster_results and self.cluster_metrics
        # This avoids duplicating the complex detection algorithm
        detected_terms = self.detect_ambiguous_terms()
        
        # Get the detector version
        version = get_detector_version()
        
        # Flag to control inclusion of resource details
        include_resource_details = False  # TODO: Make this a parameter of the detector
        max_resources_per_cluster = 50    # Limit resources to avoid overly large JSON files
        
        # Convert to evidence builders
        evidence_builders = []
        
        # Process both detected terms and any others with cluster results
        # This allows the splitter to potentially use terms that have cluster
        # information but weren't automatically classified as ambiguous
        for term in self.cluster_results.keys():
            # Get labels and metrics
            labels = self.cluster_results.get(term, [])
            metrics = self.cluster_metrics.get(term, {})
            
            # Skip if no clustering information
            if not labels or not metrics:
                continue
                
            # Get the term level
            level = metrics.get("level")
            
            # Calculate confidence based on separation score and number of clusters
            separation_score = metrics.get("separation_score", 0.0)
            silhouette = metrics.get("silhouette_score", 0.0)
            cluster_count = metrics.get("num_clusters", 0)
            
            # Apply heuristic: 0.2 base + 0.6*separation_score + 0.2*silhouette
            # The 0.2 base ensures even poor clusters get transmitted to splitter
            base_confidence = 0.2
            separation_weight = min(0.6, separation_score * 0.8) if separation_score is not None else 0
            silhouette_weight = min(0.2, max(0, silhouette * 0.4)) if silhouette is not None else 0
            
            # Higher confidence for more clusters (diminishing returns)
            cluster_bonus = min(0.2, (cluster_count - 1) * 0.1) if cluster_count > 1 else 0
            
            confidence = min(1.0, base_confidence + separation_weight + silhouette_weight + cluster_bonus)
            
            # Add confidence boost from TF-IDF if available 
            tfidf_confidence = metrics.get("tfidf_confidence", 0.0)
            if tfidf_confidence > 0.5:
                tfidf_boost = min(0.2, (tfidf_confidence - 0.5) * 0.4)  # Max boost of 0.2
                confidence = min(1.0, confidence + tfidf_boost)
                
            # Prepare metrics dictionary (clean up unnecessary keys)
            api_metrics = {
                "separation_score": separation_score,
                "silhouette_score": silhouette,
                "num_clusters": cluster_count,
                "noise_points": metrics.get("noise_points", 0),
                "valid_snippets": metrics.get("valid_snippets", 0),
                "tfidf_confidence": tfidf_confidence
            }
            
            # Prepare payload dictionary
            payload = {
                "cluster_labels": labels,
                "eps": metrics.get("eps", self.dbscan_eps),
                "min_samples": metrics.get("min_samples", self.dbscan_min_samples)
            }
            
            # Add cluster details (resources) if enabled - cap at max_resources_per_cluster per cluster
            if include_resource_details:
                # Use information from self.term_details to get resources
                term_data = self.term_details.get(term, {})
                resources = term_data.get('resources', [])
                
                # Skip if resource count mismatches label count
                if len(resources) != len(labels):
                    continue
                
                # Group resources by cluster label
                cluster_details = defaultdict(list)
                
                for i, (resource, label) in enumerate(zip(resources, labels)):
                    # Skip noise points
                    if label < 0:
                        continue
                        
                    # Limit resources per cluster for manageable filesize
                    if len(cluster_details[str(label)]) >= max_resources_per_cluster:
                        continue
                    
                    # Create minimal resource object
                    r_obj = {
                        "url": resource.get("url", ""),
                        "title": resource.get("title", "")
                    }
                    
                    # Include truncated content
                    content = resource.get("processed_content", "")
                    if content:
                        if len(content) > 300:
                            r_obj["processed_content"] = content[:300] + "..."
                        else:
                            r_obj["processed_content"] = content
                    
                    # Add to cluster
                    cluster_details[str(label)].append(r_obj)
                
                # Add to payload
                payload["cluster_details"] = dict(cluster_details)
            
            # Create the evidence builder
            builder = EvidenceBuilder.create(
                term=term,
                level=level,
                source="resource_cluster",
                detector_version=version,
                confidence=confidence,
                metrics=api_metrics,
                payload=payload
            )
            
            evidence_builders.append(builder)
        
        return evidence_builders 