import json
import glob
import os
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import datetime
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

# Import the vector store
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

class GlobalResourceClusterer:
    """
    Clusters all resources across the hierarchy together to identify global patterns.
    
    This approach differs from term-specific clustering by analyzing all resources
    in a single embedding space, allowing for more consistent clustering and
    better identification of cross-term relationships.
    """
    
    def __init__(self, 
                 hierarchy_file_path: str,
                 final_term_files_pattern: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 clustering_algorithm: str = 'dbscan',
                 dbscan_eps: float = 0.5,
                 dbscan_min_samples: int = 3,
                 hdbscan_min_cluster_size: int = 3,
                 hdbscan_min_samples: int = 3,
                 vector_store_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the global resource clusterer.
        
        Args:
            hierarchy_file_path: Path to the hierarchy.json file.
            final_term_files_pattern: Glob pattern for canonical term files.
            model_name: Name of the sentence-transformer model to use.
            clustering_algorithm: Which clustering algorithm to use ('dbscan' or 'hdbscan').
            dbscan_eps: Maximum distance between samples for DBSCAN.
            dbscan_min_samples: Minimum samples in neighborhood for DBSCAN.
            hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN.
            hdbscan_min_samples: Minimum samples parameter for HDBSCAN.
            vector_store_path: Path to store vectors. If None, uses default path.
            output_dir: Directory to save results. If None, uses default path.
        """
        self.hierarchy_file_path = hierarchy_file_path
        self.final_term_files_pattern = final_term_files_pattern
        self.model_name = model_name
        
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
        
        # Clustering parameters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        
        # Setup paths
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if vector_store_path is None:
            self.vector_store_path = os.path.join(repo_root, "data", "global_vector_store")
        else:
            self.vector_store_path = vector_store_path
            
        if output_dir is None:
            self.output_dir = os.path.join(repo_root, "data", "global_clustering_results")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = PersistentVectorStore(persist_dir=self.vector_store_path)
        
        # Initialize the embedding model
        try:
            logging.info(f"Loading sentence transformer model '{model_name}'...")
            self.embedding_model = SentenceTransformer(model_name)
            logging.info(f"Model '{model_name}' loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model '{model_name}': {e}")
            logging.error("Please ensure 'sentence-transformers' and its dependencies are installed.")
            self.embedding_model = None
            
        # Data storage
        self.hierarchy_data = None
        self.canonical_terms = set()
        self.term_details = None
        self._loaded = False
        
        # Results storage
        self.resource_embeddings = []
        self.resource_metadata = []
        self.cluster_labels = None
        self.cluster_metrics = {}
        
    def _load_data(self) -> bool:
        """
        Load the hierarchy data and canonical term lists.
        
        Returns:
            Boolean indicating success
        """
        if self._loaded:
            return True

        # Load hierarchy
        logging.info(f"[GlobalResourceClusterer] Loading hierarchy from {self.hierarchy_file_path}...")
        try:
            with open(self.hierarchy_file_path, 'r') as f:
                self.hierarchy_data = json.load(f)
            self.term_details = self.hierarchy_data.get('terms', {})
            if not self.term_details:
                logging.warning("[GlobalResourceClusterer] Hierarchy file loaded, but 'terms' dictionary is missing or empty.")
                return False
            logging.info(f"[GlobalResourceClusterer] Loaded {len(self.term_details)} terms from hierarchy.")
        except FileNotFoundError:
            logging.error(f"[GlobalResourceClusterer] Hierarchy file not found: {self.hierarchy_file_path}")
            return False
        except json.JSONDecodeError:
            logging.error(f"[GlobalResourceClusterer] Error decoding JSON from {self.hierarchy_file_path}")
            return False

        # Load canonical terms
        logging.info(f"[GlobalResourceClusterer] Loading canonical terms from pattern: {self.final_term_files_pattern}")
        final_term_files = glob.glob(self.final_term_files_pattern)
        if not final_term_files:
            logging.warning(f"[GlobalResourceClusterer] No files found matching pattern: {self.final_term_files_pattern}")

        for file_path in final_term_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            self.canonical_terms.add(term)
            except Exception as e:
                logging.error(f"[GlobalResourceClusterer] Error reading final term file {file_path}: {e}")

        logging.info(f"[GlobalResourceClusterer] Loaded {len(self.canonical_terms)} unique canonical terms.")
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
    
    def collect_and_embed_resources(self) -> int:
        """
        Collect all resources from the hierarchy and generate embeddings.
        Uses vector store for caching to avoid redundant embedding computation.
        
        Returns:
            Number of resources processed
        """
        if not self._load_data():
            logging.error("[GlobalResourceClusterer] Failed to load data. Aborting resource collection.")
            return 0
            
        if not self.embedding_model:
            logging.error("[GlobalResourceClusterer] Embedding model not loaded. Aborting.")
            return 0
            
        logging.info("[GlobalResourceClusterer] Starting resource collection and embedding...")
        
        # Reset storage
        self.resource_embeddings = []
        self.resource_metadata = []
        
        # Process each term's resources
        total_resources = 0
        processed_resources = 0
        skipped_resources = 0
        canonical_terms_with_resources = 0
        
        # First pass: count total resources for progress reporting
        for term, term_data in self.term_details.items():
            if term in self.canonical_terms:
                resources = term_data.get('resources', [])
                total_resources += len(resources)
                
        logging.info(f"[GlobalResourceClusterer] Found {total_resources} total resources across {len(self.canonical_terms)} canonical terms")
        
        # Second pass: process resources
        for term, term_data in self.term_details.items():
            # Only process canonical terms
            if term not in self.canonical_terms:
                continue
                
            # Get term metadata
            term_level = term_data.get('level')
            resources = term_data.get('resources', [])
            
            if not resources:
                continue
                
            canonical_terms_with_resources += 1
            term_processed_resources = 0
            
            # Process each resource
            for res_idx, resource in enumerate(resources):
                # Skip resources without content
                if not resource.get('processed_content'):
                    skipped_resources += 1
                    continue
                    
                # Extract and process content
                content = resource.get('processed_content')
                processed_content = self._extract_informative_content(content)
                
                if len(processed_content.strip()) <= 10:
                    skipped_resources += 1
                    continue
                
                # Check if we already have this embedding in the vector store
                existing_embedding = self.vector_store.get(processed_content)
                
                if existing_embedding is not None:
                    # Use cached embedding
                    embedding = existing_embedding
                else:
                    # Generate new embedding
                    try:
                        embedding = self.embedding_model.encode(processed_content)
                        # Store in vector store for future use
                        self.vector_store.put(
                            processed_content, 
                            embedding,
                            auxiliary_data={
                                "term": term, 
                                "level": term_level,
                                "resource_idx": res_idx
                            }
                        )
                    except Exception as e:
                        logging.error(f"[GlobalResourceClusterer] Error embedding resource for term '{term}': {e}")
                        skipped_resources += 1
                        continue
                
                # Store embedding and metadata
                self.resource_embeddings.append(embedding)
                
                # Create metadata for this resource
                metadata = {
                    "term": term,
                    "level": term_level,
                    "resource_idx": res_idx,
                    "url": resource.get("url", ""),
                    "title": resource.get("title", ""),
                    "content_snippet": processed_content[:200] + "..." if len(processed_content) > 200 else processed_content,
                }
                
                self.resource_metadata.append(metadata)
                processed_resources += 1
                term_processed_resources += 1
                
                # Log progress periodically
                if processed_resources % 1000 == 0:
                    progress = (processed_resources / total_resources) * 100
                    logging.info(f"[GlobalResourceClusterer] Processed {processed_resources}/{total_resources} resources ({progress:.1f}%)")
            
            # Log per-term summary periodically
            if canonical_terms_with_resources % 500 == 0:
                logging.info(f"[GlobalResourceClusterer] Processed {canonical_terms_with_resources} canonical terms with resources")
                
        # Save vector store
        self.vector_store.save()
        
        # Convert embeddings to numpy array for clustering
        if self.resource_embeddings:
            self.resource_embeddings = np.array(self.resource_embeddings)
            
        logging.info(f"[GlobalResourceClusterer] Resource collection complete:")
        logging.info(f"  - Processed {processed_resources} resources for {canonical_terms_with_resources} canonical terms")
        logging.info(f"  - Skipped {skipped_resources} resources due to insufficient content or errors")
        logging.info(f"  - Total embedding matrix shape: {self.resource_embeddings.shape if isinstance(self.resource_embeddings, np.ndarray) else 'empty'}")
        
        return processed_resources
    
    def cluster_resources(self) -> Dict[str, Any]:
        """
        Perform global clustering on all collected resource embeddings.
        
        Returns:
            Dictionary of clustering metrics and statistics
        """
        if not isinstance(self.resource_embeddings, np.ndarray) or len(self.resource_embeddings) == 0:
            logging.error("[GlobalResourceClusterer] No resource embeddings available for clustering")
            return {"error": "No embeddings available"}
            
        logging.info(f"[GlobalResourceClusterer] Starting global clustering using {self.clustering_algorithm.upper()}")
        logging.info(f"[GlobalResourceClusterer] Clustering {len(self.resource_embeddings)} resource embeddings")
        
        # Clustering metrics
        metrics = {
            "algorithm": self.clustering_algorithm,
            "total_resources": len(self.resource_embeddings),
            "timestamp": datetime.datetime.now().isoformat(),
            "model_name": self.model_name,
        }
        
        # Perform clustering based on selected algorithm
        if self.clustering_algorithm == 'dbscan':
            logging.info(f"[GlobalResourceClusterer] Running DBSCAN with eps={self.dbscan_eps}, min_samples={self.dbscan_min_samples}")
            
            clusterer = DBSCAN(
                eps=self.dbscan_eps, 
                min_samples=self.dbscan_min_samples, 
                metric='cosine',
                n_jobs=-1  # Use all available cores
            )
            
            self.cluster_labels = clusterer.fit_predict(self.resource_embeddings)
            
            # Add DBSCAN-specific metrics
            metrics.update({
                "dbscan_eps": self.dbscan_eps,
                "dbscan_min_samples": self.dbscan_min_samples,
            })
            
        elif self.clustering_algorithm == 'hdbscan':
            logging.info(f"[GlobalResourceClusterer] Running HDBSCAN with min_cluster_size={self.hdbscan_min_cluster_size}, min_samples={self.hdbscan_min_samples}")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                min_samples=self.hdbscan_min_samples,
                metric='euclidean',
                cluster_selection_method='eom'  # Excess of Mass method
            )
            
            self.cluster_labels = clusterer.fit_predict(self.resource_embeddings)
            
            # Add HDBSCAN-specific metrics
            metrics.update({
                "hdbscan_min_cluster_size": self.hdbscan_min_cluster_size,
                "hdbscan_min_samples": self.hdbscan_min_samples,
            })
            
            # Add probabilities and outlier scores if available
            if hasattr(clusterer, 'probabilities_'):
                # Just store summary statistics to avoid huge data
                probs = clusterer.probabilities_
                metrics["probability_stats"] = {
                    "min": float(np.min(probs)),
                    "max": float(np.max(probs)),
                    "mean": float(np.mean(probs)),
                    "median": float(np.median(probs))
                }
                
            if hasattr(clusterer, 'outlier_scores_'):
                # Just store summary statistics to avoid huge data
                scores = clusterer.outlier_scores_
                metrics["outlier_score_stats"] = {
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "mean": float(np.mean(scores)),
                    "median": float(np.median(scores))
                }
        
        # Calculate cluster statistics
        unique_clusters = set(self.cluster_labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        noise_points = sum(1 for label in self.cluster_labels if label == -1)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            if cluster_id != -1:  # Skip noise cluster
                size = sum(1 for label in self.cluster_labels if label == cluster_id)
                cluster_sizes[str(cluster_id)] = size
        
        # Add general clustering metrics
        metrics.update({
            "num_clusters": n_clusters,
            "noise_points": noise_points,
            "noise_percentage": (noise_points / len(self.resource_embeddings)) * 100,
            "cluster_sizes": cluster_sizes,
        })
        
        logging.info(f"[GlobalResourceClusterer] Clustering complete: {n_clusters} clusters found")
        logging.info(f"[GlobalResourceClusterer] Noise points: {noise_points} ({metrics['noise_percentage']:.1f}%)")
        
        # Store metrics for access after clustering
        self.cluster_metrics = metrics
        
        return metrics
    
    def analyze_clusters(self) -> Dict[str, Any]:
        """
        Analyze formed clusters to extract insights about term relationships.
        
        Returns:
            Dictionary of analysis results
        """
        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            logging.error("[GlobalResourceClusterer] No clustering results available to analyze")
            return {"error": "No clustering results"}
            
        if len(self.resource_metadata) != len(self.cluster_labels):
            logging.error("[GlobalResourceClusterer] Mismatch between resource metadata and cluster labels")
            return {"error": "Metadata and label count mismatch"}
        
        logging.info("[GlobalResourceClusterer] Starting cluster analysis...")
        
        # Initialize analysis containers
        cluster_term_counts = defaultdict(lambda: defaultdict(int))
        term_cluster_counts = defaultdict(lambda: defaultdict(int))
        cluster_level_counts = defaultdict(lambda: defaultdict(int))
        term_cluster_distribution = {}
        ambiguous_terms = set()
        
        # Process each resource and its cluster assignment
        for i, label in enumerate(self.cluster_labels):
            if label == -1:  # Skip noise points for this analysis
                continue
                
            metadata = self.resource_metadata[i]
            term = metadata.get("term", "")
            level = metadata.get("level", -1)
            
            # Count term occurrences in each cluster
            cluster_term_counts[str(label)][term] += 1
            
            # Count cluster occurrences for each term
            term_cluster_counts[term][str(label)] += 1
            
            # Track level distribution within clusters
            cluster_level_counts[str(label)][str(level)] += 1
        
        # Identify potentially ambiguous terms (those spread across multiple clusters)
        for term, clusters in term_cluster_counts.items():
            if len(clusters) > 1:
                # Calculate entropy/spread of the term across clusters
                total_resources = sum(clusters.values())
                largest_cluster_size = max(clusters.values())
                largest_cluster_percentage = (largest_cluster_size / total_resources) * 100
                
                # If the term's resources are significantly split (no dominant cluster)
                # Consider it potentially ambiguous
                if largest_cluster_percentage < 70 and total_resources >= 3:
                    ambiguous_terms.add(term)
                    
                # Store the distribution data for all terms across clusters
                term_cluster_distribution[term] = {
                    "total_resources": total_resources,
                    "clusters": dict(clusters),
                    "largest_cluster_percentage": largest_cluster_percentage,
                    "is_ambiguous": largest_cluster_percentage < 70 and total_resources >= 3
                }
        
        # Extract top terms for each cluster
        cluster_top_terms = {}
        for cluster_id, terms in cluster_term_counts.items():
            # Sort terms by frequency
            sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)
            # Keep top N terms
            cluster_top_terms[cluster_id] = sorted_terms[:20]  # Top 20 terms per cluster
        
        # Prepare analysis results
        analysis_results = {
            "total_clusters": len(cluster_term_counts),
            "total_analyzed_terms": len(term_cluster_counts),
            "potentially_ambiguous_terms": list(ambiguous_terms),
            "ambiguous_term_count": len(ambiguous_terms),
            "cluster_top_terms": cluster_top_terms,
            "term_cluster_distribution": term_cluster_distribution,
            "cluster_level_distribution": {k: dict(v) for k, v in cluster_level_counts.items()},
        }
        
        logging.info(f"[GlobalResourceClusterer] Analysis complete:")
        logging.info(f"  - Analyzed {len(term_cluster_counts)} terms across {len(cluster_term_counts)} clusters")
        logging.info(f"  - Identified {len(ambiguous_terms)} potentially ambiguous terms")
        
        return analysis_results
    
    def save_results(self, analysis_results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save clustering and analysis results to JSON files.
        
        Args:
            analysis_results: Results from analyze_clusters method
            filename: Optional custom base filename
            
        Returns:
            Path to the saved results file
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"global_clustering_{self.clustering_algorithm}_{timestamp}.json"
            
        output_path = os.path.join(self.output_dir, filename)
        
        # Prepare the full results object
        full_results = {
            "clustering_metrics": self.cluster_metrics,
            "analysis_results": analysis_results,
            "parameters": {
                "model_name": self.model_name,
                "clustering_algorithm": self.clustering_algorithm,
                "dbscan_eps": self.dbscan_eps,
                "dbscan_min_samples": self.dbscan_min_samples,
                "hdbscan_min_cluster_size": self.hdbscan_min_cluster_size,
                "hdbscan_min_samples": self.hdbscan_min_samples,
            }
        }
        
        # Save the main results file
        try:
            with open(output_path, 'w') as f:
                json.dump(full_results, f, indent=2)
            logging.info(f"[GlobalResourceClusterer] Saved results to {output_path}")
        except Exception as e:
            logging.error(f"[GlobalResourceClusterer] Error saving results: {e}")
            return None
            
        # Also save the potentially ambiguous terms to a simple text file
        if analysis_results.get("potentially_ambiguous_terms"):
            ambiguous_terms_path = os.path.join(
                self.output_dir, 
                f"potentially_ambiguous_terms_{self.clustering_algorithm}_{timestamp}.txt"
            )
            
            try:
                with open(ambiguous_terms_path, 'w') as f:
                    for term in sorted(analysis_results["potentially_ambiguous_terms"]):
                        f.write(f"{term}\n")
                logging.info(f"[GlobalResourceClusterer] Saved potentially ambiguous terms to {ambiguous_terms_path}")
            except Exception as e:
                logging.error(f"[GlobalResourceClusterer] Error saving ambiguous terms list: {e}")
        
        return output_path
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the entire global clustering pipeline: collect resources, cluster, analyze.
        
        Returns:
            Dictionary with all results and statistics
        """
        logging.info("[GlobalResourceClusterer] Starting complete global resource analysis...")
        
        # Step 1: Collect and embed resources
        resource_count = self.collect_and_embed_resources()
        if resource_count == 0:
            return {"error": "No resources collected"}
            
        # Step 2: Cluster the resources
        clustering_metrics = self.cluster_resources()
        if "error" in clustering_metrics:
            return clustering_metrics
            
        # Step 3: Analyze the clusters
        analysis_results = self.analyze_clusters()
        if "error" in analysis_results:
            return analysis_results
            
        # Step 4: Save results
        results_path = self.save_results(analysis_results)
        
        # Return comprehensive results
        return {
            "resource_count": resource_count,
            "clustering_metrics": clustering_metrics,
            "analysis_results": analysis_results,
            "results_file": results_path
        }

# If directly executed
if __name__ == '__main__':
    # Default paths
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_hierarchy_path = os.path.join(repo_root, "data", "hierarchy.json")
    default_final_terms_pattern = os.path.join(repo_root, "data", "final", "lv*", "lv*_final.txt")
    output_dir = os.path.join(repo_root, "data", "global_clustering_results")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run with DBSCAN
    print("Running global clustering with DBSCAN...")
    dbscan_clusterer = GlobalResourceClusterer(
        hierarchy_file_path=default_hierarchy_path,
        final_term_files_pattern=default_final_terms_pattern,
        clustering_algorithm='dbscan',
        dbscan_eps=0.5,
        dbscan_min_samples=3,
        output_dir=output_dir
    )
    
    dbscan_results = dbscan_clusterer.run_complete_analysis()
    
    # Run with HDBSCAN if available
    if HDBSCAN_AVAILABLE:
        print("\nRunning global clustering with HDBSCAN...")
        hdbscan_clusterer = GlobalResourceClusterer(
            hierarchy_file_path=default_hierarchy_path,
            final_term_files_pattern=default_final_terms_pattern,
            clustering_algorithm='hdbscan',
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=3,
            output_dir=output_dir
        )
        
        hdbscan_results = hdbscan_clusterer.run_complete_analysis()
    else:
        print("\nSkipping HDBSCAN clustering (not installed)") 