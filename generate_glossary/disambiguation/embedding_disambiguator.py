"""
Embedding-based disambiguation using semantic clustering.

Detects ambiguous terms by clustering resource embeddings to find
when web content about a term forms distinct semantic groups.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from .utils import (
    extract_informative_content,
    calculate_confidence_score
)

# Try to import HDBSCAN if available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.debug("HDBSCAN not available, using DBSCAN only")


def detect(
    terms: List[str],
    web_content: Dict[str, Any],
    hierarchy: Dict[str, Any],
    model_name: str = "all-MiniLM-L6-v2",
    clustering_algorithm: str = "dbscan",
    eps: float = 0.45,
    min_samples: int = 2,
    min_resources: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Detect ambiguous terms by clustering their resource embeddings.
    
    This function:
    1. Extracts content from each term's resources
    2. Generates embeddings for the content
    3. Clusters embeddings to find semantic groups
    4. Flags terms with multiple clusters as ambiguous
    
    Args:
        terms: List of terms to analyze
        web_content: Web resources for each term
        hierarchy: Hierarchy data with term information
        model_name: Sentence transformer model to use
        clustering_algorithm: Algorithm for clustering ('dbscan' or 'hdbscan')
        eps: DBSCAN epsilon parameter (max distance between samples)
        min_samples: Minimum samples to form a cluster
        min_resources: Minimum resources required for analysis
        
    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    logging.info(f"Detecting ambiguity using embeddings for {len(terms)} terms")
    
    # Load embedding model
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model {model_name}: {e}")
        return {}
    
    results = {}
    terms_processed = 0
    
    for term in terms:
        # Skip if insufficient resources
        if term not in web_content:
            continue
            
        term_resources = web_content[term]
        if isinstance(term_resources, dict):
            term_resources = term_resources.get("resources", [])
        
        if len(term_resources) < min_resources:
            continue
        
        # Extract content from resources
        contents = []
        for resource in term_resources:
            content = extract_informative_content(resource)
            if content:
                contents.append(content)
        
        if len(contents) < min_resources:
            continue
        
        # Generate embeddings
        try:
            embeddings = model.encode(contents, show_progress_bar=False)
        except Exception as e:
            logging.warning(f"Failed to generate embeddings for {term}: {e}")
            continue
        
        # Cluster embeddings
        clusters, silhouette = cluster_embeddings(
            embeddings,
            clustering_algorithm,
            eps,
            min_samples
        )
        
        # Check if ambiguous (multiple clusters)
        unique_clusters = set(c for c in clusters if c != -1)
        if len(unique_clusters) > 1:
            # Calculate cluster info
            cluster_info = []
            for cluster_id in unique_clusters:
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                cluster_size = len(cluster_indices)
                
                # Get sample resources from this cluster
                sample_resources = [term_resources[i] for i in cluster_indices[:3]]
                
                cluster_info.append({
                    "cluster_id": int(cluster_id),
                    "size": cluster_size,
                    "percentage": cluster_size / len(clusters),
                    "sample_resources": sample_resources
                })
            
            # Calculate confidence based on clustering quality
            confidence = calculate_confidence_score({
                "num_clusters": len(unique_clusters),
                "silhouette_score": silhouette,
                "largest_cluster_ratio": max(c["percentage"] for c in cluster_info),
                "total_resources": len(contents)
            })
            
            results[term] = {
                "term": term,
                "method": "embedding",
                "num_clusters": len(unique_clusters),
                "clusters": cluster_info,
                "silhouette_score": silhouette,
                "confidence": confidence,
                "total_resources": len(contents),
                "evidence": {
                    "clustering_algorithm": clustering_algorithm,
                    "eps": eps,
                    "min_samples": min_samples,
                    "model": model_name
                }
            }
            
            terms_processed += 1
            
            if terms_processed % 10 == 0:
                logging.info(f"Processed {terms_processed} ambiguous terms")
    
    logging.info(f"Found {len(results)} ambiguous terms via embeddings")
    return results


def cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: str,
    eps: float,
    min_samples: int
) -> tuple[np.ndarray, float]:
    """
    Cluster embeddings using specified algorithm.
    
    Args:
        embeddings: Embedding vectors to cluster
        algorithm: Clustering algorithm to use
        eps: DBSCAN epsilon parameter
        min_samples: Minimum samples for a cluster
        
    Returns:
        Tuple of (cluster_labels, silhouette_score)
    """
    if algorithm == "hdbscan" and HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_samples,
            min_samples=min_samples,
            cluster_selection_epsilon=eps
        )
        clusters = clusterer.fit_predict(embeddings)
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = clusterer.fit_predict(embeddings)
    
    # Calculate silhouette score if we have clusters
    silhouette = 0.0
    unique_clusters = set(clusters)
    if len(unique_clusters) > 1 and -1 not in unique_clusters:
        try:
            silhouette = silhouette_score(embeddings, clusters)
        except:
            silhouette = 0.0
    
    return clusters, silhouette