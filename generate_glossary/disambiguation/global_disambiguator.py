"""
Global clustering-based disambiguation across all terms.

Detects ambiguous terms by clustering resources globally to identify
when resources for a single term belong to different global topics.
"""

import logging
from typing import Dict, List, Any, Optional, Set
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from .utils import (
    extract_informative_content,
    calculate_confidence_score
)

# Type aliases
Terms = List[str]
WebContent = Dict[str, Any]
Hierarchy = Dict[str, Any]
DetectionResults = Dict[str, Dict[str, Any]]


def detect_ambiguous_by_global_clustering(
    terms: Terms,
    web_content: WebContent,
    hierarchy: Hierarchy,
    model_name: str = "all-MiniLM-L6-v2",
    eps: float = 0.3,
    min_samples: int = 3,
    min_resources: int = 5,
    max_resources_per_term: int = 10
) -> DetectionResults:
    """
    Detect ambiguous terms using global resource clustering.
    
    This approach:
    1. Pools resources from all terms
    2. Clusters them globally to find topics
    3. Identifies terms whose resources span multiple global clusters
    
    Args:
        terms: List of terms to analyze
        web_content: Web resources for each term
        hierarchy: Hierarchy data
        model_name: Sentence transformer model
        eps: DBSCAN epsilon for tighter global clustering
        min_samples: Minimum samples for global clusters
        min_resources: Minimum resources per term
        max_resources_per_term: Maximum resources to use per term
        
    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    logging.info(f"Detecting ambiguity using global clustering for {len(terms)} terms")
    
    # Load embedding model
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model {model_name}: {e}")
        return {}
    
    # Collect all resources with term mapping
    all_contents = []
    resource_to_term = []
    term_resource_indices = defaultdict(list)
    
    for term in terms:
        if term not in web_content:
            continue
            
        term_resources = web_content[term]
        if isinstance(term_resources, dict):
            term_resources = term_resources.get("resources", [])
        
        if len(term_resources) < min_resources:
            continue
        
        # Sample resources if too many
        if len(term_resources) > max_resources_per_term:
            import random
            term_resources = random.sample(term_resources, max_resources_per_term)
        
        # Extract content
        for resource in term_resources:
            content = extract_informative_content(resource)
            if content:
                idx = len(all_contents)
                all_contents.append(content)
                resource_to_term.append(term)
                term_resource_indices[term].append(idx)
    
    if len(all_contents) < 50:  # Need enough for meaningful global clustering
        logging.warning("Insufficient resources for global clustering")
        return {}
    
    logging.info(f"Clustering {len(all_contents)} resources globally")
    
    # Generate embeddings for all content
    try:
        all_embeddings = model.encode(all_contents, show_progress_bar=False)
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}")
        return {}
    
    # Perform global clustering
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    global_clusters = clusterer.fit_predict(all_embeddings)
    
    # Calculate global silhouette score
    unique_clusters = set(c for c in global_clusters if c != -1)
    global_silhouette = 0.0
    if len(unique_clusters) > 1:
        try:
            global_silhouette = silhouette_score(all_embeddings, global_clusters)
        except:
            pass
    
    logging.info(f"Found {len(unique_clusters)} global clusters")
    
    # Analyze term distribution across global clusters
    results = {}
    
    for term in term_resource_indices:
        term_indices = term_resource_indices[term]
        term_clusters = [global_clusters[i] for i in term_indices]
        
        # Count cluster distribution
        cluster_counts = defaultdict(int)
        for cluster_id in term_clusters:
            if cluster_id != -1:  # Ignore noise
                cluster_counts[cluster_id] += 1
        
        # Check if term spans multiple clusters
        if len(cluster_counts) > 1:
            # Calculate cluster distribution
            total_clustered = sum(cluster_counts.values())
            cluster_info = []
            
            for cluster_id, count in cluster_counts.items():
                # Find topic keywords for this cluster
                cluster_resources = [
                    all_contents[i] for i in range(len(all_contents))
                    if global_clusters[i] == cluster_id and resource_to_term[i] == term
                ]
                
                cluster_info.append({
                    "global_cluster_id": int(cluster_id),
                    "resource_count": count,
                    "percentage": count / total_clustered,
                    "sample_content": cluster_resources[:2]  # Sample content
                })
            
            # Sort by resource count
            cluster_info.sort(key=lambda x: x["resource_count"], reverse=True)
            
            # Calculate confidence
            confidence = _calculate_global_confidence(
                num_global_clusters=len(cluster_counts),
                cluster_distribution=cluster_info,
                global_silhouette=global_silhouette
            )
            
            results[term] = {
                "term": term,
                "method": "global",
                "num_global_clusters": len(cluster_counts),
                "cluster_distribution": cluster_info,
                "total_resources": len(term_indices),
                "clustered_resources": total_clustered,
                "confidence": confidence,
                "evidence": {
                    "global_clusters_total": len(unique_clusters),
                    "global_silhouette": global_silhouette,
                    "eps": eps,
                    "min_samples": min_samples
                }
            }
    
    logging.info(f"Found {len(results)} ambiguous terms via global clustering")
    return results


def _calculate_global_confidence(
    num_global_clusters: int,
    cluster_distribution: List[Dict],
    global_silhouette: float
) -> float:
    """
    Calculate confidence for global clustering detection.
    
    Args:
        num_global_clusters: Number of global clusters term appears in
        cluster_distribution: Distribution of resources across clusters
        global_silhouette: Global clustering quality score
        
    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence from number of clusters
    if num_global_clusters == 2:
        base_confidence = 0.6
    elif num_global_clusters == 3:
        base_confidence = 0.75
    else:
        base_confidence = min(0.9, 0.75 + (num_global_clusters - 3) * 0.05)
    
    # Adjust based on distribution balance
    # More balanced distribution = higher confidence
    percentages = [c["percentage"] for c in cluster_distribution]
    max_percentage = max(percentages)
    
    if max_percentage > 0.8:  # One cluster dominates
        balance_factor = 0.7
    elif max_percentage > 0.6:
        balance_factor = 0.9
    else:  # Well balanced
        balance_factor = 1.0
    
    # Include global clustering quality
    quality_factor = 1.0
    if global_silhouette > 0:
        quality_factor = 1.0 + (global_silhouette * 0.2)
    
    confidence = min(1.0, base_confidence * balance_factor * quality_factor)
    
    return confidence