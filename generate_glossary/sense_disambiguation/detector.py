"""
Detection functions for identifying ambiguous terms.

All functions are pure - no state, no classes, just data transformation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import hashlib

from .utils import (
    get_level_params,
    extract_informative_content,
    calculate_confidence_score
)

# Type aliases
Terms = List[str]
WebContent = Dict[str, Any]
Hierarchy = Dict[str, Any]
DetectionResults = Dict[str, Dict[str, Any]]
ClusteringAlgorithm = Literal["dbscan", "hdbscan"]

# Try to import HDBSCAN if available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.debug("HDBSCAN not available, using DBSCAN only")


def detect_with_embeddings(
    terms: Terms,
    web_content: WebContent,
    hierarchy: Hierarchy,
    model_name: str = "all-MiniLM-L6-v2",
    clustering_algorithm: ClusteringAlgorithm = "dbscan",
    eps: float = 0.45,
    min_samples: int = 2,
    min_resources: int = 5
) -> DetectionResults:
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
        embeddings = model.encode(contents, show_progress_bar=False)
        
        # Cluster embeddings
        cluster_labels, metrics = _cluster_embeddings(
            embeddings=embeddings,
            algorithm=clustering_algorithm,
            eps=eps,
            min_samples=min_samples
        )
        
        # Check if ambiguous (multiple clusters)
        unique_clusters = set(label for label in cluster_labels if label != -1)
        if len(unique_clusters) >= 2:
            # Calculate confidence based on cluster quality
            confidence = calculate_confidence_score(
                num_clusters=len(unique_clusters),
                silhouette=metrics.get("silhouette_score", 0),
                num_resources=len(contents)
            )
            
            # Get term level from hierarchy
            term_data = hierarchy.get("terms", {}).get(term, {})
            level = term_data.get("level")
            
            results[term] = {
                "canonical_name": term,
                "level": level,
                "overall_confidence": confidence,
                "evidence": [{
                    "source": "resource_cluster",
                    "detector_version": "2.0.0",
                    "confidence": confidence,
                    "metrics": {
                        "num_clusters": len(unique_clusters),
                        "silhouette_score": metrics.get("silhouette_score", 0),
                        "num_resources": len(contents),
                        "noise_points": sum(1 for l in cluster_labels if l == -1)
                    },
                    "payload": {
                        "cluster_labels": cluster_labels.tolist(),
                        "algorithm": clustering_algorithm,
                        "eps": eps,
                        "min_samples": min_samples
                    }
                }]
            }
        
        terms_processed += 1
        if terms_processed % 100 == 0:
            logging.debug(f"Processed {terms_processed}/{len(terms)} terms")
    
    logging.info(f"Found {len(results)} ambiguous terms via embedding clustering")
    return results


def detect_with_hierarchy(
    terms: Terms,
    hierarchy: Hierarchy
) -> DetectionResults:
    """
    Detect ambiguous terms by analyzing their parent hierarchy.
    
    A term is flagged as ambiguous if its parents trace back to:
    - Different Level 0 domains (e.g., Engineering vs Arts)
    - Same Level 0 but different Level 1 domains
    
    Args:
        terms: List of terms to analyze
        hierarchy: Hierarchy data with parent relationships
        
    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    logging.info(f"Detecting ambiguity using hierarchy for {len(terms)} terms")
    
    results = {}
    terms_dict = hierarchy.get("terms", {})
    
    for term in terms:
        if term not in terms_dict:
            continue
        
        term_data = terms_dict[term]
        parents = term_data.get("parents", [])
        
        # Need multiple parents for ambiguity
        if len(parents) < 2:
            continue
        
        # Trace ancestry for each parent
        parent_contexts = []
        for parent in parents:
            ancestry = _trace_ancestry(parent, terms_dict)
            if ancestry:
                parent_contexts.append(ancestry)
        
        # Check for divergent contexts
        if _has_divergent_contexts(parent_contexts):
            # Calculate confidence based on divergence
            confidence = _calculate_hierarchy_confidence(parent_contexts)
            
            results[term] = {
                "canonical_name": term,
                "level": term_data.get("level"),
                "overall_confidence": confidence,
                "evidence": [{
                    "source": "parent_context",
                    "detector_version": "2.0.0",
                    "confidence": confidence,
                    "metrics": {
                        "num_parents": len(parents),
                        "num_contexts": len(parent_contexts),
                        "divergent": True
                    },
                    "payload": {
                        "parents": parents,
                        "parent_contexts": parent_contexts
                    }
                }]
            }
    
    logging.info(f"Found {len(results)} ambiguous terms via hierarchy analysis")
    return results


def detect_with_global_clustering(
    terms: Terms,
    web_content: WebContent,
    hierarchy: Hierarchy,
    model_name: str = "all-MiniLM-L6-v2",
    clustering_algorithm: ClusteringAlgorithm = "dbscan",
    eps: float = 0.5,
    min_samples: int = 3
) -> DetectionResults:
    """
    Detect ambiguous terms by clustering all resources globally.
    
    This approach:
    1. Pools all resources from all terms
    2. Clusters them in a shared embedding space
    3. Analyzes how each term's resources distribute across clusters
    4. Flags terms with resources spread across multiple clusters
    
    Args:
        terms: List of terms to analyze
        web_content: Web resources for each term
        hierarchy: Hierarchy data
        model_name: Sentence transformer model
        clustering_algorithm: Algorithm for clustering
        eps: DBSCAN epsilon parameter
        min_samples: Minimum samples to form a cluster
        
    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    logging.info("Detecting ambiguity using global clustering")
    
    # Load embedding model
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model {model_name}: {e}")
        return {}
    
    # Collect all resources with term mapping
    all_contents = []
    content_to_term = []
    
    for term in terms:
        if term not in web_content:
            continue
            
        term_resources = web_content[term]
        if isinstance(term_resources, dict):
            term_resources = term_resources.get("resources", [])
        
        for resource in term_resources:
            content = extract_informative_content(resource)
            if content:
                all_contents.append(content)
                content_to_term.append(term)
    
    if len(all_contents) < 10:  # Need sufficient data for global clustering
        logging.warning("Insufficient resources for global clustering")
        return {}
    
    logging.info(f"Clustering {len(all_contents)} resources globally")
    
    # Generate embeddings for all content
    embeddings = model.encode(all_contents, show_progress_bar=False)
    
    # Cluster all embeddings together
    cluster_labels, metrics = _cluster_embeddings(
        embeddings=embeddings,
        algorithm=clustering_algorithm,
        eps=eps,
        min_samples=min_samples
    )
    
    # Analyze term distribution across clusters
    term_clusters = defaultdict(list)
    for i, term in enumerate(content_to_term):
        if cluster_labels[i] != -1:  # Ignore noise
            term_clusters[term].append(cluster_labels[i])
    
    # Find ambiguous terms (resources in multiple clusters)
    results = {}
    for term, clusters in term_clusters.items():
        unique_clusters = set(clusters)
        if len(unique_clusters) >= 2:
            # Calculate distribution metrics
            cluster_counts = {c: clusters.count(c) for c in unique_clusters}
            max_cluster_ratio = max(cluster_counts.values()) / len(clusters)
            
            # Term is ambiguous if not dominated by single cluster
            if max_cluster_ratio < 0.7:  # Less than 70% in one cluster
                confidence = 1.0 - max_cluster_ratio  # Higher spread = higher confidence
                
                term_data = hierarchy.get("terms", {}).get(term, {})
                results[term] = {
                    "canonical_name": term,
                    "level": term_data.get("level"),
                    "overall_confidence": confidence,
                    "evidence": [{
                        "source": "global_cluster",
                        "detector_version": "2.0.0",
                        "confidence": confidence,
                        "metrics": {
                            "num_clusters": len(unique_clusters),
                            "max_cluster_ratio": max_cluster_ratio,
                            "total_resources": len(clusters),
                            "cluster_distribution": cluster_counts
                        },
                        "payload": {
                            "clusters": clusters,
                            "algorithm": clustering_algorithm,
                            "eps": eps,
                            "min_samples": min_samples
                        }
                    }]
                }
    
    logging.info(f"Found {len(results)} ambiguous terms via global clustering")
    return results


def merge_detection_results(
    detection_results: List[Tuple[str, DetectionResults]]
) -> DetectionResults:
    """
    Merge results from multiple detection methods.
    
    Uses noisy-OR model for confidence aggregation:
    combined_confidence = 1 - ∏(1 - confidence_i)
    
    Args:
        detection_results: List of (method_name, results) tuples
        
    Returns:
        Merged detection results with aggregated confidence
    """
    merged = {}
    
    for method, results in detection_results:
        for term, evidence_data in results.items():
            if term not in merged:
                merged[term] = {
                    "canonical_name": evidence_data["canonical_name"],
                    "level": evidence_data.get("level"),
                    "overall_confidence": 0,
                    "evidence": []
                }
            
            # Add evidence from this method
            merged[term]["evidence"].extend(evidence_data["evidence"])
    
    # Calculate aggregated confidence using noisy-OR
    for term, data in merged.items():
        confidences = [e["confidence"] for e in data["evidence"]]
        if confidences:
            # Noisy-OR: 1 - ∏(1 - c_i)
            overall = 1.0
            for conf in confidences:
                overall *= (1.0 - conf)
            data["overall_confidence"] = 1.0 - overall
    
    return merged


# Helper functions

def _cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: ClusteringAlgorithm,
    eps: float,
    min_samples: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Cluster embeddings using specified algorithm.
    
    Returns:
        Tuple of (cluster_labels, metrics_dict)
    """
    if algorithm == "hdbscan" and HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_samples,
            min_samples=min_samples
        )
        cluster_labels = clusterer.fit_predict(embeddings)
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(embeddings)
    
    # Calculate metrics
    metrics = {}
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    if n_clusters >= 2:
        # Only calculate silhouette if we have valid clusters
        non_noise_mask = cluster_labels != -1
        if sum(non_noise_mask) > 1:
            try:
                metrics["silhouette_score"] = silhouette_score(
                    embeddings[non_noise_mask],
                    cluster_labels[non_noise_mask]
                )
            except:
                metrics["silhouette_score"] = 0
    
    metrics["n_clusters"] = n_clusters
    metrics["n_noise"] = sum(1 for l in cluster_labels if l == -1)
    
    return cluster_labels, metrics


def _trace_ancestry(
    term: str,
    terms_dict: Dict[str, Any],
    max_depth: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Trace a term's ancestry up the hierarchy.
    
    Returns:
        Dictionary with level 0 and level 1 ancestors
    """
    ancestry = {"level_0": None, "level_1": None, "path": [term]}
    current = term
    depth = 0
    
    while current and depth < max_depth:
        if current not in terms_dict:
            break
            
        term_data = terms_dict[current]
        level = term_data.get("level")
        
        if level == 0:
            ancestry["level_0"] = current
            break
        elif level == 1:
            ancestry["level_1"] = current
        
        # Move up to parent
        parents = term_data.get("parents", [])
        if parents:
            current = parents[0]  # Follow first parent
            ancestry["path"].append(current)
        else:
            break
        
        depth += 1
    
    return ancestry if ancestry["level_0"] else None


def _has_divergent_contexts(
    parent_contexts: List[Dict[str, Any]]
) -> bool:
    """
    Check if parent contexts show divergence.
    
    Returns True if:
    - Parents trace to different Level 0 domains
    - Parents trace to same Level 0 but different Level 1 domains
    """
    if len(parent_contexts) < 2:
        return False
    
    level_0_set = set()
    level_1_by_0 = defaultdict(set)
    
    for context in parent_contexts:
        l0 = context.get("level_0")
        l1 = context.get("level_1")
        
        if l0:
            level_0_set.add(l0)
            if l1:
                level_1_by_0[l0].add(l1)
    
    # Different Level 0 domains
    if len(level_0_set) > 1:
        return True
    
    # Same Level 0 but different Level 1 domains
    for l0, l1_set in level_1_by_0.items():
        if len(l1_set) > 1:
            return True
    
    return False


def _calculate_hierarchy_confidence(
    parent_contexts: List[Dict[str, Any]]
) -> float:
    """
    Calculate confidence score based on hierarchy divergence.
    
    Higher divergence = higher confidence in ambiguity.
    """
    if len(parent_contexts) < 2:
        return 0.0
    
    # Count unique Level 0 and Level 1 domains
    level_0_set = set()
    level_1_set = set()
    
    for context in parent_contexts:
        if context.get("level_0"):
            level_0_set.add(context["level_0"])
        if context.get("level_1"):
            level_1_set.add(context["level_1"])
    
    # Different Level 0 = high confidence (0.9)
    if len(level_0_set) > 1:
        return 0.9
    
    # Different Level 1 = medium confidence (0.7)
    if len(level_1_set) > 1:
        return 0.7
    
    # Some divergence but less clear
    return 0.5