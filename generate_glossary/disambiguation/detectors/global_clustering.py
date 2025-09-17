"""
Global clustering-based disambiguation across all terms.

Detects ambiguous terms by clustering resources globally to identify
when resources for a single term belong to different global topics.
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from ..types import DetectionResult, GlobalConfig
from ..utils import (
    extract_informative_content,
    calculate_confidence_score
)


# Model injection functions
def create_global_embedding_model(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """
    Create a global embedding model function for dependency injection.

    Args:
        model_name: Name of the SentenceTransformer model

    Returns:
        Function that takes list of strings and returns embeddings
    """
    model = SentenceTransformer(model_name)

    def encode_texts(texts: List[str]) -> np.ndarray:
        return model.encode(texts, show_progress_bar=False)

    return encode_texts


def with_global_embedding_model(detection_fn: Callable, model_name: str) -> Callable:
    """
    Higher-order function for global model injection.

    Args:
        detection_fn: Detection function that takes model_fn as parameter
        model_name: Name of the SentenceTransformer model

    Returns:
        Detection function with model injected
    """
    model_fn = create_global_embedding_model(model_name)

    def wrapped_detection(*args, **kwargs):
        return detection_fn(*args, model_fn=model_fn, **kwargs)

    return wrapped_detection


def create_global_detection_result(
    term: str,
    cluster_distribution: List[Dict[str, Any]],
    confidence: float,
    evidence: Dict[str, Any]
) -> DetectionResult:
    """
    Create a DetectionResult object for global clustering-based detection.

    Args:
        term: The term being analyzed
        cluster_distribution: Distribution of term resources across global clusters
        confidence: Confidence score for the detection
        evidence: Evidence supporting the detection

    Returns:
        DetectionResult object
    """
    return DetectionResult(
        term=term,
        method="global",
        confidence=confidence,
        evidence=evidence,
        clusters=cluster_distribution,
        metadata={}
    )


# Pure helper functions
def _collect_all_resources_pure(
    terms: List[str],
    web_content: Dict[str, Any],
    config: GlobalConfig
) -> Tuple[List[str], List[str], Dict[str, List[int]]]:
    """
    Collect all resources with term mappings (pure function with deterministic sampling).

    Args:
        terms: List of terms to collect resources for
        web_content: Web content for each term
        config: Configuration with resource limits

    Returns:
        Tuple of (all_contents, resource_to_term, term_resource_indices)
    """
    all_contents = []
    resource_to_term = []
    term_resource_indices = defaultdict(list)

    for term in terms:
        if term not in web_content:
            continue

        term_resources = web_content[term]
        if isinstance(term_resources, dict):
            term_resources = term_resources.get("resources", [])

        if len(term_resources) < config.min_resources:
            continue

        # Deterministic sampling - take first N resources instead of random sampling
        if len(term_resources) > config.max_resources_per_term:
            term_resources = term_resources[:config.max_resources_per_term]

        # Extract content
        for resource in term_resources:
            content = extract_informative_content(resource)
            if content:
                idx = len(all_contents)
                all_contents.append(content)
                resource_to_term.append(term)
                term_resource_indices[term].append(idx)

    return all_contents, resource_to_term, dict(term_resource_indices)


def _perform_global_clustering_pure(
    all_embeddings: np.ndarray,
    config: GlobalConfig
) -> Tuple[np.ndarray, float, int]:
    """
    Perform global clustering on all embeddings (pure function).

    Args:
        all_embeddings: All resource embeddings
        config: Configuration with clustering parameters

    Returns:
        Tuple of (cluster_labels, silhouette_score, num_unique_clusters)
    """
    # Perform global clustering
    clusterer = DBSCAN(eps=config.eps, min_samples=config.min_samples)
    global_clusters = clusterer.fit_predict(all_embeddings)

    # Calculate global silhouette score
    unique_clusters = set(c for c in global_clusters if c != -1)
    num_unique_clusters = len(unique_clusters)

    global_silhouette = 0.0
    if num_unique_clusters > 1:
        try:
            global_silhouette = silhouette_score(all_embeddings, global_clusters)
        except:
            pass

    return global_clusters, global_silhouette, num_unique_clusters


def _analyze_term_distribution_pure(
    term_indices: List[int],
    global_clusters: np.ndarray,
    all_contents: List[str],
    resource_to_term: List[str]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Analyze term distribution across global clusters (pure function).

    Args:
        term_indices: Indices of resources for a specific term
        global_clusters: Cluster assignments for all resources
        all_contents: All resource contents
        resource_to_term: Mapping from resource index to term

    Returns:
        Tuple of (cluster_info, num_clusters, total_clustered)
    """
    term_clusters = [global_clusters[i] for i in term_indices]

    # Count cluster distribution
    cluster_counts = defaultdict(int)
    for cluster_id in term_clusters:
        if cluster_id != -1:  # Ignore noise
            cluster_counts[cluster_id] += 1

    if len(cluster_counts) <= 1:
        return [], len(cluster_counts), sum(cluster_counts.values())

    # Calculate cluster distribution
    total_clustered = sum(cluster_counts.values())
    cluster_info = []

    # Convert to set for O(1) lookup performance
    term_indices_set = set(term_indices)

    for cluster_id, count in cluster_counts.items():
        # Find topic keywords for this cluster - iterate only term indices
        cluster_resources = [
            all_contents[i] for i in term_indices_set
            if global_clusters[i] == cluster_id
        ]

        cluster_info.append({
            "global_cluster_id": int(cluster_id),
            "resource_count": count,
            "percentage": count / total_clustered,
            "sample_content": cluster_resources[:2]  # Sample content
        })

    # Sort by resource count
    cluster_info.sort(key=lambda x: x["resource_count"], reverse=True)

    return cluster_info, len(cluster_counts), total_clustered


def _calculate_global_confidence_pure(
    num_global_clusters: int,
    cluster_distribution: List[Dict],
    global_silhouette: float
) -> float:
    """
    Calculate confidence for global clustering detection (pure function).

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
    max_percentage = max(percentages) if percentages else 1.0

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


def detect_global_ambiguity(
    terms: List[str],
    web_content: Dict[str, Any],
    config: GlobalConfig,
    model_fn: Callable[[List[str]], np.ndarray]
) -> List[DetectionResult]:
    """
    Pure functional detection of ambiguous terms using global clustering.

    This approach:
    1. Pools resources from all terms
    2. Clusters them globally to find topics using injected model function
    3. Identifies terms whose resources span multiple global clusters

    Args:
        terms: List of terms to analyze
        web_content: Web resources for each term
        config: GlobalConfig with detection parameters
        model_fn: Injected function for generating embeddings

    Returns:
        List of DetectionResult objects for ambiguous terms
    """
    # Collect all resources with term mapping
    all_contents, resource_to_term, term_resource_indices = _collect_all_resources_pure(
        terms, web_content, config
    )

    if len(all_contents) < config.min_total_resources:  # Need enough for meaningful global clustering
        return []

    # Generate embeddings for all content using injected model function
    try:
        all_embeddings = model_fn(all_contents)
    except Exception:
        # Skip on embedding failure - no side effects
        return []

    # Perform global clustering
    global_clusters, global_silhouette, num_unique_clusters = _perform_global_clustering_pure(
        all_embeddings, config
    )

    # Analyze term distribution across global clusters
    results = []

    for term in term_resource_indices:
        term_indices = term_resource_indices[term]

        # Analyze cluster distribution for this term
        cluster_info, num_clusters, total_clustered = _analyze_term_distribution_pure(
            term_indices, global_clusters, all_contents, resource_to_term
        )

        # Check if term spans multiple clusters
        if num_clusters > 1:
            # Calculate confidence
            confidence = _calculate_global_confidence_pure(
                num_global_clusters=num_clusters,
                cluster_distribution=cluster_info,
                global_silhouette=global_silhouette
            )

            # Create evidence dictionary
            evidence = {
                "global_clusters_total": num_unique_clusters,
                "global_silhouette": global_silhouette,
                "eps": config.eps,
                "min_samples": config.min_samples,
                "model": config.model_name,
                "num_global_clusters": num_clusters,
                "total_resources": len(term_indices),
                "clustered_resources": total_clustered
            }

            # Create detection result
            detection_result = create_global_detection_result(
                term=term,
                cluster_distribution=cluster_info,
                confidence=confidence,
                evidence=evidence
            )

            results.append(detection_result)

    return results


def detect(
    terms: List[str],
    web_content: Dict[str, Any],
    hierarchy: Dict[str, Any],
    *,
    config: Optional[GlobalConfig] = None,
    model_name: str = "all-MiniLM-L6-v2",
    eps: float = 0.3,
    min_samples: int = 3,
    min_resources: int = 5,
    max_resources_per_term: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Legacy detect function for backward compatibility (DEPRECATED).

    This function maintains backward compatibility with the original API
    while internally using the new pure functional detection.

    Args:
        terms: List of terms to analyze
        web_content: Web resources for each term
        hierarchy: Hierarchy data (unused but kept for compatibility)
        config: Optional GlobalConfig (keyword-only)
        model_name: Sentence transformer model
        eps: DBSCAN epsilon for tighter global clustering
        min_samples: Minimum samples for global clusters
        min_resources: Minimum resources per term
        max_resources_per_term: Maximum resources to use per term

    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    # Issue deprecation warning
    warnings.warn(
        "The detect() function is deprecated. Use detect_global_ambiguity() with "
        "dependency injection for better functional programming patterns.",
        DeprecationWarning,
        stacklevel=2
    )

    # Handle configuration - use config object if provided, fallback to individual parameters
    if config is not None:
        final_config = config
    else:
        final_config = GlobalConfig(
            model_name=model_name,
            eps=eps,
            min_samples=min_samples,
            min_resources=min_resources,
            max_resources_per_term=max_resources_per_term
        )

    # Load embedding model and create model function
    try:
        model_fn = create_global_embedding_model(final_config.model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model {final_config.model_name}: {e}")
        return {}

    # Call pure functional detection
    detection_results = detect_global_ambiguity(
        terms=terms,
        web_content=web_content,
        config=final_config,
        model_fn=model_fn
    )

    # Convert DetectionResult objects back to dictionary format for backward compatibility
    results = {}
    for result in detection_results:
        cluster_info = list(result.clusters) if result.clusters else []

        results[result.term] = {
            "term": result.term,
            "method": result.method,
            "num_global_clusters": result.evidence.get("num_global_clusters", 0),
            "cluster_distribution": cluster_info,
            "total_resources": result.evidence.get("total_resources", 0),
            "clustered_resources": result.evidence.get("clustered_resources", 0),
            "confidence": result.confidence,
            "evidence": dict(result.evidence)
        }

    return results


def calculate_global_confidence(
    num_global_clusters: int,
    cluster_distribution: List[Dict],
    global_silhouette: float
) -> float:
    """
    Legacy function for backward compatibility.
    """
    return _calculate_global_confidence_pure(
        num_global_clusters, cluster_distribution, global_silhouette
    )