"""
Embedding-based disambiguation using semantic clustering.

Detects ambiguous terms by clustering resource embeddings to find
when web content about a term forms distinct semantic groups.
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

from ..types import DetectionResult, EmbeddingConfig
from ..utils import (
    extract_informative_content,
    calculate_confidence_score
)

# Try to import HDBSCAN if available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# Model injection functions
def create_embedding_model(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """
    Create an embedding model function for dependency injection.

    Args:
        model_name: Name of the SentenceTransformer model

    Returns:
        Function that takes list of strings and returns embeddings
    """
    model = SentenceTransformer(model_name)

    def encode_texts(texts: List[str]) -> np.ndarray:
        return model.encode(texts, show_progress_bar=False)

    return encode_texts


def with_embedding_model(detection_fn: Callable, model_name: str) -> Callable:
    """
    Higher-order function for model injection.

    Args:
        detection_fn: Detection function that takes model_fn as parameter
        model_name: Name of the SentenceTransformer model

    Returns:
        Detection function with model injected
    """
    model_fn = create_embedding_model(model_name)

    def wrapped_detection(*args, **kwargs):
        return detection_fn(*args, model_fn=model_fn, **kwargs)

    return wrapped_detection


def create_detection_result(term: str, clusters: List[Dict[str, Any]], confidence: float, evidence: Dict[str, Any]) -> DetectionResult:
    """
    Create a DetectionResult object for embedding-based detection.

    Args:
        term: The term being analyzed
        clusters: List of cluster information dictionaries
        confidence: Confidence score for the detection
        evidence: Evidence supporting the detection

    Returns:
        DetectionResult object
    """
    return DetectionResult(
        term=term,
        method="embedding",
        confidence=confidence,
        evidence=evidence,
        clusters=clusters,
        metadata={}
    )


# Pure helper functions
def _extract_term_contents(terms: List[str], web_content: Dict[str, Any], min_resources: int) -> Tuple[Dict[str, List[str]], Dict[str, List[Any]]]:
    """
    Extract content from term resources (pure function).

    Args:
        terms: List of terms to extract content for
        web_content: Web resources for each term
        min_resources: Minimum resources required

    Returns:
        Tuple of (term_contents, term_filtered_resources):
        - term_contents: Dictionary mapping terms to their content lists
        - term_filtered_resources: Dictionary mapping terms to their filtered resources aligned with contents
    """
    term_contents = {}
    term_filtered_resources = {}

    for term in terms:
        if term not in web_content:
            continue

        term_resources = web_content[term]
        if isinstance(term_resources, dict):
            term_resources = term_resources.get("resources", [])

        if len(term_resources) < min_resources:
            continue

        # Extract content from resources and keep aligned filtered resources
        contents = []
        filtered_resources = []
        for resource in term_resources:
            content = extract_informative_content(resource)
            if content:
                contents.append(content)
                filtered_resources.append(resource)

        if len(contents) >= min_resources:
            term_contents[term] = contents
            term_filtered_resources[term] = filtered_resources

    return term_contents, term_filtered_resources


def _cluster_embeddings_pure(embeddings: np.ndarray, algorithm: str, eps: float, min_samples: int) -> Tuple[np.ndarray, float, str]:
    """
    Cluster embeddings using specified algorithm (pure function).

    Args:
        embeddings: Embedding vectors to cluster
        algorithm: Clustering algorithm to use
        eps: DBSCAN epsilon parameter
        min_samples: Minimum samples for a cluster

    Returns:
        Tuple of (cluster_labels, silhouette_score, algorithm_used)
    """
    if algorithm == "hdbscan" and HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_samples,
            min_samples=min_samples,
            cluster_selection_epsilon=eps
        )
        clusters = clusterer.fit_predict(embeddings)
        algorithm_used = "hdbscan"
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = clusterer.fit_predict(embeddings)
        algorithm_used = "dbscan"

    # Calculate silhouette score if we have clusters
    silhouette = 0.0
    unique_clusters = set(clusters)
    if len(unique_clusters) > 1 and -1 not in unique_clusters:
        try:
            silhouette = silhouette_score(embeddings, clusters)
        except:
            silhouette = 0.0

    return clusters, silhouette, algorithm_used


def _analyze_clusters_pure(clusters: np.ndarray, term_resources: List[Any]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Analyze cluster results to extract cluster information (pure function).

    Args:
        clusters: Cluster labels for each resource
        term_resources: Original resources for the term

    Returns:
        Tuple of (cluster_info_list, num_unique_clusters)
    """
    unique_clusters = set(c for c in clusters if c != -1)
    num_unique_clusters = len(unique_clusters)

    # Calculate total clustered points (excluding noise)
    total_clustered = sum(1 for c in clusters if c != -1)

    cluster_info = []
    for cluster_id in unique_clusters:
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_size = len(cluster_indices)

        # Get sample resources from this cluster
        sample_resources = [term_resources[i] for i in cluster_indices[:3]]

        # Use total_clustered as denominator for percentages
        percentage = cluster_size / total_clustered if total_clustered > 0 else 0

        cluster_info.append({
            "cluster_id": int(cluster_id),
            "size": cluster_size,
            "percentage": percentage,
            "sample_resources": sample_resources
        })

    return cluster_info, num_unique_clusters


def detect_embedding_ambiguity(
    terms: List[str],
    web_content: Dict[str, Any],
    config: EmbeddingConfig,
    model_fn: Callable[[List[str]], np.ndarray]
) -> List[DetectionResult]:
    """
    Pure functional detection of ambiguous terms using embedding clustering.

    This function:
    1. Extracts content from each term's resources
    2. Generates embeddings for the content using injected model function
    3. Clusters embeddings to find semantic groups
    4. Flags terms with multiple clusters as ambiguous

    Args:
        terms: List of terms to analyze
        web_content: Web resources for each term
        config: EmbeddingConfig with detection parameters
        model_fn: Injected function for generating embeddings

    Returns:
        List of DetectionResult objects for ambiguous terms
    """
    # Extract content for all terms and get filtered resources aligned with contents
    term_contents, term_filtered_resources = _extract_term_contents(terms, web_content, config.min_resources)

    results = []

    for term, contents in term_contents.items():
        # Get filtered resources aligned with contents
        filtered_resources = term_filtered_resources[term]

        # Generate embeddings using injected model function
        try:
            embeddings = model_fn(contents)
        except Exception:
            # Skip on embedding failure - no side effects
            continue

        # Cluster embeddings
        clusters, silhouette, algorithm_used = _cluster_embeddings_pure(
            embeddings,
            config.clustering_algorithm,
            config.eps,
            config.min_samples
        )

        # Analyze clusters with filtered resources
        cluster_info, num_unique_clusters = _analyze_clusters_pure(clusters, filtered_resources)

        # Check if ambiguous (multiple clusters)
        if num_unique_clusters > 1:
            # Calculate confidence based on clustering quality
            confidence = calculate_confidence_score(
                num_clusters=num_unique_clusters,
                silhouette=silhouette,
                num_resources=len(contents)
            )

            # Create evidence dictionary
            evidence = {
                "clustering_algorithm": algorithm_used,
                "eps": config.eps,
                "min_samples": config.min_samples,
                "model": config.model_name,
                "largest_cluster_ratio": max(c["percentage"] for c in cluster_info),
                "num_clusters": num_unique_clusters,
                "silhouette_score": silhouette,
                "total_resources": len(contents)
            }

            # Create detection result
            detection_result = create_detection_result(
                term=term,
                clusters=cluster_info,
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
    config: Optional[EmbeddingConfig] = None,
    model_name: str = "all-MiniLM-L6-v2",
    clustering_algorithm: str = "dbscan",
    eps: float = 0.45,
    min_samples: int = 2,
    min_resources: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Legacy detect function for backward compatibility (DEPRECATED).

    This function maintains backward compatibility with the original API
    while internally using the new pure functional detection.

    Args:
        terms: List of terms to analyze
        web_content: Web resources for each term
        hierarchy: Hierarchy data (unused but kept for compatibility)
        config: Optional EmbeddingConfig (keyword-only)
        model_name: Sentence transformer model to use
        clustering_algorithm: Algorithm for clustering ('dbscan' or 'hdbscan')
        eps: DBSCAN epsilon parameter (max distance between samples)
        min_samples: Minimum samples to form a cluster
        min_resources: Minimum resources required for analysis

    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    # Issue deprecation warning
    warnings.warn(
        "The detect() function is deprecated. Use detect_embedding_ambiguity() with "
        "dependency injection for better functional programming patterns.",
        DeprecationWarning,
        stacklevel=2
    )

    # Handle configuration - use config object if provided, fallback to individual parameters
    if config is not None:
        final_config = config
    else:
        final_config = EmbeddingConfig(
            model_name=model_name,
            clustering_algorithm=clustering_algorithm,
            eps=eps,
            min_samples=min_samples,
            min_resources=min_resources
        )

    # Load embedding model and create model function
    try:
        model_fn = create_embedding_model(final_config.model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model {final_config.model_name}: {e}")
        return {}

    # Call pure functional detection
    detection_results = detect_embedding_ambiguity(
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
            "num_clusters": result.evidence.get("num_clusters", 0),
            "clusters": cluster_info,
            "silhouette_score": result.evidence.get("silhouette_score", 0.0),
            "confidence": result.confidence,
            "total_resources": result.evidence.get("total_resources", 0),
            "evidence": dict(result.evidence)
        }

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
    clusters, silhouette, _ = _cluster_embeddings_pure(embeddings, algorithm, eps, min_samples)
    return clusters, silhouette