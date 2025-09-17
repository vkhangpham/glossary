"""Clustering utilities for disambiguation."""

from typing import List, Optional


def calculate_separation_score(
    clusters: List[List[float]],
    embeddings: Optional[List[List[float]]] = None
) -> float:
    """
    Calculate separation score between clusters.

    Args:
        clusters: List of cluster centroids or cluster assignments
        embeddings: Optional embeddings for detailed calculation

    Returns:
        Separation score between 0 and 1
    """
    if len(clusters) < 2:
        return 0.0

    # Simple implementation: use cluster count ratio
    # More clusters with even distribution = higher separation
    cluster_sizes = [len(c) if isinstance(c, list) else 1 for c in clusters]
    total = sum(cluster_sizes)

    if total == 0:
        return 0.0

    # Calculate distribution evenness
    max_size = max(cluster_sizes)
    evenness = 1.0 - (max_size / total - 1.0 / len(clusters))

    # More clusters = higher base score
    cluster_score = min(1.0, len(clusters) / 4.0)

    # Combine scores
    separation = (evenness + cluster_score) / 2.0

    return max(0.0, min(1.0, separation))