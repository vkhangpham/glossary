"""Confidence calculation utilities for disambiguation."""


def calculate_confidence_score(
    num_clusters: int,
    silhouette: float,
    num_resources: int
) -> float:
    """
    Calculate confidence score for ambiguity detection.

    Args:
        num_clusters: Number of clusters found
        silhouette: Silhouette score (-1 to 1)
        num_resources: Number of resources analyzed

    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence from number of clusters
    if num_clusters < 2:
        return 0.0
    elif num_clusters == 2:
        base_confidence = 0.6
    elif num_clusters == 3:
        base_confidence = 0.7
    else:
        base_confidence = 0.8

    # Adjust for silhouette score (good separation)
    if silhouette > 0:
        silhouette_boost = silhouette * 0.2  # Max 0.2 boost
    else:
        silhouette_boost = silhouette * 0.1  # Penalty for poor separation

    # Adjust for resource count (more resources = more confidence)
    if num_resources >= 10:
        resource_boost = 0.1
    elif num_resources >= 5:
        resource_boost = 0.05
    else:
        resource_boost = -0.1  # Penalty for few resources

    # Combine scores
    confidence = base_confidence + silhouette_boost + resource_boost

    # Clamp to [0, 1]
    return max(0.0, min(1.0, confidence))