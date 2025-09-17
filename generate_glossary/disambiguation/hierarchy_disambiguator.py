"""
Hierarchy-based disambiguation using parent context analysis.

Detects ambiguous terms by analyzing their relationships with parent
terms in the hierarchy to identify divergent contexts.
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .types import DetectionResult, HierarchyConfig
from .utils import calculate_confidence_score


def create_hierarchy_detection_result(
    term: str,
    parents: List[Dict[str, Any]],
    divergence_evidence: Dict[str, Any],
    confidence: float
) -> DetectionResult:
    """
    Create a DetectionResult object for hierarchy-based detection.

    Args:
        term: The term being analyzed
        parents: List of parent information
        divergence_evidence: Evidence of parent context divergence
        confidence: Confidence score for the detection

    Returns:
        DetectionResult object
    """
    return DetectionResult(
        term=term,
        method="hierarchy",
        confidence=confidence,
        evidence=divergence_evidence,
        clusters=None,
        metadata={"parents": parents}
    )


# Pure helper functions
def _build_parent_mappings_pure(hierarchy: Dict[str, Any]) -> Tuple[Dict[str, set], Dict[str, int]]:
    """
    Build parent-child mappings from hierarchy data (pure function).

    Args:
        hierarchy: Hierarchy data with parent-child relationships

    Returns:
        Tuple of (parent_map, level_map)
    """
    parent_map = defaultdict(set)
    level_map = {}

    for level_idx, level_data in enumerate(hierarchy.get("levels", [])):
        for term_data in level_data.get("terms", []):
            term = term_data.get("term")
            if not term:
                continue

            level_map[term] = level_idx

            # Track parents
            parents = term_data.get("parents", [])
            if isinstance(parents, str):
                parents = [parents]
            for parent in parents:
                if parent:
                    parent_map[term].add(parent)

    return dict(parent_map), level_map


def _extract_parent_context_pure(
    parent: str,
    hierarchy: Dict[str, Any],
    web_content: Optional[Dict[str, Any]],
    config: HierarchyConfig
) -> Optional[Dict[str, Any]]:
    """
    Extract context information for a parent term (pure function).

    Args:
        parent: Parent term to analyze
        hierarchy: Hierarchy data
        web_content: Optional web resources
        config: Configuration with web enhancement setting

    Returns:
        Context dictionary with keywords and relationships
    """
    context = {
        "term": parent,
        "keywords": set(),
        "children": set(),
        "level": -1
    }

    # Find parent in hierarchy
    for level_idx, level_data in enumerate(hierarchy.get("levels", [])):
        for term_data in level_data.get("terms", []):
            if term_data.get("term") == parent:
                context["level"] = level_idx

                # Extract keywords from metadata
                metadata = term_data.get("metadata", {})
                if metadata:
                    # Add keywords from various metadata fields
                    for field in ["keywords", "topics", "areas", "domains"]:
                        if field in metadata:
                            values = metadata[field]
                            if isinstance(values, list):
                                context["keywords"].update(values)
                            elif isinstance(values, str):
                                context["keywords"].add(values)

                # Track children
                children = term_data.get("children", [])
                if isinstance(children, list):
                    context["children"].update(children)

                break

    # Enhance with web content if available and enabled
    if config.enable_web_enhancement and web_content and parent in web_content:
        resources = web_content[parent]
        if isinstance(resources, dict):
            resources = resources.get("resources", [])

        # Extract keywords from resource titles/descriptions
        for resource in resources[:5]:  # Sample first 5
            if isinstance(resource, dict):
                title = resource.get("title", "")
                description = resource.get("description", "")

                # Simple keyword extraction (could be enhanced)
                text = f"{title} {description}".lower()
                words = text.split()
                # Filter to meaningful words (length > 3, not stopwords)
                keywords = {w for w in words if len(w) > 3}
                context["keywords"].update(keywords)

    return context if context["keywords"] else None


def _analyze_context_divergence_pure(
    parent_contexts: Dict[str, Dict],
    config: HierarchyConfig
) -> Dict[str, Any]:
    """
    Analyze divergence between parent contexts (pure function).

    Args:
        parent_contexts: Context data for each parent
        config: Configuration with overlap and similarity thresholds

    Returns:
        Evidence of context divergence
    """
    parents = list(parent_contexts.keys())

    if len(parents) < 2:
        return {"is_divergent": False, "divergence_score": 0.0}

    # Calculate pairwise similarities
    similarities = []
    divergent_pairs = []

    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            p1, p2 = parents[i], parents[j]
            ctx1, ctx2 = parent_contexts[p1], parent_contexts[p2]

            # Calculate keyword overlap
            keywords1 = ctx1["keywords"]
            keywords2 = ctx2["keywords"]

            if not keywords1 or not keywords2:
                continue

            intersection = keywords1 & keywords2
            union = keywords1 | keywords2

            overlap = len(intersection) / len(union) if union else 0
            similarities.append(overlap)

            # Check if this pair shows divergence
            if overlap < config.min_parent_overlap:
                divergent_pairs.append({
                    "parent1": p1,
                    "parent2": p2,
                    "overlap": overlap,
                    "unique_to_p1": keywords1 - keywords2,
                    "unique_to_p2": keywords2 - keywords1
                })

    if not similarities:
        return {"is_divergent": False, "divergence_score": 0.0}

    # Calculate divergence score
    avg_similarity = sum(similarities) / len(similarities)
    divergence_score = 1.0 - avg_similarity

    # Determine if contexts are divergent
    is_divergent = (
        avg_similarity < config.max_parent_similarity and
        len(divergent_pairs) > 0
    )

    return {
        "is_divergent": is_divergent,
        "divergence_score": divergence_score,
        "avg_similarity": avg_similarity,
        "divergent_pairs": divergent_pairs,
        "num_comparisons": len(similarities)
    }


def _calculate_hierarchy_confidence_pure(
    num_parents: int,
    divergence_score: float,
    level: int
) -> float:
    """
    Calculate confidence score for hierarchy-based detection (pure function).

    Args:
        num_parents: Number of parent terms
        divergence_score: Context divergence score (0-1)
        level: Hierarchy level

    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence from divergence
    base_confidence = divergence_score

    # Boost for multiple parents
    parent_boost = min(0.2, (num_parents - 2) * 0.1)

    # Level-based adjustment (higher levels more likely to be ambiguous)
    level_factor = 1.0
    if level >= 0:
        level_factor = 1.0 + (level * 0.05)

    confidence = min(1.0, base_confidence * level_factor + parent_boost)

    return confidence


def detect_hierarchy_ambiguity(
    terms: List[str],
    web_content: Optional[Dict[str, Any]],
    hierarchy: Dict[str, Any],
    config: HierarchyConfig
) -> List[DetectionResult]:
    """
    Pure functional detection of ambiguous terms using hierarchy analysis.

    A term is considered ambiguous if:
    1. It appears under multiple parents
    2. Those parents have low contextual overlap
    3. The term's usage differs across parent contexts

    Args:
        terms: List of terms to analyze
        web_content: Optional web resources for enhanced analysis
        hierarchy: Hierarchy data with parent-child relationships
        config: HierarchyConfig with detection parameters

    Returns:
        List of DetectionResult objects for ambiguous terms
    """
    # Build parent-child mappings
    parent_map, level_map = _build_parent_mappings_pure(hierarchy)

    results = []

    # Analyze each term
    for term in terms:
        if term not in parent_map:
            continue

        parents = parent_map[term]
        if len(parents) < 2:
            continue

        # Analyze parent contexts
        parent_contexts = {}
        for parent in parents:
            context = _extract_parent_context_pure(parent, hierarchy, web_content, config)
            if context:
                parent_contexts[parent] = context

        if len(parent_contexts) < 2:
            continue

        # Check for divergent contexts
        divergence_evidence = _analyze_context_divergence_pure(parent_contexts, config)

        if divergence_evidence["is_divergent"]:
            # Calculate confidence
            confidence = _calculate_hierarchy_confidence_pure(
                num_parents=len(parents),
                divergence_score=divergence_evidence["divergence_score"],
                level=level_map.get(term, -1)
            )

            # Create evidence dictionary
            evidence = {
                "num_parents": len(parents),
                "parent_contexts": parent_contexts,
                "min_overlap": config.min_parent_overlap,
                "max_similarity": config.max_parent_similarity,
                "level": level_map.get(term, -1),
                "divergence_score": divergence_evidence["divergence_score"],
                "avg_similarity": divergence_evidence.get("avg_similarity", 0.0),
                "divergent_pairs": divergence_evidence.get("divergent_pairs", []),
                "num_comparisons": divergence_evidence.get("num_comparisons", 0)
            }

            # Create parent information for metadata
            parent_info = [{
                "parent": parent,
                "level": parent_contexts[parent]["level"],
                "keywords": list(parent_contexts[parent]["keywords"]),
                "children": list(parent_contexts[parent]["children"])
            } for parent in parent_contexts]

            # Create detection result
            detection_result = create_hierarchy_detection_result(
                term=term,
                parents=parent_info,
                divergence_evidence=evidence,
                confidence=confidence
            )

            results.append(detection_result)

    return results


def detect(
    terms: List[str],
    web_content: Optional[Dict[str, Any]],
    hierarchy: Dict[str, Any],
    *,
    config: Optional[HierarchyConfig] = None,
    min_parent_overlap: float = 0.3,
    max_parent_similarity: float = 0.7
) -> Dict[str, Dict[str, Any]]:
    """
    Legacy detect function for backward compatibility (DEPRECATED).

    This function maintains backward compatibility with the original API
    while internally using the new pure functional detection.

    Args:
        terms: List of terms to analyze
        web_content: Optional web resources for enhanced analysis
        hierarchy: Hierarchy data with parent-child relationships
        config: Optional HierarchyConfig (keyword-only)
        min_parent_overlap: Minimum keyword overlap to consider parents related
        max_parent_similarity: Maximum similarity to consider contexts different

    Returns:
        Dictionary mapping ambiguous terms to their detection evidence
    """
    # Issue deprecation warning
    warnings.warn(
        "The detect() function is deprecated. Use detect_hierarchy_ambiguity() "
        "for better functional programming patterns.",
        DeprecationWarning,
        stacklevel=2
    )

    # Handle configuration - use config object if provided, fallback to individual parameters
    if config is not None:
        final_config = config
    else:
        final_config = HierarchyConfig(
            min_parent_overlap=min_parent_overlap,
            max_parent_similarity=max_parent_similarity,
            enable_web_enhancement=True  # Default value
        )

    # Call pure functional detection
    detection_results = detect_hierarchy_ambiguity(
        terms=terms,
        web_content=web_content,
        hierarchy=hierarchy,
        config=final_config
    )

    # Convert DetectionResult objects back to dictionary format for backward compatibility
    results = {}
    for result in detection_results:
        parent_info = result.metadata.get("parents", [])

        results[result.term] = {
            "term": result.term,
            "method": result.method,
            "parents": [p["parent"] for p in parent_info],
            "level": result.evidence.get("level", -1),
            "divergence_evidence": {
                "is_divergent": True,  # Only included if divergent
                "divergence_score": result.evidence.get("divergence_score", 0.0),
                "avg_similarity": result.evidence.get("avg_similarity", 0.0),
                "divergent_pairs": result.evidence.get("divergent_pairs", []),
                "num_comparisons": result.evidence.get("num_comparisons", 0)
            },
            "confidence": result.confidence,
            "evidence": dict(result.evidence)
        }

    logging.info(f"Found {len(results)} ambiguous terms via hierarchy")
    return results


# Legacy function aliases for backward compatibility
def extract_parent_context(
    parent: str,
    hierarchy: Dict[str, Any],
    web_content: Optional[Dict[str, Any]],
    enable_web_enhancement: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    """
    config = HierarchyConfig(enable_web_enhancement=enable_web_enhancement)
    return _extract_parent_context_pure(parent, hierarchy, web_content, config)


def analyze_context_divergence(
    parent_contexts: Dict[str, Dict],
    min_overlap: float,
    max_similarity: float
) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    """
    config = HierarchyConfig(
        min_parent_overlap=min_overlap,
        max_parent_similarity=max_similarity
    )
    return _analyze_context_divergence_pure(parent_contexts, config)


def calculate_hierarchy_confidence(
    num_parents: int,
    divergence_score: float,
    level: int
) -> float:
    """
    Legacy function for backward compatibility.
    """
    return _calculate_hierarchy_confidence_pure(num_parents, divergence_score, level)