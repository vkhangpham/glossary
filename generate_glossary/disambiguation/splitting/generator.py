"""
Split proposal generation functionality.

Generates semantic tags and split proposals for ambiguous terms.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict

from generate_glossary.llm import completion
from ..types import (
    DetectionResult,
    SplitProposal,
    SplittingConfig,
    LLMFunction,
    TagGeneratorFunction
)

# Type aliases for legacy compatibility
DetectionResults = Dict[str, Dict[str, Any]]
SplitProposals = List[Dict[str, Any]]  # Legacy format
WebContent = Dict[str, Any]
Hierarchy = Dict[str, Any]


def generate_split_proposals(
    detection_results: List[DetectionResult],
    hierarchy: Dict[str, Any],
    config: SplittingConfig,
    llm_fn: Optional[LLMFunction] = None
) -> List[SplitProposal]:
    """
    Generate sense split proposals for ambiguous terms (pure functional implementation).

    This function:
    1. Groups resources by cluster for each ambiguous term
    2. Generates semantic tags for each cluster
    3. Creates split proposals with meaningful names

    Args:
        detection_results: Detection results with cluster information
        hierarchy: Hierarchy data
        config: Splitting configuration
        llm_fn: Optional LLM function for tag generation

    Returns:
        List of immutable SplitProposal objects
    """
    proposals = []

    for detection_result in detection_results:
        # Extract cluster information from evidence
        cluster_info = _extract_cluster_info_pure(detection_result)
        if not cluster_info:
            continue

        # Get term and level
        term = detection_result.term
        level = detection_result.level

        # Group resources by cluster
        cluster_resources = _group_resources_by_cluster_pure(
            term=term,
            cluster_labels=cluster_info["labels"],
            web_content=detection_result.evidence.get("web_content")
        )

        # Skip if not enough distinct clusters
        if len(cluster_resources) < config.min_cluster_size:
            continue

        # Generate tags for each cluster
        if llm_fn and config.use_llm:
            tag_generator_fn = create_tag_generator(llm_fn, config)
            sense_tags = _generate_sense_tags_with_llm_pure(
                term=term,
                cluster_resources=cluster_resources,
                level=level,
                tag_generator_fn=tag_generator_fn
            )
        else:
            sense_tags = _generate_sense_tags_fallback_pure(
                term=term,
                cluster_resources=cluster_resources
            )

        # Create split proposal
        proposal = _create_split_proposal_pure(
            term=term,
            level=level,
            cluster_resources=cluster_resources,
            sense_tags=sense_tags,
            confidence=detection_result.confidence
        )

        proposals.append(proposal)

    return proposals


def create_tag_generator(llm_fn: LLMFunction, config: SplittingConfig) -> TagGeneratorFunction:
    """Create a tag generator function with given LLM function and config."""
    def tag_generator(term: str, resources: List[Dict], level: int) -> str:
        return _generate_tag_with_llm_pure(term, resources, level, llm_fn, config)
    return tag_generator


def with_llm_function(llm_provider: str = "gemini") -> LLMFunction:
    """Create an LLM function wrapper for the given provider."""
    def llm_fn(messages: List[Dict[str, Any]]) -> str:
        return completion(messages, model_provider=llm_provider)
    return llm_fn


def _extract_cluster_info_pure(detection_result: DetectionResult) -> Optional[Dict[str, Any]]:
    """
    Extract cluster information from detection result evidence (pure function).

    Args:
        detection_result: Detection result with evidence

    Returns:
        Dictionary with cluster labels and other info, or None if no clusters
    """
    evidence = detection_result.evidence

    # Check for embedding-based clustering results
    if "clusters" in evidence:
        clusters = evidence["clusters"]
        if isinstance(clusters, dict) and "labels" in clusters:
            return {
                "labels": clusters["labels"],
                "n_clusters": clusters.get("n_clusters", 0),
                "method": "embedding"
            }

    # Check for hierarchy-based clustering
    if "hierarchy_clusters" in evidence:
        hierarchy_clusters = evidence["hierarchy_clusters"]
        if isinstance(hierarchy_clusters, list) and len(hierarchy_clusters) > 1:
            return {
                "labels": list(range(len(hierarchy_clusters))),
                "n_clusters": len(hierarchy_clusters),
                "method": "hierarchy",
                "cluster_data": hierarchy_clusters
            }

    # Check for global clustering results
    if "global_clusters" in evidence:
        global_clusters = evidence["global_clusters"]
        if isinstance(global_clusters, dict) and "labels" in global_clusters:
            return {
                "labels": global_clusters["labels"],
                "n_clusters": global_clusters.get("n_clusters", 0),
                "method": "global"
            }

    return None


def _group_resources_by_cluster_pure(
    term: str,
    cluster_labels: List[int],
    web_content: Optional[Dict[str, Any]] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group resources by cluster labels (pure function).

    Args:
        term: The ambiguous term
        cluster_labels: List of cluster labels for each resource
        web_content: Optional web content data

    Returns:
        Dictionary mapping cluster ID to list of resources
    """
    cluster_resources = defaultdict(list)

    if not web_content or "resources" not in web_content:
        logging.warning(f"No web content resources found for term: {term}")
        return dict(cluster_resources)

    resources = web_content["resources"]

    # Ensure we have matching number of labels and resources
    min_length = min(len(cluster_labels), len(resources))
    if min_length == 0:
        return dict(cluster_resources)

    # Group resources by cluster
    for i in range(min_length):
        cluster_id = cluster_labels[i]
        resource = resources[i]
        cluster_resources[cluster_id].append(resource)

    return dict(cluster_resources)


def _generate_sense_tags_with_llm_pure(
    term: str,
    cluster_resources: Dict[int, List[Dict]],
    level: int,
    tag_generator_fn: TagGeneratorFunction
) -> Dict[int, str]:
    """
    Generate semantic tags using LLM for each cluster (pure function).

    Args:
        term: The ambiguous term
        cluster_resources: Resources grouped by cluster
        level: Hierarchy level
        tag_generator_fn: Function to generate tags

    Returns:
        Dictionary mapping cluster ID to semantic tag
    """
    sense_tags = {}

    for cluster_id, resources in cluster_resources.items():
        try:
            tag = tag_generator_fn(term, resources, level)
            sense_tags[cluster_id] = tag
        except Exception as e:
            logging.warning(f"Failed to generate LLM tag for cluster {cluster_id} of term '{term}': {e}")
            # Fallback to simple tag
            sense_tags[cluster_id] = f"{term}_{cluster_id}"

    return sense_tags


def _generate_sense_tags_fallback_pure(
    term: str,
    cluster_resources: Dict[int, List[Dict]]
) -> Dict[int, str]:
    """
    Generate fallback semantic tags without LLM (pure function).

    Args:
        term: The ambiguous term
        cluster_resources: Resources grouped by cluster

    Returns:
        Dictionary mapping cluster ID to semantic tag
    """
    sense_tags = {}

    for cluster_id in cluster_resources.keys():
        # Simple numeric suffix as fallback
        sense_tags[cluster_id] = f"{term}_{cluster_id + 1}"

    return sense_tags


def _generate_tag_with_llm_pure(
    term: str,
    resources: List[Dict],
    level: int,
    llm_fn: LLMFunction,
    config: SplittingConfig
) -> str:
    """
    Generate a single semantic tag using LLM (pure function).

    Args:
        term: The ambiguous term
        resources: Resources for this cluster
        level: Hierarchy level
        llm_fn: LLM function for generation
        config: Configuration for tag generation

    Returns:
        Generated semantic tag
    """
    # Extract content from resources
    content_pieces = []
    for resource in resources:
        if "title" in resource:
            content_pieces.append(resource["title"])
        if "content" in resource:
            content_pieces.append(resource["content"][:500])  # Limit content length

    if not content_pieces:
        return f"{term}_unknown"

    # Create prompt for tag generation
    level_names = ["college", "department", "area", "topic"]
    level_name = level_names[level] if level < len(level_names) else "concept"

    content_text = " ".join(content_pieces)
    prompt_text = f"""Given the following content about "{term}" in the context of {level_name}, generate a concise, descriptive tag that captures the specific meaning or sense of this term.

Content: {content_text}

Generate a short, descriptive tag (1-3 words) that distinguishes this sense of "{term}" from other possible meanings. The tag should be specific to the academic context shown in the content.

Tag:"""

    # Convert prompt to messages format
    messages = [{"role": "user", "content": prompt_text}]

    try:
        response = llm_fn(messages)
        # Clean and validate the response
        tag = re.sub(r'[^\w\s-]', '', response.strip())
        tag = re.sub(r'\s+', '_', tag.lower())

        # Ensure it's not empty and reasonable length
        if not tag or len(tag) > 50:
            return f"{term}_sense"

        return tag
    except Exception as e:
        logging.warning(f"LLM tag generation failed: {e}")
        return f"{term}_sense"


def _create_split_proposal_pure(
    term: str,
    level: int,
    cluster_resources: Dict[int, List[Dict]],
    sense_tags: Dict[int, str],
    confidence: float
) -> SplitProposal:
    """
    Create a split proposal from processed cluster data (pure function).

    Args:
        term: The ambiguous term
        level: Hierarchy level
        cluster_resources: Resources grouped by cluster
        sense_tags: Generated semantic tags for each cluster
        confidence: Confidence score

    Returns:
        Immutable SplitProposal object
    """
    # Create sense mappings for proposed_senses
    senses = []
    for cluster_id in cluster_resources.keys():
        tag = sense_tags.get(cluster_id, f"{term}_{cluster_id}")
        sense_dict = {
            "tag": tag,
            "cluster_id": cluster_id,
            "resources": cluster_resources[cluster_id]
        }
        senses.append(sense_dict)

    # Create evidence dictionary
    evidence = {
        "sense_tags": sense_tags,
        "method": "clustering",
        "cluster_count": len(cluster_resources)
    }

    return SplitProposal(
        original_term=term,
        level=level,
        proposed_senses=tuple(senses),
        confidence=confidence,
        evidence=evidence
    )


# Legacy wrapper function
def generate_splits(
    detection_results: DetectionResults,
    hierarchy: Hierarchy,
    web_content: Optional[WebContent] = None,
    use_llm: bool = True,
    llm_provider: str = "gemini"
) -> SplitProposals:
    """
    Generate sense split proposals for ambiguous terms (DEPRECATED).

    DEPRECATED: Use generate_split_proposals() with SplittingConfig for pure functional approach.
    This function is a legacy wrapper that will be removed in a future version.

    Args:
        detection_results: Detection results with cluster information
        hierarchy: Hierarchy data
        web_content: Optional web content for generating tags
        use_llm: Whether to use LLM for tag generation
        llm_provider: LLM provider to use

    Returns:
        List of split proposals in legacy format
    """
    from ..utils import _create_legacy_llm_function, _convert_detection_results_to_list, _convert_proposals_to_legacy_format

    # Convert legacy format to new format
    detection_result_list = _convert_detection_results_to_list(detection_results, web_content)

    # Create config
    config = SplittingConfig(
        use_llm=use_llm,
        min_cluster_size=2,
        max_clusters=5
    )

    # Create LLM function if needed
    llm_fn = None
    if use_llm:
        llm_fn = _create_legacy_llm_function(llm_provider)

    # Generate proposals using new function
    proposals = generate_split_proposals(detection_result_list, hierarchy, config, llm_fn)

    # Convert back to legacy format
    return _convert_proposals_to_legacy_format(proposals)