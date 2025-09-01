"""
Splitting functions for generating and validating sense splits.

All functions are pure - no state, no classes, just data transformation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Literal
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import re

from generate_glossary.utils.llm_simple import infer_structured, infer_text
from .utils import (
    get_level_params,
    extract_keywords,
    calculate_separation_score
)

# Type aliases
DetectionResults = Dict[str, Dict[str, Any]]
SplitProposal = Dict[str, Any]
SplitProposals = List[SplitProposal]
WebContent = Dict[str, Any]
Hierarchy = Dict[str, Any]


def generate_sense_splits(
    detection_results: DetectionResults,
    hierarchy: Hierarchy,
    web_content: Optional[WebContent] = None,
    use_llm: bool = True,
    llm_provider: str = "gemini"
) -> SplitProposals:
    """
    Generate sense split proposals for ambiguous terms.
    
    This function:
    1. Groups resources by cluster for each ambiguous term
    2. Generates semantic tags for each cluster
    3. Creates split proposals with meaningful names
    
    Args:
        detection_results: Detection results with cluster information
        hierarchy: Hierarchy data
        web_content: Optional web content for generating tags
        use_llm: Whether to use LLM for tag generation
        llm_provider: LLM provider to use
        
    Returns:
        List of split proposals
    """
    logging.info(f"Generating splits for {len(detection_results)} ambiguous terms")
    
    proposals = []
    
    for term, detection_data in detection_results.items():
        # Extract cluster information from evidence
        cluster_info = _extract_cluster_info(detection_data)
        if not cluster_info:
            continue
        
        # Get term level for context
        level = detection_data.get("level")
        if level is None:
            term_data = hierarchy.get("terms", {}).get(term, {})
            level = term_data.get("level", 2)  # Default to level 2
        
        # Group resources by cluster
        cluster_resources = _group_resources_by_cluster(
            term=term,
            cluster_labels=cluster_info["labels"],
            web_content=web_content
        )
        
        # Skip if not enough distinct clusters
        if len(cluster_resources) < 2:
            continue
        
        # Generate tags for each cluster
        sense_tags = _generate_sense_tags(
            term=term,
            cluster_resources=cluster_resources,
            level=level,
            use_llm=use_llm,
            llm_provider=llm_provider
        )
        
        # Create split proposal
        proposal = {
            "original_term": term,
            "level": level,
            "cluster_count": len(cluster_resources),
            "confidence": detection_data.get("overall_confidence", 0.5),
            "proposed_senses": []
        }
        
        for cluster_id, resources in cluster_resources.items():
            sense_tag = sense_tags.get(cluster_id, f"sense_{cluster_id}")
            
            proposal["proposed_senses"].append({
                "sense_tag": sense_tag,
                "cluster_id": cluster_id,
                "resource_count": len(resources),
                "sample_resources": resources[:3]  # Include sample resources
            })
        
        proposals.append(proposal)
    
    logging.info(f"Generated {len(proposals)} split proposals")
    return proposals


def validate_splits(
    split_proposals: SplitProposals,
    hierarchy: Hierarchy,
    web_content: Optional[WebContent] = None,
    use_llm: bool = True,
    llm_provider: str = "gemini"
) -> Tuple[SplitProposals, SplitProposals]:
    """
    Validate split proposals to determine which should be accepted.
    
    Validation checks:
    1. Semantic distinctness between proposed senses
    2. Sufficient separation in embedding space
    3. LLM validation of field distinctness (if enabled)
    
    Args:
        split_proposals: List of split proposals to validate
        hierarchy: Hierarchy data
        web_content: Optional web content for validation
        use_llm: Whether to use LLM for validation
        llm_provider: LLM provider to use
        
    Returns:
        Tuple of (accepted_proposals, rejected_proposals)
    """
    logging.info(f"Validating {len(split_proposals)} split proposals")
    
    accepted = []
    rejected = []
    
    for proposal in split_proposals:
        term = proposal["original_term"]
        level = proposal.get("level", 2)
        senses = proposal["proposed_senses"]
        
        # Need at least 2 senses to validate
        if len(senses) < 2:
            proposal["rejection_reason"] = "Insufficient senses (< 2)"
            rejected.append(proposal)
            continue
        
        # Calculate separation between senses
        separation_score = _calculate_sense_separation(
            senses=senses,
            web_content=web_content
        )
        
        # Get level-specific threshold
        level_params = get_level_params(level)
        threshold = level_params.get("separation_threshold", 0.5)
        
        # Check separation score
        if separation_score < threshold:
            proposal["rejection_reason"] = f"Insufficient separation ({separation_score:.2f} < {threshold})"
            proposal["separation_score"] = separation_score
            rejected.append(proposal)
            continue
        
        # LLM validation if enabled
        if use_llm and len(senses) == 2:  # Only validate pairs with LLM
            field1 = senses[0]["sense_tag"]
            field2 = senses[1]["sense_tag"]
            
            is_distinct = _check_field_distinctness_llm(
                term=term,
                field1=field1,
                field2=field2,
                llm_provider=llm_provider
            )
            
            if not is_distinct:
                proposal["rejection_reason"] = f"LLM validation failed: {field1} and {field2} not distinct"
                proposal["separation_score"] = separation_score
                rejected.append(proposal)
                continue
        
        # Passed all validation
        proposal["split_reason"] = f"Valid split confirmed (separation: {separation_score:.2f})"
        proposal["separation_score"] = separation_score
        accepted.append(proposal)
    
    logging.info(f"Validation complete: {len(accepted)} accepted, {len(rejected)} rejected")
    return accepted, rejected


def apply_splits_to_hierarchy(
    hierarchy_path: str,
    accepted_splits: SplitProposals
) -> Hierarchy:
    """
    Apply accepted splits to create updated hierarchy.
    
    This function:
    1. Loads the existing hierarchy
    2. Creates new sense nodes for split terms
    3. Updates parent relationships
    4. Returns modified hierarchy
    
    Args:
        hierarchy_path: Path to hierarchy.json
        accepted_splits: List of accepted split proposals
        
    Returns:
        Updated hierarchy with split terms
    """
    import json
    
    logging.info(f"Applying {len(accepted_splits)} splits to hierarchy")
    
    # Load hierarchy
    with open(hierarchy_path, 'r') as f:
        hierarchy = json.load(f)
    
    terms_dict = hierarchy.get("terms", {})
    
    for split in accepted_splits:
        original_term = split["original_term"]
        
        if original_term not in terms_dict:
            logging.warning(f"Term {original_term} not found in hierarchy")
            continue
        
        original_data = terms_dict[original_term]
        
        # Create new nodes for each sense
        for sense in split["proposed_senses"]:
            sense_tag = sense["sense_tag"]
            new_term = f"{original_term}_{sense_tag}"
            
            # Copy original data with modifications
            terms_dict[new_term] = {
                **original_data,
                "canonical_form": new_term,
                "original_term": original_term,
                "sense_tag": sense_tag,
                "is_disambiguated": True,
                "resource_subset": sense.get("sample_resources", [])
            }
            
            # Update children to point to new sense
            # (This is simplified - real implementation would need more logic)
            for child in original_data.get("children", []):
                if child in terms_dict:
                    child_data = terms_dict[child]
                    parents = child_data.get("parents", [])
                    if original_term in parents:
                        # Replace with appropriate sense
                        # (Would need more logic to determine which sense)
                        idx = parents.index(original_term)
                        parents[idx] = new_term
        
        # Mark original as ambiguous (don't remove, keep for reference)
        original_data["is_ambiguous"] = True
        original_data["split_into"] = [
            f"{original_term}_{s['sense_tag']}" 
            for s in split["proposed_senses"]
        ]
    
    logging.info(f"Applied splits, hierarchy now has {len(terms_dict)} terms")
    return hierarchy


# Helper functions

def _extract_cluster_info(
    detection_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract cluster information from detection evidence.
    
    Returns:
        Dictionary with cluster labels and metadata, or None
    """
    for evidence in detection_data.get("evidence", []):
        payload = evidence.get("payload", {})
        
        # Look for cluster labels in payload
        if "cluster_labels" in payload:
            return {
                "labels": payload["cluster_labels"],
                "algorithm": payload.get("algorithm", "dbscan"),
                "source": evidence.get("source")
            }
        elif "clusters" in payload:  # Global clustering format
            return {
                "labels": payload["clusters"],
                "algorithm": payload.get("algorithm", "dbscan"),
                "source": evidence.get("source")
            }
    
    return None


def _group_resources_by_cluster(
    term: str,
    cluster_labels: List[int],
    web_content: Optional[WebContent]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group resources by their cluster assignments.
    
    Returns:
        Dictionary mapping cluster ID to list of resources
    """
    cluster_resources = defaultdict(list)
    
    if not web_content or term not in web_content:
        # Create dummy resources based on cluster labels
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise
                cluster_resources[label].append({
                    "index": i,
                    "cluster": label
                })
        return dict(cluster_resources)
    
    # Get actual resources
    term_resources = web_content[term]
    if isinstance(term_resources, dict):
        term_resources = term_resources.get("resources", [])
    
    # Group by cluster
    for i, label in enumerate(cluster_labels):
        if label != -1 and i < len(term_resources):
            cluster_resources[label].append(term_resources[i])
    
    return dict(cluster_resources)


def _generate_sense_tags(
    term: str,
    cluster_resources: Dict[int, List[Dict[str, Any]]],
    level: int,
    use_llm: bool,
    llm_provider: str
) -> Dict[int, str]:
    """
    Generate semantic tags for each cluster.
    
    Uses multiple strategies:
    1. LLM-based tag generation (if enabled)
    2. Keyword extraction fallback
    3. Generic fallback
    
    Returns:
        Dictionary mapping cluster ID to semantic tag
    """
    tags = {}
    
    for cluster_id, resources in cluster_resources.items():
        if use_llm and resources:
            # Extract text from resources
            texts = []
            for r in resources[:5]:  # Use first 5 resources
                if isinstance(r, dict):
                    # Try different fields
                    text = r.get("content") or r.get("text") or r.get("snippet") or ""
                    if text:
                        texts.append(str(text)[:500])  # Limit length
            
            if texts:
                # Generate tag using LLM
                tag = _generate_tag_with_llm(
                    term=term,
                    texts=texts,
                    level=level,
                    llm_provider=llm_provider
                )
                if tag:
                    tags[cluster_id] = tag
                    continue
        
        # Fallback: keyword extraction
        keywords = extract_keywords(resources)
        if keywords:
            # Use top keyword as tag
            tags[cluster_id] = keywords[0].replace(" ", "_").lower()
        else:
            # Generic fallback
            tags[cluster_id] = f"sense_{cluster_id}"
    
    return tags


def _generate_tag_with_llm(
    term: str,
    texts: List[str],
    level: int,
    llm_provider: str
) -> Optional[str]:
    """
    Generate a semantic tag using LLM.
    
    Returns:
        Generated tag or None if failed
    """
    # Get level context
    level_params = get_level_params(level)
    level_desc = level_params.get("description", "academic field")
    
    prompt = f"""
    Given the term "{term}" and the following resource excerpts, generate a short academic field tag.
    
    The tag should represent a {level_desc} where this term is used.
    
    Resources:
    {chr(10).join(f'- {text[:200]}' for text in texts[:3])}
    
    Return ONLY a short tag (1-3 words, lowercase, underscores for spaces).
    Example tags: machine_learning, organic_chemistry, market_analysis
    """
    
    try:
        response = infer_text(
            prompt=prompt,
            provider=llm_provider,
            max_tokens=50
        )
        
        # Clean and validate response
        tag = response.strip().lower()
        tag = re.sub(r'[^a-z0-9_]', '_', tag)
        tag = re.sub(r'_+', '_', tag).strip('_')
        
        if tag and len(tag) < 50:  # Reasonable length
            return tag
    except Exception as e:
        logging.debug(f"LLM tag generation failed: {e}")
    
    return None


def _calculate_sense_separation(
    senses: List[Dict[str, Any]],
    web_content: Optional[WebContent]
) -> float:
    """
    Calculate separation score between proposed senses.
    
    Uses keyword overlap and resource diversity metrics.
    
    Returns:
        Separation score between 0 and 1
    """
    if len(senses) < 2:
        return 0.0
    
    # Extract keywords for each sense
    sense_keywords = []
    for sense in senses:
        resources = sense.get("sample_resources", [])
        keywords = extract_keywords(resources)
        sense_keywords.append(set(keywords))
    
    # Calculate pairwise overlaps
    overlaps = []
    for i in range(len(sense_keywords)):
        for j in range(i + 1, len(sense_keywords)):
            set1, set2 = sense_keywords[i], sense_keywords[j]
            if set1 and set2:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                overlap = intersection / union if union > 0 else 0
                overlaps.append(overlap)
    
    if not overlaps:
        return 0.5  # Default if no keywords
    
    # Separation is inverse of overlap
    avg_overlap = sum(overlaps) / len(overlaps)
    separation = 1.0 - avg_overlap
    
    return separation


def _check_field_distinctness_llm(
    term: str,
    field1: str,
    field2: str,
    llm_provider: str
) -> bool:
    """
    Use LLM to check if two fields are semantically distinct.
    
    Returns:
        True if fields are distinct, False otherwise
    """
    prompt = f"""
    For the academic term "{term}", determine if these two field tags represent DISTINCT concepts:
    
    Field 1: {field1}
    Field 2: {field2}
    
    Fields are DISTINCT if they represent fundamentally different academic domains or applications.
    Fields are NOT DISTINCT if they are subfields, closely related, or variations of the same concept.
    
    Examples of DISTINCT: (image_processing, market_segmentation), (stress_psychology, stress_mechanics)
    Examples of NOT DISTINCT: (deep_learning, neural_networks), (nlp, natural_language)
    
    Answer with only: DISTINCT or NOT_DISTINCT
    """
    
    try:
        response = infer_text(
            prompt=prompt,
            provider=llm_provider,
            max_tokens=20
        )
        
        response = response.strip().upper()
        return "DISTINCT" in response and "NOT" not in response
        
    except Exception as e:
        logging.debug(f"LLM distinctness check failed: {e}")
        # Conservative: assume not distinct if LLM fails
        return False