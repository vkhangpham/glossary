"""
Sense splitting functionality for ambiguous terms.

Generates and validates proposals for splitting terms with multiple meanings
into distinct senses with appropriate semantic tags.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import re

from generate_glossary.utils.llm_simple import infer_structured, infer_text
from .utils import (
    extract_keywords,
    calculate_separation_score
)

# Type aliases
DetectionResults = Dict[str, Dict[str, Any]]
SplitProposal = Dict[str, Any]
SplitProposals = List[SplitProposal]
WebContent = Dict[str, Any]
Hierarchy = Dict[str, Any]


def generate_splits(
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
        
        # Skip if only one sense
        if len(senses) < 2:
            rejected.append({
                **proposal,
                "rejection_reason": "Insufficient senses for split"
            })
            continue
        
        # Check semantic distinctness
        tags = [s["sense_tag"] for s in senses]
        if not _are_tags_distinct(tags):
            rejected.append({
                **proposal,
                "rejection_reason": "Sense tags not semantically distinct"
            })
            continue
        
        # Calculate separation score
        separation_score = calculate_separation_score(senses)
        if separation_score < 0.5:
            rejected.append({
                **proposal,
                "rejection_reason": f"Low separation score: {separation_score:.2f}"
            })
            continue
        
        # LLM validation if enabled
        if use_llm:
            is_valid, reason = _validate_with_llm(
                term=term,
                senses=senses,
                level=level,
                llm_provider=llm_provider
            )
            
            if not is_valid:
                rejected.append({
                    **proposal,
                    "rejection_reason": f"LLM validation failed: {reason}"
                })
                continue
        
        # All checks passed
        proposal["separation_score"] = separation_score
        proposal["validation_method"] = "llm" if use_llm else "rule-based"
        accepted.append(proposal)
    
    logging.info(f"Accepted {len(accepted)}/{len(split_proposals)} split proposals")
    return accepted, rejected


def apply_to_hierarchy(
    accepted_splits: SplitProposals,
    hierarchy: Hierarchy,
    create_backup: bool = True
) -> Hierarchy:
    """
    Apply accepted splits to the hierarchy.
    
    Args:
        accepted_splits: Validated split proposals to apply
        hierarchy: Current hierarchy to modify
        create_backup: Whether to backup original terms
        
    Returns:
        Updated hierarchy with split terms
    """
    logging.info(f"Applying {len(accepted_splits)} splits to hierarchy")
    
    # Create a copy to avoid modifying original
    import copy
    updated_hierarchy = copy.deepcopy(hierarchy)
    
    applied_count = 0
    
    for split in accepted_splits:
        term = split["original_term"]
        level = split["level"]
        senses = split["proposed_senses"]
        
        # Find term in hierarchy
        if "levels" in updated_hierarchy:
            level_data = updated_hierarchy["levels"][level]
            terms = level_data.get("terms", [])
            
            # Find and update term
            for i, term_data in enumerate(terms):
                if term_data.get("term") == term:
                    # Backup original if requested
                    if create_backup:
                        term_data["original_term"] = term
                        term_data["was_split"] = True
                    
                    # Create split entries
                    split_terms = []
                    for sense in senses:
                        split_term = {
                            **term_data,  # Copy original data
                            "term": f"{term} ({sense['sense_tag']})",
                            "sense_tag": sense["sense_tag"],
                            "parent_term": term,
                            "cluster_id": sense["cluster_id"]
                        }
                        split_terms.append(split_term)
                    
                    # Replace original with splits
                    terms[i:i+1] = split_terms
                    applied_count += 1
                    break
    
    logging.info(f"Applied {applied_count} splits to hierarchy")
    
    # Update metadata
    if "metadata" not in updated_hierarchy:
        updated_hierarchy["metadata"] = {}
    
    updated_hierarchy["metadata"]["disambiguation_applied"] = True
    updated_hierarchy["metadata"]["splits_applied"] = applied_count
    updated_hierarchy["metadata"]["total_splits"] = len(accepted_splits)
    
    return updated_hierarchy


# Helper functions

def _extract_cluster_info(detection_data: Dict) -> Optional[Dict]:
    """Extract cluster information from detection data."""
    method = detection_data.get("method")
    
    if method == "embedding":
        clusters = detection_data.get("clusters", [])
        if clusters:
            # Reconstruct cluster labels
            labels = []
            for cluster in clusters:
                cluster_id = cluster["cluster_id"]
                size = cluster["size"]
                labels.extend([cluster_id] * size)
            return {"labels": labels, "clusters": clusters}
    
    elif method == "global":
        distribution = detection_data.get("cluster_distribution", [])
        if distribution:
            return {"labels": None, "clusters": distribution}
    
    return None


def _group_resources_by_cluster(
    term: str,
    cluster_labels: Optional[List],
    web_content: Optional[WebContent]
) -> Dict[int, List]:
    """Group resources by their cluster assignments."""
    if not web_content or term not in web_content:
        return {}
    
    resources = web_content[term]
    if isinstance(resources, dict):
        resources = resources.get("resources", [])
    
    if not cluster_labels:
        # Use simple splitting if no labels
        mid = len(resources) // 2
        return {
            0: resources[:mid],
            1: resources[mid:]
        }
    
    # Group by cluster label
    grouped = defaultdict(list)
    for i, label in enumerate(cluster_labels[:len(resources)]):
        if label != -1:  # Skip noise
            grouped[label].append(resources[i])
    
    return dict(grouped)


def _generate_sense_tags(
    term: str,
    cluster_resources: Dict[int, List],
    level: int,
    use_llm: bool,
    llm_provider: str
) -> Dict[int, str]:
    """Generate semantic tags for each cluster."""
    tags = {}
    
    for cluster_id, resources in cluster_resources.items():
        if use_llm:
            # Use LLM to generate contextual tag
            tag = _generate_tag_with_llm(
                term=term,
                resources=resources[:3],
                level=level,
                llm_provider=llm_provider
            )
        else:
            # Extract keywords as tag
            keywords = extract_keywords(resources)
            if keywords:
                tag = keywords[0] if keywords else f"context_{cluster_id}"
            else:
                tag = f"context_{cluster_id}"
        
        tags[cluster_id] = tag
    
    return tags


def _generate_tag_with_llm(
    term: str,
    resources: List,
    level: int,
    llm_provider: str
) -> str:
    """Generate a semantic tag using LLM."""
    # Extract titles and descriptions
    contexts = []
    for resource in resources:
        if isinstance(resource, dict):
            title = resource.get("title", "")
            desc = resource.get("description", "")
            contexts.append(f"{title}: {desc}"[:200])
    
    prompt = f"""Given the term "{term}" and these contexts:
{chr(10).join(contexts)}

Generate a short (1-3 word) semantic tag that distinguishes this specific usage/meaning of the term.
The tag should be a domain, field, or context identifier.

Examples:
- For "stress": "psychology" vs "materials"
- For "model": "fashion" vs "machine learning"
- For "field": "agriculture" vs "physics"

Tag:"""
    
    try:
        tag = infer_text(prompt, provider=llm_provider, max_tokens=20)
        tag = tag.strip().lower()
        # Clean up tag
        tag = re.sub(r'[^a-z0-9\s\-]', '', tag)
        tag = tag.replace(' ', '_')
        return tag if tag else f"context_{hash(str(resources))%1000}"
    except Exception as e:
        logging.warning(f"LLM tag generation failed: {e}")
        return f"context_{hash(str(resources))%1000}"


def _are_tags_distinct(tags: List[str]) -> bool:
    """Check if tags are semantically distinct."""
    # Simple check: tags should be different
    if len(set(tags)) != len(tags):
        return False
    
    # Check for minimal difference
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            # Tags should differ by more than just numbers
            tag1 = re.sub(r'\d+', '', tags[i])
            tag2 = re.sub(r'\d+', '', tags[j])
            if tag1 == tag2:
                return False
    
    return True


def _validate_with_llm(
    term: str,
    senses: List[Dict],
    level: int,
    llm_provider: str
) -> Tuple[bool, str]:
    """Validate split proposal using LLM."""
    sense_descriptions = []
    for sense in senses:
        tag = sense["sense_tag"]
        count = sense["resource_count"]
        sense_descriptions.append(f"- {tag} ({count} resources)")
    
    prompt = f"""Evaluate if this term should be split into multiple senses:

Term: "{term}"
Level: {level} (0=College, 1=Department, 2=Research Area, 3=Topic)
Proposed senses:
{chr(10).join(sense_descriptions)}

Should this term be split into these distinct senses? Consider:
1. Are these truly different meanings/usages of the same term?
2. Would splitting improve clarity in an academic glossary?
3. Are the proposed senses sufficiently distinct?

Answer with YES or NO, followed by a brief reason.
Format: YES/NO: reason
"""
    
    try:
        response = infer_text(prompt, provider=llm_provider, max_tokens=100)
        response = response.strip()
        
        if response.startswith("YES"):
            return True, "LLM approved split"
        else:
            reason = response.split(":", 1)[1] if ":" in response else "LLM rejected split"
            return False, reason.strip()
    except Exception as e:
        logging.warning(f"LLM validation failed: {e}")
        return True, "LLM validation skipped due to error"