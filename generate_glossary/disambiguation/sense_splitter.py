"""
Sense splitting functionality for ambiguous terms.

Generates and validates proposals for splitting terms with multiple meanings
into distinct senses with appropriate semantic tags.
"""

import logging
import copy
import warnings
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict
import re

from generate_glossary.llm import completion
from .types import (
    DetectionResult,
    SplitProposal,
    SplittingConfig,
    LLMFunction,
    TagGeneratorFunction,
    ValidationFunction
)

# Type aliases for legacy compatibility
DetectionResults = Dict[str, Dict[str, Any]]
SplitProposals = List[Dict[str, Any]]  # Legacy format
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
        List of split proposals in legacy dictionary format
    """
    warnings.warn(
        "generate_splits() is deprecated. Use generate_split_proposals() with SplittingConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    logging.info(f"Generating splits for {len(detection_results)} ambiguous terms")
    
    # Convert legacy format to new format, propagating web_content
    detection_results_list = _convert_detection_results_to_list(detection_results, web_content)
    
    # Create configuration
    config = SplittingConfig(
        use_llm=use_llm,
        llm_provider=llm_provider
    )
    
    # Create LLM function if needed
    llm_fn = _create_legacy_llm_function(llm_provider) if use_llm else None
    
    # Call pure function
    proposals = generate_split_proposals(
        detection_results=detection_results_list,
        hierarchy=hierarchy,
        config=config,
        llm_fn=llm_fn
    )
    
    # Convert back to legacy format
    legacy_proposals = _convert_proposals_to_legacy_format(proposals)
    
    logging.info(f"Generated {len(legacy_proposals)} split proposals")
    return legacy_proposals


def validate_splits(
    split_proposals: SplitProposals,
    hierarchy: Hierarchy,
    web_content: Optional[WebContent] = None,
    use_llm: bool = True,
    llm_provider: str = "gemini"
) -> Tuple[SplitProposals, SplitProposals]:
    """
    Validate split proposals to determine which should be accepted (DEPRECATED).
    
    DEPRECATED: Use validate_split_proposals() with SplittingConfig for pure functional approach.
    This function is a legacy wrapper that will be removed in a future version.
    
    Args:
        split_proposals: List of split proposals to validate
        hierarchy: Hierarchy data
        web_content: Optional web content for validation
        use_llm: Whether to use LLM for validation
        llm_provider: LLM provider to use
        
    Returns:
        Tuple of (accepted_proposals, rejected_proposals) in legacy dictionary format
    """
    warnings.warn(
        "validate_splits() is deprecated. Use validate_split_proposals() with SplittingConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    logging.info(f"Validating {len(split_proposals)} split proposals")
    
    # Convert legacy format to SplitProposal objects
    proposals = []
    for legacy_proposal in split_proposals:
        # Create SplitProposal from legacy format
        proposal = SplitProposal(
            original_term=legacy_proposal["original_term"],
            level=legacy_proposal.get("level", 2),
            proposed_senses=tuple(legacy_proposal["proposed_senses"]),
            confidence=legacy_proposal.get("confidence", 0.5),
            evidence=legacy_proposal.get("evidence", {}),
            validation_status=legacy_proposal.get("validation_status")
        )
        proposals.append(proposal)
    
    # Create configuration
    config = SplittingConfig(
        use_llm=use_llm,
        llm_provider=llm_provider
    )
    
    # Create LLM function if needed
    llm_fn = _create_legacy_llm_function(llm_provider) if use_llm else None
    
    # Call pure function
    accepted_proposals, rejected_proposals = validate_split_proposals(
        proposals=proposals,
        hierarchy=hierarchy,
        config=config,
        llm_fn=llm_fn
    )
    
    # Convert back to legacy format
    accepted_legacy = _convert_proposals_to_legacy_format(accepted_proposals)
    rejected_legacy = _convert_proposals_to_legacy_format(rejected_proposals)
    
    logging.info(f"Accepted {len(accepted_legacy)}/{len(split_proposals)} split proposals")
    return accepted_legacy, rejected_legacy


def apply_to_hierarchy(
    accepted_splits: SplitProposals,
    hierarchy: Hierarchy,
    create_backup: bool = True
) -> Hierarchy:
    """
    Apply accepted splits to the hierarchy (DEPRECATED).
    
    DEPRECATED: Use apply_splits_to_hierarchy() with SplittingConfig for pure functional approach.
    This function is a legacy wrapper that will be removed in a future version.
    
    Args:
        accepted_splits: Validated split proposals to apply
        hierarchy: Current hierarchy to modify
        create_backup: Whether to backup original terms
        
    Returns:
        Updated hierarchy with split terms
    """
    warnings.warn(
        "apply_to_hierarchy() is deprecated. Use apply_splits_to_hierarchy() with SplittingConfig instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    logging.info(f"Applying {len(accepted_splits)} splits to hierarchy")
    
    # Convert legacy format to SplitProposal objects
    proposals = []
    for legacy_split in accepted_splits:
        # Create SplitProposal from legacy format
        proposal = SplitProposal(
            original_term=legacy_split["original_term"],
            level=legacy_split.get("level", 2),
            proposed_senses=tuple(legacy_split["proposed_senses"]),
            confidence=legacy_split.get("confidence", 0.5),
            evidence=legacy_split.get("evidence", {}),
            validation_status="approved"  # Assume accepted splits are approved
        )
        proposals.append(proposal)
    
    # Create configuration
    config = SplittingConfig(create_backup=create_backup)
    
    # Call pure function
    updated_hierarchy = apply_splits_to_hierarchy(
        proposals=proposals,
        hierarchy=hierarchy,
        config=config
    )
    
    applied_count = updated_hierarchy.get("metadata", {}).get("splits_applied", 0)
    logging.info(f"Applied {applied_count} splits to hierarchy")
    
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




# ============================================================================
# PURE FUNCTIONAL IMPLEMENTATIONS WITH DEPENDENCY INJECTION
# ============================================================================

def create_tag_generator(llm_fn: LLMFunction, config: SplittingConfig) -> TagGeneratorFunction:
    """Create a tag generator function with injected LLM dependency."""
    def tag_generator(term: str, resources: List[Dict], level: int) -> str:
        return _generate_tag_with_llm_pure(term, resources, level, llm_fn, config)
    return tag_generator


def create_llm_validator(llm_fn: LLMFunction, config: SplittingConfig) -> ValidationFunction:
    """Create a validation function with injected LLM dependency."""
    def validator(term: str, senses: List[Dict], level: int) -> Tuple[bool, str]:
        return _validate_with_llm_pure(term, senses, level, llm_fn, config)
    return validator


def with_llm_function(splitting_fn: Callable, llm_fn: LLMFunction) -> Callable:
    """Higher-order function to inject LLM dependency into splitting functions."""
    def wrapped(*args, **kwargs):
        return splitting_fn(*args, llm_fn=llm_fn, **kwargs)
    return wrapped


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


def _extract_cluster_info_pure(detection_result: DetectionResult) -> Optional[Dict[str, Any]]:
    """Extract cluster information from detection result (pure function)."""
    evidence = detection_result.evidence
    
    # Try different evidence structures
    if "clustering" in evidence:
        clustering_data = evidence["clustering"]
        if "labels" in clustering_data:
            return {
                "labels": clustering_data["labels"],
                "n_clusters": clustering_data.get("n_clusters", len(set(clustering_data["labels"])))
            }
    
    if "cluster_labels" in evidence:
        labels = evidence["cluster_labels"]
        return {
            "labels": labels,
            "n_clusters": len(set(labels)) if labels else 0
        }
    
    return None


def _group_resources_by_cluster_pure(
    term: str,
    cluster_labels: List[int],
    web_content: Optional[Dict[str, Any]]
) -> Dict[int, List[Dict]]:
    """Group resources by cluster labels (pure function)."""
    cluster_resources = defaultdict(list)
    
    if not web_content or "resources" not in web_content:
        # Generate dummy resources if no web content
        for i, label in enumerate(cluster_labels):
            cluster_resources[label].append({
                "title": f"{term} resource {i}",
                "description": f"Resource {i} for {term}",
                "cluster_id": label
            })
        return dict(cluster_resources)
    
    resources = web_content["resources"]
    
    # Group actual resources by cluster
    for i, label in enumerate(cluster_labels):
        if i < len(resources):
            resource = resources[i].copy()  # Shallow copy to avoid mutation
            resource["cluster_id"] = label
            cluster_resources[label].append(resource)
    
    return dict(cluster_resources)


def _generate_sense_tags_with_llm_pure(
    term: str,
    cluster_resources: Dict[int, List[Dict]],
    level: int,
    tag_generator_fn: TagGeneratorFunction
) -> Dict[int, str]:
    """Generate semantic tags using LLM (pure function)."""
    sense_tags = {}
    
    for cluster_id, resources in cluster_resources.items():
        try:
            tag = tag_generator_fn(term, resources, level)
            sense_tags[cluster_id] = tag
        except Exception:
            # Fallback to generic tag
            sense_tags[cluster_id] = f"context_{cluster_id}"
    
    return sense_tags


def _generate_sense_tags_fallback_pure(
    term: str,
    cluster_resources: Dict[int, List[Dict]]
) -> Dict[int, str]:
    """Generate fallback tags without LLM (pure function)."""
    sense_tags = {}
    
    for cluster_id, resources in cluster_resources.items():
        # Extract keywords from titles and descriptions
        keywords = []
        for resource in resources:
            title = resource.get("title", "")
            desc = resource.get("description", "")
            text = f"{title} {desc}".lower()
            
            # Simple keyword extraction
            words = re.findall(r'\b\w+\b', text)
            keywords.extend([w for w in words if len(w) > 3 and w != term.lower()])
        
        # Use most common keyword or fallback
        if keywords:
            from collections import Counter
            most_common = Counter(keywords).most_common(1)[0][0]
            sense_tags[cluster_id] = most_common
        else:
            sense_tags[cluster_id] = f"context_{cluster_id}"
    
    return sense_tags


def _generate_tag_with_llm_pure(
    term: str,
    resources: List[Dict],
    level: int,
    llm_fn: LLMFunction,
    config: SplittingConfig
) -> str:
    """Generate a semantic tag using LLM (pure function)."""
    # Extract titles and descriptions
    contexts = []
    for resource in resources[:config.max_sample_resources]:
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
        messages = [{"role": "user", "content": prompt}]
        tag = llm_fn(messages)
        tag = tag.strip().lower()
        # Clean up tag
        tag = re.sub(r'[^a-z0-9\s\-]', '', tag)
        tag = tag.replace(' ', '_')
        return tag if tag else f"context_{hash(str(resources))%1000}"
    except Exception:
        return f"context_{hash(str(resources))%1000}"


def _create_split_proposal_pure(
    term: str,
    level: int,
    cluster_resources: Dict[int, List[Dict]],
    sense_tags: Dict[int, str],
    confidence: float
) -> SplitProposal:
    """Create a split proposal (pure function)."""
    proposed_senses = []
    
    for cluster_id, resources in cluster_resources.items():
        sense_tag = sense_tags.get(cluster_id, f"sense_{cluster_id}")
        
        sense = {
            "sense_tag": sense_tag,
            "cluster_id": cluster_id,
            "resource_count": len(resources),
            "sample_resources": resources[:3]  # Include sample resources
        }
        proposed_senses.append(sense)
    
    evidence = {
        "cluster_count": len(cluster_resources),
        "total_resources": sum(len(resources) for resources in cluster_resources.values()),
        "sense_tags": sense_tags
    }
    
    return SplitProposal(
        original_term=term,
        level=level,
        proposed_senses=tuple(proposed_senses),
        confidence=confidence,
        evidence=evidence
    )


def validate_split_proposals(
    proposals: List[SplitProposal],
    hierarchy: Dict[str, Any],
    config: SplittingConfig,
    llm_fn: Optional[LLMFunction] = None
) -> Tuple[List[SplitProposal], List[SplitProposal]]:
    """
    Validate split proposals to determine which should be accepted (pure functional implementation).
    
    Validation checks:
    1. Semantic distinctness between proposed senses
    2. Sufficient separation in embedding space
    3. LLM validation of field distinctness (if enabled)
    
    Args:
        proposals: List of split proposals to validate
        hierarchy: Hierarchy data
        config: Splitting configuration
        llm_fn: Optional LLM function for validation
        
    Returns:
        Tuple of (accepted_proposals, rejected_proposals)
    """
    accepted = []
    rejected = []
    
    for proposal in proposals:
        term = proposal.original_term
        level = proposal.level
        senses = list(proposal.proposed_senses)  # Convert from tuple
        
        # Skip if only one sense
        if len(senses) < 2:
            rejected_proposal = _create_rejected_proposal_pure(
                proposal, "Insufficient senses for split"
            )
            rejected.append(rejected_proposal)
            continue
        
        # Check semantic distinctness
        tags = [s["sense_tag"] for s in senses]
        if not _are_tags_distinct_pure(tags):
            rejected_proposal = _create_rejected_proposal_pure(
                proposal, "Sense tags not semantically distinct"
            )
            rejected.append(rejected_proposal)
            continue
        
        # Calculate separation score
        separation_score = _calculate_separation_score_pure(senses)
        if separation_score < config.min_separation_score:
            rejected_proposal = _create_rejected_proposal_pure(
                proposal, f"Low separation score: {separation_score:.2f}"
            )
            rejected.append(rejected_proposal)
            continue
        
        # LLM validation if enabled
        if llm_fn and config.use_llm:
            validator_fn = create_llm_validator(llm_fn, config)
            validation_result = _safe_llm_validate_pure(
                term=term,
                senses=senses,
                level=level,
                validator_fn=validator_fn
            )
            
            if not validation_result["is_valid"]:
                rejected_proposal = _create_rejected_proposal_pure(
                    proposal, f"LLM validation failed: {validation_result['reason']}"
                )
                rejected.append(rejected_proposal)
                continue
        
        # All checks passed
        validated_proposal = _create_validated_proposal_pure(
            proposal=proposal,
            separation_score=separation_score,
            validation_method="llm" if (llm_fn and config.use_llm) else "rule-based",
            status="approved"
        )
        accepted.append(validated_proposal)
    
    return accepted, rejected


def _are_tags_distinct_pure(tags: List[str]) -> bool:
    """Check if tags are semantically distinct (pure function)."""
    if len(tags) != len(set(tags)):
        return False  # Duplicate tags
    
    # Check for semantic similarity
    for i, tag1 in enumerate(tags):
        for tag2 in tags[i+1:]:
            # Simple similarity checks
            if tag1.lower() in tag2.lower() or tag2.lower() in tag1.lower():
                return False
            
            # Check for common stems (very basic)
            if len(tag1) > 3 and len(tag2) > 3:
                if tag1[:3] == tag2[:3]:
                    return False
    
    return True


def _calculate_separation_score_pure(senses: List[Dict]) -> float:
    """Calculate separation score between senses (pure function)."""
    if len(senses) < 2:
        return 0.0
    
    # Simple separation score based on resource counts
    resource_counts = [sense.get("resource_count", 0) for sense in senses]
    
    if sum(resource_counts) == 0:
        return 0.0
    
    # Calculate balance score (how evenly distributed the resources are)
    total_resources = sum(resource_counts)
    expected_per_sense = total_resources / len(senses)
    
    # Calculate deviation from perfect balance
    deviations = [abs(count - expected_per_sense) for count in resource_counts]
    max_possible_deviation = expected_per_sense * (len(senses) - 1)
    
    if max_possible_deviation == 0:
        return 1.0
    
    balance_score = 1.0 - (sum(deviations) / max_possible_deviation)
    
    # Minimum separation requirement
    min_resources_per_sense = max(1, total_resources // (len(senses) * 2))
    has_min_resources = all(count >= min_resources_per_sense for count in resource_counts)
    
    return balance_score * (1.0 if has_min_resources else 0.5)


def _safe_llm_validate_pure(
    term: str,
    senses: List[Dict],
    level: int,
    validator_fn: ValidationFunction
) -> Dict[str, Any]:
    """Safely validate with LLM using functional error handling (pure function)."""
    try:
        is_valid, reason = validator_fn(term, senses, level)
        return {
            "is_valid": is_valid,
            "reason": reason,
            "error": None
        }
    except Exception as e:
        return {
            "is_valid": False,  # Conservative default when LLM fails
            "reason": "LLM validation unavailable",
            "error": str(e)
        }


def _validate_with_llm_pure(
    term: str,
    senses: List[Dict],
    level: int,
    llm_fn: LLMFunction,
    config: SplittingConfig
) -> Tuple[bool, str]:
    """Validate split proposal with LLM (pure function)."""
    sense_descriptions = []
    for i, sense in enumerate(senses):
        tag = sense.get("sense_tag", f"sense_{i}")
        resources = sense.get("sample_resources", [])
        
        # Extract sample context
        context_samples = []
        for resource in resources[:2]:  # Limit to 2 samples
            title = resource.get("title", "")
            desc = resource.get("description", "")
            context_samples.append(f"- {title}: {desc}"[:100])
        
        sense_desc = f"**{tag}**: {'; '.join(context_samples)}"
        sense_descriptions.append(sense_desc)
    
    prompt = f"""Evaluate if these different senses of "{term}" represent genuinely distinct meanings:

{chr(10).join(sense_descriptions)}

Are these senses distinct enough to warrant splitting the term?
Consider:
1. Do they represent different domains/fields?
2. Would users search for them differently?
3. Are the contexts clearly distinguishable?

Respond with "YES: [brief reason]" if distinct, or "NO: [brief reason]" if not distinct enough."""
    
    try:
        messages = [{"role": "user", "content": prompt}]
        response = llm_fn(messages).strip()
        
        if response.startswith("YES"):
            return True, "LLM approved split"
        else:
            reason = response.split(":", 1)[1] if ":" in response else "LLM rejected split"
            return False, reason.strip()
    except Exception:
        return True, "LLM validation skipped due to error"


def _create_rejected_proposal_pure(
    proposal: SplitProposal,
    rejection_reason: str
) -> SplitProposal:
    """Create a rejected proposal with reason (pure function)."""
    # Create new evidence with rejection reason
    new_evidence = dict(proposal.evidence)
    new_evidence["rejection_reason"] = rejection_reason
    
    return SplitProposal(
        original_term=proposal.original_term,
        level=proposal.level,
        proposed_senses=proposal.proposed_senses,
        confidence=proposal.confidence,
        evidence=new_evidence,
        validation_status="rejected"
    )


def _create_validated_proposal_pure(
    proposal: SplitProposal,
    separation_score: float,
    validation_method: str,
    status: str
) -> SplitProposal:
    """Create a validated proposal with validation metadata (pure function)."""
    # Create new evidence with validation metadata
    new_evidence = dict(proposal.evidence)
    new_evidence["separation_score"] = separation_score
    new_evidence["validation_method"] = validation_method
    
    return SplitProposal(
        original_term=proposal.original_term,
        level=proposal.level,
        proposed_senses=proposal.proposed_senses,
        confidence=proposal.confidence,
        evidence=new_evidence,
        validation_status=status
    )

def apply_splits_to_hierarchy(
    proposals: List[SplitProposal],
    hierarchy: Dict[str, Any],
    config: SplittingConfig
) -> Dict[str, Any]:
    """
    Apply accepted splits to the hierarchy (pure functional implementation).
    
    Args:
        proposals: Validated split proposals to apply
        hierarchy: Current hierarchy to modify
        config: Splitting configuration
        
    Returns:
        New hierarchy with split terms (does not modify input)
    """
    # Create a deep copy to avoid modifying original
    updated_hierarchy = copy.deepcopy(hierarchy)
    
    applied_count = 0
    
    for proposal in proposals:
        if proposal.validation_status != "approved":
            continue
            
        term = proposal.original_term
        level = proposal.level
        senses = list(proposal.proposed_senses)
        
        # Apply single split
        updated_hierarchy, was_applied = _apply_single_split_pure(
            hierarchy=updated_hierarchy,
            term=term,
            level=level,
            senses=senses,
            create_backup=config.create_backup
        )
        
        if was_applied:
            applied_count += 1
    
    # Update metadata
    updated_hierarchy = _update_hierarchy_metadata_pure(
        hierarchy=updated_hierarchy,
        applied_count=applied_count,
        total_splits=len(proposals)
    )
    
    return updated_hierarchy


def _apply_single_split_pure(
    hierarchy: Dict[str, Any],
    term: str,
    level: int,
    senses: List[Dict],
    create_backup: bool
) -> Tuple[Dict[str, Any], bool]:
    """Apply a single split to hierarchy (pure function).
    
    Note: Mutates the hierarchy argument (which should already be a copy owned by caller).
    """
    # Find term in hierarchy
    term_location = _find_term_in_hierarchy_pure(term, level, hierarchy)
    if not term_location:
        return hierarchy, False
    
    level_index, term_data = term_location
    
    # Create split terms
    split_terms = _create_split_terms_pure(
        original_term_data=term_data,
        senses=senses,
        create_backup=create_backup
    )
    
    # Apply splits directly to the hierarchy (which is already a copy)
    if "levels" in hierarchy:
        level_data = hierarchy["levels"][level]
        terms = level_data.get("terms", [])
        
        # Replace original term with splits
        terms[level_index:level_index+1] = split_terms
        level_data["terms"] = terms
    
    return hierarchy, True


def _find_term_in_hierarchy_pure(
    term: str,
    level: int,
    hierarchy: Dict[str, Any]
) -> Optional[Tuple[int, Dict]]:
    """Find term in hierarchy and return its location (pure function)."""
    if "levels" not in hierarchy:
        return None
    
    level_data = hierarchy["levels"].get(level)
    if not level_data:
        return None
    
    terms = level_data.get("terms", [])
    
    for i, term_data in enumerate(terms):
        if term_data.get("term") == term:
            return i, term_data
    
    return None


def _create_split_terms_pure(
    original_term_data: Dict[str, Any],
    senses: List[Dict],
    create_backup: bool
) -> List[Dict]:
    """Create split terms from original term data (pure function)."""
    split_terms = []
    
    for sense in senses:
        sense_tag = sense.get("sense_tag", "unknown")
        cluster_id = sense.get("cluster_id", 0)
        
        # Create new term data by copying original
        split_term = copy.deepcopy(original_term_data)
        
        # Update with split-specific information
        split_term["term"] = f"{original_term_data.get('term', '')} ({sense_tag})"
        split_term["sense_tag"] = sense_tag
        split_term["cluster_id"] = cluster_id
        
        if create_backup:
            split_term["parent_term"] = original_term_data.get("term", "")
            split_term["original_term"] = original_term_data.get("term", "")
            split_term["was_split"] = True
        
        split_terms.append(split_term)
    
    return split_terms


def _update_hierarchy_metadata_pure(
    hierarchy: Dict[str, Any],
    applied_count: int,
    total_splits: int
) -> Dict[str, Any]:
    """Update hierarchy metadata with split information (pure function)."""
    updated_hierarchy = copy.deepcopy(hierarchy)
    
    if "metadata" not in updated_hierarchy:
        updated_hierarchy["metadata"] = {}
    
    metadata = updated_hierarchy["metadata"]
    metadata["disambiguation_applied"] = True
    metadata["splits_applied"] = applied_count
    metadata["total_splits"] = total_splits
    
    return updated_hierarchy


# ============================================================================
# LEGACY WRAPPER FUNCTIONS (DEPRECATED)
# ============================================================================

def _create_legacy_llm_function(llm_provider: str = "gemini") -> LLMFunction:
    """Create LLM function from existing imports for legacy compatibility."""
    def llm_function(messages: List[Dict[str, Any]]) -> str:
        return completion(messages, tier="budget", max_tokens=100)
    return llm_function


def _convert_detection_results_to_list(
    detection_results: DetectionResults, 
    web_content: Optional[WebContent] = None
) -> List[DetectionResult]:
    """Convert legacy detection results format to DetectionResult list."""
    converted = []
    
    for term, detection_data in detection_results.items():
        evidence = detection_data.copy()
        confidence = evidence.pop("overall_confidence", 0.5)
        level = evidence.pop("level", 2)
        
        # Inject web_content if provided
        if web_content and term in web_content:
            evidence['web_content'] = web_content[term]
        
        detection_result = DetectionResult(
            term=term,
            level=level,
            method="legacy",  # Add default method since it's required
            confidence=confidence,
            evidence=evidence
        )
        converted.append(detection_result)
    
    return converted


def _convert_proposals_to_legacy_format(proposals: List[SplitProposal]) -> SplitProposals:
    """Convert SplitProposal objects to legacy dictionary format."""
    legacy_proposals = []
    
    for proposal in proposals:
        legacy_proposal = {
            "original_term": proposal.original_term,
            "level": proposal.level,
            "cluster_count": len(proposal.proposed_senses),
            "confidence": proposal.confidence,
            "proposed_senses": [dict(sense) for sense in proposal.proposed_senses]
        }
        
        # Add evidence fields to top level for backward compatibility
        if hasattr(proposal, 'evidence') and proposal.evidence:
            for key, value in proposal.evidence.items():
                if key not in legacy_proposal:
                    legacy_proposal[key] = value
        
        # Add validation status if present
        if proposal.validation_status:
            legacy_proposal["validation_status"] = proposal.validation_status
        
        legacy_proposals.append(legacy_proposal)
    
    return legacy_proposals
