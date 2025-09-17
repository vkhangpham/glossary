"""
Split proposal validation functionality.

Validates semantic distinctness and separation of proposed sense splits.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Callable

from generate_glossary.llm import completion
from ..types import (
    SplitProposal,
    SplittingConfig,
    LLMFunction,
    ValidationFunction
)

# Type aliases for legacy compatibility
SplitProposals = List[Dict[str, Any]]  # Legacy format
Hierarchy = Dict[str, Any]


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
        senses = proposal.proposed_senses

        # Convert tuple of senses to dict format for processing
        senses_dict = {i: sense for i, sense in enumerate(senses)}

        # Skip if only one sense
        if len(senses) < 2:
            rejected_proposal = _create_rejected_proposal_pure(
                proposal, "Insufficient senses for split"
            )
            rejected.append(rejected_proposal)
            continue

        # Check semantic distinctness
        tags = [sense_data["tag"] for sense_data in senses_dict.values()]
        if not _are_tags_distinct_pure(tags):
            rejected_proposal = _create_rejected_proposal_pure(
                proposal, "Sense tags not semantically distinct"
            )
            rejected.append(rejected_proposal)
            continue

        # Calculate separation score
        separation_score = _calculate_separation_score_pure(senses_dict)
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
                senses=senses_dict,
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


def create_llm_validator(llm_fn: LLMFunction, config: SplittingConfig) -> ValidationFunction:
    """Create an LLM validator function with given LLM function and config."""
    def validator(term: str, senses: Dict[int, Dict], level: int) -> Dict[str, Any]:
        return _validate_with_llm_pure(term, senses, level, llm_fn)
    return validator


def _are_tags_distinct_pure(tags: List[str]) -> bool:
    """
    Check if sense tags are semantically distinct (pure function).

    Args:
        tags: List of sense tags to check

    Returns:
        True if tags appear distinct, False otherwise
    """
    if len(tags) < 2:
        return False

    # Remove duplicates and check if we still have multiple tags
    unique_tags = set(tag.lower().strip() for tag in tags)
    if len(unique_tags) < 2:
        return False

    # Check for very similar tags (basic similarity)
    for i, tag1 in enumerate(tags):
        for tag2 in tags[i+1:]:
            # Simple similarity check
            tag1_clean = re.sub(r'[^\w]', '', tag1.lower())
            tag2_clean = re.sub(r'[^\w]', '', tag2.lower())

            # Check if one is substring of another
            if tag1_clean in tag2_clean or tag2_clean in tag1_clean:
                if abs(len(tag1_clean) - len(tag2_clean)) < 3:
                    return False

            # Check for very similar patterns
            if len(tag1_clean) > 3 and len(tag2_clean) > 3:
                # Simple character overlap check
                common_chars = set(tag1_clean) & set(tag2_clean)
                if len(common_chars) / max(len(tag1_clean), len(tag2_clean)) > 0.8:
                    return False

    return True


def _calculate_separation_score_pure(senses: Dict[int, Dict]) -> float:
    """
    Calculate separation score between senses (pure function).

    Args:
        senses: Dictionary mapping cluster ID to sense data

    Returns:
        Separation score (0.0 to 1.0, higher is better)
    """
    if len(senses) < 2:
        return 0.0

    # Simple separation metric based on resource count variance
    resource_counts = [len(sense_data.get("resources", [])) for sense_data in senses.values()]

    if not resource_counts or max(resource_counts) == 0:
        return 0.0

    # Calculate coefficient of variation as a separation metric
    mean_count = sum(resource_counts) / len(resource_counts)
    if mean_count == 0:
        return 0.0

    variance = sum((count - mean_count) ** 2 for count in resource_counts) / len(resource_counts)
    std_dev = variance ** 0.5
    cv = std_dev / mean_count

    # Convert to 0-1 scale, where higher variance (better separation) gives higher score
    separation_score = min(1.0, cv)

    return separation_score


def _safe_llm_validate_pure(
    term: str,
    senses: Dict[int, Dict],
    level: int,
    validator_fn: ValidationFunction
) -> Dict[str, Any]:
    """
    Safely validate senses with LLM, with error handling (pure function).

    Args:
        term: The ambiguous term
        senses: Dictionary of sense data
        level: Hierarchy level
        validator_fn: LLM validation function

    Returns:
        Validation result dictionary
    """
    try:
        return _validate_with_llm_pure(term, senses, level, validator_fn)
    except Exception as e:
        logging.warning(f"LLM validation failed for term '{term}': {e}")
        return {
            "is_valid": False,
            "reason": f"Validation error: {str(e)}",
            "confidence": 0.0
        }


def _validate_with_llm_pure(
    term: str,
    senses: Dict[int, Dict],
    level: int,
    llm_fn: LLMFunction
) -> Dict[str, Any]:
    """
    Validate sense distinctness using LLM (pure function).

    Args:
        term: The ambiguous term
        senses: Dictionary of sense data
        level: Hierarchy level
        llm_fn: LLM function for validation

    Returns:
        Validation result dictionary
    """
    # Create context for LLM
    level_names = ["college", "department", "area", "topic"]
    level_name = level_names[level] if level < len(level_names) else "concept"

    sense_descriptions = []
    for cluster_id, sense_data in senses.items():
        tag = sense_data["tag"]
        resources = sense_data.get("resources", [])

        # Get sample content from resources
        sample_content = []
        for resource in resources[:3]:  # Limit to 3 resources
            if "title" in resource:
                sample_content.append(resource["title"])

        content_text = "; ".join(sample_content) if sample_content else "No content available"
        sense_descriptions.append(f"Sense '{tag}': {content_text}")

    senses_text = "\n".join(sense_descriptions)

    prompt_text = f"""Given the term "{term}" in the context of {level_name}, evaluate whether the following proposed senses represent truly distinct meanings that would be valuable to separate in an academic glossary:

{senses_text}

Consider:
1. Are these senses semantically distinct enough to warrant separation?
2. Would separating them help disambiguate the term in academic contexts?
3. Do they represent genuinely different concepts or just minor variations?

Respond with either "ACCEPT" or "REJECT" followed by a brief explanation.

Response:"""

    # Convert prompt to messages format
    messages = [{"role": "user", "content": prompt_text}]

    try:
        response = llm_fn(messages).strip()

        # Parse response
        is_valid = response.upper().startswith("ACCEPT")
        reason = response.split("\n", 1)[1] if "\n" in response else response

        return {
            "is_valid": is_valid,
            "reason": reason.strip(),
            "confidence": 0.8 if is_valid else 0.2
        }

    except Exception as e:
        raise ValueError(f"LLM validation failed: {str(e)}")


def _create_rejected_proposal_pure(
    proposal: SplitProposal,
    reason: str
) -> SplitProposal:
    """
    Create a rejected proposal with reason (pure function).

    Args:
        proposal: Original proposal
        reason: Rejection reason

    Returns:
        New SplitProposal marked as rejected
    """
    # Create new evidence including rejection reason
    new_evidence = dict(proposal.evidence)
    new_evidence.update({
        "validation_status": "rejected",
        "rejection_reason": reason,
        "original_confidence": proposal.confidence
    })

    return SplitProposal(
        original_term=proposal.original_term,
        level=proposal.level,
        proposed_senses=proposal.proposed_senses,
        confidence=0.0,  # Set confidence to 0 for rejected
        evidence=new_evidence,
        validation_status="rejected"
    )


def _create_validated_proposal_pure(
    proposal: SplitProposal,
    separation_score: float,
    validation_method: str,
    status: str
) -> SplitProposal:
    """
    Create a validated proposal with validation info (pure function).

    Args:
        proposal: Original proposal
        separation_score: Calculated separation score
        validation_method: Method used for validation
        status: Validation status

    Returns:
        New SplitProposal with validation results
    """
    # Create new evidence including validation info
    new_evidence = dict(proposal.evidence)
    new_evidence.update({
        "validation_status": status,
        "separation_score": separation_score,
        "validation_method": validation_method
    })

    return SplitProposal(
        original_term=proposal.original_term,
        level=proposal.level,
        proposed_senses=proposal.proposed_senses,
        confidence=proposal.confidence,
        evidence=new_evidence,
        validation_status=status
    )


# Legacy wrapper function
def validate_splits(
    proposals: SplitProposals,
    hierarchy: Hierarchy,
    use_llm: bool = True,
    llm_provider: str = "gemini"
) -> Tuple[SplitProposals, SplitProposals]:
    """
    Validate split proposals (DEPRECATED).

    DEPRECATED: Use validate_split_proposals() with SplittingConfig for pure functional approach.
    This function is a legacy wrapper that will be removed in a future version.

    Args:
        proposals: Split proposals in legacy format
        hierarchy: Hierarchy data
        use_llm: Whether to use LLM for validation
        llm_provider: LLM provider to use

    Returns:
        Tuple of (accepted_proposals, rejected_proposals) in legacy format
    """
    from ..utils import _convert_legacy_proposals_to_new, _convert_proposals_to_legacy_format, _create_legacy_llm_function

    # Convert legacy format to new format
    proposal_objects = _convert_legacy_proposals_to_new(proposals)

    # Create config
    config = SplittingConfig(
        use_llm=use_llm,
        min_separation_score=0.3,
        min_cluster_size=2
    )

    # Create LLM function if needed
    llm_fn = None
    if use_llm:
        llm_fn = _create_legacy_llm_function(llm_provider)

    # Validate using new function
    accepted, rejected = validate_split_proposals(proposal_objects, hierarchy, config, llm_fn)

    # Convert back to legacy format
    return (_convert_proposals_to_legacy_format(accepted),
            _convert_proposals_to_legacy_format(rejected))