"""
Sense splitting functionality for ambiguous terms (LEGACY WRAPPERS).

This module provides backward-compatible wrapper functions that delegate to the
new pure functional implementations in the splitting module. All functions in this
module are deprecated and will be removed in a future version.

Provides legacy compatibility for:
- generate_splits()
- validate_splits()
- apply_to_hierarchy()
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple

from .splitting import (
    generate_split_proposals,
    validate_split_proposals,
    apply_splits_to_hierarchy
)
from .types import SplitProposal, SplittingConfig, LLMFunction
from .utils import (
    _convert_detection_results_to_list as convert_detection_results_to_list,
    _convert_proposals_to_legacy_format as convert_proposals_to_legacy_format,
    _convert_legacy_proposals_to_new as convert_legacy_proposals_to_new,
    _create_legacy_llm_function as create_legacy_llm_function
)

# Type aliases for legacy compatibility
DetectionResults = Dict[str, Dict[str, Any]]
SplitProposals = List[Dict[str, Any]]
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
    This function is a thin legacy wrapper that will be removed in a future version.

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

    # Convert legacy format to new format
    detection_results_list = convert_detection_results_to_list(detection_results, web_content)

    # Create configuration
    config = SplittingConfig(
        use_llm=use_llm,
        llm_provider=llm_provider
    )

    # Create LLM function if needed
    llm_fn = create_legacy_llm_function(llm_provider) if use_llm else None

    # Call pure function
    proposals = generate_split_proposals(
        detection_results=detection_results_list,
        hierarchy=hierarchy,
        config=config,
        llm_fn=llm_fn
    )

    # Convert back to legacy format
    legacy_proposals = convert_proposals_to_legacy_format(proposals)

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
    This function is a thin legacy wrapper that will be removed in a future version.

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
    proposals = convert_legacy_proposals_to_new(split_proposals)

    # Create configuration
    config = SplittingConfig(
        use_llm=use_llm,
        llm_provider=llm_provider
    )

    # Create LLM function if needed
    llm_fn = create_legacy_llm_function(llm_provider) if use_llm else None

    # Call pure function
    accepted_proposals, rejected_proposals = validate_split_proposals(
        proposals=proposals,
        hierarchy=hierarchy,
        config=config,
        llm_fn=llm_fn
    )

    # Convert back to legacy format
    accepted_legacy = convert_proposals_to_legacy_format(accepted_proposals)
    rejected_legacy = convert_proposals_to_legacy_format(rejected_proposals)

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
    This function is a thin legacy wrapper that will be removed in a future version.

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
    proposals = convert_legacy_proposals_to_new(accepted_splits)

    # Mark all proposals as approved since they're accepted splits
    approved_proposals = []
    for proposal in proposals:
        if proposal.validation_status != "approved":
            # Create new proposal with approved status
            approved_proposal = SplitProposal(
                original_term=proposal.original_term,
                level=proposal.level,
                proposed_senses=proposal.proposed_senses,
                confidence=proposal.confidence,
                evidence=proposal.evidence,
                validation_status="approved"
            )
            approved_proposals.append(approved_proposal)
        else:
            approved_proposals.append(proposal)

    # Create configuration
    config = SplittingConfig(create_backup=create_backup)

    # Call pure function
    updated_hierarchy = apply_splits_to_hierarchy(
        proposals=approved_proposals,
        hierarchy=hierarchy,
        config=config
    )

    # Log applied count from metadata
    metadata = updated_hierarchy.get("_metadata", {})
    applied_info = metadata.get("splits_applied", [])
    applied_count = applied_info[-1]["applied_count"] if applied_info else 0

    logging.info(f"Applied {applied_count} splits to hierarchy")

    return updated_hierarchy