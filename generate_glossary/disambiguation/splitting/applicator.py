"""
Split proposal application functionality.

Applies validated split proposals to hierarchy data structures.
"""

import copy
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..types import SplitProposal, SplittingConfig

# Type aliases for legacy compatibility
SplitProposals = List[Dict[str, Any]]  # Legacy format
Hierarchy = Dict[str, Any]


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
        # Check if proposal is approved
        if proposal.validation_status != 'approved':
            continue

        term = proposal.original_term
        level = proposal.level
        senses = proposal.proposed_senses

        # Convert tuple of senses to dict format for processing
        senses_dict = {i: sense for i, sense in enumerate(senses)}

        # Apply single split
        updated_hierarchy, was_applied = _apply_single_split_pure(
            hierarchy=updated_hierarchy,
            term=term,
            level=level,
            senses=senses_dict,
            create_backup=config.create_backup if hasattr(config, 'create_backup') else False
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
    senses: Dict[int, Dict],
    create_backup: bool = False
) -> Tuple[Dict[str, Any], bool]:
    """
    Apply a single term split to the hierarchy (pure function).

    Args:
        hierarchy: Hierarchy data to modify
        term: Original term to split
        level: Level of the term
        senses: Dictionary of sense data
        create_backup: Whether to backup original term

    Returns:
        Tuple of (updated_hierarchy, was_applied)
    """
    # Find the term in the hierarchy
    term_location = _find_term_in_hierarchy_pure(hierarchy, term, level)
    if not term_location:
        logging.warning(f"Could not find term '{term}' at level {level} in hierarchy")
        return hierarchy, False

    parent_container = term_location["parent"]
    term_key = term_location["key"]
    original_data = term_location["data"]

    # Create backup if requested
    if create_backup and term_key not in parent_container.get("_backups", {}):
        if "_backups" not in parent_container:
            parent_container["_backups"] = {}
        parent_container["_backups"][term_key] = copy.deepcopy(original_data)

    # Create split terms
    split_terms = _create_split_terms_pure(term, senses, original_data)

    # Remove original term and add split terms
    if term_key in parent_container:
        del parent_container[term_key]

    for split_term_name, split_term_data in split_terms.items():
        parent_container[split_term_name] = split_term_data

    return hierarchy, True


def _find_term_in_hierarchy_pure(
    hierarchy: Dict[str, Any],
    term: str,
    level: int
) -> Optional[Dict[str, Any]]:
    """
    Find a term at a specific level in the hierarchy (pure function).

    Args:
        hierarchy: Hierarchy data to search
        term: Term to find
        level: Level to search at

    Returns:
        Dictionary with parent container, key, and data, or None if not found
    """
    def search_recursive(container, current_level=0, parent=None, key=None):
        if current_level == level:
            # We're at the target level, check if term exists
            if isinstance(container, dict) and term in container:
                return {
                    "parent": container,
                    "key": term,
                    "data": container[term]
                }
            return None

        # Search deeper levels
        if isinstance(container, dict):
            for key, value in container.items():
                if key.startswith("_"):  # Skip metadata keys
                    continue
                result = search_recursive(value, current_level + 1, container, key)
                if result:
                    return result

        return None

    return search_recursive(hierarchy)


def _create_split_terms_pure(
    original_term: str,
    senses: Dict[int, Dict],
    original_data: Any
) -> Dict[str, Any]:
    """
    Create new terms from split senses (pure function).

    Args:
        original_term: The original ambiguous term
        senses: Dictionary of sense data
        original_data: Original term data from hierarchy

    Returns:
        Dictionary of new split terms
    """
    split_terms = {}

    for cluster_id, sense_data in senses.items():
        tag = sense_data["tag"]
        resources = sense_data.get("resources", [])

        # Create new term name
        split_term_name = tag if tag != original_term else f"{original_term}_{cluster_id}"

        # Create term data
        term_data = copy.deepcopy(original_data) if original_data else {}

        # Add metadata about the split
        if not isinstance(term_data, dict):
            term_data = {"value": term_data}

        term_data.update({
            "_split_from": original_term,
            "_sense_tag": tag,
            "_cluster_id": cluster_id,
            "_resources": resources,
            "_split_timestamp": "auto"  # Would be replaced with actual timestamp in real implementation
        })

        split_terms[split_term_name] = term_data

    return split_terms


def _update_hierarchy_metadata_pure(
    hierarchy: Dict[str, Any],
    applied_count: int,
    total_splits: int
) -> Dict[str, Any]:
    """
    Update hierarchy metadata with split information (pure function).

    Args:
        hierarchy: Hierarchy to update
        applied_count: Number of splits that were applied
        total_splits: Total number of splits attempted

    Returns:
        Updated hierarchy with metadata
    """
    if "_metadata" not in hierarchy:
        hierarchy["_metadata"] = {}

    metadata = hierarchy["_metadata"]

    if "splits_applied" not in metadata:
        metadata["splits_applied"] = []

    # Add information about this split operation
    split_info = {
        "applied_count": applied_count,
        "total_splits": total_splits,
        "timestamp": "auto",  # Would be actual timestamp in real implementation
        "success_rate": applied_count / total_splits if total_splits > 0 else 0.0
    }

    metadata["splits_applied"].append(split_info)
    metadata["total_splits_ever"] = metadata.get("total_splits_ever", 0) + applied_count

    return hierarchy


# Legacy wrapper function
def apply_to_hierarchy(
    proposals: SplitProposals,
    hierarchy: Hierarchy,
    create_backup: bool = True
) -> Hierarchy:
    """
    Apply splits to hierarchy (DEPRECATED).

    DEPRECATED: Use apply_splits_to_hierarchy() with SplittingConfig for pure functional approach.
    This function is a legacy wrapper that will be removed in a future version.

    Args:
        proposals: Split proposals in legacy format
        hierarchy: Hierarchy data
        create_backup: Whether to backup original terms

    Returns:
        Updated hierarchy
    """
    from ..utils import _convert_legacy_proposals_to_new

    # Convert legacy format to new format
    proposal_objects = _convert_legacy_proposals_to_new(proposals)

    # Create config
    config = SplittingConfig(
        use_llm=True,
        create_backup=create_backup,
        min_cluster_size=2
    )

    # Apply using new function
    return apply_splits_to_hierarchy(proposal_objects, hierarchy, config)