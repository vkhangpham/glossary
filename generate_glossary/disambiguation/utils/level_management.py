"""
Level management utilities for disambiguation.

Pure functions for managing hierarchy levels and their parameters.
"""

from typing import Dict, List, Any, Optional

# Level-specific parameters
LEVEL_PARAMS = {
    0: {
        "eps": 0.6,
        "min_samples": 3,
        "description": "college or broad academic domain",
        "separation_threshold": 0.7,
        "examples": "Arts and Sciences, Engineering, Medicine, Business, Law"
    },
    1: {
        "eps": 0.5,
        "min_samples": 2,
        "description": "academic department or field",
        "separation_threshold": 0.6,
        "examples": "Computer Science, Psychology, Economics, Biology"
    },
    2: {
        "eps": 0.4,
        "min_samples": 2,
        "description": "research area or specialization",
        "separation_threshold": 0.5,
        "examples": "Machine Learning, Cognitive Psychology, Microeconomics, Molecular Biology"
    },
    3: {
        "eps": 0.3,
        "min_samples": 2,
        "description": "specific research topic or method",
        "separation_threshold": 0.4,
        "examples": "Deep Learning, Memory Formation, Game Theory, CRISPR"
    }
}


def get_level_params(level: int) -> Dict[str, Any]:
    """
    Get parameters for a specific hierarchy level.

    Args:
        level: Hierarchy level (0-3)

    Returns:
        Dictionary of level-specific parameters
    """
    return LEVEL_PARAMS.get(level, LEVEL_PARAMS[2])  # Default to level 2


def get_term_level(term: str, hierarchy: Dict[str, Any]) -> Optional[int]:
    """
    Determine the hierarchical level of a term.

    Args:
        term: Term to find
        hierarchy: Hierarchy data

    Returns:
        Level (0-3) or None if not found
    """
    def search_recursive(container, current_level=0):
        if isinstance(container, dict):
            if term in container:
                return current_level
            for key, value in container.items():
                if key.startswith("_"):  # Skip metadata
                    continue
                result = search_recursive(value, current_level + 1)
                if result is not None:
                    return result
        return None

    return search_recursive(hierarchy)


def filter_terms_by_level(
    terms: List[str],
    hierarchy: Dict[str, Any],
    target_level: int
) -> List[str]:
    """
    Filter terms to only include those at specified level.

    Args:
        terms: List of terms to filter
        hierarchy: Hierarchy data
        target_level: Target level to filter for

    Returns:
        Filtered list of terms
    """
    filtered_terms = []

    for term in terms:
        term_level = get_term_level(term, hierarchy)
        if term_level == target_level:
            filtered_terms.append(term)

    return filtered_terms