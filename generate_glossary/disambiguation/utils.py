"""
Utility functions for sense disambiguation.

Pure functions for data loading, parameter management, and calculations.
"""

from typing import Dict, List, Any, Optional, Union

# Import functions from new modular structure for backward compatibility
from .utils.text import extract_informative_content, extract_keywords
from .utils.clustering import calculate_separation_score
from .utils.confidence import calculate_confidence_score
from .utils.io import load_hierarchy, load_web_content, save_results

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


# Legacy utility functions for backward compatibility with sense_splitter
def _create_legacy_llm_function(llm_provider: str = "gemini"):
    """Create legacy LLM function wrapper."""
    from generate_glossary.llm import completion

    def llm_fn(prompt: str, **kwargs) -> str:
        return completion(prompt, model_provider=llm_provider, **kwargs)
    return llm_fn


def _convert_detection_results_to_list(detection_results: Dict[str, Dict], web_content: Optional[Dict] = None):
    """Convert legacy detection results format to new list format."""
    from .types import DetectionResult

    result_list = []
    for term, result_data in detection_results.items():
        # Extract evidence data
        evidence = result_data.copy()
        if web_content and term in web_content:
            evidence["web_content"] = web_content[term]

        detection_result = DetectionResult(
            term=term,
            level=result_data.get("level", 2),
            confidence=result_data.get("confidence", 0.5),
            evidence=evidence
        )
        result_list.append(detection_result)

    return result_list


def _convert_proposals_to_legacy_format(proposals):
    """Convert new SplitProposal objects to legacy format."""
    legacy_proposals = []

    for proposal in proposals:
        legacy_proposal = {
            "term": proposal.term,
            "level": proposal.level,
            "confidence": proposal.confidence,
            "method": proposal.method,
            "senses": []
        }

        # Convert senses to legacy format
        for cluster_id, sense_data in proposal.senses.items():
            legacy_sense = {
                "sense_tag": sense_data["tag"],
                "cluster_id": cluster_id,
                "resources": sense_data.get("resources", [])
            }
            legacy_proposal["senses"].append(legacy_sense)

        # Add validation info if present
        if hasattr(proposal, 'validation_result'):
            legacy_proposal["validation"] = proposal.validation_result

        legacy_proposals.append(legacy_proposal)

    return legacy_proposals


def _convert_legacy_proposals_to_new(proposals):
    """Convert legacy format proposals to new SplitProposal objects."""
    from .types import SplitProposal

    new_proposals = []

    for proposal in proposals:
        # Convert legacy senses format
        senses = {}
        for i, sense in enumerate(proposal.get("senses", [])):
            cluster_id = sense.get("cluster_id", i)
            senses[cluster_id] = {
                "tag": sense.get("sense_tag", f"sense_{i}"),
                "resources": sense.get("resources", [])
            }

        new_proposal = SplitProposal(
            term=proposal["term"],
            level=proposal.get("level", 2),
            senses=senses,
            confidence=proposal.get("confidence", 0.5),
            method=proposal.get("method", "clustering")
        )

        # Add validation result if present
        if "validation" in proposal:
            new_proposal.validation_result = proposal["validation"]

        new_proposals.append(new_proposal)

    return new_proposals


# Export all functions for external use
__all__ = [
    # Level management
    "LEVEL_PARAMS",
    "get_level_params",
    "get_term_level",
    "filter_terms_by_level",

    # Re-exported utility functions (for backward compatibility)
    "extract_informative_content",
    "extract_keywords",
    "calculate_separation_score",
    "calculate_confidence_score",
    "load_hierarchy",
    "load_web_content",
    "save_results",

    # Legacy helper functions
    "_create_legacy_llm_function",
    "_convert_detection_results_to_list",
    "_convert_proposals_to_legacy_format",
    "_convert_legacy_proposals_to_new"
]