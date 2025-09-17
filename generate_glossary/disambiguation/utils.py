"""
Utility functions for sense disambiguation.

Pure functions for data loading, parameter management, and calculations.
"""

from typing import Dict, List, Any, Optional, Union

# Import functions from new modular structure for backward compatibility
from .utils.text import extract_informative_content, extract_keywords
from .utils.clustering import calculate_separation_score
from .utils.confidence import calculate_confidence_score

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
    import warnings
    warnings.warn("Legacy LLM function wrapper is deprecated; please migrate to functional API", DeprecationWarning, stacklevel=2)
    
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
            method=result_data.get("method", "legacy"),
            confidence=result_data.get("confidence", 0.5),
            evidence=evidence,
            clusters=result_data.get("clusters"),
            metadata=result_data.get("metadata", {})
        )
        result_list.append(detection_result)

    return result_list


def _convert_proposals_to_legacy_format(proposals):
    """Convert new SplitProposal objects to legacy format."""
    legacy_proposals = []

    for proposal in proposals:
        legacy_proposal = {
            "original_term": proposal.original_term,
            "level": proposal.level,
            "confidence": proposal.confidence,
            "proposed_senses": []
        }

        # Convert senses to legacy format
        for sense_data in proposal.proposed_senses:
            legacy_sense = {
                "sense_tag": sense_data["tag"],
                "cluster_id": sense_data.get("cluster_id", 0),
                "resources": sense_data.get("resources", [])
            }
            legacy_proposal["proposed_senses"].append(legacy_sense)

        # Add validation info if present
        if proposal.validation_status:
            legacy_proposal["validation_status"] = proposal.validation_status

        # Add evidence
        if proposal.evidence:
            legacy_proposal["evidence"] = proposal.evidence

        legacy_proposals.append(legacy_proposal)

    return legacy_proposals


def _convert_legacy_proposals_to_new(proposals):
    """Convert legacy format proposals to new SplitProposal objects."""
    from .types import SplitProposal

    new_proposals = []

    for proposal in proposals:
        # Convert legacy senses format
        proposed_senses = []
        for i, sense in enumerate(proposal.get("proposed_senses", proposal.get("senses", []))):
            sense_dict = {
                "tag": sense.get("sense_tag", f"sense_{i}"),
                "cluster_id": sense.get("cluster_id", i),
                "resources": sense.get("resources", [])
            }
            proposed_senses.append(sense_dict)

        new_proposal = SplitProposal(
            original_term=proposal.get("original_term", proposal.get("term")),
            level=proposal.get("level", 2),
            proposed_senses=tuple(proposed_senses),
            confidence=proposal.get("confidence", 0.5),
            evidence=proposal.get("evidence", {}),
            validation_status=proposal.get("validation_status")
        )

        new_proposals.append(new_proposal)

    return new_proposals


# Export all functions for external use
def _convert_legacy_detection_config(
    config_dict: Dict[str, Any], 
    method: str, 
    level: Optional[int] = None
) -> "DisambiguationConfig":
    """
    Convert legacy dictionary config to functional DisambiguationConfig.
    
    This is a pure function that creates immutable configuration objects
    from legacy dictionary configurations.
    """
    from .types import DisambiguationConfig, EmbeddingConfig, HierarchyConfig, GlobalConfig, LevelConfig
    
    # Get level-specific parameters if available
    level_params = get_level_params(level) if level is not None else {}
    
    # Create method-specific configurations
    embedding_config = EmbeddingConfig(
        model_name=config_dict.get("embedding_model", "all-MiniLM-L6-v2"),
        clustering_algorithm=config_dict.get("clustering_algorithm", "dbscan"),
        eps=config_dict.get("dbscan_eps", level_params.get("eps", 0.45)),
        min_samples=config_dict.get("dbscan_min_samples", level_params.get("min_samples", 2)),
        min_resources=config_dict.get("min_resources", 5),
        max_resources_per_term=config_dict.get("max_resources_per_term", 20)
    )
    
    hierarchy_config = HierarchyConfig(
        min_parent_overlap=config_dict.get("min_parent_overlap", 0.3),
        max_parent_similarity=config_dict.get("max_parent_similarity", 0.7),
        min_occurrence_ratio=config_dict.get("min_occurrence_ratio", 0.1)
    )
    
    global_config = GlobalConfig(
        model_name=config_dict.get("embedding_model", "all-MiniLM-L6-v2"),
        eps=config_dict.get("global_eps", 0.3),
        min_samples=config_dict.get("global_min_samples", 3),
        min_resources=config_dict.get("min_resources", 5),
        max_resources_per_term=config_dict.get("max_resources_per_term", 10)
    )
    
    # Determine which methods to enable based on the method parameter
    if method == "embedding":
        methods = ("embedding",)
    elif method == "hierarchy":
        methods = ("hierarchy",)
    elif method == "global":
        methods = ("global",)
    elif method == "hybrid":
        methods = ("embedding", "hierarchy", "global")
    else:
        methods = ("embedding", "hierarchy", "global")
    
    # Create level configs if level is specified
    level_configs = {}
    if level is not None and level_params:
        level_configs[level] = LevelConfig(
            eps=level_params.get("eps", 0.45),
            min_samples=level_params.get("min_samples", 2),
            min_resources=level_params.get("min_resources", 5)
        )
    
    return DisambiguationConfig(
        methods=methods,
        min_confidence=config_dict.get("min_confidence", 0.5),
        level_configs=level_configs,
        embedding_config=embedding_config,
        hierarchy_config=hierarchy_config,
        global_config=global_config,
        parallel_processing=config_dict.get("parallel_processing", True),
        use_cache=config_dict.get("use_cache", True)
    )


def convert_to_legacy_format(immutable_objects: Union[List, Dict]) -> Union[List[Dict], Dict]:
    """
    Convert immutable objects (DetectionResult, SplitProposal) to legacy format.
    
    This is a pure function that converts functional data structures back to
    legacy dictionary format for backward compatibility.
    """
    if isinstance(immutable_objects, list):
        # Handle list of objects
        if not immutable_objects:
            return []
        
        # Check type of first object to determine conversion method
        first_obj = immutable_objects[0]
        if hasattr(first_obj, 'term') and hasattr(first_obj, 'confidence'):
            if hasattr(first_obj, 'method'):
                # DetectionResult objects
                return _convert_detection_results_to_legacy_dict(immutable_objects)
            else:
                # SplitProposal objects
                return _convert_proposals_to_legacy_format(immutable_objects)
    
    # Handle single object or dict
    return immutable_objects


def convert_from_legacy_format(legacy_data: Union[List[Dict], Dict]) -> Union[List, Dict]:
    """
    Convert legacy dictionary format to immutable objects.
    
    This is a pure function that converts legacy data structures to
    functional immutable objects.
    """
    if isinstance(legacy_data, list):
        if not legacy_data:
            return []
        
        # Check structure to determine what to convert to
        first_item = legacy_data[0]
        if isinstance(first_item, dict):
            if "method" in first_item or "confidence" in first_item:
                # Convert to DetectionResult objects
                return _convert_detection_results_to_list(
                    {item["term"]: item for item in legacy_data}
                )
            elif "proposed_senses" in first_item or "senses" in first_item:
                # Convert to SplitProposal objects
                return _convert_legacy_proposals_to_new(legacy_data)
    
    return legacy_data


def _convert_detection_results_to_legacy_dict(detection_results: List) -> Dict[str, Dict]:
    """Convert DetectionResult list to legacy dictionary format."""
    legacy_dict = {}
    
    for result in detection_results:
        legacy_dict[result.term] = {
            "term": result.term,
            "level": result.level,
            "confidence": result.confidence,
            "method": result.method,
            "evidence": result.evidence,
            "clusters": result.clusters,
            "metadata": result.metadata
        }
        
        # Remove None values for cleaner output
        legacy_dict[result.term] = {
            k: v for k, v in legacy_dict[result.term].items() 
            if v is not None
        }
    
    return legacy_dict


# Export all functions for external use
__all__ = [
    # Level management
    "LEVEL_PARAMS",
    "get_level_params",
    "get_term_level",
    "filter_terms_by_level",

    # Pure utility functions (for backward compatibility)
    "extract_informative_content",
    "extract_keywords",
    "calculate_separation_score",
    "calculate_confidence_score",

    # Legacy helper functions
    "_create_legacy_llm_function",
    "_convert_detection_results_to_list",
    "_convert_proposals_to_legacy_format",
    "_convert_legacy_proposals_to_new",

    # Enhanced conversion functions
    "_convert_legacy_detection_config",
    "convert_to_legacy_format",
    "convert_from_legacy_format",
    "_convert_detection_results_to_legacy_dict"
]