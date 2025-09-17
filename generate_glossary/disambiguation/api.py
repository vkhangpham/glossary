"""
Public API for term disambiguation.

This module provides the essential functions for detecting and resolving
ambiguous academic terms in the glossary.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path


def disambiguate_terms(
    hierarchy_path: str,
    level: Optional[int] = None,
    web_content_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    method: str = "hybrid",
    apply_splits: bool = False,
    config: Optional[Dict[str, Any]] = None,
    return_immutable: bool = False
) -> Dict[str, Any]:
    """
    Run complete disambiguation pipeline for academic terms.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Specific hierarchy level to process (0-3), or None for all
        web_content_path: Path to web content JSON file
        output_dir: Directory for output files
        method: Detection method ("embedding", "hierarchy", "global", "hybrid")
        apply_splits: Whether to apply splits to hierarchy
        config: Additional configuration options
        return_immutable: Return immutable data structures
        
    Returns:
        Dictionary with detection results, splits, and file paths
    """
    from .main import run_disambiguation_pipeline
    from .utils import convert_from_legacy_format
    
    results = run_disambiguation_pipeline(
        hierarchy_path=hierarchy_path,
        level=level,
        web_content_path=web_content_path,
        output_dir=output_dir,
        config={
            **(config or {}),
            "detection_method": method,
            "apply_to_hierarchy": apply_splits
        }
    )
    
    # Convert results to immutable format if requested
    if return_immutable:
        if "detection_results" in results:
            results["detection_results"] = convert_from_legacy_format(
                list(results["detection_results"].values())
            )
        if "accepted_splits" in results:
            results["accepted_splits"] = convert_from_legacy_format(
                results["accepted_splits"]
            )
        if "rejected_splits" in results:
            results["rejected_splits"] = convert_from_legacy_format(
                results["rejected_splits"]
            )
    
    return results


def detect_ambiguous(
    hierarchy_path: str,
    level: Optional[int] = None,
    method: str = "hybrid",
    web_content: Optional[Dict[str, Any]] = None,
    min_confidence: float = 0.5,
    config: Optional[Dict[str, Any]] = None,
    return_immutable: bool = False
) -> Union[List, Dict[str, Dict[str, Any]]]:
    """
    Detect ambiguous terms in the hierarchy.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Specific level to check, or None for all
        method: Detection method to use
        web_content: Loaded web content dictionary
        min_confidence: Minimum confidence threshold
        config: Additional configuration options
        return_immutable: Return DetectionResult objects instead of legacy dict
        
    Returns:
        List of DetectionResult objects or dictionary mapping ambiguous terms to detection details
    """
    from .main import detect_ambiguous_terms
    from .utils import convert_from_legacy_format
    
    # Merge configuration
    full_config = {"min_confidence": min_confidence}
    if config:
        full_config.update(config)
    
    results = detect_ambiguous_terms(
        hierarchy_path=hierarchy_path,
        level=level,
        method=method,
        web_content=web_content,
        config=full_config
    )
    
    # Convert to immutable format if requested
    if return_immutable:
        return convert_from_legacy_format(list(results.values()))
    
    return results


def split_senses(
    detection_results: Union[List, Dict[str, Dict[str, Any]]],
    hierarchy_path: str,
    use_llm: bool = True,
    llm_provider: str = "gemini",
    config: Optional[Dict[str, Any]] = None,
    return_immutable: bool = False
) -> Union[Tuple[List, List], Tuple[List[Dict], List[Dict]]]:
    """
    Generate and validate sense splits for ambiguous terms.
    
    Args:
        detection_results: DetectionResult objects or output from detect_ambiguous()
        hierarchy_path: Path to hierarchy.json file  
        use_llm: Whether to use LLM validation
        llm_provider: LLM provider for validation
        config: Additional configuration options
        return_immutable: Return SplitProposal objects instead of legacy dict
        
    Returns:
        Tuple of (accepted_splits, rejected_splits) in requested format
    """
    from .main import split_ambiguous_terms
    from .utils import convert_to_legacy_format, convert_from_legacy_format
    
    # Convert immutable detection results to legacy format if needed
    legacy_detection_results = detection_results
    if isinstance(detection_results, list) and hasattr(detection_results[0], 'term'):
        # Convert DetectionResult objects to legacy format
        legacy_detection_results = convert_to_legacy_format(detection_results)
    
    # Merge configuration
    full_config = {
        "use_llm_validation": use_llm,
        "llm_provider": llm_provider
    }
    if config:
        full_config.update(config)
    
    accepted, rejected = split_ambiguous_terms(
        detection_results=legacy_detection_results,
        hierarchy_path=hierarchy_path,
        config=full_config
    )
    
    # Convert to immutable format if requested
    if return_immutable:
        accepted_immutable = convert_from_legacy_format(accepted)
        rejected_immutable = convert_from_legacy_format(rejected)
        return accepted_immutable, rejected_immutable
    
    return accepted, rejected


# Enhanced API functions with immutable structure support

def disambiguate_terms_functional(
    hierarchy_path: str,
    level: Optional[int] = None,
    web_content_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    method: str = "hybrid",
    apply_splits: bool = False,
    config: Optional["DisambiguationConfig"] = None,
    return_immutable: bool = True
) -> Dict[str, Any]:
    """
    Run complete disambiguation pipeline with functional configuration support.
    
    This function supports both legacy dictionary configuration and new immutable
    DisambiguationConfig objects, providing a smooth migration path.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Specific hierarchy level to process (0-3), or None for all
        web_content_path: Path to web content JSON file
        output_dir: Directory for output files
        method: Detection method ("embedding", "hierarchy", "global", "hybrid")
        apply_splits: Whether to apply splits to hierarchy
        config: Functional DisambiguationConfig object or legacy dict
        return_immutable: Return immutable data structures
        
    Returns:
        Dictionary with detection results, splits, and file paths
    """
    from .main import run_disambiguation_pipeline
    from .utils import convert_to_legacy_format, convert_from_legacy_format
    
    # Convert functional config to legacy format if needed
    legacy_config = {}
    if config is not None:
        if hasattr(config, 'methods'):
            # Functional config - convert to legacy
            legacy_config = {
                "detection_method": method,
                "apply_to_hierarchy": apply_splits,
                "min_confidence": config.min_confidence,
                "parallel_processing": config.parallel_processing,
                "use_cache": config.use_cache,
                # Extract method-specific settings
                "embedding_model": config.embedding_config.model_name,
                "clustering_algorithm": config.embedding_config.clustering_algorithm,
                "dbscan_eps": config.embedding_config.eps,
                "dbscan_min_samples": config.embedding_config.min_samples,
                "min_resources": config.embedding_config.min_resources,
                "min_parent_overlap": config.hierarchy_config.min_parent_overlap,
                "max_parent_similarity": config.hierarchy_config.max_parent_similarity,
                "global_eps": config.global_config.eps,
                "global_min_samples": config.global_config.min_samples,
            }
        else:
            # Legacy dict config
            legacy_config = config
    
    legacy_config.update({
        "detection_method": method,
        "apply_to_hierarchy": apply_splits
    })
    
    # Run pipeline with legacy interface
    results = run_disambiguation_pipeline(
        hierarchy_path=hierarchy_path,
        level=level,
        web_content_path=web_content_path,
        output_dir=output_dir,
        config=legacy_config
    )
    
    # Convert results to immutable format if requested
    if return_immutable:
        if "detection_results" in results:
            results["detection_results"] = convert_from_legacy_format(
                list(results["detection_results"].values())
            )
        if "accepted_splits" in results:
            results["accepted_splits"] = convert_from_legacy_format(
                results["accepted_splits"]
            )
        if "rejected_splits" in results:
            results["rejected_splits"] = convert_from_legacy_format(
                results["rejected_splits"]
            )
    
    return results


def detect_ambiguous_functional(
    hierarchy_path: str,
    level: Optional[int] = None,
    method: str = "hybrid",
    web_content: Optional[Dict[str, Any]] = None,
    config: Optional["DisambiguationConfig"] = None,
    return_immutable: bool = True
) -> Union[List["DetectionResult"], Dict[str, Dict[str, Any]]]:
    """
    Detect ambiguous terms with functional configuration support.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Specific level to check, or None for all
        method: Detection method to use
        web_content: Loaded web content dictionary
        config: Functional DisambiguationConfig object
        return_immutable: Return DetectionResult objects instead of legacy dict
        
    Returns:
        List of DetectionResult objects or legacy dictionary format
    """
    from .main import detect_ambiguous_terms
    from .utils import convert_from_legacy_format
    
    # Convert functional config to legacy format if provided
    legacy_config = {}
    if config is not None:
        if hasattr(config, 'methods'):
            legacy_config = {
                "min_confidence": config.min_confidence,
                "embedding_model": config.embedding_config.model_name,
                "clustering_algorithm": config.embedding_config.clustering_algorithm,
                "dbscan_eps": config.embedding_config.eps,
                "dbscan_min_samples": config.embedding_config.min_samples,
                "min_resources": config.embedding_config.min_resources,
                "min_parent_overlap": config.hierarchy_config.min_parent_overlap,
                "max_parent_similarity": config.hierarchy_config.max_parent_similarity,
                "global_eps": config.global_config.eps,
                "global_min_samples": config.global_config.min_samples,
            }
        else:
            legacy_config = config
    
    # Run detection with legacy interface
    results = detect_ambiguous_terms(
        hierarchy_path=hierarchy_path,
        level=level,
        method=method,
        web_content=web_content,
        config=legacy_config
    )
    
    # Convert to immutable format if requested
    if return_immutable:
        return convert_from_legacy_format(list(results.values()))
    
    return results


def split_senses_functional(
    detection_results: Union[List["DetectionResult"], Dict[str, Dict[str, Any]]],
    hierarchy_path: str,
    config: Optional[Dict[str, Any]] = None,
    return_immutable: bool = True
) -> Union[Tuple[List["SplitProposal"], List["SplitProposal"]], Tuple[List[Dict], List[Dict]]]:
    """
    Generate and validate sense splits with immutable data support.
    
    Args:
        detection_results: DetectionResult objects or legacy dict format
        hierarchy_path: Path to hierarchy.json file  
        config: Configuration for splitting and validation
        return_immutable: Return SplitProposal objects instead of legacy dict
        
    Returns:
        Tuple of (accepted_splits, rejected_splits) in requested format
    """
    from .main import split_ambiguous_terms
    from .utils import convert_to_legacy_format, convert_from_legacy_format
    
    # Convert immutable detection results to legacy format if needed
    if isinstance(detection_results, list) and hasattr(detection_results[0], 'term'):
        # Convert DetectionResult objects to legacy format
        legacy_detection_results = convert_to_legacy_format(detection_results)
    else:
        # Already legacy format
        legacy_detection_results = detection_results
    
    # Run splitting with legacy interface
    accepted, rejected = split_ambiguous_terms(
        detection_results=legacy_detection_results,
        hierarchy_path=hierarchy_path,
        config=config or {}
    )
    
    # Convert to immutable format if requested
    if return_immutable:
        accepted_immutable = convert_from_legacy_format(accepted)
        rejected_immutable = convert_from_legacy_format(rejected)
        return accepted_immutable, rejected_immutable
    
    return accepted, rejected


# Type conversion utilities for API users

def convert_to_legacy_api_format(immutable_objects) -> Union[List[Dict], Dict]:
    """
    Public API function to convert immutable objects to legacy format.
    
    Args:
        immutable_objects: DetectionResult or SplitProposal objects
        
    Returns:
        Legacy dictionary format
    """
    from .utils import convert_to_legacy_format
    return convert_to_legacy_format(immutable_objects)


def convert_from_legacy_api_format(legacy_data) -> Union[List, Dict]:
    """
    Public API function to convert legacy format to immutable objects.
    
    Args:
        legacy_data: Legacy dictionary format data
        
    Returns:
        Immutable objects (DetectionResult or SplitProposal)
    """
    from .utils import convert_from_legacy_format
    return convert_from_legacy_format(legacy_data)
