"""
Public API for term disambiguation.

This module provides the essential functions for detecting and resolving
ambiguous academic terms in the glossary.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


def disambiguate_terms(
    hierarchy_path: str,
    level: Optional[int] = None,
    web_content_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    method: str = "hybrid",
    apply_splits: bool = False,
    config: Optional[Dict[str, Any]] = None
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
        
    Returns:
        Dictionary with detection results, splits, and file paths
    """
    from .main import run_disambiguation_pipeline
    
    return run_disambiguation_pipeline(
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


def detect_ambiguous(
    hierarchy_path: str,
    level: Optional[int] = None,
    method: str = "hybrid",
    web_content: Optional[Dict[str, Any]] = None,
    min_confidence: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    """
    Detect ambiguous terms in the hierarchy.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Specific level to check, or None for all
        method: Detection method to use
        web_content: Loaded web content dictionary
        min_confidence: Minimum confidence threshold
        
    Returns:
        Dictionary mapping ambiguous terms to detection details
    """
    from .main import detect_ambiguous_terms
    
    return detect_ambiguous_terms(
        hierarchy_path=hierarchy_path,
        level=level,
        method=method,
        web_content=web_content,
        config={"min_confidence": min_confidence}
    )


def split_senses(
    detection_results: Dict[str, Dict[str, Any]],
    hierarchy_path: str,
    use_llm: bool = True,
    llm_provider: str = "gemini"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate and validate sense splits for ambiguous terms.
    
    Args:
        detection_results: Output from detect_ambiguous()
        hierarchy_path: Path to hierarchy.json file  
        use_llm: Whether to use LLM validation
        llm_provider: LLM provider for validation
        
    Returns:
        Tuple of (accepted_splits, rejected_splits)
    """
    from .main import split_ambiguous_terms
    
    return split_ambiguous_terms(
        detection_results=detection_results,
        hierarchy_path=hierarchy_path,
        config={
            "use_llm_validation": use_llm,
            "llm_provider": llm_provider
        }
    )