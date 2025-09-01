"""
Main entry point for sense disambiguation.

This module orchestrates detection and splitting of ambiguous terms
using a functional pipeline approach.
"""

import logging
from typing import Dict, List, Any, Optional, Literal, Tuple
import json
import time
from pathlib import Path

from .detector import (
    detect_with_embeddings,
    detect_with_hierarchy,
    detect_with_global_clustering,
    merge_detection_results
)
from .splitter import (
    generate_sense_splits,
    validate_splits,
    apply_splits_to_hierarchy
)
from .utils import (
    load_hierarchy,
    load_web_content,
    save_results,
    get_level_params
)

# Type aliases
Terms = List[str]
DetectionMethod = Literal["embedding", "hierarchy", "global", "hybrid"]
DetectionResults = Dict[str, Dict[str, Any]]
SplitProposals = List[Dict[str, Any]]

# Default configuration
DEFAULT_CONFIG = {
    "min_resources": 5,
    "min_confidence": 0.5,
    "embedding_model": "all-MiniLM-L6-v2",
    "clustering_algorithm": "dbscan",
    "dbscan_eps": 0.45,
    "dbscan_min_samples": 2,
    "use_llm_validation": True,
    "llm_provider": "gemini",
    "output_dir": "data/sense_disambiguation"
}


def detect_ambiguous_terms(
    hierarchy_path: str,
    terms: Optional[Terms] = None,
    level: Optional[int] = None,
    method: DetectionMethod = "hybrid",
    web_content: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> DetectionResults:
    """
    Detect potentially ambiguous terms using specified method.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        terms: Optional list of terms to analyze (if None, uses all terms)
        level: Optional hierarchy level to filter results (0-3)
        method: Detection method to use
        web_content: Optional web content for terms
        config: Configuration parameters
        
    Returns:
        Dictionary mapping terms to detection evidence
    """
    start_time = time.time()
    
    # Merge config with defaults
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Load hierarchy if not provided
    hierarchy = load_hierarchy(hierarchy_path)
    
    # Get terms from hierarchy if not provided
    if terms is None:
        terms = list(hierarchy.get("terms", {}).keys())
    
    # Filter by level if specified
    if level is not None:
        terms = [
            term for term in terms
            if hierarchy.get("terms", {}).get(term, {}).get("level") == level
        ]
    
    logging.info(f"Detecting ambiguous terms using {method} method for {len(terms)} terms")
    
    # Initialize results
    results = {}
    
    # Run detection based on method
    if method == "embedding":
        if not web_content:
            web_content = load_web_content(config.get("web_content_path"))
        results = detect_with_embeddings(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            model_name=config["embedding_model"],
            clustering_algorithm=config["clustering_algorithm"],
            eps=config["dbscan_eps"],
            min_samples=config["dbscan_min_samples"],
            min_resources=config["min_resources"]
        )
        
    elif method == "hierarchy":
        results = detect_with_hierarchy(
            terms=terms,
            hierarchy=hierarchy
        )
        
    elif method == "global":
        if not web_content:
            web_content = load_web_content(config.get("web_content_path"))
        results = detect_with_global_clustering(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            model_name=config["embedding_model"],
            clustering_algorithm=config["clustering_algorithm"],
            eps=config.get("global_eps", 0.5),
            min_samples=config.get("global_min_samples", 3)
        )
        
    elif method == "hybrid":
        # Run all methods and merge results
        detection_results = []
        
        # Embedding-based detection
        if web_content or config.get("web_content_path"):
            if not web_content:
                web_content = load_web_content(config.get("web_content_path"))
            
            embedding_results = detect_with_embeddings(
                terms=terms,
                web_content=web_content,
                hierarchy=hierarchy,
                model_name=config["embedding_model"],
                clustering_algorithm=config["clustering_algorithm"],
                eps=config["dbscan_eps"],
                min_samples=config["dbscan_min_samples"],
                min_resources=config["min_resources"]
            )
            detection_results.append(("embedding", embedding_results))
        
        # Hierarchy-based detection
        hierarchy_results = detect_with_hierarchy(
            terms=terms,
            hierarchy=hierarchy
        )
        detection_results.append(("hierarchy", hierarchy_results))
        
        # Global clustering detection (if web content available)
        if web_content:
            global_results = detect_with_global_clustering(
                terms=terms,
                web_content=web_content,
                hierarchy=hierarchy,
                model_name=config["embedding_model"],
                clustering_algorithm=config["clustering_algorithm"],
                eps=config.get("global_eps", 0.5),
                min_samples=config.get("global_min_samples", 3)
            )
            detection_results.append(("global", global_results))
        
        # Merge all detection results
        results = merge_detection_results(detection_results)
    
    # Filter by minimum confidence
    min_confidence = config.get("min_confidence", 0.5)
    filtered_results = {
        term: evidence
        for term, evidence in results.items()
        if evidence.get("overall_confidence", 0) >= min_confidence
    }
    
    elapsed = time.time() - start_time
    logging.info(
        f"Detection complete in {elapsed:.2f}s: "
        f"{len(filtered_results)} ambiguous terms found "
        f"(filtered from {len(results)} by confidence >= {min_confidence})"
    )
    
    return filtered_results


def split_ambiguous_terms(
    detection_results: DetectionResults,
    hierarchy_path: str,
    web_content: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[SplitProposals, SplitProposals]:
    """
    Generate and validate sense splits for ambiguous terms.
    
    Args:
        detection_results: Detection results from detect_ambiguous_terms
        hierarchy_path: Path to hierarchy.json file
        web_content: Optional web content for terms
        config: Configuration parameters
        
    Returns:
        Tuple of (accepted_splits, rejected_splits)
    """
    start_time = time.time()
    
    # Merge config with defaults
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Load hierarchy
    hierarchy = load_hierarchy(hierarchy_path)
    
    # Load web content if needed and not provided
    if config.get("use_llm_validation", True) and not web_content:
        web_content = load_web_content(config.get("web_content_path"))
    
    logging.info(f"Generating splits for {len(detection_results)} ambiguous terms")
    
    # Generate sense splits
    split_proposals = generate_sense_splits(
        detection_results=detection_results,
        hierarchy=hierarchy,
        web_content=web_content,
        use_llm=config.get("use_llm_validation", True),
        llm_provider=config.get("llm_provider", "gemini")
    )
    
    # Validate splits
    accepted, rejected = validate_splits(
        split_proposals=split_proposals,
        hierarchy=hierarchy,
        web_content=web_content,
        use_llm=config.get("use_llm_validation", True),
        llm_provider=config.get("llm_provider", "gemini")
    )
    
    elapsed = time.time() - start_time
    logging.info(
        f"Splitting complete in {elapsed:.2f}s: "
        f"{len(accepted)} accepted, {len(rejected)} rejected"
    )
    
    return accepted, rejected


def run_disambiguation_pipeline(
    hierarchy_path: str,
    level: Optional[int] = None,
    web_content_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    apply_to_hierarchy: bool = False
) -> Dict[str, Any]:
    """
    Run the complete sense disambiguation pipeline.
    
    This is the main entry point that combines detection and splitting.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Optional hierarchy level to process (0-3)
        web_content_path: Optional path to web content JSON
        output_dir: Directory to save results
        config: Configuration parameters
        apply_to_hierarchy: Whether to apply splits to hierarchy
        
    Returns:
        Dictionary containing results and file paths
    """
    start_time = time.time()
    
    # Merge config with defaults
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Set output directory
    output_dir = Path(output_dir or config.get("output_dir", "data/sense_disambiguation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with paths
    if web_content_path:
        config["web_content_path"] = web_content_path
    
    # Load web content once
    web_content = None
    if web_content_path:
        web_content = load_web_content(web_content_path)
        logging.info(f"Loaded web content for {len(web_content)} terms")
    
    # Step 1: Detection
    logging.info("=" * 60)
    logging.info("STEP 1: DETECTING AMBIGUOUS TERMS")
    logging.info("=" * 60)
    
    detection_results = detect_ambiguous_terms(
        hierarchy_path=hierarchy_path,
        level=level,
        method=config.get("detection_method", "hybrid"),
        web_content=web_content,
        config=config
    )
    
    # Save detection results
    detection_file = save_results(
        results=detection_results,
        output_dir=output_dir,
        prefix=f"detection_level{level}" if level is not None else "detection"
    )
    
    if not detection_results:
        logging.info("No ambiguous terms detected")
        return {
            "detection_results": {},
            "accepted_splits": [],
            "rejected_splits": [],
            "detection_file": str(detection_file),
            "elapsed_time": time.time() - start_time
        }
    
    # Step 2: Splitting
    logging.info("=" * 60)
    logging.info("STEP 2: GENERATING SENSE SPLITS")
    logging.info("=" * 60)
    
    accepted_splits, rejected_splits = split_ambiguous_terms(
        detection_results=detection_results,
        hierarchy_path=hierarchy_path,
        web_content=web_content,
        config=config
    )
    
    # Save split results
    splits_file = save_results(
        results={
            "accepted": accepted_splits,
            "rejected": rejected_splits,
            "summary": {
                "total_ambiguous": len(detection_results),
                "accepted_splits": len(accepted_splits),
                "rejected_splits": len(rejected_splits),
                "level": level,
                "config": config
            }
        },
        output_dir=output_dir,
        prefix=f"splits_level{level}" if level is not None else "splits"
    )
    
    # Step 3: Apply to hierarchy (optional)
    updated_hierarchy_path = None
    if apply_to_hierarchy and accepted_splits:
        logging.info("=" * 60)
        logging.info("STEP 3: APPLYING SPLITS TO HIERARCHY")
        logging.info("=" * 60)
        
        updated_hierarchy = apply_splits_to_hierarchy(
            hierarchy_path=hierarchy_path,
            accepted_splits=accepted_splits
        )
        
        # Save updated hierarchy
        updated_hierarchy_path = output_dir / "hierarchy_disambiguated.json"
        with open(updated_hierarchy_path, 'w') as f:
            json.dump(updated_hierarchy, f, indent=2)
        
        logging.info(f"Updated hierarchy saved to {updated_hierarchy_path}")
    
    # Summary
    elapsed = time.time() - start_time
    logging.info("=" * 60)
    logging.info("DISAMBIGUATION PIPELINE COMPLETE")
    logging.info(f"Total time: {elapsed:.2f}s")
    logging.info(f"Ambiguous terms found: {len(detection_results)}")
    logging.info(f"Accepted splits: {len(accepted_splits)}")
    logging.info(f"Rejected splits: {len(rejected_splits)}")
    logging.info("=" * 60)
    
    return {
        "detection_results": detection_results,
        "accepted_splits": accepted_splits,
        "rejected_splits": rejected_splits,
        "detection_file": str(detection_file),
        "splits_file": str(splits_file),
        "updated_hierarchy_path": str(updated_hierarchy_path) if updated_hierarchy_path else None,
        "elapsed_time": elapsed
    }