#!/usr/bin/env python3
"""
Main orchestration and CLI for term disambiguation.

Provides both orchestration functions and command-line interface for
detecting and resolving ambiguous academic terms.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Literal, Union

from .detectors import embedding as embedding_disambiguator
from .detectors import hierarchy as hierarchy_disambiguator
from .detectors import global_clustering as global_disambiguator
from .splitting import generate_splits, validate_splits
from .utils.io import load_hierarchy, load_web_content, save_results
from .utils import get_level_params

# Type aliases
Terms = List[str]
WebContent = Dict[str, Any]
Hierarchy = Dict[str, Any]
DetectionResults = Dict[str, Dict[str, Any]]
DetectionMethod = Literal["embedding", "hierarchy", "global", "hybrid"]
# Functional core imports
from .core import create_detection_pipeline, get_detection_summary
from .types import (
    DisambiguationConfig, 
    EmbeddingConfig, 
    HierarchyConfig, 
    GlobalConfig,
    DetectionResult,
    LevelConfig
)
from .config.profiles import get_profile, ACADEMIC_PROFILE
from .splitting import generate_split_proposals, validate_split_proposals, apply_splits_to_hierarchy


def detect_ambiguous_terms(
    hierarchy_path: str,
    terms: Optional[Terms] = None,
    level: Optional[int] = None,
    method: DetectionMethod = "hybrid",
    web_content: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Union[DetectionResults, Tuple[DetectionResults, str]]:
    """
    Main function to detect ambiguous terms using specified method(s).
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        terms: Optional list of specific terms to check
        level: Optional specific level to process (0-3)
        method: Detection method to use
        web_content: Optional pre-loaded web content
        config: Additional configuration parameters
        
    Returns:
        Dictionary mapping ambiguous terms to detection evidence, or
        tuple of (results, output_file_path) if output_dir is specified
    """
    config = config or {}
    
    # Load hierarchy
    hierarchy = load_hierarchy(hierarchy_path)
    
    # Get terms to process
    if terms is None:
        terms = _extract_terms_from_hierarchy(hierarchy, level)
    
    logging.info(f"Processing {len(terms)} terms with method: {method}")
    
    # Load web content if needed and not provided
    if web_content is None and method != "hierarchy":
        web_content_path = config.get("web_content_path")
        if web_content_path:
            web_content = load_web_content(web_content_path)
    
    # Convert legacy config to functional config
    disambiguation_config = _convert_legacy_config(config, method, level)
    
    # Use functional core for detection
    try:
        detection_results = create_detection_pipeline(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            config=disambiguation_config,
            timeout=config.get("timeout", 300),  # 5 minute default timeout
            combination_strategy=config.get("combination_strategy", "union")
        )
        
        # Convert functional results back to legacy format for backward compatibility
        results = _convert_detection_results_to_legacy_format(detection_results)
        
    except Exception as e:
        logging.error(f"Functional detection failed, falling back to legacy: {e}")
        # Fallback to legacy detection for backward compatibility
        results = _legacy_detect_ambiguous_terms(
            terms, hierarchy, web_content, method, config, level
        )
    
    # Filter by confidence if specified (already done in functional core, but kept for legacy compatibility)
    min_confidence = config.get("min_confidence", 0.0)
    if min_confidence > 0:
        results = {
            term: data for term, data in results.items()
            if data.get("confidence", 0) >= min_confidence or
               data.get("overall_confidence", 0) >= min_confidence
        }
    
    # Save results if output directory specified
    output_dir = config.get("output_dir")
    if output_dir:
        output_file = save_results(results, Path(output_dir), "detection")
        logging.info(f"Saved detection results to {output_file}")
        return results, str(output_file)
    
    return results


def split_ambiguous_terms(
    detection_results: DetectionResults,
    hierarchy_path: str,
    web_content: Optional[WebContent] = None,
    config: Optional[Dict[str, Any]] = None
) -> Union[Tuple[List[Dict], List[Dict]], Tuple[List[Dict], List[Dict], str]]:
    """
    Generate and validate sense splits for detected ambiguous terms.
    
    Args:
        detection_results: Detection results from detect_ambiguous_terms
        hierarchy_path: Path to hierarchy.json file
        web_content: Optional web content for split generation
        config: Additional configuration
        
    Returns:
        Tuple of (accepted_splits, rejected_splits), or
        tuple of (accepted_splits, rejected_splits, output_file_path) if output_dir is specified
    """
    config = config or {}
    
    # Load hierarchy
    hierarchy = load_hierarchy(hierarchy_path)
    
    # Try to use functional splitting first
    try:
        # Convert legacy detection results to functional format
        functional_detection_results = _convert_detection_results_to_functional_format(detection_results)

        # Build SplittingConfig from config
        from .types import SplittingConfig
        splitting_config = SplittingConfig(
            use_llm=config.get("use_llm_validation", True),
            llm_provider=config.get("llm_provider", "gemini"),
            min_cluster_size=config.get("min_cluster_size", 2),
            min_separation_score=config.get("min_separation_score", 0.5),
            max_sample_resources=config.get("max_sample_resources", 3),
            create_backup=config.get("create_backup", True),
            tag_generation_max_tokens=config.get("tag_generation_max_tokens", 20),
            validation_max_tokens=config.get("validation_max_tokens", 100)
        )

        # Create LLM function for dependency injection
        llm_fn = with_llm_function(config.get("llm_provider", "gemini"))

        # Generate split proposals using functional approach
        proposals = generate_split_proposals(
            detection_results=functional_detection_results,
            hierarchy=hierarchy,
            config=splitting_config,
            llm_fn=llm_fn
        )

        # Validate proposals using functional approach
        accepted, rejected = validate_split_proposals(
            proposals=proposals,
            hierarchy=hierarchy,
            config=splitting_config,
            llm_fn=llm_fn
        )
        
        # Convert back to legacy format for backward compatibility
        accepted_legacy = [_convert_proposal_to_legacy_format(p) for p in accepted]
        rejected_legacy = [_convert_proposal_to_legacy_format(p) for p in rejected]
        
    except Exception as e:
        logging.error(f"Functional splitting failed, falling back to legacy: {e}")
        # Fallback to legacy splitting methods
        accepted_legacy, rejected_legacy = _legacy_split_ambiguous_terms(
            detection_results, hierarchy, web_content, config
        )
    
    # Save results if output directory specified
    output_dir = config.get("output_dir")
    if output_dir:
        results = {
            "accepted": accepted_legacy,
            "rejected": rejected_legacy,
            "summary": {
                "total_proposals": len(accepted_legacy) + len(rejected_legacy),
                "accepted_count": len(accepted_legacy),
                "rejected_count": len(rejected_legacy)
            }
        }
        output_file = save_results(results, Path(output_dir), "splits")
        logging.info(f"Saved split results to {output_file}")
        return accepted_legacy, rejected_legacy, str(output_file)
    
    return accepted_legacy, rejected_legacy


def run_disambiguation_pipeline(
    hierarchy_path: str,
    level: Optional[int] = None,
    web_content_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    apply_splits: bool = False
) -> Dict[str, Any]:
    """
    Run complete disambiguation pipeline.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Optional specific level to process
        web_content_path: Optional path to web content JSON
        output_dir: Directory for output files
        config: Pipeline configuration
        apply_splits: Whether to apply splits to hierarchy
        
    Returns:
        Dictionary with complete pipeline results
    """
    config = config or {}
    start_time = time.time()
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path("data/disambiguation")
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Load web content if provided
    web_content = None
    if web_content_path:
        web_content = load_web_content(web_content_path)
        logging.info(f"Loaded web content for {len(web_content)} terms")
    
    # Step 1: Detect ambiguous terms
    logging.info("=" * 60)
    logging.info("STEP 1: DETECTING AMBIGUOUS TERMS")
    logging.info("=" * 60)
    
    detection_config = {
        **config,
        "output_dir": output_path
    }
    
    detection_result = detect_ambiguous_terms(
        hierarchy_path=hierarchy_path,
        level=level,
        method=config.get("detection_method", "hybrid"),
        web_content=web_content,
        config=detection_config
    )
    
    # Handle detection results and file path
    if isinstance(detection_result, tuple):
        detection_results, detection_file = detection_result
    else:
        detection_results = detection_result
        detection_file = None
    
    # Step 2: Generate and validate splits
    logging.info("=" * 60)
    logging.info("STEP 2: GENERATING AND VALIDATING SPLITS")
    logging.info("=" * 60)
    
    split_config = {
        **config,
        "output_dir": output_path
    }
    
    split_result = split_ambiguous_terms(
        detection_results=detection_results,
        hierarchy_path=hierarchy_path,
        web_content=web_content,
        config=split_config
    )
    
    # Handle split results and file path
    if len(split_result) == 3:
        accepted_splits, rejected_splits, splits_file = split_result
    else:
        accepted_splits, rejected_splits = split_result
        splits_file = None
    
    # Step 3: Optionally apply to hierarchy
    updated_hierarchy_path = None
    if apply_splits and accepted_splits:
        logging.info("=" * 60)
        logging.info("STEP 3: APPLYING SPLITS TO HIERARCHY")
        logging.info("=" * 60)
        
        hierarchy = load_hierarchy(hierarchy_path)
        updated_hierarchy = apply_splits_to_hierarchy(
            accepted_splits=accepted_splits,
            hierarchy=hierarchy,
            create_backup=True
        )
        
        # Save updated hierarchy
        updated_hierarchy_path = output_path / "hierarchy_with_splits.json"
        with open(updated_hierarchy_path, 'w') as f:
            json.dump(updated_hierarchy, f, indent=2)
        
        logging.info(f"Saved updated hierarchy to {updated_hierarchy_path}")
        updated_hierarchy_path = str(updated_hierarchy_path)
    
    elapsed_time = time.time() - start_time
    
    # Return complete results with actual file paths
    return {
        "detection_results": detection_results,
        "accepted_splits": accepted_splits,
        "rejected_splits": rejected_splits,
        "detection_file": detection_file,
        "splits_file": splits_file,
        "updated_hierarchy_path": updated_hierarchy_path,
        "elapsed_time": elapsed_time,
        "config": config
    }


# Helper functions

def _extract_terms_from_hierarchy(
    hierarchy: Hierarchy,
    level: Optional[int] = None
) -> Terms:
    """Extract terms from hierarchy, optionally filtering by level."""
    terms = []
    
    if "levels" in hierarchy:
        # New format with levels
        for level_idx, level_data in enumerate(hierarchy["levels"]):
            if level is not None and level_idx != level:
                continue
            
            for term_data in level_data.get("terms", []):
                term = term_data.get("term")
                if term:
                    terms.append(term)
    else:
        # Old format
        for term, data in hierarchy.get("terms", {}).items():
            if level is not None and data.get("level") != level:
                continue
            terms.append(term)
    
    return terms


def _merge_detection_results(
    results_list: List[DetectionResults],
    min_confidence: float = 0.5
) -> DetectionResults:
    """Merge detection results from multiple methods."""
    from functools import reduce
    from operator import mul
    
    merged = {}
    
    # Collect all detected terms
    all_terms = set()
    for results in results_list:
        all_terms.update(results.keys())
    
    # Merge evidence for each term
    for term in all_terms:
        term_evidence = []
        methods_detected = []
        
        for results in results_list:
            if term in results:
                evidence = results[term]
                term_evidence.append(evidence)
                methods_detected.append(evidence.get("method", "unknown"))
        
        # Calculate combined confidence using noisy-OR
        confidences = [e.get("confidence", 0.5) for e in term_evidence]
        combined_confidence = 1.0 - reduce(lambda acc, c: acc * (1.0 - c), confidences, 1.0)
        
        # Only include if above threshold
        if combined_confidence >= min_confidence:
            merged[term] = {
                "term": term,
                "methods_detected": methods_detected,
                "evidence_count": len(term_evidence),
                "evidence": term_evidence,
                "overall_confidence": combined_confidence
            }
            
            # Add level if available
            for evidence in term_evidence:
                if "level" in evidence:
                    merged[term]["level"] = evidence["level"]
                    break
    
    return merged


# CLI functionality


# Legacy conversion and compatibility functions

def _convert_legacy_config(
    config: Dict[str, Any], 
    method: DetectionMethod, 
    level: Optional[int] = None
) -> DisambiguationConfig:
    """Convert legacy dictionary config to functional DisambiguationConfig."""
    
    # Get level-specific parameters if available
    level_params = get_level_params(level) if level is not None else {}
    
    # Create method-specific configurations
    embedding_config = EmbeddingConfig(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
        clustering_algorithm=config.get("clustering_algorithm", "dbscan"),
        eps=config.get("dbscan_eps", level_params.get("eps", 0.45)),
        min_samples=config.get("dbscan_min_samples", level_params.get("min_samples", 2)),
        min_resources=config.get("min_resources", 5)
    )
    
    hierarchy_config = HierarchyConfig(
        min_parent_overlap=config.get("min_parent_overlap", 0.3),
        max_parent_similarity=config.get("max_parent_similarity", 0.7),
        enable_web_enhancement=config.get("enable_web_enhancement", True),
        max_web_resources_for_keywords=config.get("max_web_resources_for_keywords", 5)
    )
    
    global_config = GlobalConfig(
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
        eps=config.get("global_eps", 0.3),
        min_samples=config.get("global_min_samples", 3),
        min_resources=config.get("min_resources", 5),
        max_resources_per_term=config.get("max_resources_per_term", 10),
        min_total_resources=config.get("min_total_resources", 50)
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
            description=level_params.get("description", f"Level {level} configuration"),
            separation_threshold=level_params.get("separation_threshold", 0.5),
            examples=level_params.get("examples", "")
        )
    
    return DisambiguationConfig(
        methods=methods,
        min_confidence=config.get("min_confidence", 0.5),
        level_configs=level_configs,
        embedding_config=embedding_config,
        hierarchy_config=hierarchy_config,
        global_config=global_config,
        parallel_processing=config.get("parallel_processing", True),
        use_cache=config.get("use_cache", True)
    )


def _convert_detection_results_to_legacy_format(
    detection_results: List[DetectionResult]
) -> DetectionResults:
    """Convert functional DetectionResult objects to legacy dictionary format."""
    legacy_results = {}
    
    for result in detection_results:
        legacy_results[result.term] = {
            "term": result.term,
            "level": result.level,
            "confidence": result.confidence,
            "method": result.method,
            "evidence": result.evidence,
            "clusters": result.clusters,
            "metadata": result.metadata
        }
        
        # Remove None values for cleaner output
        legacy_results[result.term] = {
            k: v for k, v in legacy_results[result.term].items() 
            if v is not None
        }
    
    return legacy_results


def _legacy_detect_ambiguous_terms(
    terms: Terms,
    hierarchy: Hierarchy,
    web_content: Optional[WebContent],
    method: DetectionMethod,
    config: Dict[str, Any],
    level: Optional[int] = None
) -> DetectionResults:
    """
    Fallback to legacy detection methods for backward compatibility.
    
    This function preserves the original detection logic when the functional
    core fails, ensuring the system remains operational during migration.
    """
    import warnings
    warnings.warn("Legacy detection is deprecated; please migrate to functional API", DeprecationWarning, stacklevel=2)
    
    # Get level-specific parameters
    level_params = get_level_params(level) if level is not None else {}
    
    # Apply detection method(s) using legacy code
    if method == "embedding":
        results = embedding_disambiguator.detect(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            clustering_algorithm=config.get("clustering_algorithm", "dbscan"),
            eps=config.get("dbscan_eps", level_params.get("eps", 0.45)),
            min_samples=config.get("dbscan_min_samples", level_params.get("min_samples", 2)),
            min_resources=config.get("min_resources", 5)
        )
    
    elif method == "hierarchy":
        results = hierarchy_disambiguator.detect(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            min_parent_overlap=config.get("min_parent_overlap", 0.3),
            max_parent_similarity=config.get("max_parent_similarity", 0.7)
        )
    
    elif method == "global":
        results = global_disambiguator.detect(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            eps=config.get("global_eps", 0.3),
            min_samples=config.get("global_min_samples", 3),
            min_resources=config.get("min_resources", 5),
            max_resources_per_term=config.get("max_resources_per_term", 10)
        )
    
    elif method == "hybrid":
        # Run all methods and merge results
        results = {}
        
        # Embedding-based detection
        embedding_results = embedding_disambiguator.detect(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            clustering_algorithm=config.get("clustering_algorithm", "dbscan"),
            eps=config.get("dbscan_eps", level_params.get("eps", 0.45)),
            min_samples=config.get("dbscan_min_samples", level_params.get("min_samples", 2)),
            min_resources=config.get("min_resources", 5)
        )
        
        # Hierarchy-based detection
        hierarchy_results = hierarchy_disambiguator.detect(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            min_parent_overlap=config.get("min_parent_overlap", 0.3),
            max_parent_similarity=config.get("max_parent_similarity", 0.7)
        )
        
        # Global clustering detection
        global_results = global_disambiguator.detect(
            terms=terms,
            web_content=web_content,
            hierarchy=hierarchy,
            model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
            eps=config.get("global_eps", 0.3),
            min_samples=config.get("global_min_samples", 3),
            min_resources=config.get("min_resources", 5),
            max_resources_per_term=config.get("max_resources_per_term", 10)
        )
        
        # Merge results
        results = _merge_detection_results(
            [embedding_results, hierarchy_results, global_results],
            min_confidence=config.get("min_confidence", 0.5)
        )
    
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    return results


def _convert_detection_results_to_functional_format(
    legacy_results: DetectionResults
) -> List[DetectionResult]:
    """Convert legacy dictionary format to functional DetectionResult objects."""
    functional_results = []
    
    for term, data in legacy_results.items():
        result = DetectionResult(
            term=term,
            level=data.get("level", 2),  # Default to level 2 if not specified
            confidence=data.get("confidence", 0.5),
            method=data.get("method", "unknown"),
            evidence=data.get("evidence", {}),
            clusters=data.get("clusters"),
            metadata=data.get("metadata", {})
        )
        functional_results.append(result)
    
    return functional_results

def _convert_proposal_to_legacy_format(proposal) -> Dict[str, Any]:
    """Convert functional SplitProposal to legacy dictionary format."""
    return {
        "term": proposal.original_term,
        "senses": proposal.proposed_senses,
        "confidence": proposal.confidence,
        "evidence": proposal.evidence,
        "is_valid": proposal.validation_status == "approved",
        "validation_reason": proposal.validation_status or '',
        "validation_status": proposal.validation_status
    }


def _legacy_split_ambiguous_terms(
    detection_results: DetectionResults,
    hierarchy: Hierarchy,
    web_content: Optional[WebContent],
    config: Dict[str, Any]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Fallback to legacy splitting methods for backward compatibility.
    
    This function preserves the original splitting logic when the functional
    approach fails, ensuring the system remains operational during migration.
    """
    import warnings
    warnings.warn("Legacy splitting is deprecated; please migrate to functional API", DeprecationWarning, stacklevel=2)
    
    # Generate split proposals using legacy method
    proposals = generate_splits(
        detection_results=detection_results,
        hierarchy=hierarchy,
        web_content=web_content,
        use_llm=config.get("use_llm_validation", True),
        llm_provider=config.get("llm_provider", "gemini")
    )
    
    # Validate proposals using legacy method
    accepted, rejected = validate_splits(
        proposals=proposals,
        hierarchy=hierarchy,
        use_llm=config.get("use_llm_validation", True),
        llm_provider=config.get("llm_provider", "gemini")
    )
    
    return accepted, rejected

def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect and resolve ambiguous terms in academic glossary",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main arguments
    parser.add_argument(
        "hierarchy",
        nargs="?",
        default="data/hierarchy.json",
        help="Path to hierarchy.json file (default: data/hierarchy.json)"
    )
    
    parser.add_argument(
        "--level", "-l",
        type=int,
        choices=[0, 1, 2, 3],
        help="Specific hierarchy level to process"
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["embedding", "hierarchy", "global", "hybrid"],
        default="hybrid",
        help="Detection method (default: hybrid)"
    )
    
    parser.add_argument(
        "--web-content", "-w",
        help="Path to web content JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="data/disambiguation",
        help="Output directory (default: data/disambiguation)"
    )
    
    # Detection parameters
    parser.add_argument(
        "--min-resources",
        type=int,
        default=5,
        help="Minimum resources required (default: 5)"
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--clustering",
        choices=["dbscan", "hdbscan"],
        default="dbscan",
        help="Clustering algorithm (default: dbscan)"
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=0.45,
        help="DBSCAN epsilon parameter (default: 0.45)"
    )
    
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples for clustering (default: 2)"
    )
    
    # Split parameters
    parser.add_argument(
        "--generate-splits",
        action="store_true",
        help="Generate split proposals after detection"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM validation"
    )
    
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini"],
        default="gemini",
        help="LLM provider (default: gemini)"
    )
    
    parser.add_argument(
        "--apply-splits",
        action="store_true",
        help="Apply validated splits to hierarchy"
    )
    
    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Build configuration
    config = {
        "min_resources": args.min_resources,
        "min_confidence": args.min_confidence,
        "embedding_model": args.embedding_model,
        "clustering_algorithm": args.clustering,
        "dbscan_eps": args.eps,
        "dbscan_min_samples": args.min_samples,
        "use_llm_validation": not args.no_llm,
        "llm_provider": args.llm_provider,
        "detection_method": args.method
    }
    
    # Run pipeline or just detection
    if args.generate_splits:
        # Run full pipeline
        results = run_disambiguation_pipeline(
            hierarchy_path=args.hierarchy,
            level=args.level,
            web_content_path=args.web_content,
            output_dir=args.output,
            config=config,
            apply_splits=args.apply_splits
        )
        
        # Display summary
        print("\n" + "=" * 60)
        print("DISAMBIGUATION PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Ambiguous terms found: {len(results['detection_results'])}")
        print(f"Accepted splits: {len(results['accepted_splits'])}")
        print(f"Rejected splits: {len(results['rejected_splits'])}")
        print(f"Total time: {results['elapsed_time']:.2f}s")
        
        if results.get("updated_hierarchy_path"):
            print(f"\nUpdated hierarchy: {results['updated_hierarchy_path']}")
    else:
        # Just run detection
        web_content = None
        if args.web_content:
            web_content = load_web_content(args.web_content)
        
        config["output_dir"] = args.output
        
        results = detect_ambiguous_terms(
            hierarchy_path=args.hierarchy,
            level=args.level,
            method=args.method,
            web_content=web_content,
            config=config
        )
        
        # Display results
        print(f"\nFound {len(results)} ambiguous terms")
        
        if args.verbose and results:
            print("\nAmbiguous terms by confidence:")
            sorted_terms = sorted(
                results.items(),
                key=lambda x: x[1].get("overall_confidence", x[1].get("confidence", 0)),
                reverse=True
            )
            for term, data in sorted_terms[:10]:
                confidence = data.get("overall_confidence", data.get("confidence", 0))
                level = data.get("level", "?")
                print(f"  - {term} (L{level}, confidence: {confidence:.2f})")
    
    return 0


if __name__ == "__main__":
    import numpy as np  # Import here for CLI usage
    sys.exit(main())