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
from typing import Dict, List, Any, Optional, Tuple, Literal

from . import embedding_disambiguator
from . import hierarchy_disambiguator
from . import global_disambiguator
from .sense_splitter import generate_splits, validate_splits, apply_to_hierarchy
from .utils import (
    load_hierarchy,
    load_web_content,
    save_results,
    get_level_params
)

# Type aliases
Terms = List[str]
WebContent = Dict[str, Any]
Hierarchy = Dict[str, Any]
DetectionResults = Dict[str, Dict[str, Any]]
DetectionMethod = Literal["embedding", "hierarchy", "global", "hybrid"]


def detect_ambiguous_terms(
    hierarchy_path: str,
    terms: Optional[Terms] = None,
    level: Optional[int] = None,
    method: DetectionMethod = "hybrid",
    web_content: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> DetectionResults:
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
        Dictionary mapping ambiguous terms to detection evidence
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
    
    # Get level-specific parameters
    level_params = get_level_params(level) if level is not None else {}
    
    # Apply detection method(s)
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
    
    # Filter by confidence if specified
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
    
    return results


def split_ambiguous_terms(
    detection_results: DetectionResults,
    hierarchy_path: str,
    web_content: Optional[WebContent] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate and validate sense splits for detected ambiguous terms.
    
    Args:
        detection_results: Detection results from detect_ambiguous_terms
        hierarchy_path: Path to hierarchy.json file
        web_content: Optional web content for split generation
        config: Additional configuration
        
    Returns:
        Tuple of (accepted_splits, rejected_splits)
    """
    config = config or {}
    
    # Load hierarchy
    hierarchy = load_hierarchy(hierarchy_path)
    
    # Generate split proposals
    proposals = generate_splits(
        detection_results=detection_results,
        hierarchy=hierarchy,
        web_content=web_content,
        use_llm=config.get("use_llm_validation", True),
        llm_provider=config.get("llm_provider", "gemini")
    )
    
    # Validate proposals
    accepted, rejected = validate_splits(
        split_proposals=proposals,
        hierarchy=hierarchy,
        web_content=web_content,
        use_llm=config.get("use_llm_validation", True),
        llm_provider=config.get("llm_provider", "gemini")
    )
    
    # Save results if output directory specified
    output_dir = config.get("output_dir")
    if output_dir:
        results = {
            "accepted": accepted,
            "rejected": rejected,
            "summary": {
                "total_proposals": len(proposals),
                "accepted_count": len(accepted),
                "rejected_count": len(rejected)
            }
        }
        output_file = save_results(results, Path(output_dir), "splits")
        logging.info(f"Saved split results to {output_file}")
    
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
    Run complete disambiguation pipeline.
    
    Args:
        hierarchy_path: Path to hierarchy.json file
        level: Optional specific level to process
        web_content_path: Optional path to web content JSON
        output_dir: Directory for output files
        config: Pipeline configuration
        apply_to_hierarchy: Whether to apply splits to hierarchy
        
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
    
    detection_results = detect_ambiguous_terms(
        hierarchy_path=hierarchy_path,
        level=level,
        method=config.get("detection_method", "hybrid"),
        web_content=web_content,
        config=detection_config
    )
    
    detection_file = output_path / "detection_results.json"
    
    # Step 2: Generate and validate splits
    logging.info("=" * 60)
    logging.info("STEP 2: GENERATING AND VALIDATING SPLITS")
    logging.info("=" * 60)
    
    split_config = {
        **config,
        "output_dir": output_path
    }
    
    accepted_splits, rejected_splits = split_ambiguous_terms(
        detection_results=detection_results,
        hierarchy_path=hierarchy_path,
        web_content=web_content,
        config=split_config
    )
    
    splits_file = output_path / "split_results.json"
    
    # Step 3: Optionally apply to hierarchy
    updated_hierarchy_path = None
    if apply_to_hierarchy and accepted_splits:
        logging.info("=" * 60)
        logging.info("STEP 3: APPLYING SPLITS TO HIERARCHY")
        logging.info("=" * 60)
        
        hierarchy = load_hierarchy(hierarchy_path)
        updated_hierarchy = apply_to_hierarchy(
            accepted_splits=accepted_splits,
            hierarchy=hierarchy,
            create_backup=True
        )
        
        # Save updated hierarchy
        updated_hierarchy_path = output_path / "hierarchy_with_splits.json"
        with open(updated_hierarchy_path, 'w') as f:
            json.dump(updated_hierarchy, f, indent=2)
        
        logging.info(f"Saved updated hierarchy to {updated_hierarchy_path}")
    
    elapsed_time = time.time() - start_time
    
    # Return complete results
    return {
        "detection_results": detection_results,
        "accepted_splits": accepted_splits,
        "rejected_splits": rejected_splits,
        "detection_file": str(detection_file),
        "splits_file": str(splits_file),
        "updated_hierarchy_path": str(updated_hierarchy_path) if updated_hierarchy_path else None,
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
        combined_confidence = 1.0 - np.prod([1.0 - c for c in confidences])
        
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
            apply_to_hierarchy=args.apply_splits
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