#!/usr/bin/env python3
"""
Command-line interface for sense disambiguation.

Provides a simple interface for detecting and splitting ambiguous terms.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .main import (
    detect_ambiguous_terms,
    split_ambiguous_terms,
    run_disambiguation_pipeline
)
from .utils import load_hierarchy, load_web_content


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def cmd_detect(args):
    """Run ambiguity detection."""
    setup_logging(args.verbose)
    
    # Build config
    config = {
        "min_resources": args.min_resources,
        "min_confidence": args.min_confidence,
        "embedding_model": args.embedding_model,
        "clustering_algorithm": args.clustering,
        "dbscan_eps": args.eps,
        "dbscan_min_samples": args.min_samples,
        "output_dir": args.output
    }
    
    # Load web content if provided
    web_content = None
    if args.web_content:
        web_content = load_web_content(args.web_content)
        logging.info(f"Loaded web content for {len(web_content)} terms")
    
    # Run detection
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
            key=lambda x: x[1].get("overall_confidence", 0),
            reverse=True
        )
        for term, data in sorted_terms[:10]:
            confidence = data.get("overall_confidence", 0)
            level = data.get("level", "?")
            print(f"  - {term} (L{level}, confidence: {confidence:.2f})")
    
    # Save results
    if args.output:
        from .utils import save_results
        output_dir = Path(args.output)
        output_file = save_results(results, output_dir, "detection")
        print(f"\nResults saved to: {output_file}")
    
    return 0 if results else 1


def cmd_split(args):
    """Run sense splitting on detection results."""
    setup_logging(args.verbose)
    
    # Load detection results
    import json
    try:
        with open(args.input, 'r') as f:
            detection_results = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load detection results: {e}")
        return 1
    
    # Build config
    config = {
        "use_llm_validation": not args.no_llm,
        "llm_provider": args.llm_provider,
        "output_dir": args.output
    }
    
    # Load web content if provided
    web_content = None
    if args.web_content:
        web_content = load_web_content(args.web_content)
    
    # Run splitting
    accepted, rejected = split_ambiguous_terms(
        detection_results=detection_results,
        hierarchy_path=args.hierarchy,
        web_content=web_content,
        config=config
    )
    
    # Display results
    print(f"\nSplitting results:")
    print(f"  Accepted: {len(accepted)} splits")
    print(f"  Rejected: {len(rejected)} splits")
    
    if args.verbose and accepted:
        print("\nAccepted splits:")
        for split in accepted[:5]:
            term = split["original_term"]
            senses = [s["sense_tag"] for s in split["proposed_senses"]]
            print(f"  - {term} → {', '.join(senses)}")
    
    # Save results
    if args.output:
        from .utils import save_results
        output_dir = Path(args.output)
        
        results = {
            "accepted": accepted,
            "rejected": rejected,
            "summary": {
                "total_terms": len(detection_results),
                "accepted_splits": len(accepted),
                "rejected_splits": len(rejected)
            }
        }
        
        output_file = save_results(results, output_dir, "splits")
        print(f"\nResults saved to: {output_file}")
    
    return 0


def cmd_run(args):
    """Run complete disambiguation pipeline."""
    setup_logging(args.verbose)
    
    # Build config
    config = {
        "detection_method": args.method,
        "min_resources": args.min_resources,
        "min_confidence": args.min_confidence,
        "embedding_model": args.embedding_model,
        "clustering_algorithm": args.clustering,
        "dbscan_eps": args.eps,
        "dbscan_min_samples": args.min_samples,
        "use_llm_validation": not args.no_llm,
        "llm_provider": args.llm_provider,
        "output_dir": args.output
    }
    
    # Run pipeline
    results = run_disambiguation_pipeline(
        hierarchy_path=args.hierarchy,
        level=args.level,
        web_content_path=args.web_content,
        output_dir=args.output,
        config=config,
        apply_to_hierarchy=args.apply
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
    
    print(f"\nDetection results: {results['detection_file']}")
    print(f"Split results: {results['splits_file']}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sense disambiguation for academic glossary",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        "--hierarchy",
        default="data/hierarchy.json",
        help="Path to hierarchy.json file (default: data/hierarchy.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Detection command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect ambiguous terms"
    )
    detect_parser.add_argument(
        "--method", "-m",
        choices=["embedding", "hierarchy", "global", "hybrid"],
        default="hybrid",
        help="Detection method (default: hybrid)"
    )
    detect_parser.add_argument(
        "--level", "-l",
        type=int,
        choices=[0, 1, 2, 3],
        help="Hierarchy level to process"
    )
    detect_parser.add_argument(
        "--web-content", "-w",
        help="Path to web content JSON file"
    )
    detect_parser.add_argument(
        "--min-resources",
        type=int,
        default=5,
        help="Minimum resources required (default: 5)"
    )
    detect_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)"
    )
    detect_parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2)"
    )
    detect_parser.add_argument(
        "--clustering",
        choices=["dbscan", "hdbscan"],
        default="dbscan",
        help="Clustering algorithm (default: dbscan)"
    )
    detect_parser.add_argument(
        "--eps",
        type=float,
        default=0.45,
        help="DBSCAN epsilon parameter (default: 0.45)"
    )
    detect_parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples for clustering (default: 2)"
    )
    detect_parser.add_argument(
        "--output", "-o",
        default="data/sense_disambiguation",
        help="Output directory (default: data/sense_disambiguation)"
    )
    detect_parser.set_defaults(func=cmd_detect)
    
    # Split command
    split_parser = subparsers.add_parser(
        "split",
        help="Generate sense splits from detection results"
    )
    split_parser.add_argument(
        "input",
        help="Path to detection results JSON"
    )
    split_parser.add_argument(
        "--web-content", "-w",
        help="Path to web content JSON file"
    )
    split_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM validation"
    )
    split_parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini"],
        default="gemini",
        help="LLM provider (default: gemini)"
    )
    split_parser.add_argument(
        "--output", "-o",
        default="data/sense_disambiguation",
        help="Output directory (default: data/sense_disambiguation)"
    )
    split_parser.set_defaults(func=cmd_split)
    
    # Run command (complete pipeline)
    run_parser = subparsers.add_parser(
        "run",
        help="Run complete disambiguation pipeline"
    )
    run_parser.add_argument(
        "--level", "-l",
        type=int,
        choices=[0, 1, 2, 3],
        help="Hierarchy level to process"
    )
    run_parser.add_argument(
        "--web-content", "-w",
        help="Path to web content JSON file"
    )
    run_parser.add_argument(
        "--method", "-m",
        choices=["embedding", "hierarchy", "global", "hybrid"],
        default="hybrid",
        help="Detection method (default: hybrid)"
    )
    run_parser.add_argument(
        "--min-resources",
        type=int,
        default=5,
        help="Minimum resources required (default: 5)"
    )
    run_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)"
    )
    run_parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    run_parser.add_argument(
        "--clustering",
        choices=["dbscan", "hdbscan"],
        default="dbscan",
        help="Clustering algorithm"
    )
    run_parser.add_argument(
        "--eps",
        type=float,
        default=0.45,
        help="DBSCAN epsilon parameter"
    )
    run_parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples for clustering"
    )
    run_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM validation"
    )
    run_parser.add_argument(
        "--llm-provider",
        choices=["openai", "gemini"],
        default="gemini",
        help="LLM provider"
    )
    run_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply splits to hierarchy"
    )
    run_parser.add_argument(
        "--output", "-o",
        default="data/sense_disambiguation",
        help="Output directory"
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()