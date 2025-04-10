#!/usr/bin/env python3
"""
Command Line Interface for the Sense Disambiguation package.

This script provides a user-friendly CLI for running the various detectors
and the splitter with configurable parameters.
"""

import os
import sys
import argparse
import logging
import warnings
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

# Filter deprecation and runtime warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                       message="invalid value encountered in scalar divide")

# Import the detector and splitter modules
from generate_glossary.sense_disambiguation.detector import (
    ParentContextDetector,
    ResourceClusterDetector,
    HybridAmbiguityDetector,
    HDBSCAN_AVAILABLE
)
from generate_glossary.sense_disambiguation.splitter import SenseSplitter
from generate_glossary.sense_disambiguation.global_clustering import GlobalResourceClusterer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sense_disambiguation_cli")

def find_repo_root() -> str:
    """Find the repository root directory."""
    # Start with the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Traverse up until we find the repo root (where we expect certain directories to exist)
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check for indicators of the repo root
        if os.path.exists(os.path.join(current_dir, "generate_glossary")) and \
           os.path.exists(os.path.join(current_dir, "data")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # If we couldn't find it, default to the script's parent's parent directory
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_default_paths() -> Dict[str, str]:
    """Get default paths for common arguments."""
    repo_root = find_repo_root()
    return {
        "hierarchy_file": os.path.join(repo_root, "data", "hierarchy.json"),
        "final_terms_pattern": os.path.join(repo_root, "data", "final", "lv*", "lv*_final.txt"),
        "detector_output_dir": os.path.join(repo_root, "data", "ambiguity_detection_results"),
        "splitter_output_dir": os.path.join(repo_root, "data", "sense_disambiguation_results")
    }

def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser."""
    defaults = get_default_paths()
    
    parser.add_argument(
        "--hierarchy-file",
        default=defaults["hierarchy_file"],
        help=f"Path to the hierarchy.json file (default: {defaults['hierarchy_file']})"
    )
    parser.add_argument(
        "--final-terms-pattern",
        default=defaults["final_terms_pattern"],
        help=f"Glob pattern for lv*_final.txt files (default: {defaults['final_terms_pattern']})"
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[0, 1, 2, 3],
        help="Hierarchy level to process (0-3, omit to process all levels)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

def setup_detector_parser(subparsers) -> None:
    """Set up the parser for the detector command."""
    defaults = get_default_paths()
    
    detector_parser = subparsers.add_parser(
        "detect",
        help="Run ambiguity detection on the hierarchy",
        description="Detect potentially ambiguous terms in the glossary hierarchy"
    )
    
    # Add common arguments
    setup_common_args(detector_parser)
    
    # Add detector-specific arguments
    detector_parser.add_argument(
        "--detector",
        choices=["parent-context", "resource-cluster", "hybrid"],
        default="hybrid",
        help="Which detection method to use (default: hybrid)"
    )
    detector_parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Name of the sentence transformer model to use (default: all-MiniLM-L6-v2)"
    )
    detector_parser.add_argument(
        "--min-resources",
        type=int,
        default=5,
        help="Minimum resources required for a term to be analyzed (default: 5)"
    )
    detector_parser.add_argument(
        "--clustering",
        choices=["dbscan", "hdbscan"],
        default="dbscan",
        help="Clustering algorithm to use for resource-cluster and hybrid detectors (default: dbscan)"
    )
    # Add clustering algorithm parameters
    detector_parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.45,
        help="DBSCAN epsilon parameter - max distance between samples (default: 0.35)"
    )
    detector_parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples parameter - points required to form a core point (default: 2)"
    )
    detector_parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN min_cluster_size parameter - minimum cluster size (default: 2)"
    )
    detector_parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=2,
        help="HDBSCAN min_samples parameter - similar to DBSCAN's min_samples (default: 2)"
    )
    detector_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for hybrid detector results (default: 0.5)"
    )
    detector_parser.add_argument(
        "--output-dir",
        default=defaults["detector_output_dir"],
        help=f"Directory to save detection results (default: {defaults['detector_output_dir']})"
    )
    detector_parser.add_argument(
        "--save-comprehensive-details",
        action="store_true",
        default=True,
        help="Save comprehensive cluster details including full resource information (default: True)"
    )
    
    # Set the handler function
    detector_parser.set_defaults(func=run_detector)

def setup_splitter_parser(subparsers) -> None:
    """Set up the parser for the splitter command."""
    defaults = get_default_paths()
    
    splitter_parser = subparsers.add_parser(
        "split",
        help="Run sense splitting on ambiguous terms",
        description="Analyze detected ambiguous terms and propose sense splits"
    )
    
    # Add common arguments
    setup_common_args(splitter_parser)
    
    # Add splitter-specific arguments
    splitter_parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the detection results JSON file from detector"
    )
    splitter_parser.add_argument(
        "--cluster-details-file",
        help="Path to the comprehensive cluster details JSON file (optional, enhances analysis)"
    )
    splitter_parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Name of the sentence transformer model to use (default: all-MiniLM-L6-v2)"
    )
    splitter_parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use LLM for tagging and validation (default: True)"
    )
    splitter_parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    splitter_parser.add_argument(
        "--llm-model",
        default=None,
        help="Specific LLM model to use (provider-dependent, default: None)"
    )
    splitter_parser.add_argument(
        "--output-dir",
        default=defaults["splitter_output_dir"],
        help=f"Directory to save split proposals (default: {defaults['splitter_output_dir']})"
    )
    
    # Set the handler function
    splitter_parser.set_defaults(func=run_splitter)

def setup_global_clustering_parser(subparsers) -> None:
    """Set up the parser for the global clustering command."""
    defaults = get_default_paths()
    
    clustering_parser = subparsers.add_parser(
        "global-cluster",
        help="Run global resource clustering across all terms",
        description="Cluster all resources together to identify global patterns and term relationships"
    )
    
    # Add common arguments
    setup_common_args(clustering_parser)
    
    # Add global clustering specific arguments
    clustering_parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Name of the sentence transformer model to use (default: all-MiniLM-L6-v2)"
    )
    clustering_parser.add_argument(
        "--clustering",
        choices=["dbscan", "hdbscan"],
        default="dbscan",
        help="Clustering algorithm to use (default: dbscan)"
    )
    # Add clustering algorithm parameters
    clustering_parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon parameter - max distance between samples (default: 0.5)"
    )
    clustering_parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=3,
        help="DBSCAN min_samples parameter - points required to form a core point (default: 3)"
    )
    clustering_parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size parameter - minimum cluster size (default: 5)"
    )
    clustering_parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=3,
        help="HDBSCAN min_samples parameter - similar to DBSCAN's min_samples (default: 3)"
    )
    clustering_parser.add_argument(
        "--output-dir",
        default=os.path.join(find_repo_root(), "data", "global_clustering_results"),
        help=f"Directory to save clustering results"
    )
    clustering_parser.add_argument(
        "--vector-store-dir",
        default=os.path.join(find_repo_root(), "data", "global_vector_store"),
        help=f"Directory to store vector embeddings"
    )
    
    # Set the handler function
    clustering_parser.set_defaults(func=run_global_clustering)

def run_detector(args: argparse.Namespace) -> None:
    """Run the selected detector with the provided arguments."""
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    levels_to_process = [args.level] if args.level is not None else range(4)
    
    # Create timestamp for subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a timestamped subdirectory for all output files
    output_subdir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Check if HDBSCAN is requested but not available
    if args.clustering == "hdbscan" and not HDBSCAN_AVAILABLE:
        logger.warning("HDBSCAN requested but not available. Falling back to DBSCAN.")
        args.clustering = "dbscan"
    
    logger.info(f"Starting detection using {args.detector} detector")
    logger.info(f"Embedding model: {args.model}")
    if args.detector in ["resource-cluster", "hybrid"]:
        logger.info(f"Clustering algorithm: {args.clustering}")
        if args.clustering == "dbscan":
            logger.info(f"DBSCAN parameters: eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}")
        elif args.clustering == "hdbscan":
            logger.info(f"HDBSCAN parameters: min_cluster_size={args.hdbscan_min_cluster_size}, min_samples={args.hdbscan_min_samples}")
    logger.info(f"Results will be saved to: {output_subdir}")
    
    # Track results from all levels when processing multiple levels
    all_level_results = {}
    all_level_files = []
    
    if args.detector == "parent-context":
        # Process each requested level separately for parent context
        # (parent context doesn't benefit from cross-level analysis)
        logger.info(f"Processing hierarchy levels: {list(levels_to_process)}")
        for level in levels_to_process:
            logger.info(f"Processing level {level}")
            
            detector = ParentContextDetector(
                hierarchy_file_path=args.hierarchy_file,
                final_term_files_pattern=args.final_terms_pattern,
                level=level
            )
            
            ambiguous_terms = detector.detect_ambiguous_terms()
            
            output_filename = f"parent_context_terms_level{level}.txt"
            output_path = os.path.join(output_subdir, output_filename)
            
            with open(output_path, "w") as f:
                for term in sorted(ambiguous_terms):
                    f.write(f"{term}\n")
            
            all_level_files.append(output_path)
            logger.info(f"Found {len(ambiguous_terms)} potentially ambiguous terms at level {level}")
            logger.info(f"Saved results to {output_path}")
    
    elif args.detector == "resource-cluster":
        # Create a detector instance with level=None to cluster all terms together first
        # This provides better clustering by considering the full semantic space
        detector = ResourceClusterDetector(
            hierarchy_file_path=args.hierarchy_file,
            final_term_files_pattern=args.final_terms_pattern,
            model_name=args.model,
            min_resources=args.min_resources,
            clustering_algorithm=args.clustering,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            level=None,  # Process all levels together initially
            output_dir=output_subdir  # Set the timestamped subdir as output
        )
        
        # Run detection once to populate the cache
        logger.info("Running initial clustering on all terms...")
        all_ambiguous_terms = detector.detect_ambiguous_terms()
        logger.info(f"Found {len(all_ambiguous_terms)} potentially ambiguous terms across all levels")
        
        # Save comprehensive cluster details if requested
        if args.save_comprehensive_details:
            comprehensive_filename = f"comprehensive_cluster_details.json"
            comprehensive_path = detector.save_comprehensive_cluster_details(comprehensive_filename)
            logger.info(f"Saved comprehensive cluster details to {comprehensive_path}")
        
        # Then filter for each level
        logger.info(f"Filtering results for hierarchy levels: {list(levels_to_process)}")
        for level in levels_to_process:
            logger.info(f"Processing level {level}")
            
            # Set the level for filtering
            detector.level = level
            
            # Re-run detection (will use cached results and filter by level)
            level_ambiguous_terms = detector.detect_ambiguous_terms()
            
            # Save summary results for this level
            output_filename = f"resource_cluster_terms_level{level}.txt"
            output_path = os.path.join(output_subdir, output_filename)
            
            with open(output_path, "w") as f:
                for term in sorted(level_ambiguous_terms):
                    f.write(f"{term}\n")
            
            # Save detailed results for this level
            detail_filename = f"resource_cluster_results_level{level}.json"
            detail_path = detector.save_detailed_results(detail_filename)
            
            all_level_files.append(detail_path)
            logger.info(f"Found {len(level_ambiguous_terms)} potentially ambiguous terms at level {level}")
            logger.info(f"Saved results to {output_path} and {detail_path}")
    
    elif args.detector == "hybrid":
        # Create a hybrid detector instance
        detector = HybridAmbiguityDetector(
            hierarchy_file_path=args.hierarchy_file,
            final_term_files_pattern=args.final_terms_pattern,
            model_name=args.model,
            min_resources=args.min_resources,
            level=None,  # Process all levels together initially
            output_dir=output_subdir  # Set the timestamped subdir as output
        )
        
        # Configure the clustering algorithm for DBSCAN detector
        if hasattr(detector, 'dbscan_detector'):
            # Always set the DBSCAN parameters
            detector.dbscan_detector.dbscan_eps = args.dbscan_eps
            detector.dbscan_detector.dbscan_min_samples = args.dbscan_min_samples
            logger.info(f"Configured DBSCAN detector with eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}")
            
            # If the user requested hdbscan, swap the algorithm
            if args.clustering == 'hdbscan' and HDBSCAN_AVAILABLE:
                detector.dbscan_detector.clustering_algorithm = 'hdbscan'
                logger.info("Changed primary detector clustering algorithm to HDBSCAN")
            
        # Configure HDBSCAN detector if available
        if hasattr(detector, 'hdbscan_detector') and detector.hdbscan_detector:
            detector.hdbscan_detector.hdbscan_min_cluster_size = args.hdbscan_min_cluster_size
            detector.hdbscan_detector.hdbscan_min_samples = args.hdbscan_min_samples
            logger.info(f"Configured HDBSCAN detector with min_cluster_size={args.hdbscan_min_cluster_size}, min_samples={args.hdbscan_min_samples}")
        
        # Run detection on all levels to populate the cache
        logger.info("Running initial detection on all terms...")
        all_results = detector.detect_ambiguous_terms()
        
        # Save comprehensive cluster details if requested
        if args.save_comprehensive_details:
            # Save DBSCAN cluster details
            if hasattr(detector, 'dbscan_detector'):
                dbscan_filename = f"comprehensive_cluster_details_dbscan.json"
                dbscan_path = detector.dbscan_detector.save_comprehensive_cluster_details(dbscan_filename)
                logger.info(f"Saved DBSCAN comprehensive cluster details to {dbscan_path}")
            
            # Save HDBSCAN cluster details if available
            if hasattr(detector, 'hdbscan_detector') and detector.hdbscan_detector:
                hdbscan_filename = f"comprehensive_cluster_details_hdbscan.json"
                hdbscan_path = detector.hdbscan_detector.save_comprehensive_cluster_details(hdbscan_filename)
                logger.info(f"Saved HDBSCAN comprehensive cluster details to {hdbscan_path}")
        
        all_terms = list(all_results.keys())
        logger.info(f"Found {len(all_terms)} potentially ambiguous terms across all levels")
        
        # Apply confidence threshold and process each level
        logger.info(f"Filtering results for hierarchy levels: {list(levels_to_process)}")
        for level in levels_to_process:
            logger.info(f"Processing level {level}")
            
            # Set the level for filtering
            detector.level = level
            
            # Re-run detection (will use cached results and filter by level)
            detector.detect_ambiguous_terms()
            
            # Get level-specific confidence-filtered results
            confidence_results = detector.get_results_by_confidence(min_confidence=args.min_confidence)
            high_conf = confidence_results.get("high", [])
            med_conf = confidence_results.get("medium", [])
            low_conf = confidence_results.get("low", [])
            
            # Combine results meeting the confidence threshold
            min_conf_results = high_conf + med_conf
            if args.min_confidence < 0.5:
                min_conf_results += low_conf
            
            # Save detailed results for this level
            detail_filename = f"hybrid_detection_results_level{level}.json"
            detail_path = detector.save_results(detail_filename)
            all_level_files.append(detail_path)
            
            # Also save a simple text file with the high+medium confidence terms
            output_filename = f"hybrid_terms_level{level}.txt"
            output_path = os.path.join(output_subdir, output_filename)
            
            with open(output_path, "w") as f:
                for term in sorted(min_conf_results):
                    f.write(f"{term}\n")
            
            logger.info(f"Level {level} results: {len(high_conf)} high confidence, {len(med_conf)} medium confidence")
            logger.info(f"Found {len(min_conf_results)} terms meeting min confidence threshold {args.min_confidence}")
            logger.info(f"Saved results to {output_path} and {detail_path}")
    
    logger.info(f"Detection complete. Results saved to {output_subdir}")
    if len(all_level_files) > 0:
        logger.info(f"Output files: {', '.join(os.path.basename(f) for f in all_level_files)}")
    
    # Create a combined results file if multiple levels were processed
    if args.level is None and args.detector in ["resource-cluster", "hybrid"]:
        logger.info("Creating combined results file for all levels...")
        
        # Create a combined file for all levels
        combined_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "parameters": {
                "detector": args.detector,
                "model_name": args.model,
                "min_resources": args.min_resources,
                "min_confidence": args.min_confidence if hasattr(args, 'min_confidence') else None,
                "clustering": args.clustering
            },
            "level_files": [os.path.basename(f) for f in all_level_files],
            "combined_results": {}
        }
        
        # For hybrid detector, load and combine the results from each level file
        if args.detector == "hybrid":
            for level in levels_to_process:
                level_key = f"level{level}"
                level_file = f"hybrid_detection_results_level{level}.json"
                level_path = os.path.join(output_subdir, level_file)
                
                try:
                    with open(level_path, 'r') as f:
                        level_data = json.load(f)
                        # Extract the detailed results for this level
                        if "detailed_results" in level_data:
                            combined_data["combined_results"][level_key] = level_data["detailed_results"]
                except Exception as e:
                    logger.error(f"Error loading level {level} results: {e}")
        
        # For resource-cluster detector, load and combine the results
        elif args.detector == "resource-cluster":
            for level in levels_to_process:
                level_key = f"level{level}"
                level_file = f"resource_cluster_results_level{level}.json"
                level_path = os.path.join(output_subdir, level_file)
                
                try:
                    with open(level_path, 'r') as f:
                        level_data = json.load(f)
                        # Create a simplified structure for this level
                        combined_data["combined_results"][level_key] = {
                            "cluster_results": level_data.get("cluster_results", {}),
                            "metrics": level_data.get("metrics", {})
                        }
                except Exception as e:
                    logger.error(f"Error loading level {level} results: {e}")
        
        # Save the combined file
        combined_filename = f"{args.detector}_detection_results_combined.json"
        combined_path = os.path.join(output_subdir, combined_filename)
        
        try:
            with open(combined_path, 'w') as f:
                json.dump(combined_data, f, indent=2)
            logger.info(f"Created combined results file: {combined_path}")
        except Exception as e:
            logger.error(f"Error creating combined results file: {e}")

def run_splitter(args: argparse.Namespace) -> None:
    """Run the splitter with the provided arguments."""
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if the input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
        
    # Check if cluster details file exists, if provided
    if args.cluster_details_file and not os.path.exists(args.cluster_details_file):
        logger.error(f"Cluster details file not found: {args.cluster_details_file}")
        sys.exit(1)
    
    logger.info(f"Starting splitting with embedding model: {args.model}")
    
    # Extract timestamp from input file path (format: path/TIMESTAMP/filename)
    # Used for organizing output in the same timestamp directory
    input_dir = os.path.dirname(args.input_file)
    timestamp_dir = os.path.basename(input_dir)
    
    # Create output directory with timestamp if it looks like a timestamp
    if re.match(r'\d{8}_\d{6}', timestamp_dir):
        output_subdir = os.path.join(args.output_dir, timestamp_dir)
        logger.info(f"Using timestamp from input file: {timestamp_dir}")
    else:
        # If not a timestamp dir, create a new one
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(args.output_dir, timestamp)
        logger.info(f"Created new timestamp directory: {timestamp}")
    
    # Ensure the output directory exists
    os.makedirs(output_subdir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_subdir}")
    
    # Check if this is a combined results file
    is_combined_file = False
    combined_results = {}
    
    # Try to load the input file
    try:
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
            # Check if it's a combined file
            if "combined_results" in input_data and isinstance(input_data["combined_results"], dict):
                is_combined_file = True
                combined_results = input_data["combined_results"]
                logger.info(f"Detected combined results file with {len(combined_results)} levels")
    except json.JSONDecodeError:
        logger.error(f"Error parsing input file as JSON: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Determine which levels to process
    if args.level is not None:
        levels_to_process = [args.level]
        logger.info(f"Processing specified level: {args.level}")
    elif is_combined_file:
        # Extract levels from combined results
        levels_to_process = []
        for key in combined_results.keys():
            if key.startswith("level"):
                try:
                    level = int(key[5:])  # Extract number from "level{N}"
                    levels_to_process.append(level)
                except ValueError:
                    logger.warning(f"Skipping invalid level key: {key}")
        levels_to_process.sort()  # Process in order
        logger.info(f"Processing levels from combined file: {levels_to_process}")
    else:
        # Default to all levels
        levels_to_process = range(4)
        logger.info(f"Processing all levels: {list(levels_to_process)}")
    
    # Process each level
    for level in levels_to_process:
        logger.info(f"Processing level {level}")
        
        # Initialize splitter
        splitter = SenseSplitter(
            hierarchy_file_path=args.hierarchy_file,
            candidate_terms_list=[],  # Will be populated from input file
            cluster_results={},       # Will be populated from input file
            embedding_model_name=args.model,  # Model parameter is now guaranteed to exist
            use_llm_for_tags=args.use_llm,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            level=level,
            output_dir=output_subdir
        )
        
        # First, check if cluster details file is provided and load it
        if args.cluster_details_file:
            logger.info(f"Loading comprehensive cluster details from: {args.cluster_details_file}")
            cluster_details_loaded = splitter._load_cluster_results_from_file(args.cluster_details_file)
            if cluster_details_loaded:
                logger.info("Successfully loaded comprehensive cluster details")
            else:
                logger.warning("Failed to load comprehensive cluster details, continuing with regular processing")
        
        # Load results from the main input file
        loaded_successfully = False
        
        if is_combined_file:
            level_key = f"level{level}"
            if level_key in combined_results:
                logger.info(f"Processing combined results for level {level}")
                
                # For each level, we need to create a structure compatible with what the splitter expects
                # The splitter's _load_cluster_results_from_file method expects a file with a 'cluster_results' key
                # containing a mapping of term -> array of cluster labels
                
                # Extract level data from the combined results
                level_data = combined_results[level_key]
                
                # Create a temporary ResourceClusterDetector-like result structure
                temp_data = {
                    "timestamp": datetime.now().isoformat(),
                    "parameters": input_data.get("parameters", {}),
                    "cluster_results": {},
                    "metrics": {}
                }
                
                # Check if we're dealing with hybrid detector output format
                if all(isinstance(level_data.get(term), dict) and "metrics" in level_data.get(term, {}) 
                      for term in list(level_data.keys())[:1]):
                    
                    # We're dealing with hybrid detector format where metrics are nested
                    # We need to extract the actual cluster labels
                    cluster_count = 0
                    
                    for term, term_data in level_data.items():
                        # First check if term was detected by dbscan
                        if "detected_by" in term_data and term_data["detected_by"].get("dbscan", False):
                            # This term was detected by dbscan
                            if "metrics" in term_data and "dbscan" in term_data["metrics"]:
                                dbscan_metrics = term_data["metrics"]["dbscan"]
                                
                                # Store metrics
                                if term not in temp_data["metrics"]:
                                    temp_data["metrics"][term] = {}
                                temp_data["metrics"][term]["dbscan"] = dbscan_metrics
                                
                                # For hybrid detector, we need to create the actual cluster_results 
                                # structure based on cluster_sizes data
                                if "cluster_sizes" in dbscan_metrics:
                                    # We can reconstruct basic clustering info from cluster sizes
                                    clusters = dbscan_metrics.get("cluster_sizes", {})
                                    num_resources = dbscan_metrics.get("num_resources", 0)
                                    
                                    if clusters and num_resources:
                                        # Dynamically create dummy cluster labels for SenseSplitter
                                        # This is only for identifying which terms to process
                                        # The actual clustering will be done by the splitter
                                        dummy_labels = []
                                        for i in range(num_resources):
                                            # Default to noise cluster (-1)
                                            dummy_labels.append(-1)
                                        
                                        # Assign some resources to each valid cluster
                                        idx = 0
                                        for cluster_id_str, size in clusters.items():
                                            if cluster_id_str != "-1":  # Skip noise cluster
                                                cluster_id = int(cluster_id_str)
                                                # Assign 'size' resources to this cluster
                                                for i in range(size):
                                                    if idx < len(dummy_labels):
                                                        dummy_labels[idx] = cluster_id
                                                        idx += 1
                                        
                                        # Store reconstructed labels
                                        temp_data["cluster_results"][term] = dummy_labels
                                        cluster_count += 1
                
                logger.info(f"Reconstructed cluster information for {cluster_count} terms at level {level}")
                if cluster_count == 0:
                    logger.warning(f"Could not reconstruct cluster information for level {level}. Splitting will likely fail.")
            else:
                logger.warning(f"Unknown format in level {level} data. Cannot reconstruct cluster information.")
            
            # Create temporary file for this level's data
            temp_file = os.path.join(output_subdir, f"temp_level{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            try:
                with open(temp_file, 'w') as f:
                    json.dump(temp_data, f)
                # Only load from temp file if no cluster details were provided or they failed to load
                if not args.cluster_details_file or not hasattr(splitter, 'detailed_cluster_info') or not splitter.detailed_cluster_info:
                    loaded_successfully = splitter._load_cluster_results_from_file(temp_file)
                else:
                    # If comprehensive details were already loaded, just populate candidate terms and other essential info
                    try:
                        with open(temp_file, 'r') as f:
                            temp_data = json.load(f)
                            splitter.candidate_terms = list(temp_data.get("cluster_results", {}).keys())
                            if not splitter.cluster_results:  # Only update if not already populated from details file
                                splitter.cluster_results = temp_data.get("cluster_results", {})
                            if not splitter.cluster_metrics:  # Only update if not already populated from details file
                                splitter.cluster_metrics = temp_data.get("metrics", {})
                            loaded_successfully = True
                    except Exception as e:
                        logger.error(f"Error loading temporary file: {e}")
                
                # Clean up temporary file
                try:
                    os.remove(temp_file)
                except:
                    pass
            except Exception as e:
                logger.error(f"Error processing combined results for level {level}: {e}")
        else:
            # This is a direct, non-combined file - use it directly
            # Only load from input file if no cluster details were provided or they failed to load
            if not args.cluster_details_file or not hasattr(splitter, 'detailed_cluster_info') or not splitter.detailed_cluster_info:
                loaded_successfully = splitter._load_cluster_results_from_file(args.input_file)
            else:
                # If comprehensive details were already loaded, just populate candidate terms and other essential info
                try:
                    with open(args.input_file, 'r') as f:
                        input_data = json.load(f)
                        splitter.candidate_terms = list(input_data.get("cluster_results", {}).keys())
                        if not splitter.cluster_results:  # Only update if not already populated from details file
                            splitter.cluster_results = input_data.get("cluster_results", {})
                        if not splitter.cluster_metrics:  # Only update if not already populated from details file
                            splitter.cluster_metrics = input_data.get("metrics", {})
                        loaded_successfully = True
                except Exception as e:
                    logger.error(f"Error loading input file: {e}")
        
        if not loaded_successfully:
            logger.error(f"Failed to load detection results for level {level}")
            continue
        
        # Generate and save split proposals
        accepted, rejected, output_path = splitter.run(
            save_output=True,
            output_filename=f"split_proposals_level{level}.json"  # Simplified filename without timestamp
        )
        
        logger.info(f"Level {level}: Generated {len(accepted)} accepted and {len(rejected)} rejected proposals")
        logger.info(f"Results saved to: {output_path}")
        
        # Add a status line to separate level outputs
        logger.info(f"Completed processing for level {level}")
        logger.info("-" * 40)

def run_global_clustering(args: argparse.Namespace) -> None:
    """Run the global resource clustering."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Starting global resource clustering...")
    
    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vector_store_dir, exist_ok=True)
    
    # Initialize the global resource clusterer
    clusterer = GlobalResourceClusterer(
        hierarchy_file_path=args.hierarchy_file,
        final_term_files_pattern=args.final_terms_pattern,
        model_name=args.model,
        clustering_algorithm=args.clustering,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        vector_store_path=args.vector_store_dir,
        output_dir=args.output_dir
    )
    
    # Run the complete analysis pipeline
    logger.info("Running global clustering analysis pipeline...")
    results = clusterer.run_complete_analysis()
    
    if "error" in results:
        logger.error(f"Error during global clustering: {results['error']}")
        sys.exit(1)
    
    # Report results
    resource_count = results.get("resource_count", 0)
    clustering_metrics = results.get("clustering_metrics", {})
    analysis_results = results.get("analysis_results", {})
    
    logger.info(f"Global clustering complete:")
    logger.info(f"  - Processed {resource_count} resources")
    logger.info(f"  - Found {clustering_metrics.get('num_clusters', 0)} clusters")
    logger.info(f"  - Identified {len(analysis_results.get('potentially_ambiguous_terms', []))} potentially ambiguous terms")
    logger.info(f"  - Results saved to {results.get('results_file')}")
    
    # Create a summary file with key statistics
    summary_file = os.path.join(args.output_dir, f"global_clustering_summary_{timestamp}.txt")
    
    try:
        with open(summary_file, "w") as f:
            f.write(f"=== Global Resource Clustering Summary ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Algorithm: {args.clustering}\n\n")
            
            f.write(f"Resources processed: {resource_count}\n")
            f.write(f"Clusters found: {clustering_metrics.get('num_clusters', 0)}\n")
            f.write(f"Noise points: {clustering_metrics.get('noise_points', 0)} ({clustering_metrics.get('noise_percentage', 0):.1f}%)\n\n")
            
            f.write(f"Potentially ambiguous terms: {len(analysis_results.get('potentially_ambiguous_terms', []))}\n\n")
            
            # List top 20 potentially ambiguous terms
            terms = analysis_results.get('potentially_ambiguous_terms', [])
            if terms:
                f.write("Top potentially ambiguous terms:\n")
                for term in sorted(terms)[:20]:
                    f.write(f"  - {term}\n")
                    
            # List top clusters by size
            cluster_sizes = clustering_metrics.get('cluster_sizes', {})
            if cluster_sizes:
                f.write("\nTop clusters by size:\n")
                sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: int(x[1]), reverse=True)
                for i, (cluster_id, size) in enumerate(sorted_clusters[:10]):
                    f.write(f"  Cluster {cluster_id}: {size} resources\n")
                    
                    # Add top terms for this cluster if available
                    top_terms = analysis_results.get('cluster_top_terms', {}).get(cluster_id, [])
                    if top_terms:
                        top_5_terms = top_terms[:5]
                        f.write(f"     Top terms: {', '.join(term for term, count in top_5_terms)}\n")
                        
        logger.info(f"Summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error writing summary file: {e}")

def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Command Line Interface for the Sense Disambiguation package"
    )
    
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="sub-command help"
    )
    
    # Set up parsers for subcommands
    setup_detector_parser(subparsers)
    setup_splitter_parser(subparsers)
    setup_global_clustering_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate function
    args.func(args)

if __name__ == "__main__":
    main() 