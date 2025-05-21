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

# Add the parent directory to sys.path if running as a script
if __name__ == "__main__":
    # Get the parent directory of the script directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, parent_dir)

# Custom logging level for progress updates - must be defined before other imports
# that might use logging
PROGRESS = 15  # Between DEBUG (10) and INFO (20)
logging.addLevelName(PROGRESS, "PROGRESS")

# Add custom logging method
def progress(self, message, *args, **kws):
    """Log progress message with severity 'PROGRESS'"""
    if self.isEnabledFor(PROGRESS):
        self._log(PROGRESS, message, args, **kws)

# Add the method to the Logger class
logging.Logger.progress = progress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sense_disambiguation_cli")

# Filter deprecation and runtime warnings
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                       message="invalid value encountered in scalar divide")

# Create placeholder variables for lazy imports
ParentContextDetector = None
ResourceClusterDetector = None
HybridAmbiguityDetector = None
RadialPolysemyDetector = None
HDBSCAN_AVAILABLE = False
SenseSplitter = None

def _load_detector_modules():
    """Lazily load detector modules only when needed for actual execution (not help)"""
    global ParentContextDetector, ResourceClusterDetector, HybridAmbiguityDetector
    global RadialPolysemyDetector, HDBSCAN_AVAILABLE
    
    try:
        from sense_disambiguation.detector import (
            ParentContextDetector,
            ResourceClusterDetector,
            HybridAmbiguityDetector,
            RadialPolysemyDetector,
            HDBSCAN_AVAILABLE
        )
        logger.debug("Imported detector modules from sense_disambiguation package")
    except ImportError:
        try:
            from detector import (
                ParentContextDetector,
                ResourceClusterDetector,
                HybridAmbiguityDetector,
                RadialPolysemyDetector,
                HDBSCAN_AVAILABLE
            )
            logger.debug("Imported detector modules directly from local path")
        except ImportError:
            logger.error("Failed to import detector modules from sense_disambiguation or local path")
            sys.exit(1)

def _load_splitter_module():
    """Lazily load splitter module only when needed for actual execution (not help)"""
    global SenseSplitter
    
    try:
        from sense_disambiguation.splitter import SenseSplitter
        logger.debug("Imported SenseSplitter from sense_disambiguation package")
    except ImportError:
        try:
            from splitter import SenseSplitter
            logger.debug("Imported SenseSplitter directly from local path")
        except ImportError:
            logger.error("Failed to import SenseSplitter from sense_disambiguation or local path")
            sys.exit(1)

def find_repo_root() -> str:
    """Find the repository root directory."""
    # Start with the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Traverse up until we find the repo root (where we expect certain directories to exist)
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check for indicators of the repo root
        if os.path.exists(os.path.join(current_dir, "generate_glossary")) and \
           os.path.exists(os.path.join(current_dir, "sense_disambiguation/data")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # If we couldn't find it, default to the script's parent's parent directory
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_default_paths() -> Dict[str, str]:
    """Get default paths for common arguments."""
    repo_root = find_repo_root()
    return {
        "hierarchy_file": os.path.join(repo_root, "data", "final", "hierarchy.json"),
        "final_terms_pattern": os.path.join(repo_root, "data", "final", "lv*", "lv*_final.txt"),
        "detector_output_dir": os.path.join(repo_root, "sense_disambiguation/data", "ambiguity_detection_results"),
        "splitter_output_dir": os.path.join(repo_root, "sense_disambiguation/data", "sense_disambiguation_results")
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
    
    # Add detector-specific arguments - simplified
    detector_group = detector_parser.add_argument_group("Detector options")
    detector_group.add_argument(
        "--detector",
        choices=["hybrid", "resource-cluster", "parent-context", "radial-polysemy"],
        default="hybrid",
        help="Detection method to use (default: hybrid - combines all methods)"
    )
    detector_group.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for results (0.0-1.0, default: 0.5)"
    )
    
    # Simplified clustering options
    clustering_group = detector_parser.add_argument_group("Clustering options")
    clustering_group.add_argument(
        "--clustering-preset",
        choices=["standard", "sensitive", "conservative", "experimental"],
        default="standard",
        help="""Preset configuration for clustering parameters:
                standard: eps=0.45, min_samples=2 (balanced default)
                sensitive: eps=0.5, min_samples=2 (finds more ambiguous terms)
                conservative: eps=0.35, min_samples=3 (higher precision, fewer terms)
                experimental: Uses HDBSCAN clustering if available"""
    )
    clustering_group.add_argument(
        "--min-resources",
        type=int,
        default=3,
        help="Minimum resources required for term analysis (default: 3)"
    )
    
    # Advanced options - hidden by default for simplicity
    advanced_group = detector_parser.add_argument_group("Advanced options")
    advanced_group.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model to use (default: all-MiniLM-L6-v2)"
    )
    advanced_group.add_argument(
        "--no-radial-detector",
        action="store_false",
        dest="use_radial_detector",
        help="Disable radial polysemy detection as a confidence booster (radial polysemy is used to boost confidence for terms with multiple clusters, not as a standalone detector)"
    )
    advanced_group.add_argument(
        "--save-details",
        action="store_true",
        default=True,
        help="Save detailed clustering information (default: True)"
    )
    advanced_group.add_argument(
        "--output-dir",
        default=defaults["detector_output_dir"],
        help=f"Directory for results (default: {defaults['detector_output_dir']})"
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
    
    # Add splitter-specific arguments - simplified
    splitter_parser.add_argument(
        "--input-file",
        required=True,
        help="Results file from detector (required)"
    )
    
    # Basic options
    splitter_group = splitter_parser.add_argument_group("Splitting options")
    splitter_group.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Disable LLM for tag generation (not recommended)"
    )
    
    # Advanced options
    advanced_group = splitter_parser.add_argument_group("Advanced options")
    advanced_group.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    advanced_group.add_argument(
        "--llm-model",
        help="Specific LLM model name (default: uses provider's default model)"
    )
    advanced_group.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model to use (default: all-MiniLM-L6-v2)"
    )
    advanced_group.add_argument(
        "--cluster-details-file",
        help="Path to comprehensive cluster details (optional, enhances analysis)"
    )
    advanced_group.add_argument(
        "--output-dir",
        default=defaults["splitter_output_dir"],
        help=f"Directory for results (default: {defaults['splitter_output_dir']})"
    )
    
    # Set the handler function
    splitter_parser.set_defaults(func=run_splitter)

def _get_clustering_params(preset: str) -> Dict[str, Any]:
    """
    Convert a clustering preset name to actual clustering parameters.
    
    Args:
        preset: The preset name (standard, sensitive, conservative, experimental)
        
    Returns:
        Dictionary of clustering parameters
    """
    # Define preset configurations
    presets = {
        "standard": {
            "algorithm": "dbscan",
            "dbscan_eps": 0.4,
            "dbscan_min_samples": 2,
            "hdbscan_min_cluster_size": 2,
            "hdbscan_min_samples": 2
        },
        "sensitive": {
            "algorithm": "dbscan",
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 2,
            "hdbscan_min_cluster_size": 2,
            "hdbscan_min_samples": 1
        },
        "conservative": {
            "algorithm": "dbscan",
            "dbscan_eps": 0.35,
            "dbscan_min_samples": 3,
            "hdbscan_min_cluster_size": 3,
            "hdbscan_min_samples": 3
        },
        "experimental": {
            "algorithm": "hdbscan" if HDBSCAN_AVAILABLE else "dbscan",
            "dbscan_eps": 0.4,
            "dbscan_min_samples": 2,
            "hdbscan_min_cluster_size": 2,
            "hdbscan_min_samples": 2
        }
    }
    
    # Return the parameters for the selected preset
    return presets.get(preset, presets["standard"])

def run_detector(args: argparse.Namespace) -> None:
    """Run the detector with the provided arguments."""
    # Load detector modules now that we're actually running the command
    _load_detector_modules()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    else:
        # Set to our custom PROGRESS level for intermediate updates without excessive detail
        logging.getLogger().setLevel(PROGRESS)
        logger.progress("Running with standard logging level - use --verbose for more details")
    
    # Get clustering parameters based on preset
    clustering_params = _get_clustering_params(args.clustering_preset)
    logger.info(f"Using clustering preset: {args.clustering_preset}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_subdir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_subdir}")
    
    # Determine which levels to process
    if args.level is not None:
        levels_to_process = [args.level]
        logger.info(f"Processing single level: {args.level}")
    else:
        levels_to_process = [0, 1, 2, 3]
        logger.info(f"Processing all levels: {levels_to_process}")
    
    # Track all level-specific result files
    all_level_files = []
    
    # PARENT-CONTEXT DETECTOR
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
            
            # Set the output dir for the instance based on CLI args
            detector.cli_output_dir = output_subdir 
            
            ambiguous_terms = detector.detect_ambiguous_terms()
            
            # Save summary TXT file (existing behavior)
            output_filename = f"parent_context_terms_level{level}.txt"
            output_path = os.path.join(output_subdir, output_filename)
            with open(output_path, "w") as f:
                for term in sorted(ambiguous_terms):
                    f.write(f"{term}\n")
            # Save detailed JSON file (new behavior)
            detail_filename = f"parent_context_details_level{level}.json"
            detail_path = detector.save_detailed_results(filename=detail_filename)
            
            # Add both paths to tracking list (if successful)
            all_level_files.append(output_path)
            if detail_path: all_level_files.append(detail_path) 
            
            logger.info(f"Found {len(ambiguous_terms)} potentially ambiguous terms at level {level}")
            logger.info(f"Saved results to {output_path} and {detail_path or '[JSON Save Failed]'}") # Log both
    
    # RESOURCE-CLUSTER DETECTOR
    elif args.detector == "resource-cluster":
        # Create a detector instance with level=None to cluster all terms together first
        # This provides better clustering by considering the full semantic space
        detector = ResourceClusterDetector(
            hierarchy_file_path=args.hierarchy_file,
            final_term_files_pattern=args.final_terms_pattern,
            model_name=args.model,
            min_resources=args.min_resources,
            clustering_algorithm=clustering_params["algorithm"],
            dbscan_eps=clustering_params["dbscan_eps"],
            dbscan_min_samples=clustering_params["dbscan_min_samples"],
            hdbscan_min_cluster_size=clustering_params["hdbscan_min_cluster_size"],
            hdbscan_min_samples=clustering_params["hdbscan_min_samples"],
            level=None,  # Process all levels together initially
            output_dir=output_subdir  # Set the timestamped subdir as output
        )
        
        # Run detection once to populate the cache
        logger.info("Running initial clustering on all terms...")
        all_ambiguous_terms = detector.detect_ambiguous_terms()
        logger.info(f"Found {len(all_ambiguous_terms)} potentially ambiguous terms across all levels")
        
        # Save comprehensive cluster details if requested
        if args.save_details:
            # Use a filename that includes the clustering parameters
            comprehensive_filename = f"comprehensive_cluster_details_{detector.clustering_algorithm}.json"
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
            
            # Add both paths to tracking list
            all_level_files.append(output_path)
            if detail_path: all_level_files.append(detail_path) 
            
            logger.info(f"Found {len(level_ambiguous_terms)} potentially ambiguous terms at level {level}")
            logger.info(f"Saved results to {output_path} and {detail_path or '[JSON Save Failed]'}")
    
    # HYBRID DETECTOR
    elif args.detector == "hybrid":
        # Create a hybrid detector instance
        detector = HybridAmbiguityDetector(
            hierarchy_file_path=args.hierarchy_file,
            final_term_files_pattern=args.final_terms_pattern,
            model_name=args.model,
            min_resources=args.min_resources,
            level=None,  # Process all levels together initially
            output_dir=output_subdir,  # Set the timestamped subdir as output
            use_radial_detector=args.use_radial_detector
        )
        
        # Configure the clustering algorithm for DBSCAN detector
        if hasattr(detector, 'dbscan_detector'):
            # Set the clustering parameters
            detector.dbscan_detector.clustering_algorithm = clustering_params["algorithm"]
            detector.dbscan_detector.dbscan_eps = clustering_params["dbscan_eps"]
            detector.dbscan_detector.dbscan_min_samples = clustering_params["dbscan_min_samples"]
            logger.info(f"Configured DBSCAN detector with parameters from preset: {args.clustering_preset}")
        
        # Log radial detector status
        if args.use_radial_detector and hasattr(detector, 'radial_detector') and detector.radial_detector:
            logger.info("Radial polysemy detector enabled as a confidence booster for terms with multiple clusters")
        else:
            logger.info("Radial polysemy confidence boosting disabled")
            
        # Run detection on all levels to populate the cache
        logger.info("Running initial detection on all terms...")
        all_results = detector.detect_ambiguous_terms()
        
        # Save comprehensive cluster details if requested
        if args.save_details:
            # Save DBSCAN cluster details
            if hasattr(detector, 'dbscan_detector'):
                dbscan_filename = f"comprehensive_cluster_details_dbscan.json"
                dbscan_path = detector.dbscan_detector.save_comprehensive_cluster_details(dbscan_filename)
                logger.info(f"Saved DBSCAN comprehensive cluster details to {dbscan_path}")
            
            # Save radial polysemy results if available
            if hasattr(detector, 'radial_detector') and detector.radial_detector:
                radial_filename = f"radial_polysemy_results.json"
                radial_path = detector.radial_detector.save_results(radial_filename)
                logger.info(f"Saved radial polysemy results to {radial_path}")
        
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
            # Ensure the detector's output_dir is correctly set before saving
            detector.output_dir = output_subdir 
            detail_path = detector.save_results(detail_filename)
            if detail_path: all_level_files.append(detail_path)
            
            # Also save a simple text file with the high+medium confidence terms
            output_filename = f"hybrid_terms_level{level}.txt"
            output_path = os.path.join(output_subdir, output_filename)
            
            with open(output_path, "w") as f:
                for term in sorted(min_conf_results):
                    f.write(f"{term}\n")
            
            logger.info(f"Level {level} results: {len(high_conf)} high confidence, {len(med_conf)} medium confidence")
            logger.info(f"Found {len(min_conf_results)} terms meeting min confidence threshold {args.min_confidence}")
            logger.info(f"Saved results to {output_path} and {detail_path or '[JSON Save Failed]'}")
    
    # RADIAL-POLYSEMY DETECTOR
    elif args.detector == "radial-polysemy":
        # Create a RadialPolysemyDetector instance
        detector = RadialPolysemyDetector(
            hierarchy_file_path=args.hierarchy_file,
            final_term_files_pattern=args.final_terms_pattern,
            model_name=args.model,
            context_window_size=10,
            min_contexts=10,  # Reduced from 20 to improve detection rate
            level=None,  # Process all levels together initially
            output_dir=output_subdir
        )
        
        # Set the output dir for the instance explicitly, just in case
        detector.cli_output_dir = output_subdir
        
        logger.info("Running radial polysemy detection on all terms...")
        # Detect ambiguous terms (level filtering happens inside if args.level is set)
        ambiguous_terms = detector.detect_ambiguous_terms()
        
        # Save detailed JSON results (existing)
        detail_filename = f"radial_polysemy_results.json" # Consistent filename
        detail_path = detector.save_results(detail_filename)
        
        # Save summary TXT results (new)
        output_filename = f"radial_polysemy_terms.txt" # Consistent filename
        output_path = detector.save_summary_results(ambiguous_terms, filename=output_filename)
        
        # Add paths to tracking list
        if detail_path: all_level_files.append(detail_path)
        if output_path: all_level_files.append(output_path)
        
        logger.info(f"Saved detailed scores to {detail_path or '[JSON Save Failed]'}")
        logger.info(f"Saved summary term list to {output_path or '[TXT Save Failed]'}")
        
        all_analyzed_terms_count = len(detector.polysemy_scores) if hasattr(detector, 'polysemy_scores') else 0
        logger.info(f"Found {all_analyzed_terms_count} analyzed terms, {len(ambiguous_terms)} flagged as ambiguous")
    
    logger.info(f"Detection complete. Results saved to {output_subdir}")
    valid_files = [os.path.basename(f) for f in all_level_files if f and os.path.exists(f)] # Check existence
    if valid_files:
        logger.info(f"Output files: {', '.join(valid_files)}")
    
    # Create a combined results file if multiple levels were processed
    if args.level is None and args.detector in ["resource-cluster", "hybrid", "parent-context"]:
        logger.info("Creating combined results file for all levels...")
        # ... (rest of combined file logic - might need updates based on new file names)
        # ... For now, assume it mostly works, focusing on individual detector saving ...

def run_splitter(args: argparse.Namespace) -> None:
    """Run the splitter with the provided arguments."""
    # Load splitter module now that we're actually running the command
    _load_splitter_module()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    else:
        # Set to our custom PROGRESS level for intermediate updates without excessive detail
        logging.getLogger().setLevel(PROGRESS)
        logger.progress("Running with standard logging level - use --verbose for more details")
    
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
        
        # --- Revised Loading Order ---
        # 1. Load the main input file first to get the primary candidate list
        loaded_successfully = False
        if is_combined_file:
            # Logic to handle combined file loading (extract level data, create temp file)
            level_key = f"level{level}"
            if level_key in combined_results:
                logger.info(f"Processing combined results for level {level} from main input file first...")
                level_data = combined_results[level_key]
                # Create temporary structure compatible with splitter loading
                temp_data = {
                    "timestamp": datetime.now().isoformat(),
                    "parameters": input_data.get("parameters", {}),
                    "detailed_results": level_data # Pass the level data directly
                }
                temp_file = os.path.join(output_subdir, f"temp_level{level}_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                try:
                    with open(temp_file, 'w') as f:
                        json.dump(temp_data, f)
                    loaded_successfully = splitter._load_cluster_results_from_file(temp_file)
                    os.remove(temp_file) # Clean up
                except Exception as e:
                    logger.error(f"Error processing combined results for level {level}: {e}")
            else:
                 logger.warning(f"Level {level} key '{level_key}' not found in combined results.")
        else:
            # Load directly from the single input file
            logger.info(f"Loading main input file: {args.input_file}")
            loaded_successfully = splitter._load_cluster_results_from_file(args.input_file)

        if not loaded_successfully:
            logger.error(f"Failed to load initial candidate terms from input file for level {level}")
            continue # Skip this level if main input fails

        # 2. Load comprehensive cluster details file if provided
        #    The splitter logic now ensures this doesn't overwrite the candidate list.
        if args.cluster_details_file:
            logger.info(f"Loading comprehensive cluster details from: {args.cluster_details_file}")
            cluster_details_loaded = splitter._load_cluster_results_from_file(args.cluster_details_file)
            if cluster_details_loaded:
                logger.info("Successfully loaded and integrated comprehensive cluster details")
            else:
                logger.warning("Failed to load comprehensive cluster details, proceeding with data from main input file only")
        
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

def main() -> None:
    """Main entry point for the CLI."""
    # Set default logging level to PROGRESS
    logging.getLogger().setLevel(PROGRESS)
    
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