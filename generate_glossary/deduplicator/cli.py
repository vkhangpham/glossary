"""
Command-line interface for deduplication.

This module provides a CLI for deduplicating technical concepts using
different deduplication modes, with graph-based deduplication as the recommended approach.
"""

import argparse
import json
import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv

from generate_glossary.deduplicator import deduplication_modes
from generate_glossary.utils.web_miner import WebContent

load_dotenv('.env')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default configuration
DEFAULT_CONFIG = {
    "batch_size": 100,
    "max_workers": None,
    "min_score": 0.7,
    "min_relevance_score": 0.3,
    "log_level": "INFO",
    "use_enhanced_linguistics": True,
}

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging with the specified level and optional file output."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Basic configuration for console output
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=numeric_level, format=log_format, stream=sys.stdout)

    # If a log file is specified, add a file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")

def read_terms(filepath: str) -> List[str]:
    """Read terms from file, one term per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def read_term_json(filepath: str) -> Dict[str, Any]:
    """Read deduplication JSON file to extract terms and their variations.
    
    This function attempts to read variations data from two possible sources:
    1. The corresponding .json file with the same base name (e.g., lv0_final.json)
    2. The metadata file in the same directory (e.g., lv0_metadata.json)
    """
    json_path = Path(filepath)
    
    # Try the corresponding JSON with the same base name first
    if json_path.suffix == '.txt':
        json_path = json_path.with_suffix('.json')
    
    result = {"canonicals": [], "variations": {}}
    
    # Try reading the json file with the same base name
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            canonicals = data.get("deduplicated_terms", [])
            
            # Get variations from the variation_reasons dictionary
            variations = {}
            variation_reasons = data.get("variation_reasons", {})
            for variation, details in variation_reasons.items():
                canonical = details.get("canonical")
                if canonical:
                    if canonical not in variations:
                        variations[canonical] = set()
                    variations[canonical].add(variation)
            
            result["canonicals"] = canonicals
            result["variations"] = variations
            
            return result
        except Exception as e:
            logging.warning(f"Error reading JSON file {json_path}: {e}")
    
    # If we couldn't read or find the json file, try the metadata file
    try:
        # Construct metadata path: data/lvX/lvX_metadata.json
        dir_path = json_path.parent
        level_prefix = json_path.stem.split('_')[0]  # Extract lvX part
        metadata_path = dir_path / f"{level_prefix}_metadata.json"
        
        if metadata_path.exists():
            logging.info(f"Reading variations from metadata file: {metadata_path}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Extract terms and their variations from metadata
            canonicals = []
            variations = {}
            
            # Process each term in the metadata
            for term, term_info in metadata.items():
                if "canonical_term" in term_info:
                    # This is a variation
                    canonical = term_info["canonical_term"]
                    if canonical not in variations:
                        variations[canonical] = set()
                    variations[canonical].add(term)
                else:
                    # This is a canonical term
                    canonicals.append(term)
                    
                    # Check if it has variations
                    if "variations" in term_info:
                        if term not in variations:
                            variations[term] = set()
                        variations[term].update(term_info["variations"])
            
            result["canonicals"] = canonicals
            result["variations"] = variations
            
            return result
        else:
            logging.warning(f"Metadata file not found: {metadata_path}")
    except Exception as e:
        logging.warning(f"Error reading metadata file: {e}")
    
    # Return empty result if neither file was read successfully
    return result

def read_web_content(filepath: str) -> Dict[str, List[WebContent]]:
    """Read web content from JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading web content: {e}", file=sys.stderr)
        return {}

def write_results(results: Dict[str, Any], output_path: str, input_terms: List[str] = None) -> None:
    """Write deduplication results to both .txt and .json files.
    
    Args:
        results: Dictionary of deduplication results
        output_path: Base path for output files (without extension)
        input_terms: Original input terms to filter results (if None, all terms are included)
    """
    # Convert sets to lists for JSON serialization
    def convert_sets(obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(item) for item in obj]
        return obj
    
    # Create a set of input terms for faster lookups
    input_term_set = set(input_terms) if input_terms else None
    
    # Write the full deduplication results to JSON
    # For the JSON file, we'll include all the relationships but filter the deduplicated_terms
    json_path = f"{output_path}.json"
    json_safe_results = convert_sets(results)
    
    # Filter deduplicated_terms to only include terms from input if specified
    if input_term_set and "deduplicated_terms" in json_safe_results:
        # Get all variations of input terms to ensure we include their canonical forms
        input_related_terms = set(input_term_set)
        
        # Add canonical terms for any input term that appears as a variation
        for canonical, variations in json_safe_results.get("variations", {}).items():
            for var in variations:
                if var in input_term_set:
                    input_related_terms.add(canonical)
                    
        # Also check cross-level variations
        for canonical, level_vars in json_safe_results.get("cross_level_variations", {}).items():
            for level, vars in level_vars.items():
                for var in vars:
                    if var in input_term_set:
                        input_related_terms.add(canonical)
        
        # Filter deduplicated_terms to only include those from input or their canonical forms
        if isinstance(json_safe_results["deduplicated_terms"], list):
            json_safe_results["deduplicated_terms"] = [
                term for term in json_safe_results["deduplicated_terms"]
                if term in input_related_terms
            ]
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
    
    # Write just the deduplicated terms to text file
    txt_path = f"{output_path}.txt"
    deduplicated_terms = results.get("deduplicated_terms", [])
    if not isinstance(deduplicated_terms, list):
        # Handle case where deduplicated_terms is a dict (for cross-level mode)
        all_terms = []
        for level_terms in deduplicated_terms.values():
            all_terms.extend(level_terms)
        deduplicated_terms = all_terms
    
    # Find and remove all exact duplicates between levels
    # This is case-insensitive for comparison but preserves original case
    if input_term_set:
        # Create a map of lowercase variations to original case
        all_variations = {}
        
        # First, add all terms from the higher level terms to the variations list
        # These are the terms that should be excluded from the current level
        higher_level_terms_set = set()
        
        # Extract all higher level terms from the input
        if "higher_level_terms" in results:
            for level, terms in results.get("higher_level_terms", {}).items():
                higher_level_terms_set.update(terms)
        
        # Add all higher level terms to the variations list (these are terms to exclude)
        for term in higher_level_terms_set:
            term_lower = term.lower()
            all_variations[term_lower] = term
        
        # Also add all variations from the results
        for canonical, variations in results.get("variations", {}).items():
            canonical_lower = canonical.lower()
            all_variations[canonical_lower] = canonical
            for var in variations:
                var_lower = var.lower()
                all_variations[var_lower] = var

        # Also include cross-level variations
        for canonical, level_vars in results.get("cross_level_variations", {}).items():
            canonical_lower = canonical.lower()
            all_variations[canonical_lower] = canonical
            for level, vars in level_vars.items():
                for var in vars:
                    var_lower = var.lower()
                    all_variations[var_lower] = var
        
        # Add cross-level canonical terms (if a term is marked as canonical in a higher level)
        higher_level_canonicals = results.get("cross_level_canonicals", [])
        for term in higher_level_canonicals:
            term_lower = term.lower()
            all_variations[term_lower] = term
        
        # Explicitly check for terms from higher levels that are duplicated in the current level
        if "terms_by_level" in results:
            # Extract all terms from higher levels (lower level numbers)
            current_level = max(results.get("terms_by_level", {}).keys(), default=0)
            for level, terms in results.get("terms_by_level", {}).items():
                if level < current_level:  # Only consider higher levels (lower numbers)
                    for term in terms:
                        term_lower = term.lower()
                        all_variations[term_lower] = term
        
        # Logging for debugging
        logging.debug(f"Found {len(all_variations)} unique terms to exclude from current level")
        
        # Exclude any input term that's a case-insensitive match with any variation or higher level term
        filtered_terms = []
        for term in deduplicated_terms:
            term_lower = term.lower()
            # Include term only if it's in input and NOT a case-insensitive match with any variation or higher level term
            if term in input_term_set and term_lower not in all_variations:
                filtered_terms.append(term)
            elif term in input_term_set and term_lower in all_variations:
                logging.debug(f"Removing duplicate term '{term}' that appears in higher level as '{all_variations[term_lower]}'")
        
        deduplicated_terms = filtered_terms
    
    with open(txt_path, "w", encoding="utf-8") as f:
        for term in sorted(deduplicated_terms):
            f.write(f"{term}\n")

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deduplicate technical concepts using different modes (graph-based deduplication recommended)."
    )
    
    # Input arguments
    parser.add_argument(
        "terms",
        help="Terms to deduplicate (file path)"
    )
    parser.add_argument(
        "-t", "--higher-level-terms",
        nargs="+",
        help="Paths to higher level term files for cross-level deduplication. Format: level:path (e.g., 0:lv0.txt 1:lv1.txt)"
    )
    parser.add_argument(
        "--current-level",
        type=int,
        help="Specify the level number for the current input terms (REQUIRED for graph mode). Lower level numbers have higher priority when selecting canonical forms. For example, if 'arts' is in level 1 and 'art' is in level 2, 'arts' will be selected as the canonical form."
    )
    
    # Mode selection
    parser.add_argument(
        "-m", "--mode",
        choices=["rule", "web", "llm", "graph"],
        default="graph",
        help="Deduplication mode (default: graph - recommended)"
    )
    
    # Web content arguments
    parser.add_argument(
        "-w", "--web-content",
        help="Path to web content JSON file (recommended for graph mode, required for web mode)"
    )
    parser.add_argument(
        "-c", "--higher-level-web-content",
        nargs="+",
        help="Paths to higher level web content files for cross-level deduplication. Format: level:path (e.g., 0:lv0.json 1:lv1.json)"
    )
    
    # Common arguments
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output file (without extension)"
    )
    parser.add_argument(
        "-l", "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_CONFIG["log_level"],
        help=f"Logging level (default: {DEFAULT_CONFIG['log_level']})"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: {output}.log)"
    )
    
    # Advanced arguments (grouped)
    advanced_group = parser.add_argument_group('Advanced Options')
    
    # Performance parameters
    advanced_group.add_argument(
        "-b", "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help=f"Batch size for parallel processing (default: {DEFAULT_CONFIG['batch_size']})"
    )
    advanced_group.add_argument(
        "-x", "--max-workers",
        type=int,
        default=DEFAULT_CONFIG["max_workers"],
        help="Maximum number of worker processes (default: auto)"
    )
    
    # Mode-specific parameters
    advanced_group.add_argument(
        "-p", "--provider",
        default="gemini",
        help="LLM provider for llm mode (default: gemini)"
    )
    advanced_group.add_argument(
        "-s", "--min-score",
        type=float,
        default=DEFAULT_CONFIG["min_score"],
        help=f"Minimum score threshold for web content quality in web mode (default: {DEFAULT_CONFIG['min_score']})"
    )
    advanced_group.add_argument(
        "-r", "--min-relevance-score",
        type=float,
        default=DEFAULT_CONFIG["min_relevance_score"],
        help=f"Minimum relevance score for web content to be considered relevant to a term (default: {DEFAULT_CONFIG['min_relevance_score']})"
    )
    advanced_group.add_argument(
        "-e", "--use-enhanced-linguistics",
        action="store_true",
        default=DEFAULT_CONFIG["use_enhanced_linguistics"],
        help="Use enhanced linguistic analysis for better variation detection"
    )
    advanced_group.add_argument(
        "--graph-cache-dir",
        help="Directory to store and load graph cache for incremental deduplication (only for graph mode)"
    )
    
    args = parser.parse_args()
    
    # Determine log file path
    log_file_path = args.log_file
    if not log_file_path:
        # Default log file name based on output path
        output_dir = Path(args.output).parent
        output_basename = Path(args.output).stem
        log_file_path = output_dir / f"{output_basename}.log"
        # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    setup_logging(args.log_level, str(log_file_path))
    
    # Silence verbose logs from underlying libraries
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("google.cloud.aiplatform").setLevel(logging.WARNING)
    logging.getLogger('google_genai.models').setLevel(logging.WARNING)
    logging.getLogger('google_genai').setLevel(logging.WARNING)
    logging.getLogger("vertexai").setLevel(logging.WARNING) # Add vertexai logger
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING) # httpx uses httpcore
    logging.getLogger("openai").setLevel(logging.WARNING) # Might also log directly
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Add other library logger names here if needed
    
    try:
        # Read terms
        if Path(args.terms).exists():
            terms = read_terms(args.terms)
        else:
            logging.error(f"Input file not found: {args.terms}")
            sys.exit(1)
        
        logging.info(f"Processing {len(terms)} terms with mode: {args.mode}")
        
        # Read higher level terms and their variations if provided
        higher_level_terms = None
        higher_level_terms_with_variations = None
        
        if args.higher_level_terms:
            higher_level_terms = {}
            higher_level_terms_with_variations = {}
            
            for level_path in args.higher_level_terms:
                level, path = level_path.split(":")
                level = int(level)
                
                # Read canonical terms
                canonical_terms = read_terms(path)
                higher_level_terms[level] = canonical_terms
                
                # Read terms and their variations from the corresponding JSON file
                term_data = read_term_json(path)
                
                # Create comprehensive list of terms (canonicals + variations)
                all_terms = set(term_data["canonicals"])
                
                # Add all variations
                for canonical, variations in term_data["variations"].items():
                    all_terms.add(canonical)
                    all_terms.update(variations)
                
                higher_level_terms_with_variations[level] = list(all_terms)
                
                logging.info(f"Read {len(canonical_terms)} canonical terms and {len(all_terms) - len(canonical_terms)} variations from level {level}")
        
        # Read web content if needed
        web_content = None
        higher_level_web_content = None
        
        if args.web_content:
            web_content = read_web_content(args.web_content)
            if not web_content:
                logging.warning("Failed to load web content or empty content provided")
                if args.mode == "web":
                    raise ValueError("Web mode requires valid web content")
            else:
                logging.info(f"Loaded web content for {len(web_content)} terms")
        elif args.mode == "web":
            raise ValueError("Web mode requires --web-content argument")
        
        # Read higher level web content if provided
        if args.higher_level_web_content:
            higher_level_web_content = {}
            for level_path in args.higher_level_web_content:
                level, path = level_path.split(":")
                level = int(level)
                higher_level_web_content[level] = read_web_content(path)
                logging.info(f"Loaded web content for level {level} with {len(higher_level_web_content[level])} terms")
        
        # Run deduplication based on mode
        if args.mode == "rule":
            # Rule-based deduplication
            results = deduplication_modes.deduplicate_rule_based(
                terms,
                higher_level_terms=higher_level_terms,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                use_enhanced_linguistics=args.use_enhanced_linguistics
            )
        elif args.mode == "web":
            # Web-based deduplication
            if not web_content:
                raise ValueError("Web mode requires web content")
            
            results = deduplication_modes.deduplicate_web_based(
                terms,
                web_content=web_content,
                higher_level_terms=higher_level_terms,
                higher_level_web_content=higher_level_web_content,
                min_score=args.min_score,
                min_relevance_score=args.min_relevance_score,
                batch_size=args.batch_size,
                max_workers=args.max_workers
            )
        elif args.mode == "llm":
            # LLM-based deduplication
            results = deduplication_modes.deduplicate_llm_based(
                terms,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                provider=args.provider
            )
        elif args.mode == "graph":
            # Graph-based deduplication (recommended)
            
            # Validate that current_level is specified when using graph mode
            if args.current_level is None:
                logging.error("ERROR: --current-level parameter is required when using graph mode for proper level-based prioritization")
                logging.error("Please specify the level number for the current input terms using --current-level")
                sys.exit(1)
            
            # Create a filtered version of higher_level_terms that only includes actual canonical terms
            # This prevents variations from being incorrectly included in higher levels
            filtered_higher_level_terms = {}
            if higher_level_terms_with_variations:
                for level, all_level_terms in higher_level_terms_with_variations.items():
                    # Only include terms that actually appear in the original canonical files
                    if level in higher_level_terms:
                        canonical_set = set(higher_level_terms[level])
                        filtered_higher_level_terms[level] = higher_level_terms[level]
                
            # Use the filtered higher level terms for deduplication
            results = deduplication_modes.deduplicate_graph_based(
                terms,
                web_content=web_content,
                higher_level_terms=filtered_higher_level_terms if filtered_higher_level_terms else higher_level_terms,
                higher_level_web_content=higher_level_web_content,
                min_score=args.min_score,
                min_relevance_score=args.min_relevance_score,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                use_enhanced_linguistics=args.use_enhanced_linguistics,
                current_level=args.current_level,
                cache_dir=args.graph_cache_dir
            )
        else:
            raise ValueError(f"Invalid deduplication mode: {args.mode}")
        
        # Write results, passing the original input terms to filter the output
        write_results(results, args.output, input_terms=terms)
        logging.info(f"Results written to {args.output}.txt and {args.output}.json")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 