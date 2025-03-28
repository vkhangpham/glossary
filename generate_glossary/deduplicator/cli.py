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
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from generate_glossary.deduplicator import deduplication_modes
from generate_glossary.utils.web_miner import WebContent

load_dotenv('.env')

# Default configuration
DEFAULT_CONFIG = {
    "batch_size": 100,
    "max_workers": None,
    "min_score": 0.7,
    "min_relevance_score": 0.3,
    "log_level": "INFO",
    "use_enhanced_linguistics": True,
}

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def read_terms(filepath: str) -> List[str]:
    """Read terms from file, one term per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

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
        
        # Check for any input term that might be a match with variations
        input_term_lower_to_original = {term.lower(): term for term in input_term_set}
        
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
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(args.log_level)
    
    try:
        # Read terms
        if Path(args.terms).exists():
            terms = read_terms(args.terms)
        else:
            logging.error(f"Input file not found: {args.terms}")
            sys.exit(1)
        
        logging.info(f"Processing {len(terms)} terms with mode: {args.mode}")
        
        # Read higher level terms if provided
        higher_level_terms = None
        if args.higher_level_terms:
            higher_level_terms = {}
            for level_path in args.higher_level_terms:
                level, path = level_path.split(":")
                level = int(level)
                higher_level_terms[level] = read_terms(path)
                logging.info(f"Read {len(higher_level_terms[level])} terms from level {level}")
        
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
            results = deduplication_modes.deduplicate_graph_based(
                terms,
                web_content=web_content,
                higher_level_terms=higher_level_terms,
                higher_level_web_content=higher_level_web_content,
                min_score=args.min_score,
                min_relevance_score=args.min_relevance_score,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                use_enhanced_linguistics=args.use_enhanced_linguistics
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