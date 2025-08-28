"""
Command-line interface for concept validation.

This module provides a CLI for validating technical concepts using
different validation modes.
"""

import argparse
import json
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from generate_glossary.validator.validation_modes import WebContent, DEFAULT_MIN_SCORE, DEFAULT_MIN_RELEVANCE_SCORE
from generate_glossary.validator import validate

def read_terms(filepath: str) -> List[str]:
    """Read terms from file, one term per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def read_web_content(filepath: str) -> Dict[str, List[WebContent]]:
    """
    Read web content from JSON file.
    
    Handles both the old format (Dict[str, List[Dict]]) and the new format with WebContent objects.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
            
            # Check if this is the new format (with WebContent objects)
            # In the new format, each content item has url, title, snippet, raw_content, processed_content, score, is_verified
            if content and isinstance(content, dict):
                for term, contents in content.items():
                    if contents and isinstance(contents, list) and isinstance(contents[0], dict):
                        # Check for new format fields
                        if all(
                            isinstance(item, dict) and 
                            "url" in item and 
                            "title" in item and 
                            "processed_content" in item and
                            "score" in item and
                            "is_verified" in item
                            for item in contents
                        ):
                            # Already in the expected format
                            return content
            
            # If we get here, it's either empty or in an old format
            # Just return it as is and let the validation function handle it
            return content
    except Exception as e:
        print(f"Error loading web content: {e}", file=sys.stderr)
        return {}

def get_rejected_terms(level: int) -> Dict[str, int]:
    """
    Get all terms rejected in previous levels.
    
    Args:
        level: Current level
        
    Returns:
        Dictionary mapping rejected terms to the level they were rejected at
    """
    if level is None or level <= 1:
        return {}  # No previous levels to check
    
    rejected_terms = {}
    
    # Check all previous levels (from 0 to level-1)
    for prev_level in range(level):
        # Look for validation result files in data/lvX/lvX*_valid.json
        pattern = f"data/lv{prev_level}/lv{prev_level}*_valid.json"
        result_files = glob.glob(pattern)
        
        for result_file in result_files:
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    
                    # Add terms that were marked as invalid to the rejected dict
                    # Only add if not already in the dict (keep earliest rejection level)
                    for term, result in results.items():
                        if not result.get("is_valid", True) and term not in rejected_terms:
                            rejected_terms[term] = prev_level
            except Exception as e:
                print(f"Warning: Could not read rejected terms from {result_file}: {e}", file=sys.stderr)
    
    return rejected_terms

def write_results(results: Dict[str, Any], output_path: str) -> None:
    """Write validation results to both .txt and .json files.
    
    Args:
        results: Dictionary of validation results
        output_path: Base path for output files (without extension)
    """
    # Convert all terms to lowercase, ensuring they are strings
    # results = {str(term).lower(): result for term, result in results.items()}
    
    # Write the full validation results to JSON
    json_path = f"{output_path}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Write just the valid terms to text file
    txt_path = f"{output_path}.txt"
    valid_terms = [
        term for term, result in results.items() 
        if result.get("is_valid", False)
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        for term in sorted(valid_terms):
            f.write(f"{term}\n")

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate technical concepts using different validation modes."
    )
    
    # Input arguments
    parser.add_argument(
        "terms",
        help="Term to validate or path to file containing terms (one per line)"
    )
    
    # Validation mode
    parser.add_argument(
        "-m", "--mode",
        choices=["rule", "web", "llm"],
        default="rule",
        help="Validation mode to use (default: rule)"
    )
    
    # Level argument
    parser.add_argument(
        "-l", "--level",
        type=int,
        help="Current validation level (if specified, terms rejected in levels 1 to level-1 will be automatically skipped)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-w", "--web-content",
        help="Path to web content JSON file (required for web mode)"
    )
    parser.add_argument(
        "-s", "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum score for web content validation (default: {DEFAULT_MIN_SCORE})"
    )
    parser.add_argument(
        "-r", "--min-relevance-score",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help=f"Minimum relevance score for web content to be considered relevant to the term (default: {DEFAULT_MIN_RELEVANCE_SCORE})"
    )
    parser.add_argument(
        "-p", "--provider",
        default="gemini",
        help="LLM provider for validation (default: gemini)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Base path for output files (will create .txt and .json files)"
    )
    parser.add_argument(
        "-n", "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--save-web-content",
        help="Path to save updated web content with relevance scores (JSON file)"
    )
    parser.add_argument(
        "--update-web-content",
        action="store_true",
        default=True,
        help="Update the input web content file in-place with relevance scores (only for web mode)"
    )
    
    args = parser.parse_args()
    
    try:
        # Read terms
        if Path(args.terms).exists():
            terms = read_terms(args.terms)
        else:
            terms = [args.terms]
        
        # Get rejected terms from previous levels if level is specified
        rejected_terms = get_rejected_terms(args.level) if args.level else {}
        if rejected_terms:
            print(f"Found {len(rejected_terms)} terms rejected in previous levels")
            
            # Filter out terms that have been rejected in previous levels
            original_count = len(terms)
            terms = [term for term in terms if term not in rejected_terms]
            filtered_count = original_count - len(terms)
            
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} previously rejected terms")
        
        # Read web content if needed
        web_content = None
        llm_responses = None
        
        if args.mode == "web":
            if not args.web_content:
                raise ValueError("Web mode requires --web-content argument")
            web_content = read_web_content(args.web_content)
            if not web_content:
                raise ValueError("Failed to load web content")
        
        # Generate LLM responses if needed
        if args.mode == "llm":
            from generate_glossary.utils.llm_simple import infer_text, get_random_llm_config
            
            print(f"Generating LLM responses using {args.provider}...")
            
            llm_responses = {}
            for term in terms:
                prompt = f"Is '{term}' a valid academic discipline or field of study? Answer with 'yes' or 'no' followed by a brief explanation."
                try:
                    response = infer_text(
                        provider=args.provider or "openai",
                        prompt=prompt
                    )
                    llm_responses[term] = response.text
                except Exception as e:
                    print(f"Error getting LLM response for '{term}': {e}", file=sys.stderr)
                    llm_responses[term] = None
        
        # Map CLI mode names to internal mode names
        # The internal API uses "rules" while the CLI uses "rule" for consistency with deduplicator
        mode_mapping = {
            "rule": "rules"  # Map "rule" to "rules" for internal API compatibility
        }
        internal_mode = mode_mapping.get(args.mode, args.mode)
        
        # Run validation on filtered terms
        results = validate(
            terms,
            mode=internal_mode,
            web_content=web_content,
            llm_responses=llm_responses,
            min_score=args.min_score,
            min_relevance_score=args.min_relevance_score,
            show_progress=not args.no_progress
        )
        
        # Add rejected terms from previous levels with is_valid=False
        for term, rejection_level in rejected_terms.items():
            if term not in results:  # Avoid overwriting if somehow the term was still validated
                results[term] = {
                    "term": term,
                    "is_valid": False,
                    "mode": internal_mode,
                    "details": {
                        "reason": f"Rejected in level {rejection_level}",
                        "level_rejected": rejection_level
                    }
                }
        
        # Output results
        if args.output:
            # Remove any extension from output path
            base_path = str(Path(args.output).with_suffix(""))
            write_results(results, base_path)
        else:
            # Just print the JSON to stdout
            json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
            print()  # Add newline
            
        # Modify the main function to update web content in-place
        if args.mode == "web" and (args.save_web_content or args.update_web_content) and web_content:
            # Update web content with relevance scores from validation results
            for term, result in results.items():
                if term in web_content and "details" in result:
                    # Get the sources with relevance scores
                    verified_sources = result["details"].get("verified_sources", [])
                    unverified_sources = result["details"].get("unverified_sources", [])
                    all_sources = verified_sources + unverified_sources
                    
                    # Create a mapping from URL to relevance score
                    url_to_relevance = {}
                    for source in all_sources:
                        if "url" in source and "relevance_score" in source:
                            url_to_relevance[source["url"]] = source["relevance_score"]
                    
                    # Update the web content with relevance scores
                    for entry in web_content[term]:
                        if isinstance(entry, dict) and "url" in entry:
                            url = entry["url"]
                            if url in url_to_relevance:
                                entry["relevance_score"] = url_to_relevance[url]
            
            # Save the updated web content
            if args.update_web_content:
                # Save back to the input file
                with open(args.web_content, "w", encoding="utf-8") as f:
                    json.dump(web_content, f, indent=2, ensure_ascii=False)
                print(f"Updated web content saved in-place to {args.web_content}")
            elif args.save_web_content:
                # Save to a new file
                with open(args.save_web_content, "w", encoding="utf-8") as f:
                    json.dump(web_content, f, indent=2, ensure_ascii=False)
                print(f"Updated web content saved to {args.save_web_content}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 