"""
Main entry point for term validation.

This module provides the core validation orchestration that combines
different validation modes into a unified pipeline.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import time
from functools import lru_cache

from .rule_validator import validate_with_rules
from .web_validator import validate_with_web_content
from .llm_validator import validate_with_llm
from .utils import normalize_term
from .cache import get_cache, filter_cached_terms

# Type aliases
Terms = Union[str, List[str]]
ValidationResult = Dict[str, Any]
ValidationResults = Dict[str, ValidationResult]
WebContent = Dict[str, List[Dict[str, Any]]]

# Default configuration
DEFAULT_CONFIG = {
    "modes": ["rule"],  # Default to rule-based validation only
    "confidence_weights": {
        "rule": 0.3,
        "web": 0.5,
        "llm": 0.2
    },
    "min_confidence": 0.5,
    "min_score": 0.5,
    "min_relevance_score": 0.5,
    "parallel": True,
    "show_progress": True
}


def validate_terms(
    terms: Terms,
    modes: Optional[List[str]] = None,
    web_content: Optional[WebContent] = None,
    llm_provider: str = "gemini",
    config: Optional[Dict[str, Any]] = None,
    existing_results: Optional[ValidationResults] = None,
    show_progress: bool = True,
    use_cache: bool = True
) -> ValidationResults:
    """
    Validate terms using specified validation modes.
    
    This is the main entry point for all validation operations.
    It orchestrates different validation modes and combines their results.
    
    Args:
        terms: Single term or list of terms to validate
        modes: List of validation modes to use (rule, web, llm)
        web_content: Web content for web-based validation
        llm_provider: LLM provider for llm-based validation
        config: Validation configuration (overrides defaults)
        existing_results: Previous validation results to extend/update
        show_progress: Whether to show progress bar
        use_cache: Whether to use cached validation results
        
    Returns:
        Dictionary mapping terms to their validation results
    """
    start_time = time.time()
    
    # Handle single term input
    if isinstance(terms, str):
        terms = [terms]
    
    # Merge config with defaults
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    # Use provided modes or get from config
    modes = modes or config.get("modes", ["rule"])
    
    # Initialize results with existing results if provided
    results = existing_results.copy() if existing_results else {}
    
    # Check cache for previously validated terms
    if use_cache:
        uncached_terms, cached_results = filter_cached_terms(terms, modes)
        results.update(cached_results)
        
        if cached_results:
            logging.info(f"Found {len(cached_results)} cached results")
        
        # If all terms are cached, return early
        if not uncached_terms:
            logging.info("All terms found in cache")
            return results
        
        # Continue with uncached terms
        terms = uncached_terms
    
    # Normalize terms
    normalized_terms = [normalize_term(term) for term in terms]
    term_mapping = dict(zip(normalized_terms, terms))  # Map normalized to original
    
    logging.info(f"Validating {len(terms)} terms using modes: {modes}")
    
    # Run each validation mode
    mode_results = {}
    
    if "rule" in modes:
        logging.info("Running rule-based validation...")
        mode_results["rule"] = validate_with_rules(
            terms,
            show_progress=show_progress and len(modes) == 1
        )
    
    if "web" in modes:
        if not web_content:
            logging.warning("Web mode requested but no web content provided")
        else:
            logging.info("Running web-based validation...")
            mode_results["web"] = validate_with_web_content(
                terms,
                web_content,
                min_score=config.get("min_score", 0.5),
                min_relevance_score=config.get("min_relevance_score", 0.5),
                show_progress=show_progress and len(modes) == 1
            )
    
    if "llm" in modes:
        logging.info(f"Running LLM-based validation with {llm_provider}...")
        mode_results["llm"] = validate_with_llm(
            terms,
            provider=llm_provider,
            show_progress=show_progress and len(modes) == 1
        )
    
    
    # Combine results from all modes with early exit optimization
    for term in terms:
        term_results = {}
        validity_votes = []
        weighted_confidence = 0.0
        total_weight = 0.0
        
        # Early exit check
        early_exit = False
        
        for mode, mode_result in mode_results.items():
            if term in mode_result:
                term_results[mode] = mode_result[term]
                
                # Get mode weight
                weight = config["confidence_weights"].get(mode, 0.25)
                total_weight += weight
                
                # Calculate weighted confidence contribution
                mode_confidence = mode_result[term].get("confidence", 0.5)
                weighted_confidence += mode_confidence * weight
                
                # Track validity votes
                validity_votes.append(mode_result[term].get("is_valid", False))
                
                # Early exit if confidence already exceeds threshold
                if len(modes) > 1 and total_weight > 0:
                    current_confidence = weighted_confidence / total_weight
                    if current_confidence >= config["min_confidence"] * 1.2:  # 20% buffer
                        early_exit = True
                        logging.debug(f"Early exit for '{term}' with confidence {current_confidence:.2f}")
                        break
        
        # Calculate overall confidence (normalized by total weight)
        if total_weight > 0:
            overall_confidence = weighted_confidence / total_weight
        else:
            overall_confidence = 0.0
        
        # Determine overall validity
        # For single mode, use that mode's validity
        # For multiple modes, use weighted confidence threshold
        if len(term_results) == 1:
            is_valid = validity_votes[0] if validity_votes else False
        else:
            is_valid = overall_confidence >= config["min_confidence"]
        
        # Store combined result
        result = {
            "term": term,
            "is_valid": is_valid,
            "confidence": round(overall_confidence, 3),
            "modes_used": list(term_results.keys()),
            "mode_results": term_results,
            "timestamp": time.time(),
            "early_exit": early_exit
        }
        
        results[term] = result
        
        # Cache the result
        if use_cache:
            cache = get_cache()
            cache.add_validation_result(term, modes, result)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Validation completed in {elapsed_time:.2f} seconds")
    
    # Log summary statistics
    valid_count = len([r for r in results.values() if r["is_valid"]])
    logging.info(f"Valid terms: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    
    return results




def main():
    """
    CLI entry point for validation.
    """
    import argparse
    import json
    import sys
    from pathlib import Path
    from .api import save_validation_results
    
    parser = argparse.ArgumentParser(
        description="Validate technical concepts using different validation modes."
    )
    
    parser.add_argument("terms", help="Term or file containing terms (one per line)")
    parser.add_argument("-m", "--mode", choices=["rule", "web", "llm"], default="rule",
                       help="Validation mode (default: rule)")
    parser.add_argument("-w", "--web-content", help="Path to web content JSON file")
    parser.add_argument("-s", "--min-score", type=float, default=0.5,
                       help="Minimum score for web validation (default: 0.5)")
    parser.add_argument("-r", "--min-relevance-score", type=float, default=0.5,
                       help="Minimum relevance score (default: 0.5)")
    parser.add_argument("-p", "--provider", default="gemini",
                       help="LLM provider (default: gemini)")
    parser.add_argument("-o", "--output", help="Output path for results")
    parser.add_argument("-n", "--no-progress", action="store_true",
                       help="Disable progress bar")
    
    args = parser.parse_args()
    
    try:
        # Load terms
        terms_path = Path(args.terms)
        if terms_path.exists():
            with open(terms_path, "r", encoding="utf-8") as f:
                terms = [line.strip() for line in f if line.strip()]
        else:
            terms = [args.terms]
        
        # Load web content if needed
        web_content = None
        if args.web_content:
            with open(args.web_content, "r", encoding="utf-8") as f:
                web_content = json.load(f)
        
        # Run validation
        results = validate_terms(
            terms,
            modes=[args.mode],
            web_content=web_content,
            llm_provider=args.provider,
            config={
                "min_score": args.min_score,
                "min_relevance_score": args.min_relevance_score
            },
            show_progress=not args.no_progress
        )
        
        # Output results
        if args.output:
            save_validation_results(results, args.output, format="both")
            print(f"Results saved to {args.output}")
        else:
            json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
            print()
            
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()