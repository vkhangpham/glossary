"""
Main entry point for term validation.

This module provides the core validation orchestration that combines
different validation modes into a unified pipeline.

⚠️ DEPRECATION NOTICE:
The functions in this module are deprecated and will be removed in v4.0.0.
Please migrate to the new functional validation API for better performance and features.

Migration guide: https://docs.example.com/validation-migration
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Union
import time
from functools import lru_cache

from .rule_validator import validate_with_rules
from .web_validator import validate_with_web_content
from .llm_validator import validate_with_llm
from .utils import normalize_term
from .cache import get_cache, filter_cached_terms
from .core import (
    ValidationConfig, ValidationResult as CoreValidationResult, validate_terms_functional,
    validate_terms_with_cache, normalize_terms, get_validation_summary
)
from .cache import CacheState, load_cache_from_disk, save_cache_to_disk

# Type aliases
Terms = Union[str, List[str]]
LegacyValidationResult = Dict[str, Any]  # Renamed to avoid shadowing the dataclass
ValidationResults = Dict[str, LegacyValidationResult]
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
    use_cache: bool = True,
    suppress_warning: bool = False
) -> ValidationResults:
    """
    Validate terms using specified validation modes.

    ⚠️ DEPRECATED: This function is deprecated and will be removed in v4.0.0.
    Use validate_terms_functional() with configuration profiles instead.

    Migration example:
        # Old way:
        results = validate_terms(
            terms,
            modes=['rule', 'web'],
            config={'min_confidence': 0.7}
        )

        # New way:
        from generate_glossary.validation import validate_terms_functional, ACADEMIC_PROFILE, override_config, ValidationConfig
        config = override_config(ACADEMIC_PROFILE, modes=('rule', 'web'), min_confidence=0.7)
        results = validate_terms_functional(terms, config)

    This function is now a thin wrapper around the functional core that maintains
    backward compatibility while using the improved functional architecture.

    Args:
        terms: Single term or list of terms to validate
        modes: List of validation modes to use (rule, web, llm)
        web_content: Web content for web-based validation
        llm_provider: LLM provider for llm-based validation
        config: Validation configuration (overrides defaults)
        existing_results: Previous validation results to extend/update
        show_progress: Whether to show progress bar
        use_cache: Whether to use cached validation results
        suppress_warning: Whether to suppress the deprecation warning

    Returns:
        Dictionary mapping terms to their validation results (legacy format)
    """
    # Issue deprecation warning unless suppressed
    if not suppress_warning:
        warnings.warn(
            "validate_terms() is deprecated and will be removed in v4.0.0. "
            "Use validate_terms_functional() with configuration profiles instead. "
            "Benefits: better performance, immutable results, composition utilities, and type safety. "
            "See migration guide: https://docs.example.com/validation-migration",
            DeprecationWarning,
            stacklevel=2
        )
    start_time = time.time()
    
    # Merge config with defaults
    merged_config = {**DEFAULT_CONFIG, **(config or {})}
    modes = modes or merged_config.get("modes", ["rule"])
    
    # Create ValidationConfig for functional core
    validation_config = ValidationConfig(
        modes=tuple(modes),
        confidence_weights=merged_config["confidence_weights"],
        min_confidence=merged_config["min_confidence"],
        min_score=merged_config["min_score"],
        min_relevance_score=merged_config["min_relevance_score"],
        parallel=merged_config["parallel"],
        show_progress=show_progress,
        llm_provider=llm_provider,
        use_cache=use_cache
    )
    
    # Convert existing results to ValidationResult format if provided
    existing_functional_results = None
    if existing_results:
        existing_functional_results = convert_legacy_to_functional_results(existing_results)
    
    # Use functional validation with caching
    if use_cache:
        cache_state = load_cache_from_disk()
        validation_results, updated_cache_state = validate_terms_with_cache(
            terms, validation_config, web_content, existing_functional_results, cache_state, auto_save=True
        )
    else:
        validation_results = validate_terms_functional(
            terms, validation_config, web_content, existing_functional_results
        )
    
    # Convert ValidationResult objects back to legacy format for backward compatibility
    legacy_results = convert_functional_to_legacy_results(validation_results)
    
    # Merge with existing results if provided
    if existing_results:
        final_results = existing_results.copy()
        final_results.update(legacy_results)
        legacy_results = final_results
    
    elapsed_time = time.time() - start_time
    logging.info(f"Validation completed in {elapsed_time:.2f} seconds")
    
    # Log summary statistics
    valid_count = len([r for r in legacy_results.values() if r["is_valid"]])
    total_count = len(legacy_results)
    if total_count > 0:
        logging.info(f"Valid terms: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
    
    return legacy_results


def validate_terms_functional_api(
    terms: Terms,
    modes: Optional[List[str]] = None,
    web_content: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    llm_provider: str = "gemini",
    config: Optional[Dict[str, Any]] = None,
    cache_state: Optional[CacheState] = None,
    use_cache: bool = True,
    auto_save_cache: bool = True
) -> Union[Dict[str, CoreValidationResult], tuple[Dict[str, CoreValidationResult], CacheState]]:
    """
    Functional API for term validation with optional caching.

    ⚠️ DEPRECATED: This function is deprecated and will be removed in v4.0.0.
    Use validate_terms_functional() directly with configuration profiles instead.

    Migration example:
        # Old way:
        results = validate_terms_functional_api(
            terms,
            modes=['rule', 'web'],
            config={'min_confidence': 0.7},
            cache_state=cache_state
        )

        # New way:
        from generate_glossary.validation import validate_terms_with_cache, ACADEMIC_PROFILE, override_config, ValidationConfig
        config = override_config(ACADEMIC_PROFILE, modes=('rule', 'web'), min_confidence=0.7)
        results, new_cache_state = validate_terms_with_cache(terms, config, cache_state=cache_state)

    This function provides the new functional interface while maintaining
    backward compatibility. It can operate with or without cache state.

    Args:
        terms: Single term or list of terms to validate
        modes: List of validation modes to use (rule, web, llm)
        web_content: Web content for web-based validation
        llm_provider: LLM provider for llm-based validation
        config: Validation configuration (overrides defaults)
        cache_state: Optional cache state for caching results
        use_cache: Whether to use caching (only if cache_state provided)
        auto_save_cache: Whether to automatically save cache to disk

    Returns:
        If cache_state provided: (validation_results, updated_cache_state)
        Otherwise: validation_results only
    """
    # Issue deprecation warning
    warnings.warn(
        "validate_terms_functional_api() is deprecated and will be removed in v4.0.0. "
        "Use validate_terms_functional() or validate_terms_with_cache() directly. "
        "Benefits: cleaner API, better type safety, and more consistent behavior. "
        "See migration guide: https://docs.example.com/validation-migration",
        DeprecationWarning,
        stacklevel=2
    )
    # Merge config with defaults and create ValidationConfig
    merged_config = {**DEFAULT_CONFIG, **(config or {})}
    modes = modes or merged_config.get("modes", ["rule"])
    
    validation_config = ValidationConfig(
        modes=tuple(modes),
        confidence_weights=merged_config["confidence_weights"],
        min_confidence=merged_config["min_confidence"],
        min_score=merged_config["min_score"],
        min_relevance_score=merged_config["min_relevance_score"],
        parallel=merged_config["parallel"],
        show_progress=merged_config["show_progress"],
        llm_provider=llm_provider,
        use_cache=use_cache
    )
    
    # Use functional validation with caching if cache_state provided
    if cache_state is not None and use_cache:
        validation_results, updated_cache_state = validate_terms_with_cache(
            terms, validation_config, web_content, None, cache_state, auto_save_cache
        )
        return validation_results, updated_cache_state
    else:
        # Use regular functional validation
        validation_results = validate_terms_functional(terms, validation_config, web_content)
        if cache_state is not None:
            return validation_results, cache_state
        else:
            return validation_results


def convert_legacy_to_functional_results(
    legacy_results: ValidationResults
) -> Dict[str, CoreValidationResult]:
    """Convert legacy validation results to functional ValidationResult objects."""
    functional_results = {}
    
    for term, legacy_result in legacy_results.items():
        # Extract data from legacy format
        is_valid = legacy_result.get("is_valid", False)
        confidence = legacy_result.get("confidence", 0.0)
        score = legacy_result.get("score", confidence)  # Fallback to confidence if no score
        relevance_score = legacy_result.get("relevance_score")
        
        # Extract mode results if available
        mode_results = legacy_result.get("mode_results", {})
        rule_result = None
        web_result = None
        llm_result = None
        
        if "rule" in mode_results:
            from types import MappingProxyType
            rule_result = MappingProxyType(mode_results["rule"])
        if "web" in mode_results:
            from types import MappingProxyType
            web_result = MappingProxyType(mode_results["web"])
        if "llm" in mode_results:
            from types import MappingProxyType
            llm_result = MappingProxyType(mode_results["llm"])
        
        # Extract errors if available
        errors = tuple(legacy_result.get("errors", []))
        
        functional_results[term] = CoreValidationResult(
            term=term,
            is_valid=is_valid,
            confidence=confidence,
            score=score,
            relevance_score=relevance_score,
            rule_result=rule_result,
            web_result=web_result,
            llm_result=llm_result,
            errors=errors
        )
    
    return functional_results


def convert_functional_to_legacy_results(
    functional_results: Dict[str, CoreValidationResult],
    preserve_cache_flags: bool = True
) -> ValidationResults:
    """Convert functional ValidationResult objects to legacy dictionary format."""
    legacy_results = {}
    
    for term, result in functional_results.items():
        # Convert ValidationResult to legacy dictionary format
        mode_results = {}
        modes_used = []
        
        if result.rule_result:
            mode_results["rule"] = dict(result.rule_result)
            modes_used.append("rule")
        if result.web_result:
            mode_results["web"] = dict(result.web_result)
            modes_used.append("web")
        if result.llm_result:
            mode_results["llm"] = dict(result.llm_result)
            modes_used.append("llm")
        
        legacy_result = {
            "term": result.term,
            "is_valid": result.is_valid,
            "confidence": result.confidence,
            "score": result.score,
            "modes_used": modes_used,
            "mode_results": mode_results,
            "timestamp": time.time(),
        }
        
        # Add optional fields if they exist
        if result.relevance_score is not None:
            legacy_result["relevance_score"] = result.relevance_score
        if result.errors:
            legacy_result["errors"] = list(result.errors)
        
        # Preserve cached flag if it exists in the original mode_results
        if preserve_cache_flags and mode_results:
            for mode_data in mode_results.values():
                if isinstance(mode_data, dict) and mode_data.get("cached"):
                    legacy_result["cached"] = True
                    break
        
        legacy_results[term] = legacy_result
    
    return legacy_results


def validate_with_functional_cache(
    terms: Terms,
    modes: Optional[List[str]] = None,
    web_content: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    llm_provider: str = "gemini",
    config: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, CoreValidationResult]:
    """
    Convenience function for functional validation with automatic cache management.

    ⚠️ DEPRECATED: This function is deprecated and will be removed in v4.0.0.
    Use validate_terms_with_cache() with configuration profiles instead.

    Migration example:
        # Old way:
        results = validate_with_functional_cache(
            terms,
            modes=['rule', 'web'],
            config={'min_confidence': 0.7},
            cache_dir='/path/to/cache'
        )

        # New way:
        from generate_glossary.validation import validate_terms_with_cache, ACADEMIC_PROFILE, override_config, ValidationConfig
        from generate_glossary.validation.cache import load_cache_from_disk
        config = override_config(ACADEMIC_PROFILE, modes=('rule', 'web'), min_confidence=0.7)
        cache_state = load_cache_from_disk(Path('/path/to/cache'))
        results, new_cache_state = validate_terms_with_cache(terms, config, cache_state=cache_state)

    This function handles cache loading and saving automatically, providing a
    simple interface for functional validation with persistent caching.

    Args:
        terms: Single term or list of terms to validate
        modes: List of validation modes to use (rule, web, llm)
        web_content: Web content for web-based validation
        llm_provider: LLM provider for llm-based validation
        config: Validation configuration (overrides defaults)
        cache_dir: Directory for cache files (uses default if None)

    Returns:
        Dictionary mapping terms to ValidationResult objects
    """
    # Issue deprecation warning
    warnings.warn(
        "validate_with_functional_cache() is deprecated and will be removed in v4.0.0. "
        "Use validate_terms_with_cache() with configuration profiles instead. "
        "Benefits: explicit cache management, better error handling, and more control. "
        "See migration guide: https://docs.example.com/validation-migration",
        DeprecationWarning,
        stacklevel=2
    )
    from pathlib import Path
    
    # Load cache state
    cache_path = Path(cache_dir) if cache_dir else None
    cache_state = load_cache_from_disk(cache_path)
    
    # Perform validation with caching
    validation_results, updated_cache_state = validate_terms_functional_api(
        terms, modes, web_content, llm_provider, config, 
        cache_state, use_cache=True, auto_save_cache=True
    )
    
    return validation_results


def main():
    """
    CLI entry point for validation.

    ⚠️ NOTE: This CLI uses the legacy validation API for backward compatibility.
    For new projects, consider using the functional validation API directly:

    Python example:
        from generate_glossary.validation import validate_terms_functional, ACADEMIC_PROFILE, ValidationConfig
        results = validate_terms_functional(terms, ACADEMIC_PROFILE)

    The functional API provides better performance, type safety, and composition utilities.
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

    # Show migration notice for CLI users
    print("⚠️ NOTICE: This CLI uses the legacy validation API.", file=sys.stderr)
    print("   For new projects, consider the functional validation API with better features.", file=sys.stderr)
    print("   See migration guide: https://docs.example.com/validation-migration\n", file=sys.stderr)

    # Check for edge case: web mode without web_content
    if args.mode == "web" and not args.web_content:
        print("Error: Web validation mode requires --web-content argument", file=sys.stderr)
        sys.exit(2)
    
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