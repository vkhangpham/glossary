"""
Rule-based validation for terms.

This module provides validation based on structural and linguistic rules
without requiring external data sources.
"""

import re
import os
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .utils import clean_text, normalize_term

# Constants
MIN_TERM_LENGTH = 2
MAX_TERM_LENGTH = 100

# Compile regex patterns once for performance
INVALID_PATTERNS_COMPILED = [
    re.compile(r'^\d+$'),  # Pure numbers
    re.compile(r'^[^\w\s]+$'),  # Only special characters
    re.compile(r'^\s*$'),  # Empty or whitespace only
    re.compile(r'^-+$'),  # Only hyphens
]

# Common non-academic terms to filter out (static default)
DEFAULT_BLACKLIST_TERMS = {
    'test', 'example', 'sample', 'demo', 'none', 'null', 
    'undefined', 'unknown', 'other', 'misc', 'miscellaneous',
    'page', 'home', 'index', 'about', 'contact'
}


def rule_validate(
    terms: List[str],
    max_workers: int = 4,
    blacklist_terms: Optional[Set[str]] = None,
    min_term_length: int = 2,
    max_term_length: int = 100,
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Pure functional rule-based validation.
    
    Args:
        terms: List of terms to validate
        max_workers: Maximum number of worker threads
        blacklist_terms: Set of blacklisted terms (uses DEFAULT_BLACKLIST_TERMS if None)
        min_term_length: Minimum term length
        max_term_length: Maximum term length
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping terms to validation results
    """
    # Set default blacklist if not provided
    blacklist_terms = blacklist_terms or DEFAULT_BLACKLIST_TERMS
    
    # Use thread pool with resource limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        validation_fn = lambda term: _validate_single_term_pure(
            term, blacklist_terms, min_term_length, max_term_length
        )
        
        if show_progress:
            results = list(tqdm(
                executor.map(validation_fn, terms),
                total=len(terms),
                desc="Rule validation"
            ))
        else:
            results = list(executor.map(validation_fn, terms))
    
    # Convert to dictionary
    return {r["term"]: r for r in results}


def validate_with_rules(
    terms: List[str],
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Validate terms using structural and linguistic rules.
    
    Legacy wrapper that maintains backward compatibility.
    
    Args:
        terms: List of terms to validate
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping terms to validation results
    """
    try:
        from generate_glossary.config import get_validation_config
        validation_config = get_validation_config()
        max_workers = getattr(validation_config, 'max_workers', 4)
    except (ImportError, AttributeError):
        max_workers = 4
    
    # Get blacklist from environment if available (legacy behavior)
    blacklist_terms = set(
        os.environ.get('VALIDATOR_BLACKLIST', '').split(',') 
        if os.environ.get('VALIDATOR_BLACKLIST') 
        else DEFAULT_BLACKLIST_TERMS
    )
    
    return rule_validate(
        terms, 
        max_workers=max_workers, 
        blacklist_terms=blacklist_terms,
        show_progress=show_progress
    )


def _validate_single_term_pure(
    term: str, 
    blacklist_terms: Set[str], 
    min_term_length: int, 
    max_term_length: int
) -> Dict[str, Any]:
    """
    Pure functional validation of a single term using rules.
    
    Args:
        term: Term to validate
        blacklist_terms: Set of blacklisted terms
        min_term_length: Minimum term length
        max_term_length: Maximum term length
        
    Returns:
        Validation result dictionary
    """
    # Clean and normalize the term
    cleaned = clean_text(term, remove_common_words=False)
    normalized = normalize_term(term)
    
    # Initialize validation checks
    checks = {
        "has_minimum_length": len(cleaned) >= min_term_length,
        "has_maximum_length": len(cleaned) <= max_term_length,
        "has_letters": bool(re.search(r'[a-zA-Z]', cleaned)),
        "not_blacklisted": normalized not in blacklist_terms,
        "no_invalid_patterns": True,
        "has_valid_structure": True
    }
    
    # Check against compiled invalid patterns
    for pattern in INVALID_PATTERNS_COMPILED:
        if pattern.match(cleaned):
            checks["no_invalid_patterns"] = False
            break
    
    # Check structure
    checks["has_valid_structure"] = _check_term_structure_pure(cleaned)
    
    # Calculate confidence based on passed checks
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    confidence = passed_checks / total_checks
    
    # Term is valid if it passes all critical checks
    is_valid = (
        checks["has_minimum_length"] and
        checks["has_letters"] and
        checks["not_blacklisted"] and
        checks["no_invalid_patterns"]
    )
    
    return {
        "term": term,
        "is_valid": is_valid,
        "confidence": round(confidence, 3),
        "mode": "rule",
        "details": {
            "cleaned_term": cleaned,
            "normalized_term": normalized,
            "checks": checks,
            "passed_checks": passed_checks,
            "total_checks": total_checks
        }
    }


def _validate_single_term(term: str) -> Dict[str, Any]:
    """
    Legacy wrapper for backward compatibility.
    
    Args:
        term: Term to validate
        
    Returns:
        Validation result dictionary
    """
    return _validate_single_term_pure(
        term, DEFAULT_BLACKLIST_TERMS, MIN_TERM_LENGTH, MAX_TERM_LENGTH
    )


def _check_term_structure_pure(term: str) -> bool:
    """
    Pure function to check if term has valid academic/technical structure.
    
    Args:
        term: Cleaned term to check
        
    Returns:
        True if structure is valid
    """
    # Check for reasonable word boundaries
    words = term.split()
    
    # Single word terms should be substantial
    if len(words) == 1:
        return len(term) >= 3
    
    # Multi-word terms should have reasonable structure
    if len(words) > 10:  # Too many words
        return False
    
    # Check for repeated words
    if len(words) != len(set(words)):
        # Some repetition is ok (e.g., "time series time analysis")
        # but excessive repetition is not
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.6:
            return False
    
    # Check for overly short words in multi-word terms
    short_words = [w for w in words if len(w) <= 1]
    if len(short_words) > len(words) / 2:
        return False
    
    return True


def validate_term_format(term: str) -> Dict[str, Any]:
    """
    Quick format validation for a single term.
    
    Useful for pre-filtering before more expensive validation.
    
    Args:
        term: Term to validate
        
    Returns:
        Format validation result
    """
    result = {
        "term": term,
        "is_valid_format": True,
        "issues": []
    }
    
    # Check basic format issues
    if not term or not term.strip():
        result["is_valid_format"] = False
        result["issues"].append("empty_term")
    
    if len(term) < MIN_TERM_LENGTH:
        result["is_valid_format"] = False
        result["issues"].append("too_short")
    
    if len(term) > MAX_TERM_LENGTH:
        result["is_valid_format"] = False
        result["issues"].append("too_long")
    
    # Check for suspicious patterns
    if term.strip().startswith('-') or term.strip().endswith('-'):
        result["issues"].append("leading_trailing_hyphen")
    
    if '  ' in term:  # Multiple spaces
        result["issues"].append("multiple_spaces")
    
    if re.match(r'^\W+$', term):  # Only non-word characters
        result["is_valid_format"] = False
        result["issues"].append("only_special_chars")
    
    return result


def _check_term_structure(term: str) -> bool:
    """
    Legacy wrapper for backward compatibility.
    
    Args:
        term: Cleaned term to check
        
    Returns:
        True if structure is valid
    """
    return _check_term_structure_pure(term)