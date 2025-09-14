"""
Functional cache management for validation results.

This module provides persistent caching for rejected terms and validation results
using immutable data structures and pure functions. The functional approach enables
composition and eliminates global state while maintaining backward compatibility.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple, Callable, TypeVar, Mapping
from datetime import datetime
from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import logging

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "glossary_validator"
REJECTED_TERMS_FILE = CACHE_DIR / "rejected_terms.json"
VALIDATION_CACHE_FILE = CACHE_DIR / "validation_cache.json"
CACHE_EXPIRY_DAYS = 30  # Refresh cache after 30 days

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Type variables for higher-order functions
T = TypeVar('T')
ValidatorFunc = Callable[[List[str]], Dict[str, Any]]


@dataclass(frozen=True)
class RejectedTermsCache:
    """Immutable cache for rejected terms."""
    terms: Mapping[str, Dict[str, Any]] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class ValidationResultsCache:
    """Immutable cache for validation results."""
    results: Mapping[str, Dict[str, Any]] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class CacheState:
    """Immutable state containing all cache data."""
    rejected_terms: RejectedTermsCache = field(default_factory=RejectedTermsCache)
    validation_results: ValidationResultsCache = field(default_factory=ValidationResultsCache)
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)


# Pure cache functions

def create_cache_key(term: str, modes: List[str]) -> str:
    """
    Create a unique cache key for term and modes combination.

    Args:
        term: Term to create key for (normalized with lower() and strip())
        modes: List of validation modes

    Returns:
        SHA256 hash of normalized term and modes combination

    Note:
        Term normalization ensures consistent caching for variations like
        "Machine Learning", " machine learning ", "MACHINE LEARNING", etc.
        Empty or whitespace-only terms are preserved as empty strings.
    """
    term_lower = term.lower().strip()
    modes_str = ",".join(sorted(modes))
    key_str = f"{term_lower}:{modes_str}"
    return hashlib.sha256(key_str.encode()).hexdigest()


def is_cache_expired(timestamp: float, max_age_days: int = CACHE_EXPIRY_DAYS) -> bool:
    """Check if a cached entry is expired."""
    if timestamp == 0:
        return True
    
    age = datetime.now().timestamp() - timestamp
    max_age = max_age_days * 24 * 3600
    return age > max_age


def cache_get_rejected(cache_state: CacheState, term: str) -> Optional[Dict[str, Any]]:
    """Get rejected term info if exists and not expired."""
    term_lower = term.lower().strip()
    
    if term_lower in cache_state.rejected_terms.terms:
        rejection = cache_state.rejected_terms.terms[term_lower]
        timestamp = rejection.get("timestamp", 0)
        
        if not is_cache_expired(timestamp):
            return rejection
    
    return None


def cache_get_validation(cache_state: CacheState, term: str, modes: List[str]) -> Optional[Dict[str, Any]]:
    """Get cached validation result if exists and not expired."""
    cache_key = create_cache_key(term, modes)
    
    if cache_key in cache_state.validation_results.results:
        cached = cache_state.validation_results.results[cache_key]
        timestamp = cached.get("timestamp", 0)
        
        if not is_cache_expired(timestamp):
            # Handle both new format (nested 'result') and legacy format
            return cached.get('result', cached)
    
    return None


def cache_set_rejected(cache_state: CacheState, term: str, reason: str, mode: str, 
                      confidence: float = 0.0) -> CacheState:
    """Add rejected term to cache and return new cache state."""
    term_lower = term.lower().strip()
    
    new_rejection = {
        "term": term,
        "reason": reason,
        "mode": mode,
        "confidence": confidence,
        "timestamp": datetime.now().timestamp(),
        "date": datetime.now().isoformat()
    }
    
    new_terms = {**cache_state.rejected_terms.terms, term_lower: new_rejection}
    new_rejected_cache = RejectedTermsCache(terms=MappingProxyType(new_terms))
    
    return CacheState(
        rejected_terms=new_rejected_cache,
        validation_results=cache_state.validation_results,
        cache_dir=cache_state.cache_dir
    )


def cache_set_validation(cache_state: CacheState, term: str, modes: List[str], 
                        result: Dict[str, Any]) -> CacheState:
    """Add validation result to cache and return new cache state."""
    cache_key = create_cache_key(term, modes)
    
    new_cached_result = {
        "result": result,
        "timestamp": datetime.now().timestamp(),
        "date": datetime.now().isoformat()
    }
    
    new_results = {**cache_state.validation_results.results, cache_key: new_cached_result}
    new_validation_cache = ValidationResultsCache(results=MappingProxyType(new_results))
    
    # Start with current rejected terms
    new_rejected_terms = cache_state.rejected_terms.terms
    
    if result.get("is_valid", False):
        # If term is now valid, remove it from rejected terms if present
        term_lower = term.lower().strip()
        if term_lower in new_rejected_terms:
            new_rejected_terms = {k: v for k, v in new_rejected_terms.items() if k != term_lower}
    else:
        # Add to rejected terms if invalid
        confidence = result.get("confidence", 0.0)
        mode_str = ", ".join(modes)
        reason = f"Failed validation with modes: {mode_str}"
        term_lower = term.lower().strip()
        
        new_rejection = {
            "term": term,
            "reason": reason,
            "mode": mode_str,
            "confidence": confidence,
            "timestamp": datetime.now().timestamp(),
            "date": datetime.now().isoformat()
        }
        new_rejected_terms = {**new_rejected_terms, term_lower: new_rejection}
    
    return CacheState(
        rejected_terms=RejectedTermsCache(terms=MappingProxyType(new_rejected_terms)),
        validation_results=new_validation_cache,
        cache_dir=cache_state.cache_dir
    )


def cache_remove_expired(cache_state: CacheState) -> Tuple[CacheState, bool]:
    """Remove expired entries and return new cache state and whether state changed."""
    # Filter expired rejected terms
    valid_rejected = {
        term: data for term, data in cache_state.rejected_terms.terms.items()
        if not is_cache_expired(data.get("timestamp", 0))
    }
    
    # Filter expired validation results
    valid_validation = {
        key: data for key, data in cache_state.validation_results.results.items()
        if not is_cache_expired(data.get("timestamp", 0))
    }
    
    # Check if state actually changed
    state_changed = (
        len(valid_rejected) != len(cache_state.rejected_terms.terms) or
        len(valid_validation) != len(cache_state.validation_results.results)
    )
    
    new_state = CacheState(
        rejected_terms=RejectedTermsCache(terms=MappingProxyType(valid_rejected)),
        validation_results=ValidationResultsCache(results=MappingProxyType(valid_validation)),
        cache_dir=cache_state.cache_dir
    )
    
    return new_state, state_changed


def cache_clear(cache_state: CacheState, expired_only: bool = True) -> Tuple[CacheState, Dict[str, int]]:
    """Clear cache entries and return new state with clear counts."""
    if expired_only:
        new_state, _ = cache_remove_expired(cache_state)
        cleared = {
            "rejected": len(cache_state.rejected_terms.terms) - len(new_state.rejected_terms.terms),
            "validation": len(cache_state.validation_results.results) - len(new_state.validation_results.results)
        }
    else:
        new_state = CacheState(cache_dir=cache_state.cache_dir)
        cleared = {
            "rejected": len(cache_state.rejected_terms.terms),
            "validation": len(cache_state.validation_results.results)
        }
    
    return new_state, cleared


# Disk I/O functions

def load_cache_from_disk(cache_dir: Optional[Path] = None) -> CacheState:
    """Load cache state from disk files."""
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    rejected_file = cache_dir / "rejected_terms.json"
    validation_file = cache_dir / "validation_cache.json"
    
    # Load rejected terms
    rejected_terms = {}
    if rejected_file.exists():
        try:
            with open(rejected_file, 'r', encoding='utf-8') as f:
                rejected_terms = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load rejected terms cache: {e}")
    
    # Load validation cache
    validation_results = {}
    if validation_file.exists():
        try:
            with open(validation_file, 'r', encoding='utf-8') as f:
                raw_validation_results = json.load(f)
                # Migrate legacy cache entries that don't have nested 'result' structure
                for key, entry in raw_validation_results.items():
                    if 'result' not in entry and 'timestamp' in entry:
                        # This is a legacy format, wrap in new structure
                        validation_results[key] = {
                            'result': entry,
                            'timestamp': entry.get('timestamp', 0),
                            'date': entry.get('date', '')
                        }
                    else:
                        # This is already new format
                        validation_results[key] = entry
        except Exception as e:
            logging.warning(f"Failed to load validation cache: {e}")
    
    cache_state = CacheState(
        rejected_terms=RejectedTermsCache(terms=MappingProxyType(rejected_terms)),
        validation_results=ValidationResultsCache(results=MappingProxyType(validation_results)),
        cache_dir=cache_dir
    )
    
    # Clean expired entries immediately after loading
    cleaned_state, _ = cache_remove_expired(cache_state)
    return cleaned_state


def save_cache_to_disk(cache_state: CacheState) -> None:
    """Save cache state to disk files."""
    cache_dir = cache_state.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    rejected_file = cache_dir / "rejected_terms.json"
    validation_file = cache_dir / "validation_cache.json"
    
    # Save rejected terms (convert MappingProxyType to dict for JSON)
    try:
        with open(rejected_file, 'w', encoding='utf-8') as f:
            json.dump(dict(cache_state.rejected_terms.terms), f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save rejected terms: {e}")
    
    # Save validation cache (convert MappingProxyType to dict for JSON)
    try:
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(dict(cache_state.validation_results.results), f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save validation cache: {e}")


def get_cache_stats(cache_state: CacheState) -> Dict[str, Any]:
    """Get cache statistics."""
    valid_rejected = sum(
        1 for r in cache_state.rejected_terms.terms.values()
        if not is_cache_expired(r.get("timestamp", 0))
    )
    
    valid_cached = sum(
        1 for c in cache_state.validation_results.results.values()
        if not is_cache_expired(c.get("timestamp", 0))
    )
    
    return {
        "rejected_terms": {
            "total": len(cache_state.rejected_terms.terms),
            "valid": valid_rejected,
            "expired": len(cache_state.rejected_terms.terms) - valid_rejected
        },
        "validation_cache": {
            "total": len(cache_state.validation_results.results),
            "valid": valid_cached,
            "expired": len(cache_state.validation_results.results) - valid_cached
        },
        "cache_dir": str(cache_state.cache_dir),
        "cache_size_kb": _get_cache_size_kb(cache_state)
    }


def _get_cache_size_kb(cache_state: CacheState) -> float:
    """Get total cache size in KB."""
    total_size = 0
    rejected_file = cache_state.cache_dir / "rejected_terms.json"
    validation_file = cache_state.cache_dir / "validation_cache.json"
    
    if rejected_file.exists():
        total_size += rejected_file.stat().st_size
    if validation_file.exists():
        total_size += validation_file.stat().st_size
    
    return round(total_size / 1024, 2)


# Higher-order cache composition functions

def filter_cached_terms_functional(terms: List[str], modes: List[str], 
                                  cache_state: CacheState) -> Tuple[List[str], Dict[str, Any]]:
    """
    Functional version of cache filtering.
    
    Returns:
        Tuple of (uncached_terms, cached_results)
    """
    uncached = []
    cached_results = {}
    
    for term in terms:
        # First check if it's rejected (fast path)
        rejection = cache_get_rejected(cache_state, term)
        if rejection:
            # Convert rejection to validation result
            cached_results[term] = {
                "term": term,
                "is_valid": False,
                "confidence": rejection.get("confidence", 0.0),
                "modes_used": rejection.get("mode", "").split(", "),
                "cached": True,
                "cache_reason": rejection.get("reason", "Previously rejected")
            }
        else:
            # Check validation cache
            result = cache_get_validation(cache_state, term, modes)
            if result:
                cached_results[term] = {**result, "cached": True}
            else:
                uncached.append(term)
    
    return uncached, cached_results


def with_cache(validator_func: ValidatorFunc, cache_state: CacheState, 
               modes: List[str], auto_save: bool = True) -> Callable[[List[str]], Tuple[Dict[str, Any], CacheState]]:
    """
    Higher-order function that adds caching to any validator function.
    
    Args:
        validator_func: Function that validates terms and returns results
        cache_state: Current cache state
        modes: List of validation modes
        auto_save: Whether to automatically save cache to disk after updates
        
    Returns:
        Function that returns (validation_results, new_cache_state)
    
    Warning:
        This function captures the initial cache_state. For subsequent calls,
        use the returned updated state or use with_cache_state() instead.
    """
    def cached_validator(terms: List[str]) -> Tuple[Dict[str, Any], CacheState]:
        # Filter cached terms
        uncached_terms, cached_results = filter_cached_terms_functional(terms, modes, cache_state)
        
        # Validate uncached terms
        new_results = {}
        updated_cache_state = cache_state
        
        if uncached_terms:
            validation_results = validator_func(uncached_terms)
            
            # Add results to cache
            for term in uncached_terms:
                if term in validation_results:
                    result = validation_results[term]
                    new_results[term] = result
                    updated_cache_state = cache_set_validation(updated_cache_state, term, modes, result)
        
        # Combine cached and new results
        all_results = {**cached_results, **new_results}
        
        # Auto-save if requested
        if auto_save and updated_cache_state != cache_state:
            save_cache_to_disk(updated_cache_state)
        
        return all_results, updated_cache_state
    
    return cached_validator


def with_cache_state(validator_func: ValidatorFunc, modes: List[str], 
                     auto_save: bool = True) -> Callable[[List[str], CacheState], Tuple[Dict[str, Any], CacheState]]:
    """
    Higher-order function that adds caching to any validator function with explicit cache state parameter.
    
    This version takes cache_state as a parameter on each call, reducing the risk of
    stale cache usage compared to with_cache().
    
    Args:
        validator_func: Function that validates terms and returns results
        modes: List of validation modes
        auto_save: Whether to automatically save cache to disk after updates
        
    Returns:
        Function that takes (terms, cache_state) and returns (validation_results, new_cache_state)
    """
    def cached_validator(terms: List[str], cache_state: CacheState) -> Tuple[Dict[str, Any], CacheState]:
        # Filter cached terms
        uncached_terms, cached_results = filter_cached_terms_functional(terms, modes, cache_state)
        
        # Validate uncached terms
        new_results = {}
        updated_cache_state = cache_state
        
        if uncached_terms:
            validation_results = validator_func(uncached_terms)
            
            # Add results to cache
            for term in uncached_terms:
                if term in validation_results:
                    result = validation_results[term]
                    new_results[term] = result
                    updated_cache_state = cache_set_validation(updated_cache_state, term, modes, result)
        
        # Combine cached and new results
        all_results = {**cached_results, **new_results}
        
        # Auto-save if requested
        if auto_save and updated_cache_state != cache_state:
            save_cache_to_disk(updated_cache_state)
        
        return all_results, updated_cache_state
    
    return cached_validator


def compose_with_cache(cache_state: CacheState, modes: List[str], auto_save: bool = True):
    """
    Decorator factory for adding caching to validator functions.
    
    Usage:
        @compose_with_cache(cache_state, ["rule", "web"])
        def my_validator(terms):
            return validate_terms(terms)
    """
    def decorator(validator_func: ValidatorFunc):
        return with_cache(validator_func, cache_state, modes, auto_save)
    return decorator


# Backward compatibility layer

class ValidationCache:
    """
    Backward compatibility wrapper for the functional cache system.
    
    This class maintains the original API while using the functional cache
    implementation under the hood.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager."""
        self.cache_dir = cache_dir or CACHE_DIR
        self._cache_state = load_cache_from_disk(self.cache_dir)
        
    def _sync_state(self) -> None:
        """Synchronize internal state with functional cache and save to disk."""
        save_cache_to_disk(self._cache_state)
        
    def is_rejected(self, term: str) -> Optional[Dict[str, Any]]:
        """Check if a term was previously rejected."""
        rejection = cache_get_rejected(self._cache_state, term)
        if rejection is None:
            # Clean up expired entries
            self._cache_state, state_changed = cache_remove_expired(self._cache_state)
            if state_changed:
                self._sync_state()
        return rejection
    
    def add_rejected(self, term: str, reason: str, mode: str, confidence: float = 0.0) -> None:
        """Add a term to the rejected cache."""
        self._cache_state = cache_set_rejected(self._cache_state, term, reason, mode, confidence)
        self._sync_state()
    
    def get_validation_result(self, term: str, modes: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached validation result for a term and mode combination."""
        result = cache_get_validation(self._cache_state, term, modes)
        if result is None:
            # Clean up expired entries
            self._cache_state, state_changed = cache_remove_expired(self._cache_state)
            if state_changed:
                self._sync_state()
        else:
            logging.debug(f"Cache hit for term '{term}' with modes {modes}")
        return result
    
    def add_validation_result(self, term: str, modes: List[str], result: Dict[str, Any]) -> None:
        """Cache a validation result."""
        self._cache_state = cache_set_validation(self._cache_state, term, modes, result)
        self._sync_state()
    
    def _create_cache_key(self, term: str, modes: List[str]) -> str:
        """Create a unique cache key for term and modes combination."""
        return create_cache_key(term, modes)
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cached entry is expired."""
        return is_cache_expired(timestamp)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return get_cache_stats(self._cache_state)
    
    def _get_cache_size_kb(self) -> float:
        """Get total cache size in KB."""
        return _get_cache_size_kb(self._cache_state)
    
    def clear_cache(self, expired_only: bool = True) -> Dict[str, int]:
        """Clear cache entries."""
        self._cache_state, cleared = cache_clear(self._cache_state, expired_only)
        self._sync_state()
        return cleared
    
    # Legacy properties for direct access (deprecated)
    @property
    def rejected_terms(self) -> Dict[str, Dict[str, Any]]:
        """Legacy property access to rejected terms."""
        return self._cache_state.rejected_terms.terms
    
    @property
    def validation_cache(self) -> Dict[str, Dict[str, Any]]:
        """Legacy property access to validation cache."""
        return self._cache_state.validation_results.results


# Global cache instance
_cache_instance = None


def get_cache() -> ValidationCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ValidationCache()
    return _cache_instance


def check_rejected_terms(terms: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Check which terms are in the rejected cache.
    
    Returns dictionary mapping rejected terms to their rejection info.
    """
    cache = get_cache()
    rejected = {}
    
    for term in terms:
        rejection = cache.is_rejected(term)
        if rejection:
            rejected[term] = rejection
    
    return rejected


def filter_cached_terms(terms: List[str], modes: List[str]) -> tuple[List[str], Dict[str, Any]]:
    """
    Filter out terms that have cached results.
    
    Returns:
        Tuple of (uncached_terms, cached_results)
    """
    cache = get_cache()
    uncached = []
    cached_results = {}
    
    for term in terms:
        # First check if it's rejected (fast path)
        rejection = cache.is_rejected(term)
        if rejection:
            # Convert rejection to validation result
            cached_results[term] = {
                "term": term,
                "is_valid": False,
                "confidence": rejection.get("confidence", 0.0),
                "modes_used": rejection.get("mode", "").split(", "),
                "cached": True,
                "cache_reason": rejection.get("reason", "Previously rejected")
            }
        else:
            # Check validation cache
            result = cache.get_validation_result(term, modes)
            if result:
                cached_results[term] = {**result, "cached": True}
            else:
                uncached.append(term)
    
    return uncached, cached_results


# API wrapper functions for requested naming convention

def cache_get(cache_state: CacheState, term: str, modes: List[str]) -> Optional[Dict[str, Any]]:
    """Wrapper for cache_get_validation to match requested API naming."""
    return cache_get_validation(cache_state, term, modes)


def cache_set(cache_state: CacheState, term: str, modes: List[str], result: Dict[str, Any]) -> CacheState:
    """Wrapper for cache_set_validation to match requested API naming."""
    return cache_set_validation(cache_state, term, modes, result)


def load_cache(cache_dir: Optional[Path] = None) -> CacheState:
    """Wrapper for load_cache_from_disk to match requested API naming."""
    return load_cache_from_disk(cache_dir)


def save_cache(cache_state: CacheState) -> None:
    """Wrapper for save_cache_to_disk to match requested API naming."""
    save_cache_to_disk(cache_state)
