"""
Cache management for validation results.

This module provides persistent caching for rejected terms and validation results
to avoid recomputing expensive validations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timedelta
import hashlib
from functools import lru_cache
import logging

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "glossary_validator"
REJECTED_TERMS_FILE = CACHE_DIR / "rejected_terms.json"
VALIDATION_CACHE_FILE = CACHE_DIR / "validation_cache.json"
CACHE_EXPIRY_DAYS = 30  # Refresh cache after 30 days

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ValidationCache:
    """Manages persistent caching of validation results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager."""
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.rejected_file = self.cache_dir / "rejected_terms.json"
        self.validation_file = self.cache_dir / "validation_cache.json"
        
        # Load existing caches
        self.rejected_terms = self._load_rejected_terms()
        self.validation_cache = self._load_validation_cache()
        
        # In-memory LRU cache for fast lookups
        self._memory_cache = {}
        
    def _load_rejected_terms(self) -> Dict[str, Dict[str, Any]]:
        """Load rejected terms from disk."""
        if self.rejected_file.exists():
            try:
                with open(self.rejected_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load rejected terms cache: {e}")
        return {}
    
    def _load_validation_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load validation cache from disk."""
        if self.validation_file.exists():
            try:
                with open(self.validation_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load validation cache: {e}")
        return {}
    
    def _save_rejected_terms(self) -> None:
        """Persist rejected terms to disk."""
        try:
            with open(self.rejected_file, 'w', encoding='utf-8') as f:
                json.dump(self.rejected_terms, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save rejected terms: {e}")
    
    def _save_validation_cache(self) -> None:
        """Persist validation cache to disk."""
        try:
            with open(self.validation_file, 'w', encoding='utf-8') as f:
                json.dump(self.validation_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save validation cache: {e}")
    
    def is_rejected(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Check if a term was previously rejected.
        
        Returns rejection info if rejected, None otherwise.
        """
        term_lower = term.lower().strip()
        
        if term_lower in self.rejected_terms:
            rejection = self.rejected_terms[term_lower]
            
            # Check if cache is expired
            timestamp = rejection.get("timestamp", 0)
            if self._is_expired(timestamp):
                # Remove expired entry
                del self.rejected_terms[term_lower]
                self._save_rejected_terms()
                return None
            
            return rejection
        
        return None
    
    def add_rejected(self, term: str, reason: str, mode: str, confidence: float = 0.0) -> None:
        """Add a term to the rejected cache."""
        term_lower = term.lower().strip()
        
        self.rejected_terms[term_lower] = {
            "term": term,
            "reason": reason,
            "mode": mode,
            "confidence": confidence,
            "timestamp": datetime.now().timestamp(),
            "date": datetime.now().isoformat()
        }
        
        self._save_rejected_terms()
    
    def get_validation_result(self, term: str, modes: List[str]) -> Optional[Dict[str, Any]]:
        """
        Get cached validation result for a term and mode combination.
        
        Returns cached result if valid, None otherwise.
        """
        # Create cache key from term and modes
        cache_key = self._create_cache_key(term, modes)
        
        if cache_key in self.validation_cache:
            cached = self.validation_cache[cache_key]
            
            # Check expiry
            if not self._is_expired(cached.get("timestamp", 0)):
                logging.debug(f"Cache hit for term '{term}' with modes {modes}")
                return cached["result"]
            else:
                # Remove expired entry
                del self.validation_cache[cache_key]
        
        return None
    
    def add_validation_result(self, term: str, modes: List[str], result: Dict[str, Any]) -> None:
        """Cache a validation result."""
        cache_key = self._create_cache_key(term, modes)
        
        self.validation_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now().timestamp(),
            "date": datetime.now().isoformat()
        }
        
        # Also update rejected terms if invalid
        if not result.get("is_valid", False):
            confidence = result.get("confidence", 0.0)
            mode_str = ", ".join(modes)
            reason = f"Failed validation with modes: {mode_str}"
            self.add_rejected(term, reason, mode_str, confidence)
        
        self._save_validation_cache()
    
    def _create_cache_key(self, term: str, modes: List[str]) -> str:
        """Create a unique cache key for term and modes combination."""
        term_lower = term.lower().strip()
        modes_str = ",".join(sorted(modes))
        key_str = f"{term_lower}:{modes_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cached entry is expired."""
        if timestamp == 0:
            return True
        
        age = datetime.now().timestamp() - timestamp
        max_age = CACHE_EXPIRY_DAYS * 24 * 3600
        return age > max_age
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        valid_rejected = sum(
            1 for r in self.rejected_terms.values()
            if not self._is_expired(r.get("timestamp", 0))
        )
        
        valid_cached = sum(
            1 for c in self.validation_cache.values()
            if not self._is_expired(c.get("timestamp", 0))
        )
        
        return {
            "rejected_terms": {
                "total": len(self.rejected_terms),
                "valid": valid_rejected,
                "expired": len(self.rejected_terms) - valid_rejected
            },
            "validation_cache": {
                "total": len(self.validation_cache),
                "valid": valid_cached,
                "expired": len(self.validation_cache) - valid_cached
            },
            "cache_dir": str(self.cache_dir),
            "cache_size_kb": self._get_cache_size_kb()
        }
    
    def _get_cache_size_kb(self) -> float:
        """Get total cache size in KB."""
        total_size = 0
        if self.rejected_file.exists():
            total_size += self.rejected_file.stat().st_size
        if self.validation_file.exists():
            total_size += self.validation_file.stat().st_size
        return round(total_size / 1024, 2)
    
    def clear_cache(self, expired_only: bool = True) -> Dict[str, int]:
        """
        Clear cache entries.
        
        Args:
            expired_only: If True, only clear expired entries
            
        Returns:
            Dictionary with counts of cleared entries
        """
        cleared = {"rejected": 0, "validation": 0}
        
        if expired_only:
            # Clear expired rejected terms
            expired_rejected = [
                term for term, data in self.rejected_terms.items()
                if self._is_expired(data.get("timestamp", 0))
            ]
            for term in expired_rejected:
                del self.rejected_terms[term]
                cleared["rejected"] += 1
            
            # Clear expired validation cache
            expired_validation = [
                key for key, data in self.validation_cache.items()
                if self._is_expired(data.get("timestamp", 0))
            ]
            for key in expired_validation:
                del self.validation_cache[key]
                cleared["validation"] += 1
        else:
            # Clear everything
            cleared["rejected"] = len(self.rejected_terms)
            cleared["validation"] = len(self.validation_cache)
            self.rejected_terms = {}
            self.validation_cache = {}
        
        # Save changes
        self._save_rejected_terms()
        self._save_validation_cache()
        
        return cleared


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