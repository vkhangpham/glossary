"""
Validation modes for technical concepts.

This module provides pure functions for different validation modes:
1. Rule-based validation (basic term structure)
2. Wikipedia-based validation (semantic similarity)
3. LLM-based validation (concept verification)
4. Web-based validation (content verification)

Note: The internal API uses 'rules' (plural) for rule-based validation,
while the CLI uses 'rule' (singular) for consistency with the deduplicator.
The CLI maps between these naming conventions.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from functools import lru_cache
import re
from rapidfuzz import fuzz
from nltk.stem import WordNetLemmatizer
from unicodedata import normalize

from .validation_utils import calculate_relevance_score

# Type aliases
ValidationResult = Dict[str, Any]
WikiData = Dict[str, List[Dict[str, Any]]]
WebContent = Dict[str, Any]  # {url: str, title: str, content: str}

# Constants
TOKEN_SET_RATIO_THRESHOLD = 80
SEMANTIC_THRESHOLD = 0.55
DEFAULT_MIN_SCORE = 0.7
DEFAULT_MIN_RELEVANCE_SCORE = 0.8

# Common words to remove in multi-word terms
COMMON_WORDS = {
    "of", "and", "or", "in", "the", "a", "an", "for", 
    "to", "by", "with", "on", "at", "&", "vs", "versus"
}

def clean_text(text: str, remove_common_words: bool = True) -> str:
    """Clean and normalize text for comparison."""
    # Normalize unicode characters
    text = normalize("NFKC", text.lower())
    
    # Remove punctuation and extra whitespace
    text = re.sub(r"[^\w\s-]", " ", text)
    words = text.split()
    
    # Remove common words for multi-word terms
    if remove_common_words and len(words) > 1:
        words = [w for w in words if w not in COMMON_WORDS]
    
    return " ".join(words).strip()

def validate_rules(term: str) -> ValidationResult:
    """Validate a term using basic rules."""
    cleaned = clean_text(term)
    
    # Basic validation rules
    is_valid = bool(
        cleaned and                     # Not empty after cleaning
        len(cleaned) >= 2 and          # At least 2 chars
        not cleaned.isdigit() and      # Not just numbers
        not cleaned.startswith("-")    # Not starting with hyphen
    )
    
    return {
        "term": term,
        "is_valid": is_valid,
        "mode": "rules",
        "details": {
            "cleaned_term": cleaned,
            "length": len(cleaned),
            "has_letters": not cleaned.isdigit()
        }
    }

def validate_wiki(term: str, wiki_data: Optional[WikiData] = None) -> ValidationResult:
    """
    Validate a term using Wikipedia data.
    
    Args:
        term: The term to validate
        wiki_data: Dictionary mapping terms to their Wikipedia entries
    """
    # First apply basic rules
    base_result = validate_rules(term)
    if not base_result["is_valid"]:
        return {**base_result, "mode": "wiki"}
        
    # Check Wikipedia data
    if not wiki_data:
        return {
            "term": term,
            "is_valid": False,
            "mode": "wiki",
            "details": {
                "error": "No Wikipedia data provided",
                "has_wiki_pages": False,
                "num_pages": 0,
                "pages": []
            }
        }
        
    # Check if term has Wikipedia entries
    wiki_pages = wiki_data.get(term, [])
    has_wiki = bool(wiki_pages)
    
    return {
        "term": term,
        "is_valid": has_wiki,
        "mode": "wiki",
        "details": {
            "has_wiki_pages": has_wiki,
            "num_pages": len(wiki_pages),
            "pages": wiki_pages
        }
    }

def validate_llm(term: str, llm_response: str) -> ValidationResult:
    """
    Validate a term using LLM response.
    
    Args:
        term: The term to validate
        llm_response: Response from LLM about term validity
    """
    # First apply basic rules
    base_result = validate_rules(term)
    if not base_result["is_valid"]:
        return {**base_result, "mode": "llm"}
    
    # Check if LLM response is provided
    if not llm_response:
        return {
            "term": term,
            "is_valid": False,
            "mode": "llm",
            "details": {
                "error": "No LLM response provided",
                "llm_response": None
            }
        }
    
    # Parse LLM response (expecting "yes" or "no" at the start)
    is_valid = llm_response.lower().startswith("yes")
    
    return {
        "term": term,
        "is_valid": is_valid,
        "mode": "llm",
        "details": {
            "llm_response": llm_response
        }
    }

def validate_web(
    term: str, 
    web_contents: List[Any], 
    min_score: float = DEFAULT_MIN_SCORE,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE
) -> ValidationResult:
    """
    Validate a term using web content.
    
    Args:
        term: The term to validate
        web_contents: List of web content objects (WebContent model from web_miner.py)
        min_score: Minimum score for content to be considered valid
        min_relevance_score: Minimum relevance score for content to be considered relevant
    """
    # First apply basic rules
    base_result = validate_rules(term)
    if not base_result["is_valid"]:
        return {**base_result, "mode": "web"}
    
    # Check if web content is provided
    if not web_contents:
        return {
            "term": term,
            "is_valid": False,
            "mode": "web",
            "details": {
                "error": "No web content provided",
                "num_sources": 0,
                "verified_sources": [],
                "unverified_sources": [],
                "relevant_sources": []
            }
        }
    
    # Process each content source
    verified_sources = []
    unverified_sources = []
    relevant_sources = []
    
    for content in web_contents:
        # Get content fields, handling both dict and object formats
        url = content.get("url", "") if isinstance(content, dict) else getattr(content, "url", "")
        title = content.get("title", "") if isinstance(content, dict) else getattr(content, "title", "")
        
        # Get score, handling both formats
        content_score = 0.5  # Default score
        if isinstance(content, dict):
            content_score = content.get("score", 0.5)
            is_verified = content.get("is_verified", False)
        else:
            content_score = getattr(content, "score", 0.5)
            is_verified = getattr(content, "is_verified", False)
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(term, content)
        
        # Create source info dictionary
        source_info = {
            "url": url,
            "title": title,
            "score": round(content_score, 2),  # Round to 2 decimal places
            "relevance_score": round(relevance_score, 2)  # Add relevance score
        }
        
        # Add to appropriate lists based on verification and relevance status
        if is_verified and content_score >= min_score:
            verified_sources.append(source_info)
            
            # Also track relevant sources separately
            if relevance_score >= min_relevance_score:
                relevant_sources.append(source_info)
        else:
            unverified_sources.append(source_info)
    
    # Sort sources by combined score (content score * relevance score) in descending order
    verified_sources.sort(key=lambda x: x["score"] * x["relevance_score"], reverse=True)
    unverified_sources.sort(key=lambda x: x["score"] * x["relevance_score"], reverse=True)
    relevant_sources.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # A term is valid if it has at least one verified AND relevant source
    is_valid = bool(relevant_sources)
    
    return {
        "term": term,
        "is_valid": is_valid,
        "mode": "web",
        "details": {
            "num_sources": len(web_contents),
            "verified_sources": verified_sources,
            "unverified_sources": unverified_sources,
            "relevant_sources": relevant_sources,
            "has_relevant_sources": bool(relevant_sources),
            "highest_relevance_score": relevant_sources[0]["relevance_score"] if relevant_sources else 0.0
        }
    } 