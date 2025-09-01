"""
Utility functions for validation module.
"""

import re
from typing import Dict, Any, Tuple, Set
from unicodedata import normalize
from collections import Counter
from rapidfuzz import fuzz

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


def normalize_term(term: str) -> str:
    """Normalize a term for consistent comparison."""
    normalized = clean_text(term, remove_common_words=False)
    normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
    return normalized.strip()


def get_fuzzy_score(text1: str, text2: str) -> float:
    """Calculate fuzzy matching score between two texts."""
    return fuzz.token_set_ratio(text1, text2)


def extract_web_content_fields(content: Dict[str, Any]) -> Tuple[str, str, float, bool]:
    """Extract key fields from web content object."""
    # Handle both dict and object formats
    if isinstance(content, dict):
        url = content.get("url", "")
        title = content.get("title", "")
        score = content.get("score", 0.5)
        is_verified = content.get("is_verified", False)
    else:
        url = getattr(content, "url", "")
        title = getattr(content, "title", "")
        score = getattr(content, "score", 0.5)
        is_verified = getattr(content, "is_verified", False)
    
    return url, title, score, is_verified


def extract_web_content_text(content: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract text content fields from web content."""
    # Handle both dict and object formats
    if isinstance(content, dict):
        title = content.get("title", "")
        snippet = content.get("snippet", "")
        processed_content = content.get("processed_content", "")
    else:
        title = getattr(content, "title", "")
        snippet = getattr(content, "snippet", "")
        processed_content = getattr(content, "processed_content", "")
    
    return title, snippet, processed_content


def calculate_relevance_score(term: str, web_content: Dict[str, Any]) -> float:
    """
    Calculate relevance of web content to a term.
    
    Returns relevance score between 0 and 1.
    """
    # Extract content fields
    title, snippet, processed_content = extract_web_content_text(web_content)
    
    # Clean and normalize term
    clean_term = clean_text(term, remove_common_words=False)
    term_tokens = set(clean_term.split())
    
    # Calculate component scores
    title_score = _calculate_text_relevance(clean_term, term_tokens, title)
    snippet_score = _calculate_text_relevance(clean_term, term_tokens, snippet)
    content_score = _calculate_content_relevance(term_tokens, processed_content)
    
    # Weighted combination
    final_score = (0.5 * title_score) + (0.3 * snippet_score) + (0.2 * content_score)
    
    return min(1.0, max(0.0, final_score))


def _calculate_text_relevance(clean_term: str, term_tokens: Set[str], text: str) -> float:
    """Calculate relevance based on text field."""
    if not text:
        return 0.0
    
    clean_text_str = clean_text(text, remove_common_words=False)
    
    # Exact match
    if clean_term in clean_text_str:
        return 1.0
    
    # Fuzzy match
    token_ratio = get_fuzzy_score(clean_term, clean_text_str) / 100.0
    
    # Word overlap
    text_tokens = set(clean_text_str.split())
    word_overlap = len(term_tokens.intersection(text_tokens)) / max(1, len(term_tokens))
    
    return max(token_ratio, word_overlap)


def _calculate_content_relevance(term_tokens: Set[str], content: str) -> float:
    """Calculate relevance based on processed content."""
    if not content:
        return 0.0
    
    # Clean and tokenize content
    clean_content = clean_text(content, remove_common_words=False)
    content_tokens = clean_content.split()
    
    # Count term token occurrences
    content_counter = Counter(content_tokens)
    term_occurrences = sum(content_counter.get(token, 0) for token in term_tokens)
    
    if not term_tokens:
        return 0.0
    
    # Calculate term frequency with diminishing returns
    content_length_factor = min(1.0, 100 / max(1, len(content_tokens)))
    term_frequency = term_occurrences / max(1, len(term_tokens))
    
    return min(1.0, term_frequency * (0.5 + 0.5 * content_length_factor))