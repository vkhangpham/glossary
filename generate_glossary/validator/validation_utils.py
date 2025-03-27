"""
Validation utilities.

This module provides pure utility functions for text processing and comparison
specific to the validation process. All functions are stateless and follow
functional programming principles.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import re
import sys
from unicodedata import normalize
from rapidfuzz import fuzz
from collections import Counter

# Try to initialize WordNet lemmatizer with fallback
try:
    from nltk.stem import WordNetLemmatizer
    # Initialize lemmatizer
    wnl = WordNetLemmatizer()
    # Test if it works properly
    wnl.lemmatize("test")
    WORDNET_AVAILABLE = True
except (ImportError, LookupError, AttributeError) as e:
    print(f"Warning: WordNet lemmatizer not available: {e}. Using simple lemmatization fallback.", file=sys.stderr)
    WORDNET_AVAILABLE = False
    wnl = None

# Common words to remove in multi-word terms
COMMON_WORDS = {
    "of", "and", "or", "in", "the", "a", "an", "for", 
    "to", "by", "with", "on", "at", "&", "vs", "versus"
}

def clean_text(text: str, remove_common_words: bool = True) -> str:
    """
    Clean and normalize text for comparison.
    
    Args:
        text (str): Text to clean
        remove_common_words (bool): Whether to remove common words
        
    Returns:
        str: Cleaned text
    """
    # Normalize unicode characters
    text = normalize("NFKC", text.lower())
    
    # Remove punctuation and extra whitespace
    text = re.sub(r"[^\w\s-]", " ", text)
    words = text.split()
    
    # Remove common words for multi-word terms
    if remove_common_words and len(words) > 1:
        words = [w for w in words if w not in COMMON_WORDS]
    
    return " ".join(words).strip()

def get_token_set_ratio(a: str, b: str) -> float:
    """
    Get fuzzy token set ratio between two strings.
    
    Args:
        a (str): First string
        b (str): Second string
        
    Returns:
        float: Token set ratio score (0-100)
    """
    return fuzz.token_set_ratio(a, b)

def simple_lemmatize(word: str) -> str:
    """
    Simple lemmatization function that handles common plural forms.
    
    Args:
        word (str): Word to lemmatize
        
    Returns:
        str: Lemmatized word
    """
    if not word:
        return word
        
    # Handle common plural endings
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    elif word.endswith('es') and len(word) > 2:
        return word[:-2]
    elif word.endswith('s') and len(word) > 1 and not word.endswith('ss'):
        return word[:-1]
    
    return word

def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using WordNet lemmatizer with fallback.
    
    Args:
        text (str): Text to lemmatize
        
    Returns:
        str: Lemmatized text
    """
    words = text.split()
    
    if WORDNET_AVAILABLE:
        try:
            return " ".join(wnl.lemmatize(w) for w in words)
        except Exception as e:
            print(f"Warning: WordNet lemmatization failed: {e}. Using fallback.", file=sys.stderr)
            # Fall through to fallback
    
    # Fallback: simple lemmatization
    return " ".join(simple_lemmatize(w) for w in words)

def format_validation_result(
    term: str,
    is_valid: bool,
    mode: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format validation result consistently.
    
    Args:
        term (str): Term being validated
        is_valid (bool): Whether term is valid
        mode (str): Validation mode used
        details (Dict[str, Any]): Additional details
        
    Returns:
        Dict[str, Any]: Formatted validation result
    """
    return {
        "term": term,
        "is_valid": is_valid,
        "mode": mode,
        "details": details or {}
    }

def extract_web_content_text(web_content: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract key text fields from web content.
    
    Args:
        web_content (Dict[str, Any]): Web content object
        
    Returns:
        Tuple[str, str, str]: Title, snippet, and processed content
    """
    # Handle both dict and object formats
    if isinstance(web_content, dict):
        title = web_content.get("title", "")
        snippet = web_content.get("snippet", "")
        processed_content = web_content.get("processed_content", "")
    else:
        title = getattr(web_content, "title", "")
        snippet = getattr(web_content, "snippet", "")
        processed_content = getattr(web_content, "processed_content", "")
    
    return title, snippet, processed_content

def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenization function that splits text on whitespace.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        List[str]: List of tokens
    """
    # First clean the text to remove punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter out empty strings
    return [token for token in cleaned.split() if token]

def calculate_relevance_score(term: str, web_content: Dict[str, Any]) -> float:
    """
    Calculate the relevance of web content to a term.
    
    This function uses multiple strategies to assess relevance:
    1. Term presence in title (highest weight)
    2. Term presence in snippet (medium weight)
    3. Term frequency in processed content (lower weight)
    4. Fuzzy matching for term variations
    
    Args:
        term (str): The term to check relevance for
        web_content (Dict[str, Any]): Web content object
        
    Returns:
        float: Relevance score between 0 and 1
    """
    # Extract content fields
    title, snippet, processed_content = extract_web_content_text(web_content)
    
    # Clean and normalize term
    clean_term = clean_text(term, remove_common_words=False)
    lemmatized_term = lemmatize_text(clean_term)
    term_tokens = set(simple_tokenize(lemmatized_term))
    
    # Calculate component scores
    title_score = _calculate_title_relevance(clean_term, lemmatized_term, title)
    snippet_score = _calculate_snippet_relevance(clean_term, lemmatized_term, snippet)
    content_score = _calculate_content_relevance(term_tokens, processed_content)
    
    # Weighted combination of scores
    # Title has highest weight, followed by snippet, then content
    final_score = (0.5 * title_score) + (0.3 * snippet_score) + (0.2 * content_score)
    
    return min(1.0, max(0.0, final_score))  # Ensure score is between 0 and 1

def _calculate_title_relevance(clean_term: str, lemmatized_term: str, title: str) -> float:
    """Calculate relevance based on title."""
    if not title:
        return 0.0
    
    # Clean and normalize title
    clean_title = clean_text(title, remove_common_words=False)
    lemmatized_title = lemmatize_text(clean_title)
    
    # Check for exact match (highest score)
    if clean_term in clean_title or lemmatized_term in lemmatized_title:
        return 1.0
    
    # Check for fuzzy match
    token_ratio = get_token_set_ratio(lemmatized_term, lemmatized_title) / 100.0
    
    # Check if term words are in title
    term_words = set(lemmatized_term.split())
    title_words = set(lemmatized_title.split())
    word_overlap = len(term_words.intersection(title_words)) / max(1, len(term_words))
    
    return max(token_ratio, word_overlap)

def _calculate_snippet_relevance(clean_term: str, lemmatized_term: str, snippet: str) -> float:
    """Calculate relevance based on snippet."""
    if not snippet:
        return 0.0
    
    # Clean and normalize snippet
    clean_snippet = clean_text(snippet, remove_common_words=False)
    lemmatized_snippet = lemmatize_text(clean_snippet)
    
    # Check for exact match
    if clean_term in clean_snippet or lemmatized_term in lemmatized_snippet:
        return 1.0
    
    # Check for fuzzy match
    token_ratio = get_token_set_ratio(lemmatized_term, lemmatized_snippet) / 100.0
    
    # Check if term words are in snippet
    term_words = set(lemmatized_term.split())
    snippet_words = set(lemmatized_snippet.split())
    word_overlap = len(term_words.intersection(snippet_words)) / max(1, len(term_words))
    
    return max(token_ratio, word_overlap)

def _calculate_content_relevance(term_tokens: set, content: str) -> float:
    """Calculate relevance based on processed content."""
    if not content:
        return 0.0
    
    # Clean and normalize content
    clean_content = clean_text(content, remove_common_words=False)
    lemmatized_content = lemmatize_text(clean_content)
    content_tokens = simple_tokenize(lemmatized_content)
    
    # Count term token occurrences in content
    content_counter = Counter(content_tokens)
    term_occurrences = sum(content_counter.get(token, 0) for token in term_tokens)
    
    # Calculate term frequency
    if not term_tokens:
        return 0.0
    
    # Normalize by content length with diminishing returns for very long content
    content_length_factor = min(1.0, 100 / max(1, len(content_tokens)))
    term_frequency = term_occurrences / max(1, len(term_tokens))
    
    # Combine frequency with content length factor
    return min(1.0, term_frequency * (0.5 + 0.5 * content_length_factor)) 