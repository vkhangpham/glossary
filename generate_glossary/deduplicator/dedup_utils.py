"""
Common utilities for deduplication.

This module provides pure utility functions for text processing and term comparison.
All functions are stateless and follow functional programming principles.
"""

from typing import List, Set, Dict, Union, Tuple, Optional, Any, Callable, TypeVar
from collections import defaultdict
import re
from unicodedata import normalize
from functools import wraps
import time
import os
import logging
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import sys

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.warning("spaCy model 'en_core_web_sm' not found. Using basic NLP.")
        nlp = spacy.blank("en")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not installed. Advanced lemmatization will be limited.")

from concurrent.futures import ProcessPoolExecutor, as_completed

# Skip NLTK downloads entirely and use fallback mechanisms
NLTK_RESOURCES_AVAILABLE = False
try:
    # Try to import wordnet to check if it's available
    from nltk.corpus import wordnet
    # Test if wordnet is actually available
    try:
        test_synsets = wordnet.synsets('test')
        if test_synsets:
            NLTK_RESOURCES_AVAILABLE = True
            logging.info("NLTK WordNet resources are available.")
        else:
            logging.warning("WordNet appears to be installed but not functioning properly.")
    except Exception as e:
        logging.warning(f"Error accessing WordNet: {e}")
except (ImportError, LookupError) as e:
    logging.warning(f"WordNet not available: {e}. Using simple lemmatization fallback.")

# Initialize lemmatizer with fallback
lemmatizer = None
try:
    if NLTK_RESOURCES_AVAILABLE:
        lemmatizer = WordNetLemmatizer()
        # Test the lemmatizer
        lemmatizer.lemmatize("test")
        logging.info("WordNet lemmatizer initialized successfully.")
except Exception as e:
    logging.warning(f"WordNet lemmatizer not available: {e}. Using simple lemmatization fallback.")
    lemmatizer = None

# Academic domain-specific constants
ACADEMIC_LEMMA_EXCEPTIONS = {
    "data": "data",  # Prevent "datum" lemmatization
    "media": "media",  # Prevent "medium" lemmatization
    "criteria": "criteria",  # Prevent "criterion" lemmatization
    "analyses": "analysis",  # Correct plural form
    "hypotheses": "hypothesis",  # Correct plural form
    "phenomena": "phenomenon",  # Correct plural form
    "curricula": "curriculum",  # Correct plural form
    "syllabi": "syllabus",  # Correct plural form
    "theses": "thesis",  # Correct plural form
    "indices": "index",  # Correct plural form
    "matrices": "matrix",  # Correct plural form
}

# British/American spelling variations - these are standardized
SPELLING_VARIATIONS = {
    "behaviour": "behavior",
    "analyse": "analyze",
    "modelling": "modeling",
    "defence": "defense",
    "centre": "center",
    "programme": "program",
    "catalogue": "catalog",
    "colour": "color",
    "organisation": "organization",
    "specialisation": "specialization",
    "labour": "labor"
}

UNWANTED_WORDS = [
    "research", "technology",
    "theory", "method", "technique", "approach", "application"
]

T = TypeVar('T')
R = TypeVar('R')

def timing_decorator(func):
    """Decorator to measure and log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        
        if isinstance(result, dict):
            result["execution_time"] = execution_time
            
        return result
    return wrapper

def process_in_parallel(
    items: List[T],
    process_func: Callable[[List[T]], R],
    batch_size: int = 20,
    max_workers: Optional[int] = None,
    desc: str = "Processing",
    error_handling: str = "continue"
) -> List[R]:
    """Process items in parallel batches with error handling."""
    start_time = time.time()
    
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]
    
    results = []
    errors = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_func, batch): i 
            for i, batch in enumerate(batches)
        }
        
        for future in tqdm(
            as_completed(future_to_batch),
            total=len(batches),
            desc=desc
        ):
            batch_index = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                error_msg = f"Error processing batch {batch_index}: {str(e)}"
                errors.append(error_msg)
                
                if error_handling == "raise":
                    raise RuntimeError(f"Batch processing failed: {error_msg}")
                elif error_handling == "log":
                    logging.error(error_msg)
                continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info(f"Processed {len(items)} items in {len(batches)} batches")
    logging.info(f"Total processing time: {total_time:.4f} seconds")
    logging.info(f"Success rate: {len(results)}/{len(batches)} batches ({len(results)/len(batches)*100:.2f}%)")
    
    if errors:
        logging.warning(f"Encountered {len(errors)} errors during processing")
    
    return results

def get_wordnet_pos(word: str) -> str:
    """Map POS tag to first character used by WordNet."""
    if not NLTK_RESOURCES_AVAILABLE:
        return 'n'  # Default to noun if NLTK is not available
        
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def simple_lemmatize(word: str) -> str:
    """Simple lemmatization function that handles common plural forms."""
    if not word:
        return word
        
    # Check academic exceptions first
    if word.lower() in ACADEMIC_LEMMA_EXCEPTIONS:
        return ACADEMIC_LEMMA_EXCEPTIONS[word.lower()]
        
    # Handle common plural endings
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    elif word.endswith('es') and len(word) > 2:
        return word[:-2]
    elif word.endswith('s') and len(word) > 1 and not word.endswith('ss'):
        return word[:-1]
    
    return word

def get_lemma_with_pos(word: str) -> str:
    """Get lemma with part-of-speech tagging."""
    if not word:
        return word
        
    # Check academic exceptions first
    if word.lower() in ACADEMIC_LEMMA_EXCEPTIONS:
        return ACADEMIC_LEMMA_EXCEPTIONS[word.lower()]
    
    # Use WordNet lemmatizer if available
    if NLTK_RESOURCES_AVAILABLE and lemmatizer:
        try:
            pos = get_wordnet_pos(word)
            return lemmatizer.lemmatize(word.lower(), pos)
        except Exception as e:
            logging.debug(f"Error using WordNet lemmatizer: {e}. Falling back to simple lemmatization.")
            return simple_lemmatize(word.lower())
    else:
        # Fallback to simple lemmatization
        return simple_lemmatize(word.lower())

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Normalize unicode characters
    text = normalize("NFKC", text.lower())
    
    # Replace special characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Lemmatize each word
    words = text.split()
    lemmatized = [get_lemma_with_pos(word) for word in words]
    
    return " ".join(lemmatized).strip()

def get_plural_variations(term: str) -> Set[str]:
    """Get plural/singular variations of a term."""
    variations = {term}
    
    # Split term into words
    words = term.split()
    if not words:
        return variations
    
    # For single-word terms
    if len(words) == 1:
        word = words[0]
        
        # Check academic exceptions first
        if word.lower() in ACADEMIC_LEMMA_EXCEPTIONS:
            singular = ACADEMIC_LEMMA_EXCEPTIONS[word.lower()]
            variations.add(singular)
            return variations
        
        # Get singular form
        if NLTK_RESOURCES_AVAILABLE and lemmatizer:
            try:
                singular = lemmatizer.lemmatize(word.lower(), 'n')
                if singular != word.lower():
                    variations.add(singular)
            except Exception:
                # Fallback to simple lemmatization
                singular = simple_lemmatize(word.lower())
                if singular != word.lower():
                    variations.add(singular)
        else:
            # Use simple lemmatization
            singular = simple_lemmatize(word.lower())
            if singular != word.lower():
                variations.add(singular)
        
        # Get plural form (simple heuristic)
        if word.lower() == singular:  # Word is already singular
            if word.endswith('y'):
                plural = word[:-1] + 'ies'
            elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
                plural = word + 'es'
            else:
                plural = word + 's'
            variations.add(plural)
    
    # For multi-word terms, only modify the last word
    else:
        last_word = words[-1]
        other_words = words[:-1]
        
        # Get variations of the last word
        last_word_variations = get_plural_variations(last_word)
        
        # Combine with other words
        for var in last_word_variations:
            if var != last_word:
                new_term = " ".join(other_words + [var])
                variations.add(new_term)
    
    return variations

def get_spelling_variations(term: str) -> Set[str]:
    """Get British/American spelling variations."""
    variations = {term}
    term_lower = term.lower()
    
    # Check each spelling variation
    for british, american in SPELLING_VARIATIONS.items():
        if british in term_lower:
            variations.add(term_lower.replace(british, american))
        elif american in term_lower:
            variations.add(term_lower.replace(american, british))
    
    return variations

def get_dash_space_variations(term: str) -> Set[str]:
    """Get variations with dashes replaced by spaces and vice versa."""
    variations = {term}
    
    # Replace dashes with spaces
    if '-' in term:
        variations.add(term.replace('-', ' '))
    
    # Replace spaces with dashes (only for multi-word terms)
    if ' ' in term:
        variations.add(term.replace(' ', '-'))
    
    return variations

def get_term_variations(term: str) -> Set[str]:
    """Get all possible variations of a term using reliable academic patterns.
    
    This function applies different variation patterns in a specific order:
    1. Basic text normalization
    2. Plural/singular forms using lemmatization
    3. British/American spelling variations
    4. Dash-space variations
    
    Each variation type is handled separately to maintain clarity and control.
    """
    variations = {term}
    
    # 1. Start with normalized text
    normalized = normalize_text(term)
    variations.add(normalized)
    
    # 2. Get plural/singular variations
    plural_vars = set()
    for var in variations.copy():
        plural_vars.update(get_plural_variations(var))
    variations.update(plural_vars)
    
    # 3. Get spelling variations
    spelling_vars = set()
    for var in variations.copy():
        spelling_vars.update(get_spelling_variations(var))
    variations.update(spelling_vars)
    
    # 4. Get dash-space variations
    dash_vars = set()
    for var in variations.copy():
        dash_vars.update(get_dash_space_variations(var))
    variations.update(dash_vars)
    
    return frozenset(str(v) for v in variations)

def is_compound_term(compound_term: str, term_list: List[str]) -> Dict[str, Any]:
    """
    Check if a term is a compound term and handle appropriately.
    
    Args:
        compound_term: The potential compound term to check
        term_list: List of all terms being processed
    
    Returns:
        Dict with keys:
            - is_compound: Whether this is a compound term
            - should_remove: Whether to remove this term (if all atomic terms are in term_list)
            - atomic_terms: List of atomic terms found
            - missing_terms: List of atomic terms not found in term_list
    """
    compound_norm = normalize_text(compound_term)
    
    # Check if this term has "and" or commas (indicators of compound terms)
    if ' and ' not in compound_norm and ',' not in compound_norm:
        return {
            "is_compound": False,
            "should_remove": False,
            "atomic_terms": [],
            "missing_terms": []
        }
    
    # Extract all parts from the compound term
    parts = []
    for and_part in compound_norm.split(' and '):
        parts.extend(part.strip() for part in and_part.split(','))
    
    # Clean up parts and remove empty ones
    atomic_terms = [part.strip() for part in parts if part.strip()]
    
    # Check which atomic terms appear in the term list
    normalized_term_list = [normalize_text(t) for t in term_list]
    found_terms = [term for term in atomic_terms if term in normalized_term_list]
    missing_terms = [term for term in atomic_terms if term not in normalized_term_list]
    
    # If all atomic terms are present, mark for removal
    should_remove = len(found_terms) == len(atomic_terms) and len(atomic_terms) > 0
    
    return {
        "is_compound": True,
        "should_remove": should_remove,
        "atomic_terms": atomic_terms,
        "missing_terms": missing_terms
    }

def get_standardized_path(
    path: Union[str, Path], 
    create_if_missing: bool = False,
    is_file: bool = False
) -> Path:
    """Standardize path handling across all levels."""
    path_obj = Path(path).expanduser().resolve()
    
    if create_if_missing:
        if is_file:
            os.makedirs(path_obj.parent, exist_ok=True)
        else:
            os.makedirs(path_obj, exist_ok=True)
    
    return path_obj 