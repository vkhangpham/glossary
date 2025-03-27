"""
Public API for concept validation.

This module provides a simple interface for validating technical concepts
using different validation modes.
"""

from typing import List, Dict, Any, Optional, Union
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .validation_modes import (
    validate_rules,
    validate_wiki,
    validate_llm,
    validate_web,
    ValidationResult,
    WebContent,
    DEFAULT_MIN_SCORE,
    DEFAULT_MIN_RELEVANCE_SCORE
)

# Type aliases
Terms = List[str]
ValidationResults = Dict[str, ValidationResult]
WikiData = Dict[str, List[Dict[str, Any]]]
WebContents = Dict[str, List[Union[Dict[str, Any], WebContent]]]

def validate(
    terms: Union[str, Terms],
    mode: str = "rules",
    wiki_data: Optional[WikiData] = None,
    llm_responses: Optional[Dict[str, str]] = None,
    web_content: Optional[WebContents] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    show_progress: bool = True
) -> Union[ValidationResult, ValidationResults]:
    """
    Validate technical concepts using specified mode.
    
    Args:
        terms (Union[str, Terms]): Term or list of terms to validate
        mode (str): Validation mode ('rules', 'wiki', 'llm', or 'web')
               Note: The CLI uses 'rule' (singular) for consistency with deduplicator,
               but the internal API uses 'rules' (plural). The CLI maps between these.
        wiki_data (WikiData): Wikipedia data for wiki validation mode
        llm_responses (Dict[str, str]): LLM responses for llm validation mode
        web_content (WebContents): Web contents for web validation mode
                                  Can be in either old format or new WebContent format
        min_score (float): Minimum score for web content validation
        min_relevance_score (float): Minimum relevance score for web content to be considered relevant
        show_progress (bool): Whether to show progress bar
        
    Returns:
        Union[ValidationResult, ValidationResults]: Validation results
    """
    # Handle single term
    if isinstance(terms, str):
        return _validate_single(terms, mode, wiki_data, llm_responses, web_content, min_score, min_relevance_score)
    
    # Handle multiple terms
    return _validate_batch(terms, mode, wiki_data, llm_responses, web_content, min_score, min_relevance_score, show_progress)

def _validate_single(
    term: str,
    mode: str,
    wiki_data: Optional[WikiData] = None,
    llm_responses: Optional[Dict[str, str]] = None,
    web_content: Optional[WebContents] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE
) -> ValidationResult:
    """Validate a single term."""
    if mode == "rules":
        return validate_rules(term)
    elif mode == "wiki":
        return validate_wiki(term, wiki_data)
    elif mode == "llm":
        llm_response = llm_responses.get(term) if llm_responses else None
        return validate_llm(term, llm_response)
    elif mode == "web":
        contents = web_content.get(term, []) if web_content else []
        return validate_web(term, contents, min_score, min_relevance_score)
    else:
        raise ValueError(f"Invalid validation mode: {mode}")

def _validate_batch(
    terms: Terms,
    mode: str,
    wiki_data: Optional[WikiData] = None,
    llm_responses: Optional[Dict[str, str]] = None,
    web_content: Optional[WebContents] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    show_progress: bool = True
) -> ValidationResults:
    """Validate multiple terms in parallel."""
    # Create validator function
    validator = partial(
        _validate_single,
        mode=mode,
        wiki_data=wiki_data,
        llm_responses=llm_responses,
        web_content=web_content,
        min_score=min_score,
        min_relevance_score=min_relevance_score
    )
    
    # Process terms in parallel
    with ThreadPoolExecutor() as executor:
        if show_progress:
            results = list(tqdm(
                executor.map(validator, terms),
                total=len(terms),
                desc=f"Validating terms ({mode})"
            ))
        else:
            results = list(executor.map(validator, terms))
    
    # Return results as dictionary
    return {r["term"]: r for r in results}

# Version
__version__ = "1.0.0"

# Public API
__all__ = ["validate"] 