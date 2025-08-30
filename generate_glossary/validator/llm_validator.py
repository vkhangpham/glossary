"""
LLM-based validation for terms.

This module uses Large Language Models to validate terms
based on semantic understanding and domain knowledge.
"""

from typing import List, Dict, Any, Optional
import logging
import time
from functools import lru_cache
from tqdm import tqdm

from .llm_utils import (
    exponential_backoff_retry,
    track_llm_cost,
    get_cost_summary,
    RateLimiter
)

# Default prompt template
DEFAULT_VALIDATION_PROMPT = """
Is '{term}' a valid academic discipline, field of study, or technical concept?

Please answer with 'yes' or 'no' followed by a brief explanation.
Consider:
1. Is it a recognized academic or technical term?
2. Is it specific enough to be meaningful?
3. Is it used in academic or professional contexts?

Format: yes/no - explanation
"""

BATCH_VALIDATION_PROMPT = """
For each of the following terms, determine if it is a valid academic discipline, field of study, or technical concept.

Terms:
{terms_list}

For each term, provide:
- Valid: yes/no
- Confidence: 0.0-1.0
- Reason: brief explanation

Return as JSON array with format:
[{{"term": "...", "valid": true/false, "confidence": 0.X, "reason": "..."}}]
"""


def validate_with_llm(
    terms: List[str],
    provider: str = "gemini",
    batch_size: int = 10,
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Validate terms using LLM.
    
    Args:
        terms: List of terms to validate
        provider: LLM provider to use (openai, gemini)
        batch_size: Number of terms to validate in one LLM call
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping terms to validation results
    """
    try:
        from generate_glossary.utils.llm_simple import infer_text
    except ImportError:
        logging.error("LLM utilities not available")
        return {term: _create_error_result(term, "LLM not available") for term in terms}
    
    results = {}
    
    # Process in batches for efficiency
    if batch_size > 1 and len(terms) > batch_size:
        results = _validate_batch(terms, provider, batch_size, show_progress)
    else:
        # Single term validation
        iterator = tqdm(terms, desc="LLM validation") if show_progress else terms
        for term in iterator:
            results[term] = _validate_single_term_llm(term, provider)
    
    # Log cost summary
    from .llm_utils import get_cost_summary
    cost_summary = get_cost_summary()
    if cost_summary["total_calls"] > 0:
        logging.info(
            f"LLM validation costs: ${cost_summary['total_cost_usd']:.4f} "
            f"for {cost_summary['total_calls']} calls "
            f"(avg: ${cost_summary['avg_cost_per_call']:.4f}/call)"
        )
    
    return results


# Global rate limiter
_rate_limiter = RateLimiter(calls_per_minute=60)


@exponential_backoff_retry(max_retries=3, base_delay=1.0)
def _validate_single_term_llm(term: str, provider: str) -> Dict[str, Any]:
    """
    Validate a single term using LLM with retry and cost tracking.
    
    Args:
        term: Term to validate
        provider: LLM provider
        
    Returns:
        Validation result dictionary
    """
    try:
        from generate_glossary.utils.llm_simple import infer_text
        
        # Rate limiting
        _rate_limiter.wait_if_needed()
        
        # Generate prompt
        prompt = DEFAULT_VALIDATION_PROMPT.format(term=term)
        
        # Get LLM response
        response = infer_text(
            provider=provider,
            prompt=prompt,
            max_tokens=100
        )
        
        # Parse response
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Track cost
        model = getattr(response, 'model', 'unknown')
        cost = track_llm_cost(provider, model, prompt, response_text)
        
        # Parse response
        is_valid, confidence, reason = _parse_llm_response(response_text)
        
        return {
            "term": term,
            "is_valid": is_valid,
            "confidence": confidence,
            "mode": "llm",
            "details": {
                "provider": provider,
                "response": response_text,
                "reason": reason,
                "cost_usd": round(cost, 6)
            }
        }
        
    except Exception as e:
        logging.error(f"LLM validation failed for '{term}': {e}")
        return _create_error_result(term, str(e))


def _validate_batch(
    terms: List[str],
    provider: str,
    batch_size: int,
    show_progress: bool
) -> Dict[str, Dict[str, Any]]:
    """
    Validate terms in batches using LLM.
    
    Args:
        terms: List of terms to validate
        provider: LLM provider
        batch_size: Batch size
        show_progress: Whether to show progress
        
    Returns:
        Dictionary of validation results
    """
    try:
        from generate_glossary.utils.llm_simple import infer_text
        import json
    except ImportError:
        return {term: _create_error_result(term, "Dependencies not available") for term in terms}
    
    results = {}
    batches = [terms[i:i+batch_size] for i in range(0, len(terms), batch_size)]
    
    iterator = tqdm(batches, desc="LLM batch validation") if show_progress else batches
    
    for batch in iterator:
        try:
            # Create batch prompt
            terms_list = "\n".join(f"- {term}" for term in batch)
            prompt = BATCH_VALIDATION_PROMPT.format(terms_list=terms_list)
            
            # Get LLM response
            response = infer_text(
                provider=provider,
                prompt=prompt,
                max_tokens=500
            )
            
            # Parse batch response
            response_text = response.text if hasattr(response, 'text') else str(response)
            batch_results = _parse_batch_response(response_text, batch)
            
            # Add to results
            for term in batch:
                if term in batch_results:
                    results[term] = batch_results[term]
                else:
                    results[term] = _create_error_result(term, "Not in batch response")
                    
        except Exception as e:
            logging.error(f"Batch validation failed: {e}")
            # Fallback to individual validation for this batch
            for term in batch:
                results[term] = _validate_single_term_llm(term, provider)
    
    return results


def _parse_llm_response(response: str) -> tuple:
    """
    Parse LLM validation response.
    
    Args:
        response: LLM response text
        
    Returns:
        Tuple of (is_valid, confidence, reason)
    """
    response_lower = response.lower().strip()
    
    # Check for yes/no at the start
    is_valid = response_lower.startswith("yes")
    
    # Try to extract confidence (look for patterns like "0.8", "80%", "high confidence")
    confidence = 0.5  # Default confidence
    
    if "high confidence" in response_lower or "very confident" in response_lower:
        confidence = 0.9
    elif "medium confidence" in response_lower or "moderate confidence" in response_lower:
        confidence = 0.6
    elif "low confidence" in response_lower or "uncertain" in response_lower:
        confidence = 0.3
    
    # If valid, boost confidence slightly
    if is_valid:
        confidence = min(1.0, confidence + 0.1)
    
    # Extract reason (everything after yes/no)
    parts = response.split('-', 1)
    reason = parts[1].strip() if len(parts) > 1 else response
    
    return is_valid, confidence, reason


def _parse_batch_response(response: str, terms: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse batch LLM response.
    
    Args:
        response: LLM response text
        terms: List of terms in the batch
        
    Returns:
        Dictionary mapping terms to results
    """
    import json
    
    results = {}
    
    try:
        # Try to parse as JSON
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            for item in parsed:
                term = item.get("term", "")
                if term in terms:
                    results[term] = {
                        "term": term,
                        "is_valid": item.get("valid", False),
                        "confidence": item.get("confidence", 0.5),
                        "mode": "llm",
                        "details": {
                            "reason": item.get("reason", ""),
                            "batch_validated": True
                        }
                    }
    except:
        # Fallback: try to parse line by line
        for term in terms:
            if term in response:
                # Simple heuristic: if term appears with "valid" or "yes" nearby
                is_valid = "valid" in response[max(0, response.index(term)-50):response.index(term)+50]
                results[term] = {
                    "term": term,
                    "is_valid": is_valid,
                    "confidence": 0.5,
                    "mode": "llm",
                    "details": {
                        "reason": "Parsed from batch response",
                        "batch_validated": True
                    }
                }
    
    return results


def _create_error_result(term: str, error: str) -> Dict[str, Any]:
    """
    Create an error result for failed validation.
    
    Args:
        term: Term that failed
        error: Error message
        
    Returns:
        Error result dictionary
    """
    return {
        "term": term,
        "is_valid": False,
        "confidence": 0.0,
        "mode": "llm",
        "details": {
            "error": error,
            "response": None
        }
    }