"""
LLM-based validation for terms.

This module uses Large Language Models to validate terms
based on semantic understanding and domain knowledge.
"""

from typing import List, Dict, Any, Optional, Callable, Union
import logging
from tqdm import tqdm

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


def _coerce_llm_response_text(response: Any) -> str:
    """
    Coerce LLM response from different providers to text.
    
    Handles common response shapes from OpenAI, Vertex AI, Claude, etc.
    
    Args:
        response: LLM response object
        
    Returns:
        Response text as string
    """
    # Direct string response
    if isinstance(response, str):
        return response
    
    # OpenAI-style response
    if hasattr(response, 'choices') and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            return choice.message.content
        elif hasattr(choice, 'text'):
            return choice.text
    
    # Vertex AI/Google-style response
    if hasattr(response, 'content') and isinstance(response.content, str):
        return response.content
    
    # Dictionary response
    if isinstance(response, dict):
        # Try common keys
        for key in ['content', 'text', 'message', 'response']:
            if key in response and isinstance(response[key], str):
                return response[key]
        # Try nested structure
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if isinstance(choice, dict):
                for key in ['message', 'text']:
                    if key in choice:
                        if isinstance(choice[key], dict) and 'content' in choice[key]:
                            return choice[key]['content']
                        elif isinstance(choice[key], str):
                            return choice[key]
    
    # Fallback to string conversion
    return str(response)


def llm_validate(
    terms: List[str],
    llm_fn: Callable,
    provider: str = "gemini",
    batch_size: int = 10,
    max_workers: int = 4,
    validation_prompt: str = DEFAULT_VALIDATION_PROMPT,
    batch_prompt: str = BATCH_VALIDATION_PROMPT,
    tier: str = "budget",
    max_tokens: int = 100,
    batch_max_tokens: int = 500,
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Pure functional LLM-based validation.
    
    Expected llm_fn signature: llm_fn(messages, tier: str, max_tokens: int) -> Union[str, dict, object]
    
    Args:
        terms: List of terms to validate
        llm_fn: LLM completion function to use
        provider: LLM provider name
        batch_size: Number of terms to validate in one LLM call
        max_workers: Maximum number of worker threads
        validation_prompt: Prompt template for single term validation
        batch_prompt: Prompt template for batch validation
        tier: LLM tier (budget, standard, premium)
        max_tokens: Maximum tokens for single term validation
        batch_max_tokens: Maximum tokens for batch validation
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping terms to validation results
    """
    results = {}
    
    # Process in batches for efficiency
    if batch_size > 1 and len(terms) >= batch_size:
        results = _validate_batch_pure(
            terms, llm_fn, provider, batch_size, batch_prompt, tier, batch_max_tokens, show_progress
        )
    else:
        # Single term validation
        iterator = tqdm(terms, desc="LLM validation") if show_progress else terms
        for term in iterator:
            results[term] = _validate_single_term_llm_pure(
                term, llm_fn, provider, validation_prompt, tier, max_tokens
            )
    
    return results


def validate_with_llm(
    terms: List[str],
    provider: str = "gemini",
    batch_size: int = 10,
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Validate terms using LLM.
    
    Legacy wrapper that maintains backward compatibility.
    
    Args:
        terms: List of terms to validate
        provider: LLM provider to use (openai, gemini)
        batch_size: Number of terms to validate in one LLM call
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping terms to validation results
    """
    try:
        from generate_glossary.llm import completion
    except ImportError:
        logging.error("LLM utilities not available")
        return {term: _create_error_result(term, "LLM not available") for term in terms}
    
    return llm_validate(
        terms, 
        completion, 
        provider=provider, 
        batch_size=batch_size, 
        show_progress=show_progress
    )


def _validate_single_term_llm_pure(
    term: str,
    llm_fn: Callable,
    provider: str,
    validation_prompt: str,
    tier: str,
    max_tokens: int
) -> Dict[str, Any]:
    """
    Pure functional validation of a single term using LLM.

    Args:
        term: Term to validate
        llm_fn: LLM completion function
        provider: LLM provider name
        validation_prompt: Prompt template for validation
        tier: LLM tier
        max_tokens: Maximum tokens for response

    Returns:
        Validation result dictionary
    """
    try:
        # Normalize and validate term input
        if term is None:
            return _create_error_result("", "Input term is None")

        # Convert to string and normalize (strip whitespace)
        normalized_term = str(term).strip()

        # Check for empty/whitespace-only terms
        if not normalized_term:
            return _create_error_result(normalized_term, "Input term is empty or whitespace-only")

        # Generate prompt
        prompt = validation_prompt.format(term=normalized_term)

        # Get LLM response
        messages = [{"role": "user", "content": prompt}]
        response = llm_fn(messages, tier=tier, max_tokens=max_tokens)

        # Parse response
        response_text = _coerce_llm_response_text(response)

        # Parse response
        is_valid, confidence, reason = _parse_llm_response(response_text)

        return {
            "term": normalized_term,
            "is_valid": is_valid,
            "confidence": confidence,
            "mode": "llm",
            "details": {
                "provider": provider,
                "response": response_text,
                "reason": reason
            }
        }
        
    except Exception as e:
        # Use the original term for logging context
        logging.error(f"LLM validation failed for '{term}': {e}")
        return _create_error_result(term, str(e))


def _validate_batch_pure(
    terms: List[str],
    llm_fn: Callable,
    provider: str,
    batch_size: int,
    batch_prompt: str,
    tier: str,
    batch_max_tokens: int,
    show_progress: bool
) -> Dict[str, Dict[str, Any]]:
    """
    Pure functional batch validation of terms using LLM.
    
    Args:
        terms: List of terms to validate
        llm_fn: LLM completion function
        provider: LLM provider name
        batch_size: Size of each batch
        batch_prompt: Prompt template for batch validation
        tier: LLM tier
        batch_max_tokens: Maximum tokens for batch response
        show_progress: Whether to show progress
        
    Returns:
        Dictionary of validation results
    """
    try:
        import json
    except ImportError:
        return {term: _create_error_result(term, "Dependencies not available") for term in terms}

    # Normalize and validate input terms
    normalized_terms = []
    results = {}

    for term in terms:
        # Validate and normalize each term
        if term is None:
            results[term] = _create_error_result("", "Input term is None")
            continue

        # Convert to string and normalize (strip whitespace)
        normalized_term = str(term).strip()

        # Check for empty/whitespace-only terms
        if not normalized_term:
            results[term] = _create_error_result(normalized_term, "Input term is empty or whitespace-only")
            continue

        normalized_terms.append((term, normalized_term))  # Keep original for results mapping

    # Only proceed with valid normalized terms
    if not normalized_terms:
        return results  # All terms were invalid

    batches = [normalized_terms[i:i+batch_size] for i in range(0, len(normalized_terms), batch_size)]
    
    iterator = tqdm(batches, desc="LLM batch validation") if show_progress else batches
    
    for batch in iterator:
        try:
            # Extract normalized terms for the prompt
            normalized_batch_terms = [normalized_term for _, normalized_term in batch]
            original_to_normalized = {original_term: normalized_term for original_term, normalized_term in batch}

            # Create batch prompt using normalized terms
            terms_list = "\n".join(f"- {normalized_term}" for normalized_term in normalized_batch_terms)
            prompt = batch_prompt.format(terms_list=terms_list)

            # Get LLM response
            messages = [{"role": "user", "content": prompt}]
            response = llm_fn(messages, tier=tier, max_tokens=batch_max_tokens)

            # Parse batch response
            response_text = _coerce_llm_response_text(response)
            batch_results = _parse_batch_response(response_text, normalized_batch_terms)

            # Map results back to original terms
            for original_term, normalized_term in batch:
                if normalized_term in batch_results:
                    # Update result to use original term key but normalized term in data
                    result = batch_results[normalized_term]
                    result["term"] = normalized_term  # Keep normalized term in result data
                    results[original_term] = result
                else:
                    results[original_term] = _create_error_result(original_term, "Not in batch response")

        except Exception as e:
            logging.error(f"Batch validation failed: {e}")
            # Fallback to individual validation for this batch
            for original_term, normalized_term in batch:
                results[original_term] = _validate_single_term_llm_pure(
                    original_term, llm_fn, provider, DEFAULT_VALIDATION_PROMPT, tier, 100
                )
    
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


def _validate_single_term_llm(term: str, provider: str) -> Dict[str, Any]:
    """
    Legacy wrapper for backward compatibility.
    
    Args:
        term: Term to validate
        provider: LLM provider
        
    Returns:
        Validation result dictionary
    """
    try:
        from generate_glossary.llm import completion
        return _validate_single_term_llm_pure(
            term, completion, provider, DEFAULT_VALIDATION_PROMPT, "budget", 100
        )
    except ImportError:
        logging.error("LLM utilities not available")
        return _create_error_result(term, "LLM not available")


def _validate_batch(
    terms: List[str],
    provider: str,
    batch_size: int,
    show_progress: bool
) -> Dict[str, Dict[str, Any]]:
    """
    Legacy wrapper for backward compatibility.
    
    Args:
        terms: List of terms to validate
        provider: LLM provider
        batch_size: Batch size
        show_progress: Whether to show progress
        
    Returns:
        Dictionary of validation results
    """
    try:
        from generate_glossary.llm import completion
        return _validate_batch_pure(
            terms, completion, provider, batch_size, BATCH_VALIDATION_PROMPT, "budget", 500, show_progress
        )
    except ImportError:
        return {term: _create_error_result(term, "Dependencies not available") for term in terms}