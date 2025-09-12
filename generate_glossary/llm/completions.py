"""
Main completion functions, prompt optimization integration, and statistics tracking.

This module contains the core completion functions and all functionality related to
prompt optimization integration, testing, validation, and statistics tracking.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import dspy
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from generate_glossary.config import get_llm_config
from generate_glossary.utils.failure_tracker import save_failure
from generate_glossary.llm.dspy_config import _ensure_dspy_configured, configure_for_optimization, inform_successful_prompt_load
from generate_glossary.llm.helpers import (
    _resolve_model, _resolve_temperature, _get_lm, _apply_optimized_prompts, _join_messages
)
from generate_glossary.llm.signatures import (
    _parse_signature_from_prompt, _create_dynamic_signature, 
    _extract_content_for_signature, _parse_structured_output
)

# Try to import prompt optimization, but handle gracefully if not available
try:
    from prompt_optimization.core import load_prompt
    PROMPT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PROMPT_OPTIMIZATION_AVAILABLE = False
    load_prompt = None

logger = logging.getLogger(__name__)

# Module-level configuration variables
_optimized_prompt_config = {
    'enable_chain_of_thought': True,
    'enable_signature_optimization': True,
    'fallback_to_text': True
}

_prompt_usage_stats = {
    'signature_attempts': 0,
    'signature_successes': 0,
    'fallback_count': 0,
    'not_found_count': 0,
    'use_cases_tested': [],
    'common_errors': {},
    'metadata_usage_count': 0,
    'text_inference_count': 0,
    'predictor_type_distribution': {}
}


def _create_optimized_predictor(
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    use_chain_of_thought: bool = True,
    signature_metadata: Optional[Dict[str, Any]] = None
) -> Union[dspy.Predict, dspy.ChainOfThought]:
    """
    Create DSPy predictor from optimized prompts using signature-based approach.
    
    When signature_metadata is provided, it's used directly instead of text inference.
    TypedPredictor may not exist in some DSPy versions; falls back to Predict for structured output.
    """
    try:
        if signature_metadata:
            signature_class = _create_dynamic_signature(signature_metadata, response_model)
            recommended_type = signature_metadata.get('predictor_type', 'Predict')
        else:
            signature_info = _parse_signature_from_prompt(system_prompt, user_prompt, signature_metadata)
            signature_class = _create_dynamic_signature(signature_info, response_model)
            recommended_type = 'Predict'
        
        typed_predictor_cls = getattr(dspy, 'TypedPredictor', None)
        
        if recommended_type == "ChainOfThought":
            if response_model is not None:
                try:
                    if typed_predictor_cls is not None:
                        predictor = dspy.ChainOfThought(signature_class)
                        logger.debug("Created ChainOfThought predictor with structured output (CoT+typed pattern)")
                    else:
                        predictor = dspy.ChainOfThought(signature_class)
                        logger.warning("ChainOfThought predictor created but TypedPredictor not available for structured output")
                except Exception as e:
                    logger.warning(f"CoT+typed pattern failed: {e}. Falling back to regular ChainOfThought")
                    predictor = dspy.ChainOfThought(signature_class)
            else:
                predictor = dspy.ChainOfThought(signature_class)
        elif response_model is not None and typed_predictor_cls is not None:
            predictor = typed_predictor_cls(signature_class)
        elif response_model is not None:
            predictor = dspy.Predict(signature_class)
        elif use_chain_of_thought:
            predictor = dspy.ChainOfThought(signature_class)
        else:
            predictor = dspy.Predict(signature_class)
        
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to create optimized predictor: {e}")
        raise


def _apply_optimized_signature(
    messages: List[Dict[str, str]],
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    signature_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Union[dspy.Predict, Any, dspy.ChainOfThought], Dict[str, str]]:
    """
    Apply optimized prompts using DSPy signature-based approach.
    
    Returns tuple of (predictor, prepared_inputs) where predictor is configured DSPy predictor
    ready for execution and prepared_inputs maps signature field names to content.
    """
    try:
        signature_info = _parse_signature_from_prompt(system_prompt, user_prompt, signature_metadata)
        
        predictor = _create_optimized_predictor(
            system_prompt, user_prompt, response_model, use_chain_of_thought=True, signature_metadata=signature_metadata
        )
        
        prepared_inputs = _extract_content_for_signature(messages, signature_info)
        
        return predictor, prepared_inputs
        
    except Exception as e:
        logger.error(f"Failed to apply optimized signature: {e}")
        raise


def _try_load_optimized_prompt(use_case: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Attempt to load optimized prompts for a given use case.
    
    Args:
        use_case: Use case identifier (e.g., "lv0_s1")
        
    Returns:
        Tuple of (system_prompt, user_prompt_template, signature_metadata) or (None, None, None) if not found
    """
    if not PROMPT_OPTIMIZATION_AVAILABLE or not use_case:
        return None, None, None
    
    try:
        # Import load_prompt instead of get_prompt_content
        # Map use_case to prompt keys
        system_key = f"{use_case}_system"
        user_key = f"{use_case}_user"
        
        # Try to load both prompts with full metadata
        system_data = None
        user_data = None
        
        try:
            system_data = load_prompt(system_key)
            system_prompt = system_data.get("content")
        except (FileNotFoundError, KeyError):
            system_prompt = None
            
        try:
            user_data = load_prompt(user_key)
            user_prompt = user_data.get("content")
        except (FileNotFoundError, KeyError):
            user_prompt = None
        
        # Extract signature metadata from either system or user data (top-level, not in metadata)
        signature_metadata = None
        system_signature_metadata = None
        user_signature_metadata = None
        
        # Check for signature metadata in both system and user prompt data
        if system_data and "signature_metadata" in system_data:
            system_signature_metadata = system_data["signature_metadata"]
        
        if user_data and "signature_metadata" in user_data:
            user_signature_metadata = user_data["signature_metadata"]
        
        # Handle conflicts when both system and user contain signature metadata
        if system_signature_metadata and user_signature_metadata:
            if system_signature_metadata != user_signature_metadata:
                logger.warning(
                    f"Conflicting signature metadata found for {use_case}: "
                    f"system and user prompts contain different metadata. "
                    f"Using user metadata (deterministic choice). "
                    f"System metadata: {system_signature_metadata.get('signature_str', 'N/A')}, "
                    f"User metadata: {user_signature_metadata.get('signature_str', 'N/A')}"
                )
            signature_metadata = user_signature_metadata  # Deterministically pick user
        elif system_signature_metadata:
            signature_metadata = system_signature_metadata
        elif user_signature_metadata:
            signature_metadata = user_signature_metadata
        
        # Ensure we have at least one prompt
        if system_prompt or user_prompt:
            logger.debug(f"Loaded optimized prompts for {use_case}, metadata: {signature_metadata is not None}")
            # Note: signature_metadata may be absent in simplified prompt files and the pipeline
            # intentionally falls back to regex-based parsing in signatures._parse_signature_from_prompt
            
            # Inform dspy_config about successful prompt loading to avoid detection probes
            inform_successful_prompt_load(signature_metadata is not None)
            
            return system_prompt, user_prompt, signature_metadata
        else:
            return None, None, None
        
    except (FileNotFoundError, KeyError, AttributeError) as e:
        logger.debug(f"No optimized prompts found for {use_case}: {e}")
        return None, None, None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_case: Optional[str] = None,
) -> str:
    """
    Perform text-only LLM completion using DSPy framework with automatic prompt optimization.
    
    This function automatically attempts to use optimized prompts when a use_case is provided,
    falling back to the original message-based approach when optimized prompts are not available.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Explicit model name (mutually exclusive with tier)
        tier: Model tier for automatic selection (mutually exclusive with model)
        temperature: Temperature setting (optional, uses config default)
        max_tokens: Maximum tokens (optional)
        use_case: Use case identifier for specialized configuration and prompt optimization
        
    Returns:
        String response from the LLM
        
    Raises:
        ValueError: For invalid parameter combinations or empty messages
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    # Try to load optimized prompts first if use_case is provided
    # This informs dspy_config about signature metadata presence before configuration
    system_prompt, user_prompt, signature_metadata = _try_load_optimized_prompt(use_case) if use_case else (None, None, None)
    
    # Ensure DSPy is configured (now with knowledge of signature metadata availability)
    _ensure_dspy_configured()
    
    # Resolve model and temperature
    resolved_model = _resolve_model(model, tier, use_case)
    resolved_temperature = _resolve_temperature(resolved_model, temperature, use_case)
    
    # Get LM instance
    lm = _get_lm(resolved_model, resolved_temperature, max_tokens, use_case)
    
    if system_prompt or user_prompt:
        try:
            # Use DSPy-native signature approach
            predictor, inputs = _apply_optimized_signature(
                messages, system_prompt, user_prompt, None, signature_metadata
            )
            with dspy.context(lm=lm):
                result = predictor(**inputs)
                # Handle different predictor response formats
                used_metadata = signature_metadata is not None
                predictor_type = type(predictor).__name__
                _update_prompt_stats(use_case, success=True, used_metadata=used_metadata, predictor_type=predictor_type)
                if hasattr(result, 'response'):
                    return result.response
                elif hasattr(result, 'reasoning') and hasattr(result, 'response'):
                    # ChainOfThought predictor with reasoning
                    return result.response
                else:
                    return str(result)
        except Exception as e:
            logger.warning(f"Signature-based approach failed: {e}. Falling back to text substitution.")
            save_failure("generate_glossary.llm.completions", "completion", type(e).__name__, str(e), 
                        {"use_case": use_case, "model": model, "tier": tier})
            _update_prompt_stats(use_case, success=False, error=str(e))
            # Fallback to original approach
            prompt = _apply_optimized_prompts(messages, system_prompt, user_prompt)
            with dspy.context(lm=lm):
                signature = "prompt -> response"
                predictor = dspy.Predict(signature)
                result = predictor(prompt=prompt)
                return result.response
    else:
        # Track when no optimized prompts were found
        if use_case:
            _update_not_found_stats(use_case)
        # Fallback to original message joining
        prompt = _join_messages(messages)
        logger.debug(f"Using original message-based prompts for use_case: {use_case}")
        
        # Create text predictor and execute
        with dspy.context(lm=lm):
            signature = "prompt -> response"
            predictor = dspy.Predict(signature)
            result = predictor(prompt=prompt)
            return result.response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def structured_completion(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    model: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_case: Optional[str] = None,
) -> BaseModel:
    """
    Perform structured LLM completion using DSPy framework with automatic prompt optimization.
    
    This function uses a predictor compatible with structured output; falls back to `Predict` with post-parse when `TypedPredictor` isn't available.
    Automatically attempts to use optimized prompts when a use_case is provided.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        response_model: Pydantic model class for structured output
        model: Explicit model name (mutually exclusive with tier)
        tier: Model tier for automatic selection (mutually exclusive with model)
        temperature: Temperature setting (optional, uses config default)
        max_tokens: Maximum tokens (optional)
        use_case: Use case identifier for specialized configuration and prompt optimization
        
    Returns:
        Pydantic model instance with structured response
        
    Raises:
        ValueError: For invalid parameter combinations or empty messages
        TypeError: If response_model is not a Pydantic BaseModel subclass
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    # Validate response_model is a Pydantic BaseModel subclass
    if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
        raise TypeError("response_model must be a subclass of pydantic.BaseModel")
    
    # Try to load optimized prompts first if use_case is provided
    # This informs dspy_config about signature metadata presence before configuration
    system_prompt, user_prompt, signature_metadata = _try_load_optimized_prompt(use_case) if use_case else (None, None, None)
    
    # Ensure DSPy is configured (now with knowledge of signature metadata availability)
    _ensure_dspy_configured()
    
    # Resolve model and temperature
    resolved_model = _resolve_model(model, tier, use_case)
    resolved_temperature = _resolve_temperature(resolved_model, temperature, use_case)
    
    # Get LM instance
    lm = _get_lm(resolved_model, resolved_temperature, max_tokens, use_case)
    
    if system_prompt or user_prompt:
        try:
            # Use DSPy-native signature approach with structured output
            predictor, inputs = _apply_optimized_signature(
                messages, system_prompt, user_prompt, response_model, signature_metadata
            )
            with dspy.context(lm=lm):
                result = predictor(**inputs)
                used_metadata = signature_metadata is not None
                predictor_type = type(predictor).__name__
                _update_prompt_stats(use_case, success=True, used_metadata=used_metadata, predictor_type=predictor_type)
                # Parse result into structured format
                return _parse_structured_output(result, response_model)
        except Exception as e:
            logger.warning(f"Signature-based structured approach failed: {e}. Falling back to text substitution.")
            save_failure("generate_glossary.llm.completions", "structured_completion", type(e).__name__, str(e), 
                        {"use_case": use_case, "model": model, "tier": tier, "response_model": response_model.__name__})
            _update_prompt_stats(use_case, success=False, error=str(e))
            # Fallback to original approach
            prompt = _apply_optimized_prompts(messages, system_prompt, user_prompt)
            with dspy.context(lm=lm):
                signature = "prompt -> output"
                predictor = dspy.Predict(signature)
                result = predictor(prompt=prompt)
                return _parse_structured_output(result, response_model)
    else:
        # Fallback to original message joining
        prompt = _join_messages(messages)
        logger.debug(f"Using original message-based prompts for structured completion with use_case: {use_case}")
        
        # Create structured predictor and execute
        with dspy.context(lm=lm):
            signature = "prompt -> output"
            predictor = dspy.Predict(signature)
            result = predictor(prompt=prompt)
            return _parse_structured_output(result, response_model)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def async_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_case: Optional[str] = None,
) -> str:
    """
    Perform asynchronous text-only LLM completion using DSPy framework with automatic prompt optimization.
    
    This function automatically attempts to use optimized prompts when a use_case is provided,
    maintaining the async pattern with DSPy's acall method.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Explicit model name (mutually exclusive with tier)
        tier: Model tier for automatic selection (mutually exclusive with model)
        temperature: Temperature setting (optional, uses config default)
        max_tokens: Maximum tokens (optional)
        use_case: Use case identifier for specialized configuration and prompt optimization
        
    Returns:
        String response from the LLM
        
    Raises:
        ValueError: For invalid parameter combinations or empty messages
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    # Try to load optimized prompts first if use_case is provided
    # This informs dspy_config about signature metadata presence before configuration
    system_prompt, user_prompt, signature_metadata = _try_load_optimized_prompt(use_case) if use_case else (None, None, None)
    
    # Ensure DSPy is configured (now with knowledge of signature metadata availability)
    _ensure_dspy_configured()
    
    # Resolve model and temperature
    resolved_model = _resolve_model(model, tier, use_case)
    resolved_temperature = _resolve_temperature(resolved_model, temperature, use_case)
    
    # Get LM instance
    lm = _get_lm(resolved_model, resolved_temperature, max_tokens, use_case)
    
    if system_prompt or user_prompt:
        try:
            # Use DSPy-native signature approach asynchronously
            predictor, inputs = _apply_optimized_signature(
                messages, system_prompt, user_prompt, None, signature_metadata
            )
            with dspy.context(lm=lm):
                # Check if predictor supports acall, fallback to sync call wrapped in async
                if hasattr(predictor, 'acall'):
                    result = await predictor.acall(**inputs)
                else:
                    result = await asyncio.to_thread(lambda: predictor(**inputs))
                # Handle different predictor response formats
                used_metadata = signature_metadata is not None
                predictor_type = type(predictor).__name__
                _update_prompt_stats(use_case, success=True, used_metadata=used_metadata, predictor_type=predictor_type)
                if hasattr(result, 'response'):
                    return result.response
                elif hasattr(result, 'reasoning') and hasattr(result, 'response'):
                    # ChainOfThought predictor with reasoning
                    return result.response
                else:
                    return str(result)
        except Exception as e:
            logger.warning(f"Async signature-based approach failed: {e}. Falling back to text substitution.")
            save_failure("generate_glossary.llm.completions", "async_completion", type(e).__name__, str(e), 
                        {"use_case": use_case, "model": model, "tier": tier})
            _update_prompt_stats(use_case, success=False, error=str(e))
            # Fallback to original approach
            prompt = _apply_optimized_prompts(messages, system_prompt, user_prompt)
            with dspy.context(lm=lm):
                signature = "prompt -> response"
                predictor = dspy.Predict(signature)
                if hasattr(predictor, 'acall'):
                    result = await predictor.acall(prompt=prompt)
                else:
                    result = await asyncio.to_thread(lambda: predictor(prompt=prompt))
                return result.response
    else:
        # Fallback to original message joining
        prompt = _join_messages(messages)
        logger.debug(f"Using original message-based prompts for async completion with use_case: {use_case}")
        
        # Create text predictor and execute asynchronously
        with dspy.context(lm=lm):
            signature = "prompt -> response"
            predictor = dspy.Predict(signature)
            result = await predictor.acall(prompt=prompt) if hasattr(predictor, 'acall') else await asyncio.to_thread(lambda: predictor(prompt=prompt))
            return result.response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
async def async_structured_completion(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    model: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_case: Optional[str] = None,
) -> BaseModel:
    """
    Perform asynchronous structured LLM completion using DSPy framework with TypedPredictor and automatic prompt optimization.
    
    This function automatically attempts to use optimized prompts when a use_case is provided,
    maintaining compatibility with structured output requirements in async context.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        response_model: Pydantic model class for structured output
        model: Explicit model name (mutually exclusive with tier)
        tier: Model tier for automatic selection (mutually exclusive with model)
        temperature: Temperature setting (optional, uses config default)
        max_tokens: Maximum tokens (optional)
        use_case: Use case identifier for specialized configuration and prompt optimization
        
    Returns:
        Pydantic model instance with structured response
        
    Raises:
        ValueError: For invalid parameter combinations or empty messages
        TypeError: If response_model is not a Pydantic BaseModel subclass
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    # Validate response_model is a Pydantic BaseModel subclass
    if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
        raise TypeError("response_model must be a subclass of pydantic.BaseModel")
    
    # Try to load optimized prompts first if use_case is provided
    # This informs dspy_config about signature metadata presence before configuration
    system_prompt, user_prompt, signature_metadata = _try_load_optimized_prompt(use_case) if use_case else (None, None, None)
    
    # Ensure DSPy is configured (now with knowledge of signature metadata availability)
    _ensure_dspy_configured()
    
    # Resolve model and temperature
    resolved_model = _resolve_model(model, tier, use_case)
    resolved_temperature = _resolve_temperature(resolved_model, temperature, use_case)
    
    # Get LM instance
    lm = _get_lm(resolved_model, resolved_temperature, max_tokens, use_case)
    
    if system_prompt or user_prompt:
        try:
            # Use DSPy-native signature approach with async structured output
            predictor, inputs = _apply_optimized_signature(
                messages, system_prompt, user_prompt, response_model, signature_metadata
            )
            with dspy.context(lm=lm):
                # Check if predictor supports acall, fallback to sync call wrapped in async
                if hasattr(predictor, 'acall'):
                    result = await predictor.acall(**inputs)
                else:
                    result = await asyncio.to_thread(lambda: predictor(**inputs))
                used_metadata = signature_metadata is not None
                predictor_type = type(predictor).__name__
                _update_prompt_stats(use_case, success=True, used_metadata=used_metadata, predictor_type=predictor_type)
                # Parse result into structured format
                return _parse_structured_output(result, response_model)
        except Exception as e:
            logger.warning(f"Async signature-based structured approach failed: {e}. Falling back to text substitution.")
            save_failure("generate_glossary.llm.completions", "async_structured_completion", type(e).__name__, str(e), 
                        {"use_case": use_case, "model": model, "tier": tier, "response_model": response_model.__name__})
            _update_prompt_stats(use_case, success=False, error=str(e))
            # Fallback to original approach
            prompt = _apply_optimized_prompts(messages, system_prompt, user_prompt)
            with dspy.context(lm=lm):
                signature = "prompt -> output"
                predictor = dspy.Predict(signature)
                if hasattr(predictor, 'acall'):
                    result = await predictor.acall(prompt=prompt)
                else:
                    result = await asyncio.to_thread(lambda: predictor(prompt=prompt))
                return _parse_structured_output(result, response_model)
    else:
        # Fallback to original message joining
        prompt = _join_messages(messages)
        logger.debug(f"Using original message-based prompts for async structured completion with use_case: {use_case}")
        
        # Create structured predictor and execute asynchronously
        with dspy.context(lm=lm):
            signature = "prompt -> output"
            predictor = dspy.Predict(signature)
            result = await predictor.acall(prompt=prompt) if hasattr(predictor, 'acall') else await asyncio.to_thread(lambda: predictor(prompt=prompt))
            return _parse_structured_output(result, response_model)


def test_optimized_prompt_integration(
    use_case: str,
    test_messages: List[Dict[str, str]],
    expected_fields: Optional[List[str]] = None,
    dry_run: bool = True
) -> Dict[str, Any]:
    """
    Test integration with existing optimized prompts.
    
    Args:
        use_case: Use case identifier (e.g., "lv0_s1")
        test_messages: Test messages to use for integration testing
        expected_fields: Optional list of expected field names to validate
        dry_run: If True, don't actually call LM (default: True)
        
    Returns:
        Dictionary with test results:
        {
            'success': bool,
            'signature_created': bool,
            'fields_extracted': List[str],
            'signature_str': str,
            'errors': List[str],
            'warnings': List[str]
        }
        
    Examples:
        test_messages = [{'role': 'user', 'content': 'Test content'}]
        result = test_optimized_prompt_integration("lv0_s1", test_messages)
        print(f"Test successful: {result['success']}")
    """
    results = {
        'success': False,
        'signature_created': False,
        'fields_extracted': [],
        'signature_str': '',
        'errors': [],
        'warnings': []
    }
    
    try:
        # Try to load optimized prompts
        system_prompt, user_prompt, signature_metadata = _try_load_optimized_prompt(use_case)
        
        if not (system_prompt or user_prompt):
            results['errors'].append(f"No optimized prompts found for use_case: {use_case}")
            return results
        
        # Handle partial prompt availability
        if not system_prompt:
            logger.debug(f"Only user prompt available for {use_case}")
        elif not user_prompt:
            logger.debug(f"Only system prompt available for {use_case}")
        
        # Test signature parsing
        try:
            signature_info = _parse_signature_from_prompt(system_prompt, user_prompt, signature_metadata)
            results['signature_str'] = signature_info['signature_str']
            results['signature_created'] = True
            logger.debug(f"Signature created: {signature_info['signature_str']}")
        except Exception as e:
            results['errors'].append(f"Signature parsing failed: {e}")
            return results
        
        # Test content extraction
        try:
            field_mapping = _extract_content_for_signature(test_messages, signature_info)
            results['fields_extracted'] = list(field_mapping.keys())
            logger.debug(f"Fields extracted: {results['fields_extracted']}")
        except Exception as e:
            results['errors'].append(f"Content extraction failed: {e}")
            return results
        
        # Validate expected fields if provided
        if expected_fields:
            missing_fields = [f for f in expected_fields if f not in results['fields_extracted']]
            if missing_fields:
                results['warnings'].append(f"Missing expected fields: {missing_fields}")
        
        # Test predictor creation
        try:
            predictor = _create_optimized_predictor(system_prompt, user_prompt, signature_metadata=signature_metadata)
            logger.debug("Predictor created successfully")
            
            # If not dry_run, test actual execution
            if not dry_run:
                try:
                    # Create mock LM or use real one
                    _ensure_dspy_configured()
                    config = get_llm_config()
                    test_model = _resolve_model(None, "budget", None)  # Use budget tier for testing
                    test_temp = _resolve_temperature(test_model, 0.1, None)  # Low temperature for consistency
                    lm = _get_lm(test_model, test_temp, 100)  # Limit tokens for testing
                    
                    with dspy.context(lm=lm):
                        result = predictor(**field_mapping)
                        logger.debug(f"Test execution successful, result type: {type(result)}")
                        results['test_execution'] = 'successful'
                except Exception as e:
                    results['warnings'].append(f"Test execution failed: {e}")
                    results['test_execution'] = 'failed'
            else:
                results['test_execution'] = 'skipped (dry_run=True)'
                
        except Exception as e:
            results['errors'].append(f"Predictor creation failed: {e}")
            return results
        
        results['success'] = True
        logger.info(f"Integration test successful for use_case: {use_case}")
        
    except Exception as e:
        results['errors'].append(f"Test failed with unexpected error: {e}")
    
    return results


def validate_prompt_format(
    system_prompt: str,
    user_prompt: str,
    signature_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate optimized prompt format for DSPy integration.
    
    Args:
        system_prompt: System prompt to validate
        user_prompt: User prompt to validate
        signature_metadata: Optional metadata containing signature information
        
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'issues': List[str],
            'recommendations': List[str],
            'signature_info': Dict[str, Any],
            'metadata_used': bool
        }
        
    Examples:
        result = validate_prompt_format(system_prompt, user_prompt)
        if result['valid']:
            print("Prompt format is valid for DSPy integration")
    """
    validation_results = {
        'valid': True,
        'issues': [],
        'recommendations': [],
        'signature_info': {},
        'metadata_used': signature_metadata is not None
    }
    
    try:
        # Parse signature information
        signature_info = _parse_signature_from_prompt(system_prompt, user_prompt, signature_metadata)
        
        # Indicate whether metadata was used or regex fallback was used
        if signature_metadata is None:
            validation_results['recommendations'].append(
                "No signature metadata found - using regex inference as fallback. "
                "Consider adding signature metadata for deterministic parsing."
            )
        validation_results['signature_info'] = signature_info
        
        # Validate input fields
        input_fields = signature_info.get('input_fields', {})
        if not input_fields:
            validation_results['issues'].append("No input fields detected")
            validation_results['valid'] = False
            validation_results['recommendations'].append(
                "Ensure prompt mentions specific input fields like 'text', 'term', or 'content'"
            )
        elif len(input_fields) > 1:
            # Warn about multiple inputs as mentioned in comment 2
            validation_results['recommendations'].append(
                f"Multiple input fields detected ({list(input_fields.keys())}). " +
                "Ensure this matches your intended signature structure."
            )
        
        # Validate output fields
        output_fields = signature_info.get('output_fields', {})
        if not output_fields:
            validation_results['issues'].append("No output fields detected")
            validation_results['valid'] = False
            validation_results['recommendations'].append(
                "Consider adding output structure like 'reasoning' and 'response'"
            )
        
        # Check for common patterns that work well with DSPy
        combined_text = f"{system_prompt} {user_prompt}".lower()
        
        # Check if prompt text suggests multiple inputs but only one was inferred
        input_indicators = ['field named', 'input field', 'given', 'provided', 'analyze', 'process']
        input_mentions = sum(1 for indicator in input_indicators if indicator in combined_text)
        
        if input_mentions > 1 and len(input_fields) == 1:
            validation_results['recommendations'].append(
                f"Prompt text suggests multiple inputs (found {input_mentions} input indicators) " +
                f"but only {len(input_fields)} field was inferred. Consider reviewing signature parsing."
            )
        
        if "step by step" in combined_text or "reasoning" in combined_text:
            validation_results['recommendations'].append(
                "Good: Prompt encourages reasoning, works well with ChainOfThought"
            )
        
        if "analysis" in combined_text and "conclusion" in combined_text:
            validation_results['recommendations'].append(
                "Good: Structured analysis pattern detected"
            )
        
        # Validate instruction clarity
        instructions = signature_info.get('instructions', '')
        if len(instructions) < 50:
            validation_results['issues'].append("Instructions may be too brief")
            validation_results['recommendations'].append(
                "Consider adding more detailed instructions for better signature context"
            )
        
        logger.debug(f"Prompt validation completed: {len(validation_results['issues'])} issues found")
        
    except Exception as e:
        validation_results['valid'] = False
        validation_results['issues'].append(f"Validation failed: {e}")
    
    return validation_results


def benchmark_integration(
    use_cases: List[str],
    test_messages: List[Dict[str, str]],
    run_lm: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark old vs new integration approaches.
    
    Args:
        use_cases: List of use case identifiers to test
        test_messages: Messages to use for benchmarking
        run_lm: If True, actually execute with real LM (default: False)
        
    Returns:
        Dictionary with benchmark results per use case:
        {
            'use_case': {
                'signature_success_rate': float,
                'fallback_rate': float,
                'avg_signature_creation_time': float,
                'errors': List[str]
            }
        }
        
    Examples:
        use_cases = ["lv0_s1", "lv0_s3"]
        test_messages = [{'role': 'user', 'content': 'Test'}]
        results = benchmark_integration(use_cases, test_messages)
    """
    benchmark_results = {}
    
    for use_case in use_cases:
        results = {
            'signature_success_rate': 0.0,
            'fallback_rate': 0.0,
            'avg_signature_creation_time': 0.0,
            'errors': []
        }
        
        try:
            # Test signature-based approach
            start_time = time.time()
            
            system_prompt, user_prompt, signature_metadata = _try_load_optimized_prompt(use_case)
            
            # Handle partial prompt availability
            if not system_prompt:
                system_prompt = "Process the input according to the instructions."
                logger.debug(f"No system prompt for {use_case}, using default")
            if not user_prompt:
                user_prompt = "Please process the provided input."
                logger.debug(f"No user prompt for {use_case}, using default")
            
            if system_prompt or user_prompt:
                try:
                    signature_info = _parse_signature_from_prompt(system_prompt, user_prompt, signature_metadata)
                    predictor = _create_optimized_predictor(system_prompt, user_prompt, signature_metadata=signature_metadata)
                    field_mapping = _extract_content_for_signature(test_messages, signature_info)
                    
                    # Optionally test actual LM execution
                    if run_lm:
                        try:
                            _ensure_dspy_configured()
                            config = get_llm_config()
                            test_model = _resolve_model(None, "budget", None)
                            test_temp = _resolve_temperature(test_model, 0.1, None)
                            lm = _get_lm(test_model, test_temp, 50)  # Small limit for benchmarking
                            
                            with dspy.context(lm=lm):
                                result = predictor(**field_mapping)
                                logger.debug(f"Benchmark LM execution successful for {use_case}")
                        except Exception as e:
                            results['errors'].append(f"LM execution failed: {e}")
                    
                    results['signature_success_rate'] = 1.0
                    results['fallback_rate'] = 0.0
                    
                except Exception as e:
                    results['signature_success_rate'] = 0.0
                    results['fallback_rate'] = 1.0
                    results['errors'].append(f"Signature approach failed: {e}")
            else:
                results['errors'].append(f"No optimized prompts found for {use_case}")
            
            end_time = time.time()
            results['avg_signature_creation_time'] = end_time - start_time
            
        except Exception as e:
            results['errors'].append(f"Benchmark failed for {use_case}: {e}")
        
        benchmark_results[use_case] = results
        logger.debug(f"Benchmark completed for {use_case}")
    
    return benchmark_results


def configure_for_optimized_prompts(
    enable_chain_of_thought: bool = True,
    enable_signature_optimization: bool = True,
    fallback_to_text: bool = True
) -> None:
    """
    Configure DSPy for optimal performance with optimized prompts.
    
    Args:
        enable_chain_of_thought: Enable ChainOfThought predictor by default
        enable_signature_optimization: Enable signature-based optimization
        fallback_to_text: Enable fallback to text-based approach on failure
        
    Examples:
        # Configure for signature-based optimization
        configure_for_optimized_prompts()
        
        # Configure with text fallback disabled
        configure_for_optimized_prompts(fallback_to_text=False)
    """
    global _optimized_prompt_config
    
    # Store configuration in module-level variable
    _optimized_prompt_config = {
        'enable_chain_of_thought': enable_chain_of_thought,
        'enable_signature_optimization': enable_signature_optimization,
        'fallback_to_text': fallback_to_text
    }
    
    # Configure DSPy settings for optimization
    configure_for_optimization(disable_cache=True, trace=False)
    
    logger.info(f"Configured DSPy for optimized prompts: {_optimized_prompt_config}")


def get_optimized_prompt_stats() -> Dict[str, Any]:
    """
    Get statistics about optimized prompt usage and performance.
    
    Returns:
        Dictionary with statistics:
        {
            'signature_attempts': int,
            'signature_successes': int,
            'fallback_count': int,
            'success_rate': float,
            'use_cases_tested': List[str],
            'common_errors': Dict[str, int],
            'metadata_usage_count': int,
            'text_inference_count': int,
            'metadata_usage_rate': float,
            'predictor_type_distribution': Dict[str, int]
        }
        
    Examples:
        stats = get_optimized_prompt_stats()
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Metadata usage rate: {stats['metadata_usage_rate']:.2%}")
    """
    global _prompt_usage_stats
    
    # Initialize stats if not available
    if '_prompt_usage_stats' not in globals():
        _prompt_usage_stats = {
            'signature_attempts': 0,
            'signature_successes': 0,
            'fallback_count': 0,
            'not_found_count': 0,
            'use_cases_tested': [],
            'common_errors': {},
            'metadata_usage_count': 0,
            'text_inference_count': 0,
            'predictor_type_distribution': {}
        }
    
    stats = _prompt_usage_stats.copy()
    
    # Calculate success rate
    if stats['signature_attempts'] > 0:
        stats['success_rate'] = stats['signature_successes'] / stats['signature_attempts']
    else:
        stats['success_rate'] = 0.0
    
    # Calculate metadata usage rate
    total_signature_creation = stats['metadata_usage_count'] + stats['text_inference_count']
    if total_signature_creation > 0:
        stats['metadata_usage_rate'] = stats['metadata_usage_count'] / total_signature_creation
    else:
        stats['metadata_usage_rate'] = 0.0
    
    return stats


def _update_prompt_stats(use_case: str, success: bool, error: Optional[str] = None, used_metadata: bool = False, predictor_type: str = None) -> None:
    """
    Update internal prompt usage statistics.
    
    Args:
        use_case: Use case identifier
        success: Whether signature-based approach succeeded
        error: Error message if failed
        used_metadata: Whether signature metadata was used (vs text inference)
        predictor_type: Type of predictor created
    """
    global _prompt_usage_stats
    
    # Initialize stats if not available
    if '_prompt_usage_stats' not in globals():
        _prompt_usage_stats = {
            'signature_attempts': 0,
            'signature_successes': 0,
            'fallback_count': 0,
            'not_found_count': 0,
            'use_cases_tested': [],
            'common_errors': {},
            'metadata_usage_count': 0,
            'text_inference_count': 0,
            'predictor_type_distribution': {}
        }
    
    _prompt_usage_stats['signature_attempts'] += 1
    
    if success:
        _prompt_usage_stats['signature_successes'] += 1
        
        # Track signature creation method
        if used_metadata:
            _prompt_usage_stats['metadata_usage_count'] += 1
        else:
            _prompt_usage_stats['text_inference_count'] += 1
            
        # Track predictor type distribution
        if predictor_type:
            _prompt_usage_stats['predictor_type_distribution'][predictor_type] = (
                _prompt_usage_stats['predictor_type_distribution'].get(predictor_type, 0) + 1
            )
            
    else:
        _prompt_usage_stats['fallback_count'] += 1
        
        if error:
            error_key = str(error)[:100]  # Truncate long errors
            _prompt_usage_stats['common_errors'][error_key] = (
                _prompt_usage_stats['common_errors'].get(error_key, 0) + 1
            )
    
    if use_case and use_case not in _prompt_usage_stats['use_cases_tested']:
        _prompt_usage_stats['use_cases_tested'].append(use_case)


def _update_not_found_stats(use_case: str) -> None:
    """
    Update stats when no optimized prompts are found.
    
    Args:
        use_case: Use case identifier
    """
    global _prompt_usage_stats
    
    # Initialize stats if not available
    if '_prompt_usage_stats' not in globals():
        _prompt_usage_stats = {
            'signature_attempts': 0,
            'signature_successes': 0,
            'fallback_count': 0,
            'not_found_count': 0,
            'use_cases_tested': [],
            'common_errors': {}
        }
    
    _prompt_usage_stats['not_found_count'] += 1
    
    if use_case and use_case not in _prompt_usage_stats['use_cases_tested']:
        _prompt_usage_stats['use_cases_tested'].append(use_case)