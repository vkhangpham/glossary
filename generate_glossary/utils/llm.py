"""
LLM utilities using LiteLLM and Instructor with production-ready features.

Features:
- Error handling and retry logic
- Optional caching for cost savings
- Async support for batch operations
- Response validation
- Centralized configuration
"""

import functools
import hashlib
import json
import asyncio
import threading
import re
from typing import Optional, Type, Dict, List, Tuple, Union
from collections import Counter

import instructor
import litellm
from litellm import completion, acompletion
from litellm.exceptions import (
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    Timeout,
    APIConnectionError,
)
from pydantic import BaseModel

from .logger import setup_logger

logger = setup_logger("llm")

litellm.enable_json_schema_validation = True


def _sanitize_error_message(error_msg: str) -> str:
    """Remove potential API keys and sensitive information from error messages."""
    if not isinstance(error_msg, str):
        error_msg = str(error_msg)

    # Remove common API key patterns
    # OpenAI keys: sk-...
    error_msg = re.sub(r"\bsk-[A-Za-z0-9]{20,}\b", "[REDACTED_API_KEY]", error_msg)
    # Generic long alphanumeric strings that might be keys
    error_msg = re.sub(r"\b[A-Za-z0-9_-]{32,}\b", "[REDACTED_TOKEN]", error_msg)
    # Remove any Bearer tokens
    error_msg = re.sub(r"\bBearer\s+[A-Za-z0-9_-]+\b", "Bearer [REDACTED]", error_msg)

    return error_msg


def _validate_messages(messages: List[Dict[str, str]]) -> None:
    """Validate message structure and content to prevent injection attacks."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a non-empty list")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"Message {i} must be a dict")

        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Message {i} must have 'role' and 'content' keys")

        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            raise ValueError(f"Message {i} 'role' and 'content' must be strings")

        # Validate role values
        valid_roles = {"system", "user", "assistant", "function", "tool"}
        if msg["role"] not in valid_roles:
            raise ValueError(
                f"Message {i} role '{msg['role']}' not in valid roles: {valid_roles}"
            )

        # Limit message content length to prevent abuse
        if len(msg["content"]) > 100000:  # 100KB limit
            raise ValueError(
                f"Message {i} content too long: {len(msg['content'])} chars (max 100000)"
            )


def _validate_parameters(
    model: str,
    temperature: float = None,
    num_responses: int = None,
    max_tokens: int = None,
) -> None:
    """Validate common function parameters."""
    if not model or not isinstance(model, str):
        raise ValueError("Model must be a non-empty string")

    if "/" not in model:
        raise ValueError(
            "Model must be in format 'provider/model' (e.g., 'openai/gpt-4')"
        )

    if temperature is not None:
        if not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be a number between 0.0 and 2.0")

    if num_responses is not None:
        if not isinstance(num_responses, int) or num_responses <= 0:
            raise ValueError("num_responses must be a positive integer")
        if num_responses > 10:
            raise ValueError("num_responses too high (max 10 to prevent abuse)")

    if max_tokens is not None:
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if max_tokens > 128000:  # Reasonable upper limit
            raise ValueError("max_tokens too high (max 128000)")


def _safe_serialize(obj):
    """Safely serialize objects for caching, filtering sensitive data."""
    if hasattr(obj, "__dict__"):
        # For objects with attributes, just return type info
        return f"<{type(obj).__name__}>"
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        # Filter out keys that might contain sensitive data
        sensitive_keys = {"api_key", "token", "password", "secret", "auth", "bearer"}
        return {
            k: (
                "[REDACTED]"
                if any(sensitive in k.lower() for sensitive in sensitive_keys)
                else _safe_serialize(v)
            )
            for k, v in obj.items()
        }
    else:
        return f"<{type(obj).__name__}>"


class LLMClient:
    """Thread-safe lazy-initialized LLM client manager."""

    _sync_client = None
    _async_client = None
    _lock = threading.Lock()

    @classmethod
    def get_sync(cls):
        """Get or create synchronous client in a thread-safe manner."""
        if cls._sync_client is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._sync_client is None:
                    cls._sync_client = instructor.from_litellm(completion)
        return cls._sync_client

    @classmethod
    def get_async(cls):
        """Get or create asynchronous client in a thread-safe manner."""
        if cls._async_client is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._async_client is None:
                    cls._async_client = instructor.from_litellm(acompletion)
        return cls._async_client


def _create_cache_key(
    model: str,
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    **kwargs,
) -> str:
    """Create a secure cache key from request parameters."""
    # Use faster Blake2b hash
    hasher = hashlib.blake2b(digest_size=16)

    hasher.update(model.encode())
    hasher.update(json.dumps(messages, sort_keys=True, separators=(",", ":")).encode())
    hasher.update(response_model.__name__.encode())

    if kwargs:
        # Use safe serialization to avoid exposing sensitive data
        safe_kwargs = _safe_serialize(kwargs)
        hasher.update(
            json.dumps(safe_kwargs, sort_keys=True, separators=(",", ":")).encode()
        )

    return hasher.hexdigest()


@functools.lru_cache(maxsize=128)
def _cached_structured_completion(
    cache_key: str,
    model: str,
    messages_json: str,
    response_model: Type[BaseModel],
    **kwargs,
) -> BaseModel:
    """Cached version of structured completion."""
    messages = json.loads(messages_json)
    return _structured_completion_impl(model, messages, response_model, **kwargs)


def _structured_completion_impl(
    model: str,
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    **kwargs,
) -> BaseModel:
    """Internal implementation of structured completion."""
    client = LLMClient.get_sync()
    return client.chat.completions.create(
        model=model, messages=messages, response_model=response_model, **kwargs
    )


def structured_completion(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    use_cache: bool = False,
) -> BaseModel:
    """
    Get structured completion using instructor + litellm with error handling.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_model: Pydantic model class for structured output
        model: Model name in format "provider/model" (required)
        temperature: Model temperature
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure
        use_cache: Whether to use caching (default: False)

    Returns:
        Parsed response as the specified Pydantic model

    Raises:
        ValueError: When input parameters are invalid
        RateLimitError: When rate limit is exceeded
        AuthenticationError: When API key is invalid
        BadRequestError: When request parameters are invalid
    """
    _validate_messages(messages)
    _validate_parameters(model, temperature, max_tokens=max_tokens)

    kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }

    try:
        if use_cache:
            cache_key = _create_cache_key(model, messages, response_model, **kwargs)
            messages_json = json.dumps(messages)
            return _cached_structured_completion(
                cache_key, model, messages_json, response_model, **kwargs
            )
        else:
            return _structured_completion_impl(
                model, messages, response_model, **kwargs
            )

    except RateLimitError as e:
        logger.warning(f"Rate limit hit for {model}: {_sanitize_error_message(str(e))}")
        logger.info("Consider implementing exponential backoff or switching providers")
        raise
    except AuthenticationError as e:
        logger.error(
            f"Authentication failed for {model}: {_sanitize_error_message(str(e))}"
        )
        logger.info("Check your API keys in environment variables")
        raise
    except BadRequestError as e:
        logger.error(f"Bad request to {model}: {_sanitize_error_message(str(e))}")
        logger.info("Check your request parameters and model compatibility")
        raise
    except Timeout as e:
        logger.warning(
            f"Request to {model} timed out: {_sanitize_error_message(str(e))}"
        )
        logger.info("Consider increasing timeout or reducing response size")
        raise
    except APIConnectionError as e:
        logger.error(f"Connection error to {model}: {_sanitize_error_message(str(e))}")
        logger.info("Check network connection and API endpoint status")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error with {model}: {type(e).__name__}: {_sanitize_error_message(str(e))}"
        )
        raise


async def async_structured_completion(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
) -> BaseModel:
    """
    Async version of structured completion for better performance in batch operations.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_model: Pydantic model class for structured output
        model: Model name in format "provider/model" (required)
        temperature: Model temperature
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure

    Returns:
        Parsed response as the specified Pydantic model

    Raises:
        ValueError: When input parameters are invalid
    """
    _validate_messages(messages)
    _validate_parameters(model, temperature, max_tokens=max_tokens)

    client = LLMClient.get_async()

    try:
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    except Exception as e:
        logger.error(
            f"Async call failed for {model}: {_sanitize_error_message(str(e))}"
        )
        raise


def text_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Get simple text completion using litellm with error handling.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name in format "provider/model" (required)
        temperature: Model temperature
        max_tokens: Max tokens to generate

    Returns:
        Response text content

    Raises:
        ValueError: When input parameters are invalid
    """
    _validate_messages(messages)
    _validate_parameters(model, temperature, max_tokens=max_tokens)

    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(
            f"Text completion failed for {model}: {_sanitize_error_message(str(e))}"
        )
        raise


def structured_completion_consensus(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    model: str,
    num_responses: int = 3,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    use_cache: bool = False,
    return_all: bool = False,
) -> Union[BaseModel, Tuple[BaseModel, List[BaseModel]]]:
    """
    Generate multiple LLM responses and return consensus or all responses.

    This is useful for getting more reliable results through voting/consensus,
    especially for extraction or classification tasks.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_model: Pydantic model class for structured output
        model: Model name in format "provider/model" (required)
        num_responses: Number of responses to generate (default: 3)
        temperature: Model temperature (higher = more variation)
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure per response
        use_cache: Whether to use caching (default: False)
        return_all: If True, returns (consensus, all_responses), else just consensus

    Returns:
        If return_all=False: The most common response (consensus)
        If return_all=True: Tuple of (consensus_response, list_of_all_responses)

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When insufficient responses succeed

    Example:
        # Get consensus from 5 responses
        result = structured_completion_consensus(
            messages=messages,
            response_model=ClassificationResult,
            num_responses=5
        )

        # Get consensus and see all responses
        consensus, all_results = structured_completion_consensus(
            messages=messages,
            response_model=ExtractedData,
            num_responses=3,
            return_all=True
        )
    """
    _validate_messages(messages)
    _validate_parameters(model, temperature, num_responses, max_tokens)

    responses = []
    errors = []
    successful_responses = 0
    required_minimum = max(1, num_responses // 2)  # At least half must succeed

    logger.info(
        f"Generating {num_responses} responses for consensus (minimum {required_minimum} required)"
    )

    for i in range(num_responses):
        try:
            response = structured_completion(
                messages=messages,
                response_model=response_model,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                use_cache=use_cache and i == 0,  # Only cache first call
            )
            responses.append(response)
            successful_responses += 1
            logger.debug(f"Generated response {i+1}/{num_responses} successfully")

        except Exception as e:
            sanitized_error = _sanitize_error_message(str(e))
            logger.warning(
                f"Failed to generate response {i+1}/{num_responses}: {sanitized_error}"
            )
            errors.append((i + 1, type(e).__name__, sanitized_error))

            remaining_attempts = num_responses - (i + 1)
            if successful_responses + remaining_attempts < required_minimum:
                logger.error(
                    f"Early termination: Cannot reach minimum {required_minimum} responses"
                )
                break
            continue

    if successful_responses < required_minimum:
        raise RuntimeError(
            f"Insufficient successful responses: {successful_responses}/{num_responses} "
            f"(minimum {required_minimum} required). Errors: {errors}"
        )

    response_strings = [
        response.model_dump_json(exclude_none=True) for response in responses
    ]
    response_counter = Counter(response_strings)
    most_common_json, count = response_counter.most_common(1)[0]
    consensus_index = response_strings.index(most_common_json)
    consensus = responses[consensus_index]

    logger.info(f"Consensus reached: {count}/{len(responses)} responses agree")

    if return_all:
        return consensus, responses
    return consensus


async def async_structured_completion_consensus(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    model: str,
    num_responses: int = 3,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    return_all: bool = False,
) -> Union[BaseModel, Tuple[BaseModel, List[BaseModel]]]:
    """
    Async version: Generate multiple LLM responses in parallel for consensus.

    Much faster than sync version as all responses are generated concurrently.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_model: Pydantic model class for structured output
        model: Model name in format "provider/model" (required)
        num_responses: Number of responses to generate (default: 3)
        temperature: Model temperature (higher = more variation)
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure per response
        return_all: If True, returns (consensus, all_responses), else just consensus

    Returns:
        If return_all=False: The most common response (consensus)
        If return_all=True: Tuple of (consensus_response, list_of_all_responses)

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When insufficient responses succeed

    Example:
        # Fast parallel consensus
        result = await async_structured_completion_consensus(
            messages=messages,
            response_model=ClassificationResult,
            num_responses=5
        )
    """
    _validate_messages(messages)
    _validate_parameters(model, temperature, num_responses, max_tokens)

    required_minimum = max(1, num_responses // 2)
    logger.info(
        f"Generating {num_responses} responses in parallel for consensus (minimum {required_minimum} required)"
    )

    tasks = []
    for i in range(num_responses):
        task = async_structured_completion(
            messages=messages,
            response_model=response_model,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    responses = []
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            sanitized_error = _sanitize_error_message(str(result))
            logger.warning(f"Response {i+1}/{num_responses} failed: {sanitized_error}")
            errors.append((i + 1, type(result).__name__, sanitized_error))
        else:
            responses.append(result)
            logger.debug(f"Response {i+1}/{num_responses} succeeded")

    successful_responses = len(responses)
    if successful_responses < required_minimum:
        raise RuntimeError(
            f"Insufficient successful responses: {successful_responses}/{num_responses} "
            f"(minimum {required_minimum} required). Errors: {errors}"
        )

    response_strings = [
        response.model_dump_json(exclude_none=True) for response in responses
    ]
    response_counter = Counter(response_strings)
    most_common_json, count = response_counter.most_common(1)[0]
    consensus_index = response_strings.index(most_common_json)
    consensus = responses[consensus_index]

    logger.info(f"Consensus reached: {count}/{len(responses)} responses agree")

    if return_all:
        return consensus, responses
    return consensus


def text_completion_consensus(
    messages: List[Dict[str, str]],
    model: str,
    num_responses: int = 3,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    return_all: bool = False,
) -> Union[str, Tuple[str, List[str]]]:
    """
    Generate multiple text responses and return consensus or all responses.

    For text responses, consensus is the most frequently occurring exact response.
    For more nuanced text consensus, consider using structured_completion_consensus
    with a Pydantic model that extracts key information.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name in format "provider/model" (required)
        num_responses: Number of responses to generate (default: 3)
        temperature: Model temperature
        max_tokens: Max tokens to generate
        return_all: If True, returns (consensus, all_responses)

    Returns:
        If return_all=False: The most common response string
        If return_all=True: Tuple of (consensus_response, list_of_all_responses)

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When insufficient responses succeed
    """
    _validate_messages(messages)
    _validate_parameters(model, temperature, num_responses, max_tokens)

    responses = []
    errors = []
    successful_responses = 0
    required_minimum = max(1, num_responses // 2)

    logger.info(
        f"Generating {num_responses} text responses for consensus (minimum {required_minimum} required)"
    )

    for i in range(num_responses):
        try:
            response = text_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            responses.append(response.strip())  # Strip whitespace for comparison
            successful_responses += 1
            logger.debug(f"Generated text response {i+1}/{num_responses} successfully")

        except Exception as e:
            sanitized_error = _sanitize_error_message(str(e))
            logger.warning(
                f"Failed to generate text response {i+1}/{num_responses}: {sanitized_error}"
            )
            errors.append((i + 1, type(e).__name__, sanitized_error))

            remaining_attempts = num_responses - (i + 1)
            if successful_responses + remaining_attempts < required_minimum:
                logger.error(
                    f"Early termination: Cannot reach minimum {required_minimum} responses"
                )
                break
            continue

    if successful_responses < required_minimum:
        raise RuntimeError(
            f"Insufficient successful text responses: {successful_responses}/{num_responses} "
            f"(minimum {required_minimum} required). Errors: {errors}"
        )

    response_counter = Counter(responses)
    consensus, count = response_counter.most_common(1)[0]

    logger.info(f"Text consensus: {count}/{len(responses)} responses agree")

    if return_all:
        return consensus, responses
    return consensus
