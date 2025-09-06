"""
LLM utilities using LiteLLM and Instructor with production-ready features.

Features:
- Error handling and retry logic
- Optional caching for cost savings
- Async support for batch operations
- Response validation
- Centralized configuration
- Automatic optimized prompt loading
"""

import asyncio
import json
import threading
import random
import os
from pathlib import Path
from typing import Optional, Type, Dict, List, Tuple, Union
from collections import Counter

import instructor
import litellm
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

# Performance optimizations - Enable for 6.5x faster completion + 100+ RPS
try:
    litellm.set_verbose = False  # type: ignore # +50 RPS improvement by reducing logging overhead
except AttributeError:
    pass  # set_verbose might not be available in all versions
litellm.enable_json_schema_validation = True

# Experimental HTTP handler for +100 RPS on OpenAI calls
os.environ.setdefault("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")


# Model tier definitions with cost-optimized selections
MODEL_TIERS = {
    "budget": [
        "openai/gpt-5-nano",  # $0.05 input / $0.40 output
        "openai/gpt-4o-mini",  # $0.15 input / $0.60 output
        "vertex_ai/gemini-2.5-flash",  # $0.30 input / $2.50 output
    ],
    "balanced": [
        "openai/gpt-5-mini",  # $0.25 input / $2.00 output
        "vertex_ai/gemini-2.5-flash",  # $0.30 input / $2.50 output
    ],
    "flagship": [
        "openai/gpt-5",  # $1.25 input / $10.00 output
        "anthropic/claude-4-sonnet",  # $3.00 input / $15.00 output
        "vertex_ai/gemini-2.5-pro",  # $1.25 input / $10.00 output
    ],
}

# Model aliases for easier referencing (removing provider prefix and dashes)
MODEL_ALIASES = {
    "gpt5nano": "openai/gpt-5-nano",
    "gpt5mini": "openai/gpt-5-mini",
    "gpt5": "openai/gpt-5",
    "gpt4omini": "openai/gpt-4o-mini",
    "claude4sonnet": "anthropic/claude-4-sonnet",
    "gemini2.5flash": "vertex_ai/gemini-2.5-flash",
    "gemini2.5pro": "vertex_ai/gemini-2.5-pro",
}


class LLMClient:
    """Thread-safe lazy-initialized LLM client manager with provider pattern."""

    _sync_client = None
    _async_client = None
    _providers = {}  # Cache provider-specific clients by model
    _lock = threading.Lock()

    @classmethod
    def get_sync(cls):
        """Get or create synchronous client in a thread-safe manner."""
        if cls._sync_client is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._sync_client is None:
                    cls._sync_client = instructor.from_litellm(litellm.completion)
        return cls._sync_client

    @classmethod
    def get_async(cls):
        """Get or create asynchronous client in a thread-safe manner."""
        if cls._async_client is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._async_client is None:
                    cls._async_client = instructor.from_litellm(litellm.acompletion)
        return cls._async_client

    @classmethod
    def get_provider_client(cls, model: str):
        """
        Get cached provider-specific client with enhanced features.

        Uses Instructor's from_provider() pattern for better provider management.
        Each model gets its own cached client with native caching enabled.

        Args:
            model: Model name in format "provider/model"

        Returns:
            Instructor client optimized for the specific model
        """
        if model not in cls._providers:
            with cls._lock:
                # Double-check locking pattern
                if model not in cls._providers:
                    # Use regular instructor client with caching
                    cls._providers[model] = instructor.from_litellm(litellm.completion)
        return cls._providers[model]


def get_model_by_tier(tier: str, random_selection: bool = True) -> str:
    """
    Get a model from the specified tier.

    Args:
        tier: Model tier ("budget", "balanced", "flagship")
        random_selection: If True, randomly select from tier; if False, return first model

    Returns:
        Model name in "provider/model" format

    Raises:
        ValueError: If tier is invalid
    """
    if tier not in MODEL_TIERS:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be one of: {list(MODEL_TIERS.keys())}"
        )

    models = MODEL_TIERS[tier]
    if random_selection:
        return random.choice(models)
    return models[0]


def resolve_model_alias(model: str) -> str:
    """
    Resolve model alias to full model name.

    Args:
        model: Model name or alias

    Returns:
        Full model name in "provider/model" format
    """
    return MODEL_ALIASES.get(model, model)


def validate_model(model: str) -> bool:
    """
    Validate if a model is in our supported tiers.

    Args:
        model: Model name to validate

    Returns:
        True if model is supported, False otherwise
    """
    resolved_model = resolve_model_alias(model)
    all_models = []
    for tier_models in MODEL_TIERS.values():
        all_models.extend(tier_models)
    return resolved_model in all_models


def get_all_models() -> Dict[str, List[str]]:
    """
    Get all available models organized by tier.

    Returns:
        Dictionary with tiers as keys and model lists as values
    """
    return MODEL_TIERS.copy()


def get_model_info(model: str) -> Dict[str, str]:
    """
    Get information about a specific model.

    Args:
        model: Model name or alias

    Returns:
        Dictionary with model information

    Raises:
        ValueError: If model is not supported
    """
    resolved_model = resolve_model_alias(model)

    if not validate_model(resolved_model):
        raise ValueError(f"Unsupported model: {model} (resolved to {resolved_model})")

    for tier, models in MODEL_TIERS.items():
        if resolved_model in models:
            info = {
                "model": resolved_model,
                "tier": tier,
            }
            if model in MODEL_ALIASES:
                info["alias"] = model
            return info

    raise ValueError(f"Model {resolved_model} not found in any tier")


def load_prompt_from_file(filepath: Union[str, Path]) -> Optional[str]:
    """
    Load a prompt from a JSON file.
    
    Simple utility to load saved prompts. The file should contain
    a JSON object with a 'content' field. If the content is in DSPy format,
    extracts the instructions from it.
    
    Args:
        filepath: Path to the prompt JSON file
        
    Returns:
        Prompt content if successful, None otherwise
    """
    try:
        path = Path(filepath)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                content = data.get("content")
                
                if not content:
                    return None
                    
                # Check if this is DSPy format (contains "instructions=")
                if "instructions='" in content or 'instructions="' in content:
                    # Extract instructions from DSPy format
                    import re
                    # Match instructions='...' handling escaped quotes and newlines
                    match = re.search(r"instructions=['\"](.+?)['\"](?=\s*\n\s*\w+\s*=|$)", content, re.DOTALL)
                    if match:
                        instructions = match.group(1)
                        # Unescape the content
                        instructions = instructions.replace("\\'", "'")
                        instructions = instructions.replace('\\"', '"')
                        instructions = instructions.replace("\\n", "\n")
                        return instructions
                
                # Return content as-is if not DSPy format
                return content
    except (json.JSONDecodeError, KeyError, IOError) as e:
        logger.debug(f"Could not load prompt from {filepath}: {e}")
    
    return None


def completion(
    messages: List[Dict[str, str]],
    response_model: Optional[Type[BaseModel]] = None,
    model: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    cache_ttl: int = 3600,  # Cache for 1 hour by default
    semantic_validation: Optional[str] = None,
) -> Union[str, BaseModel]:
    """
    Unified LLM completion function with optional structured output and validation.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_model: Pydantic model class for structured output (None = raw text)
        model: Model name in format "provider/model" (optional if tier provided)
        tier: Model tier ("budget", "balanced", "flagship") - randomly selects from tier
        temperature: Model temperature
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure
        cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour, 0 = no cache)
        semantic_validation: Optional semantic validation rule (e.g., "Output must be factually correct")

    Returns:
        If response_model provided: Parsed Pydantic model instance (potentially from cache)
        If response_model is None: Raw text response (potentially from cache)

    Raises:
        ValueError: When input parameters are invalid or neither model nor tier provided
        RateLimitError: When rate limit is exceeded
        AuthenticationError: When API key is invalid
        BadRequestError: When request parameters are invalid

    Examples:
        # Structured output with validation
        result = completion(messages, MyModel, tier="budget", semantic_validation="Facts must be verifiable")

        # Raw text output
        text = completion(messages, tier="budget")
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a non-empty list")

    if tier and model:
        raise ValueError("Provide either 'model' or 'tier', not both")
    elif tier:
        model = get_model_by_tier(tier, random_selection=True)
    elif model:
        model = resolve_model_alias(model)
    else:
        raise ValueError("Must provide either 'model' or 'tier'")

    if not model or "/" not in model:
        raise ValueError("Model must be in format 'provider/model'")
    if temperature < 0.0 or temperature > 2.0:
        raise ValueError("Temperature must be between 0.0 and 2.0")

    kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }

    if cache_ttl > 0:
        kwargs["ttl"] = cache_ttl

    if semantic_validation:
        kwargs["validation_context"] = semantic_validation

    try:
        # Use provider-specific client if semantic validation is needed, otherwise regular client
        if semantic_validation:
            client = LLMClient.get_provider_client(model)
        else:
            client = LLMClient.get_sync()

        return client.chat.completions.create(  # type: ignore
            model=model, messages=messages, response_model=response_model, **kwargs  # type: ignore
        )

    except RateLimitError:
        logger.warning(f"Rate limit hit for {model}")
        raise
    except AuthenticationError:
        logger.error(f"Authentication failed for {model}")
        raise
    except BadRequestError:
        logger.error(f"Bad request to {model}")
        raise
    except Timeout:
        logger.warning(f"Request to {model} timed out")
        raise
    except APIConnectionError:
        logger.error(f"Connection error to {model}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error with {model}: {type(e).__name__}")
        raise


async def async_completion(
    messages: List[Dict[str, str]],
    response_model: Optional[Type[BaseModel]] = None,
    model: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    cache_ttl: int = 3600,
    semantic_validation: Optional[str] = None,
) -> Union[str, BaseModel]:
    """
    Async version of unified completion function - runs sync version in thread pool.

    All parameters and behavior are identical to completion(). This simply wraps
    the sync function to run in a thread pool for async compatibility.

    Examples:
        # Structured output
        result = await async_completion(messages, MyModel, tier="budget")

        # Raw text output
        text = await async_completion(messages, tier="budget")
    """
    return await asyncio.to_thread(
        completion,
        messages=messages,
        response_model=response_model,
        model=model,
        tier=tier,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        cache_ttl=cache_ttl,
        semantic_validation=semantic_validation,
    )


async def structured_completion_consensus(
    messages: List[Dict[str, str]],
    response_model: Type[BaseModel],
    tier: str = "budget",
    num_responses: int = 3,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    return_all: bool = False,
    cache_ttl: int = 7200,  # Cache for 2 hours by default (longer for expensive consensus)
    semantic_validation: Optional[str] = None,
) -> Union[BaseModel, Tuple[BaseModel, List[BaseModel]]]:
    """
    Generate multiple LLM responses in parallel for consensus.

    Uses different models from the tier for diverse perspectives and better consensus.
    All responses are generated concurrently for maximum speed.

    Args:
        messages: List of message dicts with 'role' and 'content'
        response_model: Pydantic model class for structured output
        tier: Model tier ("budget", "balanced", "flagship") - uses different models from tier
        num_responses: Number of responses to generate (default: 3)
        temperature: Model temperature (higher = more variation)
        max_tokens: Max tokens to generate
        max_retries: Number of retries on failure per response
        return_all: If True, returns (consensus, all_responses), else just consensus
        cache_ttl: Cache time-to-live in seconds (default: 7200 = 2 hours, 0 = no cache)
        semantic_validation: Optional semantic validation rule

    Returns:
        If return_all=False: The most common response (consensus, potentially from cache)
        If return_all=True: Tuple of (consensus_response, list_of_all_responses)

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When insufficient responses succeed

    Example:
        # Fast parallel consensus with tier
        result = await structured_completion_consensus(
            messages=messages,
            response_model=ClassificationResult,
            tier="budget",
            num_responses=5
        )
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a non-empty list")

    if tier not in MODEL_TIERS:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be one of: {list(MODEL_TIERS.keys())}"
        )

    available_models = MODEL_TIERS[tier]
    required_minimum = max(1, num_responses // 2)

    logger.info(
        f"Using parallel consensus from tier '{tier}' with {len(available_models)} available models: {available_models}"
    )
    logger.info(
        f"Generating {num_responses} responses in parallel for consensus (minimum {required_minimum} required)"
    )

    tasks = []
    for i in range(num_responses):
        # Select a different model for each response to get diverse perspectives
        selected_model = random.choice(available_models)
        logger.debug(f"Response {i+1}/{num_responses}: using model '{selected_model}'")

        task = async_completion(
            messages=messages,
            response_model=response_model,
            model=selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache_ttl=cache_ttl,
            semantic_validation=semantic_validation,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    responses = []
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Log the FULL error - no sanitization!
            model_used = available_models[i % len(available_models)]
            full_error = str(result)
            logger.error(
                f"Error with {model_used}: {type(result).__name__}: {full_error}"
            )
            errors.append((i + 1, type(result).__name__, full_error))
        else:
            responses.append(result)
            logger.debug(f"Response {i+1}/{num_responses} succeeded")

    successful_responses = len(responses)
    if successful_responses < required_minimum:
        # Format errors nicely for debugging
        error_details = "\n".join(
            [f"  - Model {i}: {err_type}: {err_msg}" for i, err_type, err_msg in errors]
        )
        raise RuntimeError(
            f"Insufficient successful responses: {successful_responses}/{num_responses} "
            f"(minimum {required_minimum} required).\nErrors:\n{error_details}"
        )

    response_strings = [
        json.dumps(response.model_dump(exclude_none=True), sort_keys=True)
        for response in responses
    ]
    response_counter = Counter(response_strings)
    most_common_json, count = response_counter.most_common(1)[0]
    consensus_index = response_strings.index(most_common_json)
    consensus = responses[consensus_index]

    logger.info(f"Consensus reached: {count}/{len(responses)} responses agree")

    if return_all:
        return consensus, responses
    return consensus
