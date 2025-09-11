"""
Helper functions for model resolution, prompt utilities, and async support.

This module contains utility functions for model name resolution, prompt processing,
template formatting, and safe async execution.
"""

import asyncio
import functools
import logging
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import dspy

from generate_glossary.config import get_llm_config
from generate_glossary.llm.dspy_config import _ensure_dspy_configured

# Singleton thread pool for async utilities
_singleton_executor = ThreadPoolExecutor(max_workers=1)

logger = logging.getLogger(__name__)


def _is_gpt5_family(model: str) -> bool:
    """
    Check if model is from GPT-5 family.
    
    Args:
        model: Model string to check
        
    Returns:
        True if model is GPT-5 family
    """
    return "gpt-5" in model.lower()


def _is_likely_openai_model(model: str, config) -> bool:
    """
    Check if model is likely from OpenAI based on config and common patterns.
    
    Args:
        model: Model string to check
        config: LLM configuration object
        
    Returns:
        True if model is likely OpenAI
    """
    # Check config-driven OpenAI prefixes first
    openai_model_prefixes = getattr(config, "openai_model_prefixes", [])
    model_lower = model.lower()
    
    for pattern in openai_model_prefixes:
        if model_lower.startswith(pattern.lower()):
            return True
    
    # Common OpenAI patterns as fallback
    common_openai_patterns = ["gpt-", "text-davinci-", "text-curie-", "text-babbage-", "text-ada-", "whisper-", "o1", "embedding-"]
    for pattern in common_openai_patterns:
        if model_lower.startswith(pattern):
            return True
    
    return False


def _normalize_model_name(model: str, config) -> str:
    """
    Ensure model name is fully qualified with provider prefix using config-driven approach.
    
    Args:
        model: Model string that may or may not have provider prefix
        config: LLM configuration object
        
    Returns:
        Fully qualified model string with provider prefix
    """
    # If already has provider prefix, return as is
    if "/" in model:
        return model
    
    # First check config-driven aliases for better maintainability
    model_aliases = getattr(config, "model_aliases", {})
    if model in model_aliases:
        aliased_model = model_aliases[model]
        # If alias is fully qualified, return it; otherwise, continue processing
        if "/" in aliased_model:
            return aliased_model
        model = aliased_model  # Use alias for further processing
    
    # If still unprefixed after alias resolution, check provider prefix mappings
    model_provider_prefixes = getattr(config, "model_provider_prefixes", {})
    for prefix, provider_id in model_provider_prefixes.items():
        if model.startswith(prefix):
            return f"{provider_id}{model}"
    
    # Check if model matches known OpenAI patterns from config
    openai_model_prefixes = getattr(config, "openai_model_prefixes", [])
    model_lower = model.lower()
    for pattern in openai_model_prefixes:
        if model_lower.startswith(pattern):
            return f"openai/{model}"
    
    # For truly unknown patterns, return as-is and let DSPy handle it
    return model


def _resolve_model(model: Optional[str], tier: Optional[str], use_case: Optional[str]) -> str:
    """
    Resolve model string from tier or explicit model name.
    
    Args:
        model: Explicit model name (e.g., "openai/gpt-4o")
        tier: Model tier name (e.g., "budget", "performance") 
        use_case: Use case for per-case model selection
        
    Returns:
        Fully qualified model string in format "provider/model"
        
    Raises:
        ValueError: If both model and tier are provided, or neither is provided when use_case is not used
    """
    config = get_llm_config()
    
    # Handle use-case specific model selection first
    per_use_case_models = getattr(config, "per_use_case_models", {})
    if use_case and use_case in per_use_case_models:
        resolved_model = per_use_case_models[use_case]
        return _normalize_model_name(resolved_model, config)
    
    # Validate that exactly one of model or tier is set when use_case is not used
    if model and tier:
        raise ValueError("Cannot specify both 'model' and 'tier' parameters")
    if not model and not tier:
        # Try default tier fallback when both are None
        default_tier = getattr(config, "default_tier", None)
        if default_tier and default_tier in getattr(config, "model_tiers", {}):
            tier = default_tier
        else:
            raise ValueError("Must specify either 'model' or 'tier' parameter")
    
    # Handle explicit model specification
    if model:
        # Resolve alias if present
        model_aliases = getattr(config, "model_aliases", {})
        if model in model_aliases:
            resolved_model = model_aliases[model]
        else:
            resolved_model = model
        return _normalize_model_name(resolved_model, config)
    
    # Handle tier-based selection
    model_tiers = getattr(config, "model_tiers", {})
    if tier not in model_tiers:
        raise ValueError(f"Unknown tier: {tier}. Available tiers: {list(model_tiers.keys())}")
    
    tier_models = model_tiers[tier]
    if not tier_models:
        raise ValueError(f"No models configured for tier: {tier}")
    
    # Check for deterministic selection
    deterministic_tier_selection = getattr(config, "deterministic_tier_selection", False)
    if deterministic_tier_selection:
        random_seed = getattr(config, "random_seed", None)
        if random_seed is not None:
            # Create a seeded random instance for deterministic selection
            rng = random.Random(random_seed)
            selected_model = rng.choice(tier_models)
        else:
            # Deterministic selection without seed - use first model
            selected_model = tier_models[0]
    else:
        # Random selection from tier models (default behavior)
        selected_model = random.choice(tier_models)
    
    # Resolve alias if present
    model_aliases = getattr(config, "model_aliases", {})
    if selected_model in model_aliases:
        resolved_model = model_aliases[selected_model]
    else:
        resolved_model = selected_model
    
    return _normalize_model_name(resolved_model, config)


def _resolve_temperature(model: str, temperature: Optional[float], use_case: Optional[str]) -> float:
    """
    Resolve temperature setting for model.
    
    Args:
        model: Model string
        temperature: Explicit temperature value
        use_case: Use case for per-case temperature selection
        
    Returns:
        Temperature value to use
    """
    config = get_llm_config()
    
    # Handle GPT-5 temperature override using helper function
    if _is_gpt5_family(model):
        return 1.0
    
    # Handle use-case specific temperature
    per_use_case_temperatures = getattr(config, "per_use_case_temperatures", {})
    if use_case and use_case in per_use_case_temperatures:
        return per_use_case_temperatures[use_case]
    
    # Use provided temperature or default
    default_temperature = getattr(config, "temperature", 1.0)
    return temperature if temperature is not None else default_temperature


@functools.lru_cache(maxsize=32)
def _get_lm(model: str, temperature: float, max_tokens: Optional[int], use_case: Optional[str] = None) -> dspy.LM:
    """
    Get cached DSPy LM instance with enhanced GPT-5 handling.
    
    Args:
        model: Model string (e.g., "openai/gpt-4o")
        temperature: Temperature setting
        max_tokens: Maximum tokens (optional)
        use_case: Use case for specialized token limits
        
    Returns:
        Configured DSPy LM instance
    """
    try:
        kwargs = {
            "model": model,
            "temperature": temperature,
            "cache": False,  # Disable DSPy's built-in caching
        }
        
        # Handle GPT-5 specific requirements
        if _is_gpt5_family(model):
            kwargs["temperature"] = 1.0  # Always use 1.0 for GPT-5
            
            # Set appropriate token limits for GPT-5
            if max_tokens is None:
                if use_case and "reflection" in use_case.lower():
                    kwargs["max_tokens"] = 32000  # Higher limit for reflection tasks
                else:
                    kwargs["max_tokens"] = 16000  # Default for generation tasks
            else:
                kwargs["max_tokens"] = max_tokens
        elif max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        logger.debug(f"Creating LM instance: {model} with {kwargs}")
        return dspy.LM(**kwargs)
        
    except Exception as e:
        logger.error(f"Failed to create LM instance for {model}: {e}")
        raise


def _safe_format_template(template: str, variables: Dict[str, str]) -> str:
    """
    Safely format a template string, leaving unmatched variables intact.
    
    Args:
        template: Template string with {variable} placeholders
        variables: Dictionary of variable mappings
        
    Returns:
        Formatted string with matched variables substituted
        
    Examples:
        template = "Process {text} and analyze {unknown}"
        variables = {"text": "input data"}
        # Returns: "Process input data and analyze {unknown}"
    """
    class SafeFormatter(string.Formatter):
        def get_value(self, key, args, kwargs):
            if isinstance(key, str):
                try:
                    return kwargs[key]
                except KeyError:
                    return '{' + key + '}'
            else:
                return string.Formatter.get_value(key, args, kwargs)
    
    formatter = SafeFormatter()
    return formatter.format(template, **variables)


def _extract_prompt_variables(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Extract variables from messages for template substitution.
    
    Args:
        messages: Original messages list
        
    Returns:
        Dictionary of variable mappings for template substitution
    """
    variables = {}
    
    # Combine all message content for common variables
    all_content = []
    for message in messages:
        content = message.get("content", "")
        if content and isinstance(content, str):
            all_content.append(content.strip())
    
    combined_content = "\n\n".join(all_content)
    
    # Common template variables
    variables["text"] = combined_content
    variables["input"] = combined_content
    variables["content"] = combined_content
    
    # Look for user messages specifically
    user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    if user_messages:
        variables["user_input"] = "\n\n".join(user_messages)
    
    return variables


def _strip_role_markers(text: str) -> str:
    """
    Strip leading role markers from prompt text to avoid duplication.
    
    Args:
        text: Text that may contain role markers like "System:" or "User:"
        
    Returns:
        Text with leading role markers removed
        
    Examples:
        "System: You are helpful" -> "You are helpful"
        "User: Process this text" -> "Process this text"
        "Regular text" -> "Regular text"
    """
    if not text:
        return text
    
    text = text.strip()
    
    # Common role marker patterns
    role_patterns = [
        r'^System:\s*',
        r'^User:\s*',
        r'^Assistant:\s*',
        r'^Human:\s*',
        r'^AI:\s*',
        r'^Bot:\s*'
    ]
    
    for pattern in role_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def _apply_optimized_prompts(
    messages: List[Dict[str, str]], 
    system_prompt: Optional[str], 
    user_prompt_template: Optional[str]
) -> str:
    """
    Apply optimized prompts to create final prompt string.
    
    Strips any existing role markers from prompts to avoid duplication like
    "System: System: ..." when prompts already contain role prefixes.
    
    Args:
        messages: Original messages
        system_prompt: Optimized system prompt
        user_prompt_template: Optimized user prompt template
        
    Returns:
        Combined prompt string for DSPy predictor
    """
    parts = []
    
    # Add system prompt if available, stripping any existing role markers
    if system_prompt:
        cleaned_system = _strip_role_markers(system_prompt)
        parts.append(f"System: {cleaned_system}")
    
    # Process user prompt template if available
    if user_prompt_template:
        # Extract variables for template substitution
        variables = _extract_prompt_variables(messages)
        logger.debug(f"Template variables available: {list(variables.keys())}")
        
        try:
            # Safe template substitution that leaves unmatched variables intact
            formatted_user_prompt = _safe_format_template(user_prompt_template, variables)
            # Strip role markers from the formatted prompt
            cleaned_user = _strip_role_markers(formatted_user_prompt)
            parts.append(f"User: {cleaned_user}")
        except (KeyError, ValueError) as e:
            # Fallback to original messages if template substitution fails
            logger.warning(f"Template substitution failed: {e}. Falling back to original messages.")
            return _join_messages(messages)
    else:
        # No user template, use original user messages
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        for msg in user_messages:
            content = msg.get("content", "")
            if content:
                parts.append(f"User: {content}")
    
    # If no optimized prompts were used, fallback to original
    if not parts:
        return _join_messages(messages)
    
    return "\n\n".join(parts)


def _join_messages(messages: List[Dict[str, str]]) -> str:
    """
    Convert OpenAI-style messages to single prompt string.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Joined prompt string
    """
    parts = []
    
    for message in messages:
        # Safely handle role and content fields
        raw_role = message.get("role")
        raw_content = message.get("content")
        
        # Convert None to empty string and ensure strings before processing
        role = raw_role if isinstance(raw_role, str) else ""
        content = raw_content if isinstance(raw_content, str) else ""
        
        # Process role and content safely
        role = role.lower().strip()
        content = content.strip()
        
        # Skip entries with empty or whitespace-only content
        if not content:
            continue
        
        # Handle recognized roles
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role:
            # Unrecognized but non-empty role - include content
            parts.append(content)
        else:
            # Empty role - just include content
            parts.append(content)
    
    return "\n\n".join(parts)


@contextmanager
def with_lm_context(
    model: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_case: Optional[str] = None
):
    """
    Context manager for temporary LM switching.
    
    Args:
        model: Model name to use in context
        tier: Model tier for automatic selection (mutually exclusive with model)
        temperature: Temperature setting
        max_tokens: Maximum tokens
        use_case: Use case for specialized configuration
        
    Yields:
        Context with configured LM
        
    Examples:
        with with_lm_context(model="openai/gpt-4o", temperature=0.5):
            result = completion(messages)
        with with_lm_context(tier="flagship", temperature=0.5):
            result = completion(messages)
    """
    if model is None and tier is None:
        # No specific model or tier, just yield current context
        yield
        return
    
    # Ensure DSPy is configured
    _ensure_dspy_configured()
    
    # Resolve model and temperature
    config = get_llm_config()
    if model is None and tier is not None:
        resolved_model = _resolve_model(None, tier, use_case)
    else:
        resolved_model = _normalize_model_name(model, config)
    resolved_temp = temperature if temperature is not None else _resolve_temperature(resolved_model, None, use_case)
    
    # Create LM instance for context
    lm = _get_lm(resolved_model, resolved_temp, max_tokens, use_case)
    
    # Use DSPy context manager
    with dspy.context(lm=lm):
        yield


def is_async_context() -> bool:
    """
    Check if code is running in an async context with an event loop.
    
    Returns:
        True if an event loop is running, False otherwise
    """
    try:
        loop = asyncio.get_running_loop()
        return loop is not None and loop.is_running()
    except RuntimeError:
        return False


def run_async_safely(async_func, *args, **kwargs):
    """
    Safely run an async function from sync context without event loop conflicts.
    
    This function detects if there's already a running event loop and uses
    a thread pool executor to run the async function in a new thread when needed.
    This prevents "asyncio.run() cannot be called from a running event loop" errors.
    
    Args:
        async_func: The async function to run
        *args: Positional arguments for the async function
        **kwargs: Keyword arguments for the async function
        
    Returns:
        The result of the async function
        
    Examples:
        # From sync context without event loop
        result = run_async_safely(my_async_func, arg1, arg2, kwarg1=value1)
        
        # From sync context with existing event loop (e.g., Jupyter)
        result = run_async_safely(my_async_func, arg1, arg2)
    """
    if is_async_context():
        # We're in an async context, use singleton thread pool to isolate
        future = _singleton_executor.submit(asyncio.run, async_func(*args, **kwargs))
        return future.result()
    else:
        # No event loop running, safe to use asyncio.run directly
        return asyncio.run(async_func(*args, **kwargs))