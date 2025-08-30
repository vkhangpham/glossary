"""
LLM utilities including cost tracking and retry logic.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

# LLM pricing (per 1K tokens)
LLM_PRICING = {
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    },
    "gemini": {
        "gemini-pro": {"input": 0.00025, "output": 0.0005},
        "gemini-1.5-pro": {"input": 0.0035, "output": 0.014},
    }
}

# Global cost tracker
_cost_tracker = {
    "total_cost": 0.0,
    "total_tokens": 0,
    "total_calls": 0,
    "cost_by_provider": {},
    "tokens_by_provider": {},
    "calls_by_provider": {}
}


def exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
    exponential_base: float = 2.0
):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logging.error(f"Failed after {max_retries + 1} attempts: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    logging.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Rough approximation: 1 token ≈ 4 characters
    """
    return len(text) // 4


def track_llm_cost(
    provider: str,
    model: str,
    input_text: str,
    output_text: str
) -> float:
    """
    Track LLM API call costs.
    
    Args:
        provider: LLM provider (openai, gemini)
        model: Model name
        input_text: Input prompt text
        output_text: Output response text
        
    Returns:
        Estimated cost in USD
    """
    # Estimate tokens
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    total_tokens = input_tokens + output_tokens
    
    # Get pricing
    provider_pricing = LLM_PRICING.get(provider, {})
    
    # Try to find exact model or use default
    model_pricing = None
    for model_key in provider_pricing:
        if model_key in model.lower():
            model_pricing = provider_pricing[model_key]
            break
    
    if not model_pricing:
        # Use conservative estimate if model not found
        model_pricing = {"input": 0.01, "output": 0.03}
        logging.warning(f"Unknown model {model} for {provider}, using default pricing")
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    # Update global tracker
    _cost_tracker["total_cost"] += total_cost
    _cost_tracker["total_tokens"] += total_tokens
    _cost_tracker["total_calls"] += 1
    
    # Track by provider
    if provider not in _cost_tracker["cost_by_provider"]:
        _cost_tracker["cost_by_provider"][provider] = 0.0
        _cost_tracker["tokens_by_provider"][provider] = 0
        _cost_tracker["calls_by_provider"][provider] = 0
    
    _cost_tracker["cost_by_provider"][provider] += total_cost
    _cost_tracker["tokens_by_provider"][provider] += total_tokens
    _cost_tracker["calls_by_provider"][provider] += 1
    
    # Log if cost is significant
    if total_cost > 0.01:  # Log if cost > 1 cent
        logging.info(
            f"LLM call cost: ${total_cost:.4f} "
            f"({input_tokens} in, {output_tokens} out tokens)"
        )
    
    return total_cost


def get_cost_summary() -> Dict[str, Any]:
    """Get summary of LLM costs."""
    return {
        "total_cost_usd": round(_cost_tracker["total_cost"], 4),
        "total_tokens": _cost_tracker["total_tokens"],
        "total_calls": _cost_tracker["total_calls"],
        "avg_cost_per_call": (
            round(_cost_tracker["total_cost"] / _cost_tracker["total_calls"], 4)
            if _cost_tracker["total_calls"] > 0 else 0
        ),
        "by_provider": {
            provider: {
                "cost_usd": round(cost, 4),
                "tokens": _cost_tracker["tokens_by_provider"].get(provider, 0),
                "calls": _cost_tracker["calls_by_provider"].get(provider, 0)
            }
            for provider, cost in _cost_tracker["cost_by_provider"].items()
        }
    }


def reset_cost_tracker():
    """Reset the cost tracker."""
    global _cost_tracker
    _cost_tracker = {
        "total_cost": 0.0,
        "total_tokens": 0,
        "total_calls": 0,
        "cost_by_provider": {},
        "tokens_by_provider": {},
        "calls_by_provider": {}
    }


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        """Initialize rate limiter."""
        self.calls_per_minute = calls_per_minute
        self.call_times = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # Check if we need to wait
        if len(self.call_times) >= self.calls_per_minute:
            # Calculate how long to wait
            oldest_call = min(self.call_times)
            wait_time = 60 - (now - oldest_call) + 0.1  # Add small buffer
            
            if wait_time > 0:
                logging.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        # Record this call
        self.call_times.append(now)