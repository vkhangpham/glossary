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
import hashlib
import pickle
import time
import atexit
import sys
from pathlib import Path
from typing import Optional, Type, Dict, List, Tuple, Union
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import instructor
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Singleton executor for run_async_safely to avoid creating new executor per call
_singleton_executor = ThreadPoolExecutor(max_workers=1)
from litellm.exceptions import (
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    Timeout,
    APIConnectionError,
)
from pydantic import BaseModel

from .logger import get_logger
from .failure_tracker import save_failure
from .optimization_tracker import record_llm_operation
from generate_glossary.config import get_llm_config

logger = get_logger("llm")

# Performance optimizations - Enable for 6.5x faster completion + 100+ RPS
try:
    litellm.set_verbose = False  # type: ignore # +50 RPS improvement by reducing logging overhead
except AttributeError:
    pass  # set_verbose might not be available in all versions
litellm.enable_json_schema_validation = True

# Experimental HTTP handler for +100 RPS on OpenAI calls
os.environ.setdefault("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")


# Model tier definitions and aliases are now centralized in config.py
# Functions below will read from get_llm_config() for consistency


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


class EnhancedCache:
    """Enhanced caching system with semantic similarity matching and persistent storage."""
    
    def __init__(self, storage_path: Optional[str] = None, enable_persistent: bool = True):
        self.storage_path = Path(storage_path) if storage_path else Path("data/cache")
        self.enable_persistent = enable_persistent
        self._lock = threading.Lock()
        self._memory_cache = {}  # In-memory cache for fast access
        self._cache_metadata = {}  # Metadata for cache entries
        
        if self.enable_persistent:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
    
    def _create_cache_key(self, messages: List[Dict], model: str, temperature: float, **kwargs) -> str:
        """Create a deterministic cache key from request parameters."""
        # Sort messages and kwargs for consistent hashing
        sorted_messages = json.dumps(messages, sort_keys=True)
        sorted_kwargs = json.dumps(sorted(kwargs.items()), sort_keys=True)
        
        # For batch requests, add normalized batch signature to strengthen matching
        batch_signature = ""
        response_model = kwargs.get("response_model")
        if response_model and "List" in str(response_model):
            # Extract batch items from messages for signature
            batch_items = []
            for msg in messages:
                content = msg.get("content", "")
                # Simple extraction - look for numbered items
                import re
                items = re.findall(r"\d+\.\s*(.+)", content)
                if items:
                    batch_items.extend(items)
            if batch_items:
                batch_signature = f"|batch:{hash(tuple(sorted(batch_items)))}"
        
        # Prefer tier over model for cache key when tier is available (for consensus consistency)
        tier = kwargs.get("tier")
        model_identifier = tier if tier else model
        
        content = f"{sorted_messages}|{model_identifier}|{temperature}|{sorted_kwargs}{batch_signature}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using simple heuristics."""
        # Simple similarity calculation (can be enhanced with embeddings in the future)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_semantic_match(self, messages: List[Dict], similarity_threshold: float) -> Optional[Tuple[str, any]]:
        """Find a cached entry that semantically matches the current request."""
        current_text = " ".join(msg.get("content", "") for msg in messages)
        
        for cache_key, metadata in self._cache_metadata.items():
            if cache_key not in self._memory_cache:
                continue
                
            cached_text = metadata.get("message_text", "")
            similarity = self._calculate_semantic_similarity(current_text, cached_text)
            
            if similarity >= similarity_threshold:
                return cache_key, self._memory_cache[cache_key]
        
        return None
    
    def get(self, messages: List[Dict], model: str, temperature: float, 
            enable_semantic: bool = True, semantic_threshold: float = 0.95, 
            **kwargs) -> Optional[Tuple[any, bool]]:
        """Get cached result, optionally using semantic matching."""
        with self._lock:
            # Try exact match first
            cache_key = self._create_cache_key(messages, model, temperature, **kwargs)
            
            if cache_key in self._memory_cache:
                metadata = self._cache_metadata.get(cache_key, {})
                if not self._is_expired(metadata):
                    record_llm_operation(
                        operation_type="cache_hit",
                        model=model,
                        response_time=0.0,
                        success=True,
                        cache_hit=True
                    )
                    return self._memory_cache[cache_key], False  # False = not semantic match
            
            # Try semantic matching if enabled
            # For batch requests, disable semantic matching to avoid unsafe hits
            response_model = kwargs.get("response_model")
            is_batch_request = response_model and "List" in str(response_model)
            
            if enable_semantic and not is_batch_request:
                semantic_match = self._find_semantic_match(messages, semantic_threshold)
                if semantic_match:
                    cache_key, result = semantic_match
                    metadata = self._cache_metadata.get(cache_key, {})
                    if not self._is_expired(metadata):
                        record_llm_operation(
                            operation_type="cache_hit",
                            model=model,
                            response_time=0.0,
                            success=True,
                            cache_hit=True,
                            semantic_cache_hit=True
                        )
                        return result, True  # True = semantic match
            
            return None
    
    def put(self, messages: List[Dict], model: str, temperature: float, 
            result: any, ttl: int = 3600, **kwargs):
        """Store result in cache with metadata."""
        with self._lock:
            cache_key = self._create_cache_key(messages, model, temperature, **kwargs)
            
            # Store in memory
            self._memory_cache[cache_key] = result
            
            # Store metadata
            message_text = " ".join(msg.get("content", "") for msg in messages)
            metadata = {
                "timestamp": time.time(),
                "ttl": ttl,
                "model": model,
                "temperature": temperature,
                "message_text": message_text[:500],  # Truncate for storage efficiency
                "message_count": len(messages)
            }
            
            # Add batch metadata for validation
            response_model = kwargs.get("response_model")
            if response_model and "List" in str(response_model):
                # Extract batch items for validation
                batch_items = []
                for msg in messages:
                    content = msg.get("content", "")
                    # Simple extraction - look for numbered items
                    import re
                    items = re.findall(r"\d+\.\s*(.+)", content)
                    if items:
                        batch_items.extend(items)
                metadata["batch_items"] = sorted(batch_items)[:10]  # Store first 10 for validation
                metadata["is_batch_request"] = True
            else:
                metadata["is_batch_request"] = False
            
            self._cache_metadata[cache_key] = metadata
            
            # Save to persistent storage if enabled
            if self.enable_persistent:
                self._save_to_persistent(cache_key, result, self._cache_metadata[cache_key])
    
    def _is_expired(self, metadata: Dict) -> bool:
        """Check if a cache entry has expired."""
        if not metadata:
            return True
        timestamp = metadata.get("timestamp", 0)
        ttl = metadata.get("ttl", 3600)
        return time.time() - timestamp > ttl
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        try:
            cache_file = self.storage_path / "cache_data.pkl"
            metadata_file = self.storage_path / "cache_metadata.json"
            
            if cache_file.exists() and metadata_file.exists():
                with open(cache_file, 'rb') as f:
                    self._memory_cache = pickle.load(f)
                
                with open(metadata_file, 'r') as f:
                    self._cache_metadata = json.load(f)
                
                # Remove expired entries
                expired_keys = [
                    key for key, metadata in self._cache_metadata.items()
                    if self._is_expired(metadata)
                ]
                
                for key in expired_keys:
                    self._memory_cache.pop(key, None)
                    self._cache_metadata.pop(key, None)
                
                logger.debug(f"Loaded {len(self._memory_cache)} cache entries from persistent storage")
                
        except Exception as e:
            logger.debug(f"Could not load persistent cache: {e}")
    
    def _save_to_persistent(self, cache_key: str, result: any, metadata: Dict):
        """Save single cache entry to persistent storage."""
        try:
            # For now, we'll save the entire cache periodically rather than per-entry
            # to avoid excessive I/O
            pass
        except Exception as e:
            logger.debug(f"Could not save to persistent cache: {e}")
    
    def save_all_to_persistent(self):
        """Save all cache data to persistent storage."""
        if not self.enable_persistent:
            return
        
        try:
            cache_file = self.storage_path / "cache_data.pkl"
            metadata_file = self.storage_path / "cache_metadata.json"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self._memory_cache, f)
            
            with open(metadata_file, 'w') as f:
                json.dump(self._cache_metadata, f, indent=2, default=str)
                
            logger.debug(f"Saved {len(self._memory_cache)} cache entries to persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._memory_cache)
            expired_count = sum(
                1 for metadata in self._cache_metadata.values()
                if self._is_expired(metadata)
            )
            
            return {
                "total_entries": total_entries,
                "active_entries": total_entries - expired_count,
                "expired_entries": expired_count,
                "memory_usage_mb": (
                    sys.getsizeof(self._memory_cache) + 
                    sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self._memory_cache.items())
                ) / (1024 * 1024)
            }
    
    def clear_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = [
                key for key, metadata in self._cache_metadata.items()
                if self._is_expired(metadata)
            ]
            
            for key in expired_keys:
                self._memory_cache.pop(key, None)
                self._cache_metadata.pop(key, None)
            
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")


# Global enhanced cache instance
_global_cache: Optional[EnhancedCache] = None
_cache_lock = threading.Lock()


def get_enhanced_cache() -> EnhancedCache:
    """Get the global enhanced cache instance."""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                llm_config = get_llm_config()
                storage_path = getattr(llm_config, 'cache_storage_path', 'data/cache')
                enable_persistent = getattr(llm_config, 'enable_persistent_cache', True)
                _global_cache = EnhancedCache(storage_path, enable_persistent)
                
                # Register cleanup function to save cache on exit
                if enable_persistent and not hasattr(get_enhanced_cache, '_exit_handler_registered'):
                    atexit.register(_save_cache_on_exit)
                    get_enhanced_cache._exit_handler_registered = True
    return _global_cache


def _save_cache_on_exit():
    """Save cache to persistent storage on process exit."""
    global _global_cache
    if _global_cache:
        try:
            _global_cache.save_all_to_persistent()
        except Exception as e:
            # Use print instead of logger since logging may be shutdown
            print(f"Warning: Could not save cache on exit: {e}")


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
    llm = get_llm_config()
    if tier not in llm.model_tiers:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be one of: {list(llm.model_tiers.keys())}"
        )

    models = llm.model_tiers[tier]
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
    llm = get_llm_config()
    return llm.model_aliases.get(model, model)


def validate_model(model: str) -> bool:
    """
    Validate if a model is in our supported tiers.

    Args:
        model: Model name to validate

    Returns:
        True if model is supported, False otherwise
    """
    resolved_model = resolve_model_alias(model)
    llm = get_llm_config()
    all_models = []
    for tier_models in llm.model_tiers.values():
        all_models.extend(tier_models)
    return resolved_model in all_models


def get_all_models() -> Dict[str, List[str]]:
    """
    Get all available models organized by tier.

    Returns:
        Dictionary with tiers as keys and model lists as values
    """
    llm = get_llm_config()
    return llm.model_tiers.copy()


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
    llm = get_llm_config()
    resolved_model = resolve_model_alias(model)

    if not validate_model(resolved_model):
        raise ValueError(f"Unsupported model: {model} (resolved to {resolved_model})")

    for tier, models in llm.model_tiers.items():
        if resolved_model in models:
            info = {
                "model": resolved_model,
                "tier": tier,
                "is_default": tier == llm.default_tier,
            }
            if model in llm.model_aliases:
                info["alias"] = model
            return info

    raise ValueError(f"Model {resolved_model} not found in any tier")


def load_prompt_from_file(filepath: Union[str, Path]) -> Optional[str]:
    """
    Load a prompt from a JSON file.
    
    Simple utility to load saved prompts. The file should contain
    a JSON object with a 'content' field. If the content is in DSPy format,
    extracts the instructions from it.
    
    Supports multiple JSON formats:
    1. Simplified format: {"content": "prompt text", "metadata": {...}}
    2. DSPy format: {"content": "ChainOfThought(instructions='...')"}
    3. Direct instructions: {"instructions": "prompt text"}
    4. Legacy direct content: "prompt text" (raw string in file)
    
    Args:
        filepath: Path to the prompt JSON file
        
    Returns:
        Prompt content if successful, None otherwise
    """
    try:
        path = Path(filepath)
        logger.debug(f"Attempting to load prompt from: {path.absolute()}")
        
        if not path.exists():
            logger.debug(f"Prompt file does not exist: {path}")
            return None
            
        with open(path, 'r', encoding='utf-8') as f:
            # Try to parse as JSON first
            try:
                data = json.load(f)
                logger.debug(f"Successfully parsed JSON with keys: {list(data.keys()) if isinstance(data, dict) else 'non-dict'}")
            except json.JSONDecodeError:
                # If not valid JSON, treat the entire file content as the prompt
                f.seek(0)
                content = f.read().strip()
                if content:
                    logger.debug(f"Loaded raw text content from {path} (non-JSON format)")
                    return content
                logger.warning(f"Empty or invalid content in {path}")
                return None
            
            # Handle different JSON structures
            if isinstance(data, str):
                # Direct string content (backward compatibility)
                logger.debug(f"Loaded direct string content from {path}")
                return data
                
            if not isinstance(data, dict):
                logger.warning(f"Unexpected data type in {path}: {type(data)}")
                return None
                
            # First check for direct instructions field in metadata (highest priority)
            if "metadata" in data and "instructions" in data["metadata"]:
                logger.debug(f"Found instructions in metadata for {path}")
                return data["metadata"]["instructions"]
            
            # Check for direct instructions field at root level
            if "instructions" in data:
                logger.debug(f"Found instructions at root level for {path}")
                return data["instructions"]
            
            # Check for nested DSPy format (single key with nested instructions)
            if len(data) == 1:
                nested = next(iter(data.values()))
                if isinstance(nested, dict) and 'instructions' in nested:
                    logger.debug(f"Found nested instructions in single-key DSPy-like format for {path}")
                    return nested['instructions']
            
            # Check for content field (simplified format)
            content = data.get("content") or data.get("prompt")
            if content is None:
                logger.debug(f"No 'content' or 'prompt' field found in {path}")
                return None
            if not isinstance(content, str):
                logger.warning(f"Content field is not a string in {path}; ignoring file")
                return None
            if not content.strip():
                logger.warning(f"Empty content string in {path}")
                return None
                
            # Check if this is DSPy format (contains "instructions=")
            if isinstance(content, str) and "instructions=" in content:
                logger.debug(f"Detected DSPy format in {path}, extracting instructions")
                # Extract instructions from DSPy format with more permissive regex
                import re
                
                # Try multiple regex patterns for different formatting styles
                patterns = [
                    # Standard single/double quoted with lookahead
                    r"instructions=['\"](.+?)['\"](?=\s*\n\s*\w+\s*=|$)",
                    # Triple quoted strings
                    r"instructions='''(.+?)'''",
                    r'instructions="""(.+?)"""',
                    # Parenthesis-wrapped strings (for multi-line)
                    r"instructions=\(['\"](.+?)['\"]\)",
                    # More permissive pattern without lookahead
                    r"instructions=['\"]([^'\"]+)['\"]",
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        instructions = match.group(1)
                        # Unescape the content
                        instructions = instructions.replace("\\'", "'")
                        instructions = instructions.replace('\\"', '"')
                        instructions = instructions.replace("\\n", "\n")
                        instructions = instructions.replace("\\\\n", "\n")
                        instructions = instructions.replace("\\t", "\t")
                        logger.debug(f"Successfully extracted DSPy instructions from {path}")
                        return instructions
                
                # If no pattern matched but we know there's instructions=, 
                # log a warning for debugging
                logger.warning(f"Found 'instructions=' in {filepath} but couldn't extract content")
            
            # Return content as-is if not DSPy format
            logger.debug(f"Loaded simplified format content from {path}")
            return content
            
    except (json.JSONDecodeError, KeyError, IOError) as e:
        logger.debug(f"Could not load prompt from {filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading prompt from {filepath}: {e}")
    
    return None


def get_contextual_config(use_case: Optional[str] = None) -> Tuple[str, float]:
    """Get model and temperature based on use case configuration."""
    llm_config = get_llm_config()
    
    if use_case and hasattr(llm_config, 'per_use_case_models'):
        model = llm_config.per_use_case_models.get(use_case)
        temperature = llm_config.per_use_case_temperatures.get(use_case, llm_config.temperature)
        if model:
            return model, temperature
    
    # Fallback to default configuration
    return get_model_by_tier(llm_config.default_tier), llm_config.temperature


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError, Timeout, APIConnectionError)),
    reraise=True
)
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
    use_case: Optional[str] = None,  # For contextual configuration
    enable_enhanced_cache: Optional[bool] = None,  # Override cache setting
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
        use_case: Optional use case for contextual configuration
        enable_enhanced_cache: Optional override for enhanced caching

    Returns:
        If response_model provided: Parsed Pydantic model instance (potentially from cache)
        If response_model is None: Raw text response (potentially from cache)

    Raises:
        ValueError: When input parameters are invalid or neither model nor tier provided
        RateLimitError: When rate limit is exceeded
        AuthenticationError: When API key is invalid
        BadRequestError: When request parameters are invalid

    Examples:
        # Structured output with validation and use case
        result = completion(messages, MyModel, use_case="concept_extraction")

        # Raw text output with tier
        text = completion(messages, tier="budget")
    """
    start_time = time.time()
    llm_config = get_llm_config()
    
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a non-empty list")

    # Apply use case configuration if available
    if use_case and not model and not tier:
        contextual_model, contextual_temp = get_contextual_config(use_case)
        model = contextual_model
        temperature = contextual_temp

    if tier and model:
        raise ValueError("Provide either 'model' or 'tier', not both")
    elif tier:
        model = get_model_by_tier(tier, random_selection=False)  # Deterministic for better caching
    elif model:
        model = resolve_model_alias(model)
    else:
        raise ValueError("Must provide either 'model' or 'tier'")

    if not model or "/" not in model:
        raise ValueError("Model must be in format 'provider/model'")
    if temperature < 0.0 or temperature > 2.0:
        raise ValueError("Temperature must be between 0.0 and 2.0")
    
    # Set temperature to 1.0 for GPT-5 models
    if "gpt-5" in model.lower():
        temperature = 1.0

    # Check enhanced cache if enabled
    cache_result = None
    semantic_cache_hit = False
    
    if enable_enhanced_cache is None:
        enable_enhanced_cache = getattr(llm_config, 'enable_enhanced_cache', True)
    
    if enable_enhanced_cache and cache_ttl > 0:
        enhanced_cache = get_enhanced_cache()
        cache_result = enhanced_cache.get(
            messages=messages,
            model=model,
            temperature=temperature,
            enable_semantic=True,
            semantic_threshold=getattr(llm_config, 'semantic_similarity_threshold', 0.95),
            response_model=response_model.__name__ if response_model else None,
            max_tokens=max_tokens,
            semantic_validation=semantic_validation,
            tier=tier  # Pass tier to cache for deterministic key creation
        )
        
        if cache_result:
            result, semantic_cache_hit = cache_result
            response_time = time.time() - start_time
            logger.debug(f"Cache hit for {model} ({'semantic' if semantic_cache_hit else 'exact'}) - {response_time:.3f}s")
            return result

    # Prepare API call parameters
    kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }

    if cache_ttl > 0 and not enable_enhanced_cache:
        # Only use litellm TTL caching if enhanced cache is disabled
        kwargs["ttl"] = cache_ttl

    if semantic_validation:
        kwargs["validation_context"] = semantic_validation

    try:
        # Use provider-specific client if semantic validation is needed, otherwise regular client
        if semantic_validation:
            client = LLMClient.get_provider_client(model)
        else:
            client = LLMClient.get_sync()

        # Make the API call
        result = client.chat.completions.create(  # type: ignore
            model=model, messages=messages, response_model=response_model, **kwargs  # type: ignore
        )
        
        # Record successful operation
        response_time = time.time() - start_time
        record_llm_operation(
            operation_type="single",
            model=model,
            response_time=response_time,
            success=True,
            context={
                "use_case": use_case,
                "cache_ttl": cache_ttl,
                "semantic_validation": bool(semantic_validation),
                "response_model": response_model.__name__ if response_model else None
            }
        )
        
        # Store in enhanced cache if enabled
        if enable_enhanced_cache and cache_ttl > 0:
            enhanced_cache = get_enhanced_cache()
            enhanced_cache.put(
                messages=messages,
                model=model,
                temperature=temperature,
                result=result,
                ttl=cache_ttl,
                response_model=response_model.__name__ if response_model else None,
                max_tokens=max_tokens,
                semantic_validation=semantic_validation,
                tier=tier  # Pass tier to cache for deterministic key creation
            )
        
        logger.debug(f"Successful completion with {model} - {response_time:.3f}s")
        return result

    except RateLimitError as e:
        response_time = time.time() - start_time
        logger.warning(f"Rate limit hit for {model}")
        record_llm_operation(
            operation_type="single",
            model=model,
            response_time=response_time,
            success=False,
            context={"error_type": "RateLimitError", "use_case": use_case}
        )
        save_failure(
            module="generate_glossary.utils.llm",
            function="completion",
            error_type="RateLimitError",
            error_message=str(e),
            context={"model": model, "messages_count": len(messages)}
        )
        raise
    except AuthenticationError as e:
        response_time = time.time() - start_time
        logger.error(f"Authentication failed for {model}")
        record_llm_operation(
            operation_type="single",
            model=model,
            response_time=response_time,
            success=False,
            context={"error_type": "AuthenticationError", "use_case": use_case}
        )
        save_failure(
            module="generate_glossary.utils.llm",
            function="completion",
            error_type="AuthenticationError",
            error_message=str(e),
            context={"model": model}
        )
        raise
    except BadRequestError as e:
        response_time = time.time() - start_time
        logger.error(f"Bad request to {model}")
        record_llm_operation(
            operation_type="single",
            model=model,
            response_time=response_time,
            success=False,
            context={"error_type": "BadRequestError", "use_case": use_case}
        )
        save_failure(
            module="generate_glossary.utils.llm",
            function="completion",
            error_type="BadRequestError",
            error_message=str(e),
            context={"model": model, "messages_count": len(messages)}
        )
        raise
    except Timeout as e:
        response_time = time.time() - start_time
        logger.warning(f"Request to {model} timed out")
        record_llm_operation(
            operation_type="single",
            model=model,
            response_time=response_time,
            success=False,
            context={"error_type": "Timeout", "use_case": use_case}
        )
        save_failure(
            module="generate_glossary.utils.llm",
            function="completion",
            error_type="Timeout",
            error_message=str(e),
            context={"model": model, "max_tokens": max_tokens}
        )
        raise
    except APIConnectionError as e:
        response_time = time.time() - start_time
        logger.error(f"Connection error to {model}")
        record_llm_operation(
            operation_type="single",
            model=model,
            response_time=response_time,
            success=False,
            context={"error_type": "APIConnectionError", "use_case": use_case}
        )
        save_failure(
            module="generate_glossary.utils.llm",
            function="completion",
            error_type="APIConnectionError",
            error_message=str(e),
            context={"model": model}
        )
        raise
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Unexpected error with {model}: {type(e).__name__}")
        record_llm_operation(
            operation_type="single",
            model=model,
            response_time=response_time,
            success=False,
            context={"error_type": type(e).__name__, "use_case": use_case}
        )
        save_failure(
            module="generate_glossary.utils.llm",
            function="completion",
            error_type=type(e).__name__,
            error_message=str(e),
            context={"model": model}
        )
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
    use_case: Optional[str] = None,  # For contextual configuration
    enable_enhanced_cache: Optional[bool] = None,  # Override cache setting
) -> Union[str, BaseModel]:
    """
    Async version of unified completion function - runs sync version in thread pool.

    All parameters and behavior are identical to completion(). This simply wraps
    the sync function to run in a thread pool for async compatibility.
    
    Note: Temperature is automatically set to 1.0 for GPT-5 models.

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
        use_case=use_case,
        enable_enhanced_cache=enable_enhanced_cache,
    )


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


def calculate_confidence_score(responses: List[BaseModel]) -> float:
    """Calculate confidence score based on response agreement."""
    if len(responses) < 2:
        return 1.0
    
    # Convert responses to comparable format
    response_strings = [
        json.dumps(response.model_dump(exclude_none=True), sort_keys=True)
        for response in responses
    ]
    
    response_counter = Counter(response_strings)
    most_common_count = response_counter.most_common(1)[0][1]
    
    # Confidence is the ratio of most common response
    return most_common_count / len(responses)


def calculate_agreement_score(responses: List[BaseModel]) -> float:
    """Calculate agreement score based on response diversity."""
    if len(responses) < 2:
        return 1.0
    
    response_strings = [
        json.dumps(response.model_dump(exclude_none=True), sort_keys=True)
        for response in responses
    ]
    
    unique_responses = len(set(response_strings))
    total_responses = len(responses)
    
    # Agreement score: higher when fewer unique responses
    return 1.0 - (unique_responses - 1) / (total_responses - 1)


async def smart_structured_completion_consensus(
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
    use_case: Optional[str] = None,  # For contextual configuration
    enable_enhanced_cache: Optional[bool] = None,  # Override cache setting
    enable_smart_consensus: Optional[bool] = None,  # Override smart consensus setting
) -> Union[BaseModel, Tuple[BaseModel, List[BaseModel]]]:
    """
    Generate multiple LLM responses with smart consensus and enhanced caching.

    Uses smart consensus to reduce API calls when early responses show high agreement.
    All responses are generated concurrently for maximum speed, with early stopping
    when consensus is reached above confidence threshold.

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
        use_case: Optional use case for contextual configuration
        enable_enhanced_cache: Optional override for enhanced caching
        enable_smart_consensus: Optional override for smart consensus

    Returns:
        If return_all=False: The most common response (consensus, potentially from cache)
        If return_all=True: Tuple of (consensus_response, list_of_all_responses)

    Raises:
        ValueError: When input parameters are invalid
        RuntimeError: When insufficient responses succeed

    Example:
        # Smart consensus with early stopping
        result = await smart_structured_completion_consensus(
            messages=messages,
            response_model=ClassificationResult,
            tier="budget",
            num_responses=5,
            use_case="concept_extraction"
        )
    """
    start_time = time.time()
    
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a non-empty list")

    llm_config = get_llm_config()
    if tier not in llm_config.model_tiers:
        raise ValueError(
            f"Invalid tier '{tier}'. Must be one of: {list(llm_config.model_tiers.keys())}"
        )

    # Get configuration settings
    if enable_smart_consensus is None:
        enable_smart_consensus = getattr(llm_config, 'enable_smart_consensus', True)
    
    confidence_threshold = getattr(llm_config, 'confidence_threshold', 0.85)
    agreement_threshold = getattr(llm_config, 'agreement_threshold', 0.8)
    min_responses = max(getattr(llm_config, 'min_responses', 2), 2)
    max_responses = min(num_responses, getattr(llm_config, 'max_responses', 3))

    available_models = llm_config.model_tiers[tier]
    required_minimum = max(1, min_responses)

    logger.info(
        f"Smart consensus from tier '{tier}' with {len(available_models)} models: {available_models}"
    )
    logger.info(
        f"Smart consensus enabled: {enable_smart_consensus}, "
        f"confidence threshold: {confidence_threshold}, "
        f"min responses: {min_responses}"
    )

    responses = []
    errors = []
    api_calls_made = 0
    api_calls_saved = 0
    
    # Start with minimum number of responses
    current_batch_size = min_responses
    
    while api_calls_made < max_responses and len(responses) < max_responses:
        # Determine how many more responses we need for this batch
        remaining_needed = min(current_batch_size - len(responses), max_responses - len(responses))
        
        if remaining_needed <= 0:
            break
            
        logger.debug(f"Requesting {remaining_needed} responses (total so far: {len(responses)})")
        
        # Create tasks for this batch
        tasks = []
        models_for_batch = []
        for i in range(remaining_needed):
            # Use deterministic model selection based on stable hash
            seed = int(hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(), 16)
            # Increment seed for subsequent attempts to ensure different models
            deterministic_index = (seed + api_calls_made + i) % len(available_models)
            selected_model = available_models[deterministic_index]
            models_for_batch.append(selected_model)
            logger.debug(f"Response {api_calls_made + i + 1}: using model '{selected_model}' (deterministic)")
            
            # Override temperature for GPT-5 models
            actual_temperature = 1.0 if "gpt-5" in selected_model.lower() else temperature

            task = async_completion(
                messages=messages,
                response_model=response_model,
                model=selected_model,
                tier=tier,  # Pass tier for cache key consistency
                temperature=actual_temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                cache_ttl=cache_ttl,
                semantic_validation=semantic_validation,
                use_case=use_case,
                enable_enhanced_cache=enable_enhanced_cache,
            )
            tasks.append(task)

        # Execute this batch of requests
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        api_calls_made += len(batch_results)
        
        # Process results from this batch
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                model_used = models_for_batch[i]
                full_error = str(result)
                logger.error(f"Error with {model_used}: {type(result).__name__}: {full_error}")
                errors.append((len(responses) + len(errors) + 1, type(result).__name__, full_error))
                
                save_failure(
                    module="generate_glossary.utils.llm",
                    function="smart_structured_completion_consensus",
                    error_type=type(result).__name__,
                    error_message=full_error,
                    context={
                        "model": model_used,
                        "tier": tier,
                        "attempt": len(responses) + len(errors),
                        "total_attempts": num_responses,
                        "smart_consensus": enable_smart_consensus
                    }
                )
            else:
                responses.append(result)
                logger.debug(f"Response {len(responses)} succeeded")

        # Check if we can stop early with smart consensus
        if enable_smart_consensus and len(responses) >= min_responses:
            confidence_score = calculate_confidence_score(responses)
            agreement_score = calculate_agreement_score(responses)
            
            logger.debug(f"After {len(responses)} responses: confidence={confidence_score:.3f}, agreement={agreement_score:.3f}")
            
            # Early stopping conditions
            if confidence_score >= confidence_threshold and agreement_score >= agreement_threshold:
                api_calls_saved = max_responses - api_calls_made
                logger.info(
                    f"Early consensus reached with {len(responses)} responses "
                    f"(confidence: {confidence_score:.3f}, agreement: {agreement_score:.3f}). "
                    f"Saved {api_calls_saved} API calls"
                )
                break
        
        # If we haven't reached consensus, prepare for next batch (if smart consensus enabled)
        if enable_smart_consensus and len(responses) >= min_responses:
            current_batch_size = min(current_batch_size + 1, max_responses)

    # Check if we have enough successful responses
    successful_responses = len(responses)
    if successful_responses < required_minimum:
        response_time = time.time() - start_time
        record_llm_operation(
            operation_type="consensus",
            model=f"tier-{tier}",
            response_time=response_time,
            success=False,
            api_calls_made=api_calls_made,
            api_calls_saved=api_calls_saved,
            context={"error": "insufficient_responses", "use_case": use_case}
        )
        
        error_details = "\n".join(
            [f"  - Model {i}: {err_type}: {err_msg}" for i, err_type, err_msg in errors]
        )
        raise RuntimeError(
            f"Insufficient successful responses: {successful_responses}/{num_responses} "
            f"(minimum {required_minimum} required).\nErrors:\n{error_details}"
        )

    # Find consensus response
    response_strings = [
        json.dumps(response.model_dump(exclude_none=True), sort_keys=True)
        for response in responses
    ]
    response_counter = Counter(response_strings)
    most_common_json, count = response_counter.most_common(1)[0]
    consensus_index = response_strings.index(most_common_json)
    consensus = responses[consensus_index]

    # Calculate final scores
    final_confidence = calculate_confidence_score(responses)
    final_agreement = calculate_agreement_score(responses)
    
    response_time = time.time() - start_time
    optimization_type = "smart_consensus" if api_calls_saved > 0 else None
    
    # Record the consensus operation
    record_llm_operation(
        operation_type="consensus",
        model=f"tier-{tier}",
        response_time=response_time,
        success=True,
        optimization_type=optimization_type,
        api_calls_made=api_calls_made,
        api_calls_saved=api_calls_saved,
        confidence_score=final_confidence,
        agreement_score=final_agreement,
        context={
            "use_case": use_case,
            "total_responses": len(responses),
            "consensus_count": count,
            "smart_consensus_enabled": enable_smart_consensus
        }
    )

    logger.info(
        f"Consensus reached: {count}/{len(responses)} responses agree "
        f"(confidence: {final_confidence:.3f}, agreement: {final_agreement:.3f}, "
        f"API calls: {api_calls_made}, saved: {api_calls_saved})"
    )

    if return_all:
        return consensus, responses
    return consensus


# Legacy function name for backward compatibility
async def structured_completion_consensus(*args, **kwargs):
    """Legacy wrapper for smart_structured_completion_consensus."""
    return await smart_structured_completion_consensus(*args, **kwargs)
