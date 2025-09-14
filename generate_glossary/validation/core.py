"""
Functional validation core module.

This module provides a pure functional interface for term validation
using immutable data structures and pure functions.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Mapping
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from types import MappingProxyType
from collections import defaultdict
import logging
import time
from contextlib import contextmanager

from .rule_validator import rule_validate
from .web_validator import web_validate
from .llm_validator import llm_validate
from .utils import normalize_term
from .cache import (
    CacheState, load_cache_from_disk, save_cache_to_disk,
    filter_cached_terms_functional, with_cache, with_cache_state, cache_set_validation,
    cache_get_validation, cache_get_rejected
)

# Type aliases
Terms = Union[str, List[str]]
ValidatorResult = Dict[str, Any]
ValidatorFn = Callable[[List[str]], Dict[str, ValidatorResult]]
WebContent = Dict[str, List[Dict[str, Any]]]


@dataclass(frozen=True)
class ValidationConfig:
    """Immutable validation configuration."""
    modes: Tuple[str, ...] = ("rule",)
    confidence_weights: Mapping[str, float] = None
    min_confidence: float = 0.5
    min_score: float = 0.5
    min_relevance_score: float = 0.5
    parallel: bool = True
    show_progress: bool = True
    llm_provider: str = "gemini"
    use_cache: bool = True
    max_workers_rule: int = 4
    max_workers_web: int = 4
    
    def __post_init__(self):
        """Set default confidence weights if not provided."""
        if self.confidence_weights is None:
            object.__setattr__(self, 'confidence_weights', MappingProxyType({
                "rule": 0.3,
                "web": 0.5,
                "llm": 0.2
            }))


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result."""
    term: str
    is_valid: bool
    confidence: float
    score: float
    relevance_score: Optional[float]
    rule_result: Optional[Mapping[str, Any]] = None
    web_result: Optional[Mapping[str, Any]] = None
    llm_result: Optional[Mapping[str, Any]] = None
    errors: Tuple[str, ...] = ()
    
    @property
    def combined_score(self) -> float:
        """
        Get the precomputed combined score from all validation modes.

        Returns the weighted score that was calculated during ValidationResult
        creation using calculate_combined_score() and stored in the score field.
        This property is a synonym for the score field and does NOT recalculate
        from mode results.

        For dynamic calculation from mode results, use calculate_combined_score()
        directly with appropriate ValidationConfig weights.

        Returns:
            float: The precomputed weighted score (same as self.score)

        Note:
            This property maintains consistency with the functional approach
            where scores are calculated once during result creation rather
            than on-demand to ensure immutability and performance.
        """
        return self.score


def normalize_terms(terms: Terms) -> List[str]:
    """Normalize input terms to a list of strings."""
    return [terms] if isinstance(terms, str) else list(terms)


def create_validator_functions(
    config: ValidationConfig,
    web_content: Optional[WebContent] = None
) -> Dict[str, ValidatorFn]:
    """Create validator functions based on configuration."""
    validators = {}

    if "rule" in config.modes:
        validators["rule"] = lambda terms_list: rule_validate(
            terms_list,
            max_workers=config.max_workers_rule,
            min_term_length=2,
            max_term_length=100,
            show_progress=config.show_progress
        )

    if "web" in config.modes:
        if web_content is None:
            logging.warning(
                "Web validation mode enabled but web_content is None. "
                "Web validation will be skipped. Provide web_content to enable web validation."
            )
        else:
            validators["web"] = lambda terms_list: web_validate(
                terms_list,
                web_content,
                min_score=config.min_score,
                min_relevance_score=config.min_relevance_score,
                min_relevant_sources=1,
                high_quality_content_threshold=0.7,
                high_quality_relevance_threshold=0.7,
                max_workers=config.max_workers_web,
                show_progress=config.show_progress
            )
    
    if "llm" in config.modes:
        # Import LLM completion function for dependency injection
        try:
            from generate_glossary.llm import completion
            validators["llm"] = lambda terms_list: llm_validate(
                terms_list, 
                completion, 
                provider=config.llm_provider,
                batch_size=10,
                max_workers=4,
                tier="budget",
                max_tokens=100,
                batch_max_tokens=500,
                show_progress=config.show_progress
            )
        except ImportError:
            logging.warning("LLM utilities not available, skipping LLM validation")
    
    return validators




def parallel_validate(
    terms_list: List[str],
    validators: Dict[str, ValidatorFn],
    config: ValidationConfig,
    timeout: Optional[float] = None
) -> Dict[str, Dict[str, ValidatorResult]]:
    """
    Execute validators in parallel with enhanced error handling and resource management.

    This function runs multiple validators concurrently to improve performance while
    providing robust timeout and error handling. Failed or timed-out validators
    gracefully return empty results rather than crashing the entire validation.

    Args:
        terms_list: List of terms to validate
        validators: Dictionary of validator functions (mode_name -> validator_func)
        config: Validation configuration containing parallel settings
        timeout: Optional timeout in seconds for individual validators.
                 If specified, each validator must complete within this time.
                 Timed-out validators return empty results and log warnings.

    Returns:
        Dictionary mapping mode names to validation results.
        Modes that fail or timeout return empty dictionaries.

    Timeout Behavior:
        - Each validator runs with the specified individual timeout
        - Timed-out validators are cancelled and return empty results
        - Timeout events are logged as warnings with validator details
        - Resource cleanup ensures no hanging threads or futures

    Error Handling:
        - Validators that raise exceptions return empty results
        - All errors are logged with full details
        - Other validators continue running despite individual failures
        - The function never raises exceptions from validator failures

    Performance:
        - Falls back to sequential execution if config.parallel=False
        - Limits concurrent validators to prevent resource exhaustion
        - Uses ThreadPoolExecutor with proper resource management

    Example:
        >>> validators = {"rule": rule_validator, "web": web_validator}
        >>> results = parallel_validate(
        ...     terms_list=["term1", "term2"],
        ...     validators=validators,
        ...     config=config,
        ...     timeout=60.0  # 60 second timeout per validator
        ... )
        >>> # results = {"rule": {...}, "web": {...}}
    """
    if not config.parallel or len(validators) == 1:
        return sequential_validate(terms_list, validators)
    
    results = {}
    max_workers = min(len(validators), 4)  # Limit concurrent validators
    
    with validation_context(f"Parallel validation with {len(validators)} validators"):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all validator tasks
            future_to_mode = {}
            for mode, validator_fn in validators.items():
                try:
                    future = executor.submit(validator_fn, terms_list)
                    future_to_mode[future] = mode
                except Exception as e:
                    logging.error(f"Failed to submit validator {mode}: {e}")
                    results[mode] = {}
            
            # Collect results with timeout support
            try:
                for future in as_completed(future_to_mode):
                    mode = future_to_mode[future]
                    try:
                        result = future.result(timeout=timeout)
                        results[mode] = result
                        logging.debug(f"Validator {mode} completed successfully")
                    except TimeoutError:
                        logging.warning(f"Validator {mode} timed out")
                        results[mode] = {}
                    except Exception as e:
                        logging.error(f"Validator {mode} failed: {e}")
                        results[mode] = {}
            except TimeoutError:
                # Handle global timeout - cancel remaining futures and set empty results
                logging.warning("Global timeout reached, canceling remaining validators")
                for future, mode in future_to_mode.items():
                    if not future.done():
                        future.cancel()
                        if mode not in results:
                            results[mode] = {}
    
    return results


def sequential_validate(
    terms_list: List[str],
    validators: Dict[str, ValidatorFn]
) -> Dict[str, Dict[str, ValidatorResult]]:
    """Execute validators sequentially and return combined results."""
    results = {}
    for mode, validator_fn in validators.items():
        try:
            results[mode] = validator_fn(terms_list)
        except Exception as e:
            logging.error(f"Validator {mode} failed: {e}")
            results[mode] = {}
    
    return results


def calculate_combined_confidence(
    mode_results: Dict[str, ValidatorResult],
    config: ValidationConfig
) -> float:
    """Calculate weighted confidence score from multiple validation modes."""
    total_weight = 0.0
    weighted_sum = 0.0
    
    for mode, result in mode_results.items():
        if mode in config.confidence_weights and 'confidence' in result:
            weight = config.confidence_weights[mode]
            confidence = result['confidence']
            weighted_sum += weight * confidence
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def calculate_combined_score(
    mode_results: Dict[str, ValidatorResult],
    config: ValidationConfig
) -> float:
    """Calculate weighted score from multiple validation modes."""
    total_weight = 0.0
    weighted_sum = 0.0
    
    for mode, result in mode_results.items():
        if mode in config.confidence_weights:
            weight = config.confidence_weights[mode]
            score = result.get('score')
            if score is None:
                if mode == 'web':
                    score = result.get('details', {}).get('avg_content_score')
                if score is None:
                    score = result.get('confidence')
            if score is not None:
                weighted_sum += weight * float(score)
                total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def extract_relevance_score(mode_results: Dict[str, ValidatorResult]) -> Optional[float]:
    """Extract relevance score from validation results."""
    for mode, result in mode_results.items():
        if 'relevance_score' in result:
            return result['relevance_score']
        if mode == 'web':
            val = result.get('details', {}).get('avg_relevance_score')
            if val is not None:
                return float(val)
    return None


def collect_errors(mode_results: Dict[str, ValidatorResult]) -> Tuple[str, ...]:
    """Collect all errors from validation results."""
    errors = []
    for mode, result in mode_results.items():
        if 'error' in result:
            errors.append(f"{mode}: {result['error']}")
        elif 'errors' in result and isinstance(result['errors'], list):
            for error in result['errors']:
                errors.append(f"{mode}: {error}")
        
        # Check details for additional errors
        d = result.get('details', {})
        if isinstance(d, dict):
            if 'error' in d and d['error']:
                errors.append(f"{mode}: {d['error']}")
            if 'errors' in d and isinstance(d['errors'], (list, tuple)):
                errors.extend(f"{mode}: {e}" for e in d['errors'])
    return tuple(errors)


def combine_results(
    results_by_mode: Dict[str, Dict[str, ValidatorResult]],
    conflict_resolution: str = "highest_confidence",
    merge_strategy: str = "union"
) -> Dict[str, Dict[str, ValidatorResult]]:
    """
    Combine validation results from multiple modes with sophisticated aggregation.
    
    Args:
        results_by_mode: Results organized by validation mode
        conflict_resolution: How to resolve conflicts ('highest_confidence', 'majority', 'weighted')
        merge_strategy: How to merge results ('union', 'intersection', 'weighted_union')
        
    Returns:
        Combined results organized by term
    """
    combined: Dict[str, Dict[str, ValidatorResult]] = defaultdict(dict)
    
    # Basic combination - collect all results per term
    for mode, per_term in results_by_mode.items():
        for term, result in per_term.items():
            combined[term][mode] = result
    
    # Apply conflict resolution and merge strategies
    if conflict_resolution == "highest_confidence":
        # Keep results from mode with highest confidence for each term
        for term, mode_results in combined.items():
            if len(mode_results) > 1:
                best_mode = max(
                    mode_results.keys(),
                    key=lambda m: mode_results[m].get('confidence', 0.0)
                )
                # Create copies of results to avoid mutating originals
                new_mode_results = {}
                for mode, result in mode_results.items():
                    result_copy = dict(result)
                    if mode == best_mode:
                        result_copy['is_primary'] = True
                    new_mode_results[mode] = result_copy
                combined[term] = new_mode_results
    
    elif conflict_resolution == "majority":
        # Use majority vote for validity determination
        for term, mode_results in combined.items():
            if len(mode_results) > 1:
                valid_votes = sum(
                    1 for result in mode_results.values()
                    if result.get('is_valid', False)
                )
                majority_valid = valid_votes > len(mode_results) / 2
                
                # Create copies of results to avoid mutating originals
                new_mode_results = {}
                for mode, result in mode_results.items():
                    result_copy = dict(result)
                    result_copy['majority_valid'] = majority_valid
                    new_mode_results[mode] = result_copy
                combined[term] = new_mode_results
    
    return dict(combined)


def create_validation_result(
    term: str,
    validation_results: Dict[str, Dict[str, ValidatorResult]],
    config: ValidationConfig
) -> ValidationResult:
    """Create a ValidationResult from raw validation results."""
    # Extract results for this term from each mode
    term_results = {}
    for mode, mode_data in validation_results.items():
        if term in mode_data:
            term_results[mode] = mode_data[term]
    
    # Calculate combined metrics
    confidence = calculate_combined_confidence(term_results, config)
    score = calculate_combined_score(term_results, config)
    relevance_score = extract_relevance_score(term_results)
    errors = collect_errors(term_results)
    
    # Determine validity based on thresholds
    is_valid = (
        confidence >= config.min_confidence and
        score >= config.min_score and
        (relevance_score is None or relevance_score >= config.min_relevance_score)
    )
    
    return ValidationResult(
        term=term,
        is_valid=is_valid,
        confidence=confidence,
        score=score,
        relevance_score=relevance_score,
        rule_result=MappingProxyType(term_results["rule"]) if "rule" in term_results else None,
        web_result=MappingProxyType(term_results["web"]) if "web" in term_results else None,
        llm_result=MappingProxyType(term_results["llm"]) if "llm" in term_results else None,
        errors=errors
    )


def validate_terms_functional(
    terms: Terms,
    config: Optional[ValidationConfig] = None,
    web_content: Optional[WebContent] = None,
    existing_results: Optional[Dict[str, ValidationResult]] = None,
    cache_state: Optional[CacheState] = None
) -> Dict[str, ValidationResult]:
    """
    Pure functional term validation.
    
    Args:
        terms: Terms to validate (string or list of strings)
        config: Validation configuration (uses defaults if None)
        web_content: Web content for web validation mode
        existing_results: Existing validation results to merge with
        cache_state: Optional cache state for caching results
        
    Returns:
        Dictionary mapping terms to ValidationResult objects
    """
    if config is None:
        config = ValidationConfig()
    
    # Normalize input terms
    terms_list = normalize_terms(terms)
    
    # Filter terms that already have results
    if existing_results:
        terms_list = [term for term in terms_list if term not in existing_results]
    
    # Early return if no terms to validate
    if not terms_list:
        return existing_results or {}
    
    # Create validator functions
    validators = create_validator_functions(config, web_content)
    
    if not validators:
        logging.warning("No validators configured")
        return existing_results or {}
    
    # Execute validation
    validation_results = parallel_validate(terms_list, validators, config)
    
    # Create immutable results
    results = {}
    for term in terms_list:
        results[term] = create_validation_result(term, validation_results, config)
    
    # Merge with existing results
    if existing_results:
        final_results = dict(existing_results)
        final_results.update(results)
        return final_results
    
    return results


def validate_terms_with_cache(
    terms: Terms,
    config: Optional[ValidationConfig] = None,
    web_content: Optional[WebContent] = None,
    existing_results: Optional[Dict[str, ValidationResult]] = None,
    cache_state: Optional[CacheState] = None,
    auto_save: bool = True
) -> Tuple[Dict[str, ValidationResult], Optional[CacheState]]:
    """
    Functional term validation with caching support.
    
    Args:
        terms: Terms to validate (string or list of strings)
        config: Validation configuration (uses defaults if None)
        web_content: Web content for web validation mode
        existing_results: Existing validation results to merge with
        cache_state: Cache state for caching results
        auto_save: Whether to automatically save cache to disk
        
    Returns:
        Tuple of (validation_results, updated_cache_state)
        If no cache_state provided, second element is None
    """
    if config is None:
        config = ValidationConfig()
    
    # Normalize input terms
    terms_list = normalize_terms(terms)
    
    if not terms_list:
        return existing_results or {}, cache_state
    
    # Filter terms that already have results in existing_results
    if existing_results:
        terms_list = [term for term in terms_list if term not in existing_results]
        # Early return if all terms are already processed
        if not terms_list:
            return existing_results, cache_state
    
    # If cache is provided and enabled, use cached validation
    if cache_state and config.use_cache:
        modes = list(config.modes)
        
        # Filter cached terms
        uncached_terms, cached_results = filter_cached_terms_functional(terms_list, modes, cache_state)
        
        # Convert cached results to ValidationResult objects
        validation_results = {}
        for term, cached_result in cached_results.items():
            # Restore per-mode details if they exist in cache
            rule_result = None
            web_result = None
            llm_result = None
            
            mode_results = cached_result.get("mode_results")
            if mode_results:
                if "rule" in mode_results:
                    rule_result = MappingProxyType(mode_results["rule"])
                if "web" in mode_results:
                    web_result = MappingProxyType(mode_results["web"])
                if "llm" in mode_results:
                    llm_result = MappingProxyType(mode_results["llm"])
            
            validation_results[term] = ValidationResult(
                term=term,
                is_valid=cached_result.get("is_valid", False),
                confidence=cached_result.get("confidence", 0.0),
                score=cached_result.get("score", cached_result.get("confidence", 0.0)),
                relevance_score=cached_result.get("relevance_score"),
                rule_result=rule_result,
                web_result=web_result,
                llm_result=llm_result,
                errors=tuple(cached_result.get("errors", [])) if cached_result.get("errors") else ()
            )
        
        # Validate uncached terms
        updated_cache_state = cache_state
        if uncached_terms:
            new_results = validate_terms_functional(
                uncached_terms, config, web_content
            )
            
            # Add new results to cache and merge with cached results
            for term, result in new_results.items():
                validation_results[term] = result
                # Convert ValidationResult to dict for cache storage
                mode_results = {}
                if result.rule_result:
                    mode_results["rule"] = dict(result.rule_result)
                if result.web_result:
                    mode_results["web"] = dict(result.web_result)
                if result.llm_result:
                    mode_results["llm"] = dict(result.llm_result)
                
                result_dict = {
                    "term": result.term,
                    "is_valid": result.is_valid,
                    "confidence": result.confidence,
                    "score": result.score,
                    "relevance_score": result.relevance_score,
                    "modes_used": modes,
                    "mode_results": mode_results,
                    "errors": list(result.errors) if result.errors else []
                }
                updated_cache_state = cache_set_validation(updated_cache_state, term, modes, result_dict)
        
        # Auto-save if requested
        if auto_save and updated_cache_state != cache_state:
            save_cache_to_disk(updated_cache_state)
        
        # Merge with existing results
        if existing_results:
            final_results = dict(existing_results)
            final_results.update(validation_results)
            return final_results, updated_cache_state
        
        return validation_results, updated_cache_state
    
    else:
        # No caching, use regular validation
        results = validate_terms_functional(terms_list, config, web_content, existing_results)
        return results, cache_state


def with_cache_support(
    validator_func: Callable[[Terms], Dict[str, ValidationResult]], 
    cache_state: CacheState, 
    modes: List[str],
    auto_save: bool = True
) -> Callable[[Terms], Tuple[Dict[str, ValidationResult], CacheState]]:
    """
    Add caching support to any validator function.
    
    This function delegates to with_cache() to avoid code duplication.
    It handles the conversion between ValidationResult objects and the dict format
    expected by the lower-level cache functions.
    
    Args:
        validator_func: Function that validates terms and returns ValidationResult objects
        cache_state: Current cache state
        modes: List of validation modes
        auto_save: Whether to automatically save cache to disk after updates
        
    Returns:
        Function that returns (validation_results, new_cache_state)
    
    Warning:
        This function captures the initial cache_state. For subsequent calls,
        use the returned updated state or consider using with_cache_state() instead.
    """
    # Adapter function to convert ValidationResult objects to dict format
    def dict_validator(terms_list: List[str]) -> Dict[str, Any]:
        validation_results = validator_func(terms_list)
        # Convert ValidationResult objects to dict format for cache compatibility
        dict_results = {}
        for term, result in validation_results.items():
            # Preserve mode_results if they exist
            mode_results = {}
            if result.rule_result:
                mode_results["rule"] = dict(result.rule_result)
            if result.web_result:
                mode_results["web"] = dict(result.web_result)
            if result.llm_result:
                mode_results["llm"] = dict(result.llm_result)

            dict_results[term] = {
                "term": result.term,
                "is_valid": result.is_valid,
                "confidence": result.confidence,
                "score": result.score,
                "relevance_score": result.relevance_score,
                "modes_used": modes,
                "mode_results": mode_results,
                "errors": list(result.errors) if result.errors else []
            }
        return dict_results
    
    # Get cached validator from with_cache
    cached_dict_validator = with_cache(dict_validator, cache_state, modes, auto_save)
    
    def cached_validator(terms: Terms) -> Tuple[Dict[str, ValidationResult], CacheState]:
        terms_list = normalize_terms(terms)
        
        # Get dict results and updated cache state
        dict_results, updated_cache_state = cached_dict_validator(terms_list)
        
        # Convert dict results back to ValidationResult objects
        validation_results = {}
        for term, dict_result in dict_results.items():
            # Restore mode_results if they exist
            rule_result = None
            web_result = None
            llm_result = None

            mode_results = dict_result.get("mode_results", {})
            if "rule" in mode_results:
                rule_result = MappingProxyType(mode_results["rule"])
            if "web" in mode_results:
                web_result = MappingProxyType(mode_results["web"])
            if "llm" in mode_results:
                llm_result = MappingProxyType(mode_results["llm"])

            validation_results[term] = ValidationResult(
                term=term,
                is_valid=dict_result.get("is_valid", False),
                confidence=dict_result.get("confidence", 0.0),
                score=dict_result.get("score", dict_result.get("confidence", 0.0)),
                relevance_score=dict_result.get("relevance_score"),
                rule_result=rule_result,
                web_result=web_result,
                llm_result=llm_result,
                errors=tuple(dict_result.get("errors", [])) if dict_result.get("errors") else ()
            )
        
        return validation_results, updated_cache_state
    
    return cached_validator


def compose_validators(
    validators: Dict[str, ValidatorFn],
    error_strategy: str = "skip",
    timeout: Optional[float] = None
) -> Callable[[List[str]], Dict[str, Dict[str, ValidatorResult]]]:
    """
    Compose multiple validator functions with error handling and timeout support.
    
    Args:
        validators: Dictionary mapping mode names to validator functions
        error_strategy: How to handle errors ('skip', 'fail', 'return_empty')
        timeout: Optional timeout in seconds for each validator
        
    Returns:
        Composed validator function
    """
    def composed(terms_list: List[str]) -> Dict[str, Dict[str, ValidatorResult]]:
        results = {}
        
        for mode, fn in validators.items():
            try:
                if timeout:
                    result = with_timeout(fn, timeout)(terms_list)
                else:
                    result = fn(terms_list)
                results[mode] = result
            except TimeoutError:
                logging.warning(f"Validator {mode} timed out after {timeout}s")
                if error_strategy == "fail":
                    raise
                elif error_strategy == "return_empty":
                    results[mode] = {}
                # skip strategy - don't add to results
            except Exception as e:
                logging.error(f"Validator {mode} failed: {e}")
                if error_strategy == "fail":
                    raise
                elif error_strategy == "return_empty":
                    results[mode] = {}
                # skip strategy - don't add to results
        
        return results
    return composed


def with_timeout(func: Callable, timeout_seconds: float) -> Callable:
    """
    Add timeout support to any function.

    Wraps a function to automatically timeout and cancel execution if it takes
    longer than the specified duration. This is essential for preventing
    validators from hanging indefinitely, especially with network or LLM calls.

    Args:
        func: The function to wrap with timeout support
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Wrapped function that raises TimeoutError if execution exceeds timeout

    Raises:
        TimeoutError: If function execution exceeds timeout_seconds

    Example:
        >>> timeout_validator = with_timeout(some_validator, 30.0)
        >>> results = timeout_validator(terms)  # Times out after 30 seconds

    Note:
        The wrapped function runs in a separate thread to enable cancellation.
        This adds minimal overhead but ensures clean resource cleanup.
    """
    def wrapped(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError:
                future.cancel()
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
    return wrapped


def with_retry(
    func: Callable, 
    max_retries: int = 3, 
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,)
) -> Callable:
    """Add retry logic to any function with exponential backoff."""
    def wrapped(*args, **kwargs):
        last_exception = None
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt == max_retries:
                    break
                
                logging.warning(f"Function {func.__name__} failed (attempt {attempt + 1}): {e}")
                time.sleep(current_delay)
                current_delay *= backoff_factor
        
        raise last_exception
    return wrapped


def create_validation_pipeline(
    config: ValidationConfig,
    web_content: Optional[WebContent] = None,
    timeout: Optional[float] = None,
    retry_config: Optional[Dict[str, Any]] = None
) -> Callable[[List[str]], Dict[str, ValidationResult]]:
    """
    Create a complex validation pipeline with error handling and retries.
    
    Args:
        config: Validation configuration
        web_content: Web content for web validation
        timeout: Optional timeout for individual validators
        retry_config: Optional retry configuration
        
    Returns:
        Pipeline function that validates terms and returns ValidationResult objects
    """
    # Create base validators
    validators = create_validator_functions(config, web_content)
    
    # Apply timeout if specified
    if timeout:
        validators = {
            mode: with_timeout(fn, timeout)
            for mode, fn in validators.items()
        }
    
    # Apply retry if specified
    if retry_config:
        validators = {
            mode: with_retry(fn, **retry_config)
            for mode, fn in validators.items()
        }
    
    # Compose validators
    composed_validator = compose_validators(validators, error_strategy="skip", timeout=timeout)
    
    def pipeline(terms_list: List[str]) -> Dict[str, ValidationResult]:
        """Execute the validation pipeline."""
        # Execute composed validator to get results by mode
        results_by_mode = composed_validator(terms_list)
        
        # Create ValidationResult objects
        results = {}
        for term in terms_list:
            results[term] = create_validation_result(term, results_by_mode, config)
        
        return results
    
    return pipeline


def conditional_validate(
    condition_func: Callable[[str], bool],
    validator_func: ValidatorFn,
    default_result: Optional[ValidatorResult] = None
) -> ValidatorFn:
    """Create a conditional validator that only validates terms meeting a condition."""
    def conditional(terms_list: List[str]) -> Dict[str, ValidatorResult]:
        filtered_terms = [term for term in terms_list if condition_func(term)]
        
        if not filtered_terms:
            if default_result:
                return {term: default_result for term in terms_list}
            return {}
        
        results = validator_func(filtered_terms)
        
        # Fill in default results for terms that didn't meet condition
        if default_result:
            for term in terms_list:
                if term not in results:
                    results[term] = default_result
        
        return results
    
    return conditional


def filter_results(
    results: Dict[str, ValidationResult],
    predicate: Callable[[ValidationResult], bool]
) -> Dict[str, ValidationResult]:
    """Filter validation results based on a predicate function."""
    return {
        term: result for term, result in results.items()
        if predicate(result)
    }


def transform_results(
    results: Dict[str, ValidationResult],
    transformer: Callable[[ValidationResult], ValidationResult]
) -> Dict[str, ValidationResult]:
    """Transform validation results using a transformer function."""
    return {
        term: transformer(result) for term, result in results.items()
    }


@contextmanager
def validation_context(description: str = "Validation"):
    """Context manager for validation operations with timing and error handling."""
    start_time = time.time()
    logging.info(f"Starting {description}")
    
    try:
        yield
        duration = time.time() - start_time
        logging.info(f"Completed {description} in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Failed {description} after {duration:.2f}s: {e}")
        raise


def filter_valid_terms(
    validation_results: Dict[str, ValidationResult]
) -> List[str]:
    """Extract only valid terms from validation results."""
    return [
        term for term, result in validation_results.items()
        if result.is_valid
    ]


def filter_invalid_terms(
    validation_results: Dict[str, ValidationResult]
) -> List[str]:
    """Extract only invalid terms from validation results."""
    return [
        term for term, result in validation_results.items()
        if not result.is_valid
    ]


def get_validation_summary(
    validation_results: Dict[str, ValidationResult]
) -> Dict[str, Any]:
    """Get summary statistics from validation results."""
    total_terms = len(validation_results)
    valid_terms = len(filter_valid_terms(validation_results))
    invalid_terms = total_terms - valid_terms
    
    if total_terms > 0:
        avg_confidence = sum(r.confidence for r in validation_results.values()) / total_terms
        avg_score = sum(r.score for r in validation_results.values()) / total_terms
    else:
        avg_confidence = avg_score = 0.0
    
    return {
        "total_terms": total_terms,
        "valid_terms": valid_terms,
        "invalid_terms": invalid_terms,
        "validity_rate": valid_terms / total_terms if total_terms > 0 else 0.0,
        "average_confidence": avg_confidence,
        "average_score": avg_score
    }