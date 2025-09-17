"""
Functional disambiguation core module.

This module provides a pure functional interface for disambiguation detection
using immutable data structures and pure functions. It replicates the patterns
from the validation system's functional core.
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

from .types import DetectionResult, DisambiguationConfig, EmbeddingConfig, HierarchyConfig, GlobalConfig
from .embedding_disambiguator import detect_embedding_ambiguity, with_embedding_model
from .hierarchy_disambiguator import detect_hierarchy_ambiguity
from .global_disambiguator import detect_global_ambiguity


# Type aliases for functional composition
Terms = Union[str, List[str]]
DetectionFn = Callable[[List[str]], List[DetectionResult]]
WebContent = Dict[str, Any]
HierarchyData = Dict[str, Any]


@dataclass(frozen=True)
class Success:
    """Success result wrapper for functional error handling."""
    value: List[DetectionResult]


@dataclass(frozen=True)
class Failure:
    """Failure result wrapper for functional error handling."""
    error: str
    exception: Optional[Exception] = None


# Result type alias for functional error handling
Result = Union[Success, Failure]


def is_success(result: Result) -> bool:
    """Check if result is a success."""
    return isinstance(result, Success)


def is_failure(result: Result) -> bool:
    """Check if result is a failure."""
    return isinstance(result, Failure)


def get_value(result: Result) -> List[DetectionResult]:
    """Extract value from success result."""
    if isinstance(result, Success):
        return result.value
    raise ValueError(f"Cannot get value from failure: {result.error}")


def get_error(result: Result) -> str:
    """Extract error from failure result."""
    if isinstance(result, Failure):
        return result.error
    raise ValueError("Cannot get error from success result")


def normalize_terms(terms: Terms) -> List[str]:
    """Normalize input terms to a list of strings."""
    return [terms] if isinstance(terms, str) else list(terms)


def safe_detect(
    detection_fn: DetectionFn,
    terms: List[str],
    *args,
    **kwargs
) -> Result:
    """
    Safely execute a detection function with error isolation.

    Args:
        detection_fn: Detection function to execute
        terms: List of terms to analyze
        *args: Positional arguments for detection function
        **kwargs: Keyword arguments for detection function

    Returns:
        Success with detection results or Failure with error information
    """
    try:
        results = detection_fn(terms, *args, **kwargs)
        return Success(results)
    except Exception as e:
        error_msg = f"Detection function {detection_fn.__name__} failed: {str(e)}"
        logging.error(error_msg)
        return Failure(error_msg, e)


def parallel_detect(
    detection_functions: Dict[str, Callable],
    terms: List[str],
    config: DisambiguationConfig,
    timeout: Optional[float] = None
) -> Dict[str, Result]:
    """
    Execute detection functions in parallel with enhanced error handling.

    This function runs multiple detection methods concurrently to improve performance
    while providing robust timeout and error handling. Failed or timed-out detectors
    gracefully return Failure results rather than crashing the entire detection.

    Args:
        detection_functions: Dictionary of detection functions (method_name -> function)
        terms: List of terms to analyze
        config: Disambiguation configuration
        timeout: Optional timeout in seconds for individual detectors

    Returns:
        Dictionary mapping method names to Result objects (Success or Failure)
    """
    if not config.parallel_processing or len(detection_functions) == 1:
        return sequential_detect(detection_functions, terms)

    results = {}
    max_workers = min(len(detection_functions), 4)  # Limit concurrent detectors

    with detection_context(f"Parallel detection with {len(detection_functions)} methods"):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all detection tasks
            future_to_method = {}
            for method, detection_fn in detection_functions.items():
                try:
                    future = executor.submit(detection_fn, terms)
                    future_to_method[future] = method
                except Exception as e:
                    logging.error(f"Failed to submit detector {method}: {e}")
                    results[method] = Failure(f"Failed to submit: {str(e)}", e)

            # Collect results with timeout support
            try:
                for future in as_completed(future_to_method):
                    method = future_to_method[future]
                    try:
                        result = future.result(timeout=timeout)
                        results[method] = Success(result)
                        logging.debug(f"Detector {method} completed successfully")
                    except TimeoutError:
                        error_msg = f"Detector {method} timed out"
                        logging.warning(error_msg)
                        results[method] = Failure(error_msg)
                    except Exception as e:
                        error_msg = f"Detector {method} failed: {str(e)}"
                        logging.error(error_msg)
                        results[method] = Failure(error_msg, e)
            except TimeoutError:
                # Handle global timeout - cancel remaining futures and set failures
                logging.warning("Global timeout reached, canceling remaining detectors")
                for future, method in future_to_method.items():
                    if not future.done():
                        future.cancel()
                        if method not in results:
                            results[method] = Failure("Global timeout reached")

    return results


def sequential_detect(
    detection_functions: Dict[str, Callable],
    terms: List[str]
) -> Dict[str, Result]:
    """Execute detection functions sequentially and return results."""
    results = {}
    for method, detection_fn in detection_functions.items():
        try:
            result = detection_fn(terms)
            results[method] = Success(result)
        except Exception as e:
            error_msg = f"Detector {method} failed: {str(e)}"
            logging.error(error_msg)
            results[method] = Failure(error_msg, e)

    return results


def combine_detection_results(
    results_by_method: Dict[str, Result],
    strategy: str = "union"
) -> List[DetectionResult]:
    """
    Combine detection results from multiple methods with sophisticated aggregation.

    Args:
        results_by_method: Results organized by detection method
        strategy: How to combine results ('union', 'intersection', 'weighted_union')

    Returns:
        Combined list of DetectionResult objects
    """
    # Extract successful results
    successful_results = {}
    for method, result in results_by_method.items():
        if is_success(result):
            successful_results[method] = get_value(result)
        else:
            logging.warning(f"Skipping failed method {method}: {get_error(result)}")

    if not successful_results:
        return []

    if strategy == "union":
        return _combine_union(successful_results)
    elif strategy == "intersection":
        return _combine_intersection(successful_results)
    elif strategy == "weighted_union":
        return _combine_weighted_union(successful_results)
    else:
        raise ValueError(f"Unknown combination strategy: {strategy}")


def _combine_union(results_by_method: Dict[str, List[DetectionResult]]) -> List[DetectionResult]:
    """Combine results using union strategy - include all unique terms."""
    combined = {}

    for method, results in results_by_method.items():
        for result in results:
            term = result.term
            if term not in combined:
                combined[term] = result
            else:
                # Keep result with higher confidence
                if result.confidence > combined[term].confidence:
                    combined[term] = result

    return list(combined.values())


def _combine_intersection(results_by_method: Dict[str, List[DetectionResult]]) -> List[DetectionResult]:
    """Combine results using intersection strategy - only terms detected by all methods."""
    if len(results_by_method) < 2:
        return list(results_by_method.values())[0] if results_by_method else []

    # Find terms detected by all methods
    term_sets = [set(result.term for result in results) for results in results_by_method.values()]
    common_terms = set.intersection(*term_sets)

    # For each common term, pick the result with highest confidence
    combined = {}
    for method, results in results_by_method.items():
        for result in results:
            if result.term in common_terms:
                if result.term not in combined or result.confidence > combined[result.term].confidence:
                    combined[result.term] = result

    return list(combined.values())


def _combine_weighted_union(results_by_method: Dict[str, List[DetectionResult]]) -> List[DetectionResult]:
    """Combine results using weighted union - average confidences across methods."""
    term_results = defaultdict(list)

    # Group results by term
    for method, results in results_by_method.items():
        for result in results:
            term_results[result.term].append(result)

    # Create combined results with averaged confidence
    combined = []
    for term, results in term_results.items():
        if len(results) == 1:
            combined.append(results[0])
        else:
            # Average confidences and combine evidence
            avg_confidence = sum(r.confidence for r in results) / len(results)

            # Use the result with highest individual confidence as base
            base_result = max(results, key=lambda r: r.confidence)

            # Combine evidence from all methods
            combined_evidence = {}
            for result in results:
                combined_evidence.update(result.evidence)

            # Create combined result
            combined_result = DetectionResult(
                term=term,
                method="combined",
                confidence=avg_confidence,
                evidence=MappingProxyType(combined_evidence),
                clusters=base_result.clusters,
                metadata=MappingProxyType({
                    "methods_used": [r.method for r in results],
                    "individual_confidences": [r.confidence for r in results]
                })
            )
            combined.append(combined_result)

    return combined


def with_timeout(func: Callable, timeout_seconds: float) -> Callable:
    """
    Add timeout support to any function.

    Args:
        func: The function to wrap with timeout support
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Wrapped function that raises TimeoutError if execution exceeds timeout
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


def compose_detectors(
    detectors: Dict[str, DetectionFn],
    error_strategy: str = "skip",
    timeout: Optional[float] = None
) -> Callable[[List[str]], Dict[str, Result]]:
    """
    Compose multiple detection functions with error handling and timeout support.

    Args:
        detectors: Dictionary mapping method names to detection functions
        error_strategy: How to handle errors ('skip', 'fail', 'return_empty')
        timeout: Optional timeout in seconds for each detector

    Returns:
        Composed detector function that returns Results
    """
    def composed(terms_list: List[str]) -> Dict[str, Result]:
        results = {}

        for method, fn in detectors.items():
            try:
                if timeout:
                    wrapped_fn = with_timeout(fn, timeout)
                    result = wrapped_fn(terms_list)
                else:
                    result = fn(terms_list)
                results[method] = Success(result)
            except TimeoutError:
                error_msg = f"Detector {method} timed out after {timeout}s"
                logging.warning(error_msg)
                if error_strategy == "fail":
                    raise
                elif error_strategy == "return_empty":
                    results[method] = Success([])
                else:  # skip strategy
                    results[method] = Failure(error_msg)
            except Exception as e:
                error_msg = f"Detector {method} failed: {str(e)}"
                logging.error(error_msg)
                if error_strategy == "fail":
                    raise
                elif error_strategy == "return_empty":
                    results[method] = Success([])
                else:  # skip strategy
                    results[method] = Failure(error_msg, e)

        return results
    return composed


def extract_method_configs(config: DisambiguationConfig) -> Tuple[EmbeddingConfig, HierarchyConfig, GlobalConfig]:
    """Extract method-specific configurations from main config."""
    return config.embedding_config, config.hierarchy_config, config.global_config


def validate_config_compatibility(config: DisambiguationConfig) -> bool:
    """Validate configuration consistency across methods."""
    # Check that enabled methods have corresponding configs
    for method in config.methods:
        if method == "embedding" and not config.embedding_config:
            return False
        elif method == "hierarchy" and not config.hierarchy_config:
            return False
        elif method == "global" and not config.global_config:
            return False
    return True


def apply_level_specific_params(config: DisambiguationConfig, level: int) -> DisambiguationConfig:
    """Apply level-specific parameters from profiles if available."""
    if level in config.level_configs:
        level_config = config.level_configs[level]
        # For now, return the config as-is since level-specific overrides
        # would require creating new config objects with updated parameters
        # This can be extended based on specific level override requirements
    return config


def create_embedding_pipeline(
    config: EmbeddingConfig,
    web_content: WebContent
) -> DetectionFn:
    """Build embedding detection pipeline with model injection."""
    # Create model-injected detection function
    detection_with_model = with_embedding_model(detect_embedding_ambiguity, config.model_name)

    def pipeline(terms: List[str]) -> List[DetectionResult]:
        return detection_with_model(terms, web_content, config)

    return pipeline


def create_hierarchy_pipeline(
    config: HierarchyConfig,
    web_content: Optional[WebContent],
    hierarchy: HierarchyData
) -> DetectionFn:
    """Build hierarchy detection pipeline."""
    def pipeline(terms: List[str]) -> List[DetectionResult]:
        return detect_hierarchy_ambiguity(terms, web_content, hierarchy, config)

    return pipeline


def create_global_pipeline(
    config: GlobalConfig,
    web_content: WebContent
) -> DetectionFn:
    """Build global detection pipeline with model injection."""
    # Create model-injected detection function
    detection_with_model = with_embedding_model(detect_global_ambiguity, config.model_name)

    def pipeline(terms: List[str]) -> List[DetectionResult]:
        return detection_with_model(terms, web_content, config)

    return pipeline


def create_hybrid_pipeline(
    embedding_config: Optional[EmbeddingConfig] = None,
    hierarchy_config: Optional[HierarchyConfig] = None,
    global_config: Optional[GlobalConfig] = None,
    web_content: Optional[WebContent] = None,
    hierarchy: Optional[HierarchyData] = None
) -> Callable[[List[str]], Dict[str, Result]]:
    """
    Create a multi-method detection pipeline.

    Args:
        embedding_config: Optional embedding detection configuration
        hierarchy_config: Optional hierarchy detection configuration
        global_config: Optional global detection configuration
        web_content: Web content for methods that require it
        hierarchy: Hierarchy data for hierarchy-based detection

    Returns:
        Pipeline function that returns detection results by method
    """
    detectors = {}

    if embedding_config and web_content:
        detectors["embedding"] = create_embedding_pipeline(embedding_config, web_content)

    if hierarchy_config and hierarchy:
        detectors["hierarchy"] = create_hierarchy_pipeline(hierarchy_config, web_content, hierarchy)

    if global_config and web_content:
        detectors["global"] = create_global_pipeline(global_config, web_content)

    return compose_detectors(detectors)


def create_detection_pipeline(
    terms: Terms,
    web_content: Optional[WebContent] = None,
    hierarchy: Optional[HierarchyData] = None,
    config: Optional[DisambiguationConfig] = None,
    timeout: Optional[float] = None,
    combination_strategy: str = "union"
) -> List[DetectionResult]:
    """
    Pure functional orchestration of disambiguation detection.

    This is the main entry point for functional disambiguation detection.
    It orchestrates all enabled detection methods and combines their results.

    Args:
        terms: Terms to analyze for ambiguity
        web_content: Web content for embedding and global methods
        hierarchy: Hierarchy data for hierarchy-based detection
        config: Disambiguation configuration (uses defaults if None)
        timeout: Optional timeout for individual detection methods
        combination_strategy: How to combine results from multiple methods

    Returns:
        List of DetectionResult objects for ambiguous terms
    """
    # Use default config if none provided
    if config is None:
        config = DisambiguationConfig()

    # Normalize input terms
    terms_list = normalize_terms(terms)

    if not terms_list:
        return []

    # Validate configuration
    if not validate_config_compatibility(config):
        raise ValueError("Invalid configuration: missing configs for enabled methods")

    # Extract method-specific configurations
    embedding_config, hierarchy_config, global_config = extract_method_configs(config)

    # Create detection pipeline based on enabled methods
    enabled_configs = {}
    if "embedding" in config.methods:
        enabled_configs["embedding"] = embedding_config
    if "hierarchy" in config.methods:
        enabled_configs["hierarchy"] = hierarchy_config
    if "global" in config.methods:
        enabled_configs["global"] = global_config

    # Create hybrid pipeline
    pipeline = create_hybrid_pipeline(
        embedding_config=enabled_configs.get("embedding"),
        hierarchy_config=enabled_configs.get("hierarchy"),
        global_config=enabled_configs.get("global"),
        web_content=web_content,
        hierarchy=hierarchy
    )

    # Execute detection pipeline
    if config.parallel_processing:
        results_by_method = pipeline(terms_list)
    else:
        # Force sequential execution by wrapping in sequential_detect
        detection_functions = {
            method: lambda t, m=method: pipeline(t)[m] for method in enabled_configs.keys()
        }
        results_by_method = sequential_detect(detection_functions, terms_list)

    # Combine results from all methods
    combined_results = combine_detection_results(results_by_method, combination_strategy)

    # Filter by minimum confidence
    filtered_results = [
        result for result in combined_results
        if result.confidence >= config.min_confidence
    ]

    return filtered_results


@contextmanager
def detection_context(description: str = "Detection"):
    """Context manager for detection operations with timing and error handling."""
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


def filter_ambiguous_terms(
    detection_results: List[DetectionResult]
) -> List[str]:
    """Extract only ambiguous terms from detection results."""
    return [result.term for result in detection_results]


def get_detection_summary(
    detection_results: List[DetectionResult]
) -> Dict[str, Any]:
    """Get summary statistics from detection results."""
    total_terms = len(detection_results)

    if total_terms == 0:
        return {
            "total_terms": 0,
            "methods_used": [],
            "average_confidence": 0.0,
            "confidence_distribution": {}
        }

    methods_used = list(set(result.method for result in detection_results))
    avg_confidence = sum(result.confidence for result in detection_results) / total_terms

    # Confidence distribution by method
    method_confidences = defaultdict(list)
    for result in detection_results:
        method_confidences[result.method].append(result.confidence)

    confidence_distribution = {
        method: {
            "count": len(confidences),
            "average": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences)
        }
        for method, confidences in method_confidences.items()
    }

    return {
        "total_terms": total_terms,
        "methods_used": methods_used,
        "average_confidence": avg_confidence,
        "confidence_distribution": confidence_distribution
    }