"""URL processing and optimization module.

This module provides intelligent URL discovery, filtering, deduplication, and caching
for the Firecrawl-based mining system. It includes domain classification, academic
content filtering, and performance optimization through caching.
"""

import time
import asyncio
import concurrent.futures
import logging
import threading
import weakref
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse, parse_qs

from .config import ConfigError, get_firecrawl_client
from generate_glossary.utils.error_handler import (
    ExternalServiceError,
    handle_error,
    processing_context,
)
from generate_glossary.utils.logger import (
    get_logger,
    log_processing_step,
    log_with_context,
)


logger = get_logger(__name__)

# Simple in-memory cache for mapping results
_mapping_cache: Dict[str, Dict[str, Any]] = {}

# Thread-safe access to shared objects across threads
_thread_lock = threading.Lock()

# Per-event-loop lock storage for thread-safe Firecrawl client access
# Uses WeakKeyDictionary to automatically clean up locks when loops are garbage-collected
_loop_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)


def _get_firecrawl_lock() -> asyncio.Lock:
    """Get or create the firecrawl lock for the current event loop with thread safety."""
    loop = asyncio.get_running_loop()

    with _thread_lock:
        if loop not in _loop_locks:
            _loop_locks[loop] = asyncio.Lock()
        return _loop_locks[loop]


def _resolve_firecrawl_client(
    app: Optional[Any],
    operation: str,
    correlation_id: Optional[str] = None,
) -> Optional[Any]:
    """Return an initialized Firecrawl client, handling configuration failures."""

    if app is not None:
        return app

    try:
        return get_firecrawl_client()
    except ConfigError as exc:
        handle_error(
            ExternalServiceError(str(exc), service="firecrawl"),
            context={"operation": operation},
            operation=operation,
        )
        log_with_context(
            logger,
            logging.ERROR,
            f"Unable to acquire Firecrawl client: {exc}",
            correlation_id=correlation_id,
        )
        return None


async def _map_urls_with_lock(
    app: Any, domain: str, limit: Optional[int] = None
) -> List[str]:
    """Async wrapper for map_urls_fast_enhanced with semaphore-based concurrency control."""
    # Use semaphore instead of lock to avoid blocking other coroutines
    # This allows multiple concurrent operations while still limiting resource usage
    semaphore = getattr(_map_urls_with_lock, "_semaphore", None)
    if semaphore is None:
        _map_urls_with_lock._semaphore = asyncio.Semaphore(
            5
        )  # Allow 5 concurrent operations
        semaphore = _map_urls_with_lock._semaphore

    async with semaphore:
        return await asyncio.to_thread(map_urls_fast_enhanced, app, domain, limit)


async def map_urls_concurrently(
    app: Optional[Any],
    domains: List[str],
    limit: Optional[int] = None,
    concurrency: int = 5,
) -> Dict[str, List[str]]:
    """Concurrently map multiple domains using Firecrawl v2.2.0 15x faster Map endpoint.

    Enhanced with intelligent batching, concurrent mapping, and academic content filtering.

    Args:
        app: Firecrawl client, will get default if None
        domains: List of domain URLs to map
        limit: Maximum URLs per domain
        concurrency: Number of concurrent mapping operations

    Returns:
        Dictionary mapping domain URLs to lists of discovered URLs
    """
    with processing_context("map_urls_concurrent") as correlation_id:
        app = _resolve_firecrawl_client(app, "map_urls_concurrent", correlation_id)
        if app is None:
            return {}

        log_processing_step(
            logger,
            "map_urls_concurrent",
            "started",
            {
                "domains_count": len(domains),
                "limit_per_domain": limit,
                "concurrency": concurrency,
                "v2.2.0_performance": "15x faster mapping",
            },
            correlation_id=correlation_id,
        )

        try:
            results = {}

            # Create async tasks with thread-safe Firecrawl access
            async_tasks = [
                _map_urls_with_lock(app, domain, limit) for domain in domains
            ]

            # Execute all tasks concurrently and gather results
            try:
                task_results = await asyncio.gather(
                    *async_tasks, return_exceptions=True
                )
            except Exception as e:
                log_with_context(
                    logger,
                    logging.ERROR,
                    f"Failed to execute async mapping tasks: {e}",
                    correlation_id=correlation_id,
                )
                return {}

            # Process results and handle exceptions
            for domain, task_result in zip(domains, task_results):
                if isinstance(task_result, Exception):
                    log_with_context(
                        logger,
                        logging.ERROR,
                        f"Failed to map {domain}: {task_result}",
                        correlation_id=correlation_id,
                    )
                    results[domain] = []
                else:
                    urls = task_result
                    results[domain] = urls
                    log_with_context(
                        logger,
                        logging.DEBUG,
                        f"Mapped {len(urls)} URLs from {domain}",
                        correlation_id=correlation_id,
                    )

            total_urls = sum(len(urls) for urls in results.values())
            log_processing_step(
                logger,
                "map_urls_concurrent",
                "completed",
                {
                    "domains_processed": len(results),
                    "total_urls_discovered": total_urls,
                    "avg_urls_per_domain": total_urls / max(1, len(results)),
                },
                correlation_id=correlation_id,
            )

            return results

        except Exception as e:
            handle_error(
                ExternalServiceError(
                    f"Concurrent URL mapping failed: {e}", service="firecrawl"
                ),
                context={"domains_count": len(domains), "concurrency": concurrency},
                operation="map_urls_concurrent",
            )
            log_with_context(
                logger,
                logging.ERROR,
                f"Concurrent URL mapping failed: {e}",
                correlation_id=correlation_id,
            )
            return {}


def map_urls_fast_enhanced(
    app: Any, base_url: str, limit: Optional[int] = None, return_raw: bool = False
) -> List[str]:
    """Enhanced version of fast URL mapping with intelligent URL filtering and quality scoring.

    Improvements:
    - Domain-specific mapping strategies
    - URL deduplication and quality scoring
    - Academic content relevance filtering
    - Intelligent caching with TTL

    Args:
        app: Firecrawl client
        base_url: Base URL to map
        limit: Maximum URLs to return
        return_raw: Whether to return raw results without filtering

    Returns:
        List of discovered and filtered URLs
    """
    with processing_context("map_urls_fast_enhanced") as correlation_id:
        log_processing_step(
            logger,
            "map_urls_fast_enhanced",
            "started",
            {"base_url": base_url, "limit": limit},
            correlation_id=correlation_id,
        )

        try:
            # Check cache first
            cached_urls = get_cached_mapping(base_url)
            if cached_urls is not None:
                log_with_context(
                    logger,
                    logging.DEBUG,
                    f"Using cached mapping for {base_url}",
                    correlation_id=correlation_id,
                )
                return cached_urls[:limit] if limit else cached_urls

            # Determine domain type for specialized mapping
            domain_type = classify_domain_type(base_url)

            # Build enhanced map parameters
            map_params = {"url": base_url}
            if limit:
                map_params["limit"] = limit

            # Domain-specific optimization
            if domain_type == "academic":
                # Academic sites often have deeper structures
                map_params["limit"] = min(limit or 100, 200)
            elif domain_type == "reference":
                # Reference sites need broader coverage
                map_params["limit"] = min(limit or 50, 150)

            # Execute mapping with v2.2.0 performance improvements
            start_time = time.time()
            try:
                result = app.map_url(**map_params)
                mapping_duration = time.time() - start_time

                log_with_context(
                    logger,
                    logging.DEBUG,
                    f"Firecrawl mapping completed in {mapping_duration:.2f}s (15x performance)",
                    correlation_id=correlation_id,
                )
            except Exception as e:
                log_with_context(
                    logger,
                    logging.ERROR,
                    f"Firecrawl mapping failed: {e}",
                    correlation_id=correlation_id,
                )
                return []

            # Extract URLs from response
            if isinstance(result, dict) and "links" in result:
                raw_urls = result["links"]
            elif hasattr(result, "links"):
                raw_urls = result.links
            else:
                raw_urls = []

            if not raw_urls:
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"No URLs discovered for {base_url}",
                    correlation_id=correlation_id,
                )
                return []

            # Return raw results if requested
            if return_raw:
                cache_mapping_results(base_url, raw_urls)
                return raw_urls[:limit] if limit else raw_urls

            # Apply intelligent filtering and scoring
            base_domain = urlparse(base_url).netloc

            # 1. Filter for academic relevance
            academic_urls = filter_academic_urls(raw_urls, base_domain)

            # 2. Deduplicate and score for quality
            scored_urls = deduplicate_and_score_urls(academic_urls, base_domain)

            # 3. Apply limit
            final_urls = scored_urls[:limit] if limit else scored_urls

            # Cache results
            cache_mapping_results(base_url, final_urls)

            log_processing_step(
                logger,
                "map_urls_fast_enhanced",
                "completed",
                {
                    "raw_urls": len(raw_urls),
                    "academic_filtered": len(academic_urls),
                    "deduplicated": len(scored_urls),
                    "final_urls": len(final_urls),
                    "domain_type": domain_type,
                    "mapping_duration_ms": int(mapping_duration * 1000),
                },
                correlation_id=correlation_id,
            )

            return final_urls

        except Exception as e:
            handle_error(
                ExternalServiceError(
                    f"Enhanced URL mapping failed: {e}", service="firecrawl"
                ),
                context={
                    "base_url": base_url,
                    "domain_type": classify_domain_type(base_url),
                },
                operation="map_urls_fast_enhanced",
            )
            log_with_context(
                logger,
                logging.ERROR,
                f"Enhanced URL mapping failed for {base_url}: {e}",
                correlation_id=correlation_id,
            )
            return []


def _run_async_coro(
    coro,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    timeout: Optional[float] = None,
):
    """Execute async coroutine from sync context with proper event loop handling.

    Args:
        coro: Async coroutine to execute
        loop: Optional specific event loop to use for scheduling
        timeout: Optional timeout in seconds for coroutine execution

    Returns:
        Result from executing the coroutine

    Raises:
        RuntimeError: If called from within a running event loop without explicit loop parameter
        asyncio.TimeoutError: If coroutine execution exceeds timeout
        Exception: Re-raises any exception from coroutine execution
    """
    try:
        # Check if we're already in a running loop
        current_loop = asyncio.get_running_loop()
        if loop is None:
            # We're in a running loop but no explicit loop provided - this is unsafe
            raise RuntimeError(
                "_run_async_coro called from within a running event loop. "
                "Use explicit loop parameter or run from synchronous context."
            )
        elif loop == current_loop:
            # Trying to schedule on the same loop we're running in - deadlock risk
            raise RuntimeError(
                "Cannot schedule coroutine on the same loop that is currently running. "
                "This would cause a deadlock."
            )

        # Apply timeout if specified
        if timeout is not None:
            coro = asyncio.wait_for(coro, timeout=timeout)

        # Schedule on the provided loop
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=timeout)
        except asyncio.TimeoutError:
            # Cancel the coroutine on timeout
            future.cancel()
            logger.warning(f"Coroutine execution timed out after {timeout}s")
            raise

    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # No running loop - safe to use asyncio.run
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout=timeout)
            try:
                return asyncio.run(coro)
            except asyncio.TimeoutError:
                logger.warning(f"Coroutine execution timed out after {timeout}s")
                raise
        else:
            # Re-raise other RuntimeErrors
            raise


def map_urls_fast_enhanced_bulk(
    domains: List[str],
    max_urls_per_domain: int = 20,
    use_fast_map: bool = True,
    run_in_new_thread: bool = False,
    raise_on_running_loop: bool = True,
) -> Dict[str, Any]:
    """High-level bulk mapping API compatible with tests.

    Args:
        domains: List of domain URLs to map
        max_urls_per_domain: Maximum URLs to return per domain
        use_fast_map: Whether to use fast mapping (ignored, always uses fast mapping)
        run_in_new_thread: Force execution in a new thread to avoid event loop conflicts
        raise_on_running_loop: Whether to raise error when called from running event loop

    Returns:
        Dict with keys: urls_found, domains_processed, processing_time, details
    """
    start_time = time.time()

    # Early return for empty input
    if not domains:
        logger.debug("Empty domains list provided, returning early")
        return {
            "urls_found": 0,
            "domains_processed": 0,
            "processing_time": time.time() - start_time,
            "details": {},
        }

    # Resolve client
    app = _resolve_firecrawl_client(None, "map_urls_fast_enhanced_bulk")
    if app is None:
        return {
            "urls_found": 0,
            "domains_processed": 0,
            "processing_time": time.time() - start_time,
            "details": {},
        }

    # Call async function from sync context with proper event loop handling
    async def _async_wrapper():
        return await map_urls_concurrently(app, domains, limit=max_urls_per_domain)

    try:
        # Handle event loop context based on parameters
        if run_in_new_thread:
            # Force execution in new thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(_async_wrapper()))
                result = future.result()
        else:
            # Use existing event loop handling with timeout
            result = _run_async_coro(
                _async_wrapper(), timeout=300.0
            )  # 5 minute timeout

    except RuntimeError as e:
        if "running event loop" in str(e).lower() and not raise_on_running_loop:
            logger.warning(
                f"Running in event loop context, falling back to new thread: {e}"
            )
            # Fallback to new thread execution
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(_async_wrapper()))
                result = future.result()
        else:
            raise
    except asyncio.TimeoutError:
        logger.error("URL mapping operation timed out after 5 minutes")
        return {
            "urls_found": 0,
            "domains_processed": 0,
            "processing_time": time.time() - start_time,
            "details": {},
        }
    except Exception as e:
        # Log error and return empty result
        logger.error(f"Failed to execute URL mapping: {e}")
        return {
            "urls_found": 0,
            "domains_processed": 0,
            "processing_time": time.time() - start_time,
            "details": {},
        }

    # Validate result type
    if not isinstance(result, dict):
        logger.warning(
            f"Expected dict result but got {type(result)}, returning empty result"
        )
        return {
            "urls_found": 0,
            "domains_processed": 0,
            "processing_time": time.time() - start_time,
            "details": {},
        }

    # Calculate summary statistics
    total_urls = sum(len(urls) for urls in result.values())
    processing_time = time.time() - start_time

    return {
        "urls_found": total_urls,
        "domains_processed": len(result),
        "processing_time": processing_time,
        "details": result,
    }


def classify_domain_type(url: str) -> str:
    """Classify domain type for specialized mapping strategies.

    Args:
        url: URL to classify

    Returns:
        Domain type: academic, reference, research, media, or general
    """
    try:
        domain = urlparse(url).netloc.lower()

        # Academic institutions
        if (
            ".edu" in domain
            or ".ac." in domain
            or "university" in domain
            or "college" in domain
        ):
            return "academic"

        # Reference sites
        if any(
            ref in domain
            for ref in ["wikipedia", "britannica", "reference", "encyclopedia"]
        ):
            return "reference"

        # Research repositories
        if any(
            repo in domain
            for repo in ["arxiv", "researchgate", "scholar", "pubmed", "ieee"]
        ):
            return "research"

        # News and media
        if any(
            news in domain for news in ["news", "journal", "magazine", "times", "post"]
        ):
            return "media"

        return "general"
    except Exception:
        return "general"


def filter_academic_urls(urls: List[str], base_domain: str) -> List[str]:
    """Filter URLs for academic content relevance.

    Args:
        urls: List of URLs to filter
        base_domain: Base domain for relevance checking

    Returns:
        List of URLs filtered for academic content
    """
    academic_patterns = [
        "research",
        "faculty",
        "department",
        "paper",
        "publication",
        "journal",
        "conference",
        "symposium",
        "thesis",
        "dissertation",
        "curriculum",
        "course",
        "academic",
        "scholar",
        "study",
    ]

    filtered_urls = []
    for url in urls:
        url_lower = url.lower()

        # Include if URL contains academic patterns
        if any(pattern in url_lower for pattern in academic_patterns):
            filtered_urls.append(url)
        # Or if it's from the base domain (likely relevant)
        elif base_domain.lower() in url_lower:
            filtered_urls.append(url)
        # Or if it's a PDF (likely academic content)
        elif url_lower.endswith(".pdf"):
            filtered_urls.append(url)

    # If no academic URLs found, return top URLs from original list
    if not filtered_urls and urls:
        return urls[: min(10, len(urls))]

    return filtered_urls


def deduplicate_and_score_urls(urls: List[str], base_domain: str) -> List[str]:
    """Deduplicate URLs and score them for quality, returning best URLs.

    Args:
        urls: List of URLs to deduplicate and score
        base_domain: Base domain for scoring

    Returns:
        List of URLs sorted by quality score (highest first)
    """
    seen_urls = set()
    url_scores = []

    for url in urls:
        try:
            parsed = urlparse(url)

            # Normalize URL for deduplication
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if normalized in seen_urls:
                continue
            seen_urls.add(normalized)

            # Score URL quality
            score = 0

            # Domain relevance
            if base_domain.lower() in parsed.netloc.lower():
                score += 10

            # Path depth (prefer not too deep, not too shallow)
            path_depth = len([p for p in parsed.path.split("/") if p])
            if 1 <= path_depth <= 4:
                score += 5
            elif path_depth > 4:
                score -= 2

            # File type preferences
            path_lower = parsed.path.lower()
            if path_lower.endswith(".pdf"):
                score += 8  # PDFs often contain academic content
            elif path_lower.endswith((".html", ".htm")):
                score += 3  # Standard web pages
            elif path_lower.endswith((".php", ".asp", ".aspx")):
                score += 2  # Dynamic pages

            # Academic keywords in path
            academic_keywords = [
                "research",
                "faculty",
                "department",
                "course",
                "publication",
            ]
            for keyword in academic_keywords:
                if keyword in path_lower:
                    score += 3

            # Avoid certain patterns
            avoid_patterns = ["admin", "login", "search", "contact", "privacy"]
            for pattern in avoid_patterns:
                if pattern in path_lower:
                    score -= 5

            # Query parameters (usually less valuable)
            if parsed.query:
                score -= 1

            url_scores.append((url, score))

        except Exception:
            # If parsing fails, give it a low score but don't exclude
            url_scores.append((url, 0))

    # Sort by score (highest first) and return URLs
    url_scores.sort(key=lambda x: x[1], reverse=True)
    return [url for url, score in url_scores]


def cache_mapping_results(base_url: str, urls: List[str]) -> None:
    """Cache mapping results with TTL.

    Args:
        base_url: Base URL that was mapped
        urls: List of discovered URLs to cache
    """
    _mapping_cache[base_url] = {
        "urls": urls,
        "timestamp": time.time(),
        "ttl": 3600,  # 1 hour cache
    }

    # Clean old cache entries
    current_time = time.time()
    expired_keys = [
        key
        for key, value in _mapping_cache.items()
        if current_time - value["timestamp"] > value["ttl"]
    ]
    for key in expired_keys:
        del _mapping_cache[key]


def get_cached_mapping(base_url: str) -> Optional[List[str]]:
    """Get cached mapping results if still valid.

    Args:
        base_url: Base URL to check cache for

    Returns:
        Cached URLs list or None if not cached or expired
    """
    if base_url in _mapping_cache:
        cached = _mapping_cache[base_url]
        if time.time() - cached["timestamp"] < cached["ttl"]:
            return cached["urls"]
        else:
            del _mapping_cache[base_url]
    return None


def clear_mapping_cache() -> None:
    """Clear all cached mapping results."""
    global _mapping_cache
    _mapping_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get mapping cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    current_time = time.time()
    valid_entries = 0
    expired_entries = 0
    total_urls = 0

    for value in _mapping_cache.values():
        if current_time - value["timestamp"] < value["ttl"]:
            valid_entries += 1
            total_urls += len(value["urls"])
        else:
            expired_entries += 1

    return {
        "total_entries": len(_mapping_cache),
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "total_cached_urls": total_urls,
        "cache_hit_potential": valid_entries / max(1, len(_mapping_cache)),
        "avg_urls_per_entry": total_urls / max(1, valid_entries),
    }


def optimize_url_discovery(
    domains: List[str], target_urls_per_domain: int = 20
) -> Dict[str, int]:
    """Optimize URL discovery parameters based on domain types.

    Args:
        domains: List of domains to analyze
        target_urls_per_domain: Target number of URLs per domain

    Returns:
        Dictionary mapping domains to recommended limits
    """
    optimized_limits = {}

    for domain in domains:
        domain_type = classify_domain_type(domain)

        # Adjust limits based on domain type
        if domain_type == "academic":
            # Academic sites often have rich content, can handle more URLs
            limit = min(target_urls_per_domain * 2, 100)
        elif domain_type == "research":
            # Research repositories are usually well-structured
            limit = min(target_urls_per_domain * 1.5, 75)
        elif domain_type == "reference":
            # Reference sites may have lots of links but need filtering
            limit = min(target_urls_per_domain * 3, 150)
        else:
            # General domains - use target as-is
            limit = target_urls_per_domain

        optimized_limits[domain] = int(limit)

    return optimized_limits


__all__ = [
    "map_urls_concurrently",
    "map_urls_fast_enhanced",
    "map_urls_fast_enhanced_bulk",
    "classify_domain_type",
    "filter_academic_urls",
    "deduplicate_and_score_urls",
    "cache_mapping_results",
    "get_cached_mapping",
    "clear_mapping_cache",
    "get_cache_stats",
    "optimize_url_discovery",
]
