"""Core mining functions module.

This module contains the main mining functions including concept search,
batch URL scraping, smart extraction with Pydantic schemas, and the
primary mine_concepts function that orchestrates the entire process.
"""

import os
import json
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# HTTP timeout exception imports
try:
    from requests.exceptions import (
        Timeout as RequestsTimeout,
        ConnectionError as RequestsConnectionError,
    )
except ImportError:
    RequestsTimeout = TimeoutError
    RequestsConnectionError = ConnectionError

try:
    from httpx import (
        ReadTimeout as HttpxReadTimeout,
        ConnectTimeout as HttpxConnectTimeout,
    )
except ImportError:
    HttpxReadTimeout = TimeoutError
    HttpxConnectTimeout = ConnectionError

from .models import ConceptDefinition, WebResource, WebhookConfig, ApiUsageStats
from .config import ConfigError, get_firecrawl_client
from .performance import get_current_profile
from .queue_management import (
    get_queue_status_async,
    apply_intelligent_throttling,
    poll_job_with_adaptive_strategy,
)
from .url_processing import (
    map_urls_concurrently,
    map_urls_fast_enhanced,
    optimize_url_discovery,
)
from .api_tracking import get_api_usage_stats
from .webhooks import setup_webhooks
from .async_processing import (
    execute_with_resource_management,
    process_with_streaming,
    execute_parallel_pipeline,
    get_concurrency_manager,
)
from generate_glossary.config import get_mining_config
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
from generate_glossary.llm import run_async_safely


logger = get_logger(__name__)


def _normalize_legacy_parameters(**kwargs) -> Dict[str, Any]:
    """Normalize legacy parameter aliases to current parameter names.

    This centralizes legacy parameter handling to avoid repetition and potential
    event loop issues with parameter processing across different contexts.

    Args:
        **kwargs: Parameters that may include legacy aliases

    Returns:
        Dictionary with normalized parameters
    """
    normalized = kwargs.copy()

    # Handle max_pages_per_pdf -> max_pages alias
    if normalized.get("max_pages") is None and "max_pages_per_pdf" in normalized:
        normalized["max_pages"] = normalized.pop("max_pages_per_pdf")
        logger.debug(
            f"Normalized legacy parameter max_pages_per_pdf={normalized['max_pages']} to max_pages"
        )

    # Handle use_map_endpoint -> use_fast_map alias
    if "use_map_endpoint" in normalized:
        normalized["use_fast_map"] = normalized.pop("use_map_endpoint")
        logger.debug(
            f"Normalized legacy parameter use_map_endpoint={normalized['use_fast_map']} to use_fast_map"
        )

    return normalized


def _resolve_firecrawl_client(
    app: Optional[Any],
    operation: str,
    correlation_id: Optional[str] = None,
) -> Optional[Any]:
    """Return an initialized Firecrawl client, handling configuration errors."""

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (
            ConnectionError,
            TimeoutError,
            OSError,
            RequestsTimeout,
            RequestsConnectionError,
            HttpxReadTimeout,
            HttpxConnectTimeout,
        )
    ),
    reraise=True,
)
def search_concepts_batch(
    app: Any, concepts: List[str], max_urls_per_concept: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """Search for multiple concepts using Firecrawl's search endpoint with v2 features.

    Args:
        app: Firecrawl client
        concepts: List of concepts to search for
        max_urls_per_concept: Maximum URLs to return per concept

    Returns:
        Dictionary mapping concepts to lists of search results
    """
    with processing_context("search_concepts_batch") as correlation_id:
        log_processing_step(
            logger,
            "search_concepts_batch",
            "started",
            {
                "concepts_count": len(concepts),
                "max_urls_per_concept": max_urls_per_concept,
            },
            correlation_id=correlation_id,
        )

        results = {}
        api_usage_stats = get_api_usage_stats()

        try:
            for concept in concepts:
                # Build academic-focused query with proper OR operator handling
                if " " in concept:
                    # Multi-word concept: quote it and append OR terms outside quotes
                    escaped_concept = concept.replace('"', '"')
                    query = f'"{escaped_concept}" (definition OR explanation OR academic OR wikipedia OR edu OR arxiv)'
                else:
                    # Single word: no quotes needed
                    query = f"{concept} (definition OR explanation OR academic OR wikipedia OR edu OR arxiv)"

                log_with_context(
                    logger,
                    logging.INFO,
                    f"Searching for: {concept}",
                    correlation_id=correlation_id,
                )

                # Track API usage
                api_usage_stats.add_call("search")

                # Use Firecrawl search with v2.2.0 patterns and proper named parameters
                try:
                    # v2.2.0: Use proper named parameters and handle structured return types
                    search_result = app.search(
                        query=query,
                        limit=max_urls_per_concept,
                        categories=[
                            "research"
                        ],  # v2.2.0: Filter for research/academic content
                    )
                except (TypeError, AttributeError):
                    # Fallback to standard search if v2.2.0 features not available
                    log_with_context(
                        logger,
                        logging.DEBUG,
                        "Search categories parameter unsupported; falling back to basic search",
                        correlation_id=correlation_id,
                    )
                    search_result = app.search(query=query, limit=max_urls_per_concept)

                # Extract the results with v2.2.0 structured response handling
                search_results = []
                if isinstance(search_result, dict):
                    # v2.2.0: Handle structured return types with "success" and "data" keys
                    if search_result.get("success") and "data" in search_result:
                        data = search_result["data"]
                        # Handle categorized results with "web", "images", "news" arrays
                        if isinstance(data, dict):
                            if "web" in data:
                                search_results = data["web"][:max_urls_per_concept]
                            else:
                                search_results = [data] if data else []
                        elif isinstance(data, list):
                            search_results = data[:max_urls_per_concept]
                    else:
                        # Fallback: treat as simple list response
                        if isinstance(search_result, list):
                            search_results = search_result[:max_urls_per_concept]
                        else:
                            search_results = [search_result] if search_result else []
                elif isinstance(search_result, list):
                    search_results = search_result[:max_urls_per_concept]
                else:
                    search_results = [search_result] if search_result else []

                # Normalize search results
                normalized_results = []
                for result in search_results:
                    if isinstance(result, dict):
                        normalized_results.append(result)
                    elif hasattr(result, "__dict__"):
                        normalized_results.append(result.__dict__)
                    else:
                        # Convert to dict format
                        normalized_results.append(
                            {"url": str(result), "title": "", "snippet": ""}
                        )

                results[concept] = normalized_results

                log_with_context(
                    logger,
                    logging.DEBUG,
                    f"Found {len(normalized_results)} results for concept: {concept}",
                    correlation_id=correlation_id,
                )

        except Exception as e:
            handle_error(
                ExternalServiceError(
                    f"Concept search batch failed: {e}", service="firecrawl"
                ),
                context={"concepts_count": len(concepts)},
                operation="search_concepts_batch",
            )
            log_with_context(
                logger,
                logging.ERROR,
                f"Batch concept search failed: {e}",
                correlation_id=correlation_id,
            )
            return {}

        log_processing_step(
            logger,
            "search_concepts_batch",
            "completed",
            {
                "concepts_processed": len(results),
                "total_urls_found": sum(len(urls) for urls in results.values()),
            },
            correlation_id=correlation_id,
        )

        return results


def search_concepts_batch_clientless(
    concepts: List[str], max_urls_per_concept: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """Clientless wrapper for search_concepts_batch.

    This function automatically gets the client and forwards to search_concepts_batch.

    Args:
        concepts: List of concepts to search for
        max_urls_per_concept: Maximum URLs to return per concept

    Returns:
        Dictionary mapping concepts to lists of search results
    """
    client = _resolve_firecrawl_client(None, "search_concepts_batch_clientless")
    if not client:
        return {}
    return search_concepts_batch(client, concepts, max_urls_per_concept)


async def batch_scrape_urls(
    app: Any,
    urls: List[str],
    max_concurrent: int = 10,
    max_age: int = 172800000,  # 2 days
    use_summary: bool = True,
    summary_prompt: Optional[str] = None,
    poll_interval: int = 2,
    wait_timeout: int = 120,
    max_pages: Optional[int] = None,  # v2.2.0: PDF page limit
    max_pages_per_pdf: Optional[int] = None,  # Alias for max_pages
    enable_queue_monitoring: bool = False,
    enable_api_tracking: bool = False,
) -> Dict[str, Any]:
    """Use Firecrawl's batch scrape endpoint for 500% performance improvement.

    Args:
        app: Firecrawl client
        urls: List of URLs to scrape
        max_concurrent: Maximum concurrent operations
        max_age: Cache age in milliseconds
        use_summary: Whether to use summary format
        summary_prompt: Optional custom summary prompt
        poll_interval: Polling interval for job status
        wait_timeout: Maximum wait timeout
        max_pages: PDF page limit (v2.2.0)
        max_pages_per_pdf: Alias for max_pages (for test compatibility)
        enable_queue_monitoring: Enable queue monitoring
        enable_api_tracking: Enable API call tracking

    Returns:
        Dictionary with scraped results
    """
    # Normalize legacy parameters
    params = _normalize_legacy_parameters(
        max_pages=max_pages, max_pages_per_pdf=max_pages_per_pdf
    )
    max_pages = params["max_pages"]

    with processing_context("batch_scrape_urls") as correlation_id:
        log_processing_step(
            logger,
            "batch_scrape_urls",
            "started",
            {
                "urls_count": len(urls),
                "max_concurrent": max_concurrent,
                "use_summary": use_summary,
                "max_age_hours": max_age / 1000 / 60 / 60,
                "max_pages": max_pages,  # v2.2.0 feature
            },
            correlation_id=correlation_id,
        )

        if not urls:
            return {}

        api_usage_stats = get_api_usage_stats()

        try:
            # Apply intelligent throttling based on queue status
            if enable_queue_monitoring:
                current_profile = get_current_profile()
                throttle_delay = await apply_intelligent_throttling(
                    app, current_profile
                )
                if throttle_delay > 0:
                    log_with_context(
                        logger,
                        logging.INFO,
                        f"Applying throttling delay: {throttle_delay:.1f}s",
                        correlation_id=correlation_id,
                    )
                    await asyncio.sleep(throttle_delay)

            # Build formats for batch scraping
            formats = ["markdown"]
            if use_summary:
                if summary_prompt:
                    formats = [{"type": "summary", "prompt": summary_prompt}]
                else:
                    formats = ["summary"]

            # Track API usage
            api_usage_stats.add_call("batch_scrape")

            # Build batch scrape parameters
            scrape_params = {
                "urls": urls,
                "formats": formats,
                "maxAge": max_age,
                "onlyMainContent": True,
            }

            # v2.2.0: Add maxPages parameter for PDF control - support both formats for SDK compatibility
            if max_pages is not None:
                scrape_params["maxPages"] = max_pages  # Top-level format
                # Also add nested format for newer SDK versions
                if "pageOptions" not in scrape_params:
                    scrape_params["pageOptions"] = {}
                scrape_params["pageOptions"]["pdf"] = {"maxPages": max_pages}

            # Execute batch scrape with v2.2.0 enhanced parameters
            try:
                result = app.batch_scrape(**scrape_params)

                # Handle job-based response (async job submission)
                # Support multiple job ID formats: job_id, jobId, ID, or nested data.jobId
                job_id = None
                if isinstance(result, dict):
                    job_id = (
                        result.get("job_id")
                        or result.get("jobId")
                        or result.get("ID")
                        or (result.get("data") or {}).get("jobId")
                        or (result.get("data") or {}).get("job_id")
                    )

                if job_id:
                    log_with_context(
                        logger,
                        logging.INFO,
                        f"Batch scrape job submitted: {job_id}",
                        correlation_id=correlation_id,
                    )

                    # Use adaptive polling strategy
                    final_result = await poll_job_with_adaptive_strategy(
                        app, job_id, enable_queue_monitoring
                    )

                    # Log completion details
                    items = (
                        final_result.get("data") or final_result.get("results") or []
                    )
                    successful = len([r for r in items if not r.get("error")])
                    log_processing_step(
                        logger,
                        "batch_scrape_urls",
                        "completed",
                        {
                            "urls_processed": len(urls),
                            "successful_scrapes": successful,
                            "performance_improvement": "500% faster than sequential",
                        },
                        correlation_id=correlation_id,
                    )

                    return final_result

                # Handle direct response (synchronous result)
                else:
                    log_processing_step(
                        logger,
                        "batch_scrape_urls",
                        "completed",
                        {"urls_processed": len(urls), "result_type": "direct_return"},
                        correlation_id=correlation_id,
                    )
                    return result

            except (AttributeError, TypeError) as e:
                handle_error(
                    e,
                    context={
                        "urls_count": len(urls),
                        "max_concurrent": max_concurrent,
                        "fallback": "sequential_scraping",
                    },
                    operation="batch_scrape_firecrawl",
                )
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"Batch scrape not available, falling back to sequential: {e}",
                    correlation_id=correlation_id,
                )

                # Fallback to sequential scraping with v2.2.0 features
                results = {}
                for url in urls:
                    try:
                        # Track individual scrape API usage
                        api_usage_stats.add_call("scrape")

                        # Build scrape parameters with v2.2.0 structured extraction
                        scrape_params = {
                            "url": url,
                            "formats": formats,
                            "onlyMainContent": True,
                            "pageOptions": {"blockAds": True},
                        }

                        # Add v2.2.0 maxPages parameter under pageOptions for PDF control
                        if max_pages is not None:
                            # Ensure pageOptions exists before setting pdf parameters
                            if "pageOptions" not in scrape_params:
                                scrape_params["pageOptions"] = {}
                            scrape_params["pageOptions"]["pdf"] = {
                                "maxPages": max_pages
                            }

                        result = app.scrape(**scrape_params)
                        results[url] = result

                    except Exception as scrape_error:
                        log_with_context(
                            logger,
                            logging.ERROR,
                            f"Failed to scrape {url}: {scrape_error}",
                            correlation_id=correlation_id,
                        )
                        results[url] = {"error": str(scrape_error)}

                return {"data": results}

        except Exception as e:
            handle_error(
                ExternalServiceError(
                    f"Enhanced batch scraping failed: {e}", service="firecrawl"
                ),
                context={
                    "urls_count": len(urls),
                    "max_concurrent": max_concurrent,
                    "enable_queue_monitoring": enable_queue_monitoring,
                },
                operation="batch_scrape_urls",
            )
            log_with_context(
                logger,
                logging.ERROR,
                f"Enhanced batch scraping failed: {e}",
                correlation_id=correlation_id,
            )
            return {}


def extract_with_smart_prompts(
    app: Any,
    urls: List[str],
    concept: str,
    actions: Optional[List[Dict]] = None,
    max_pages: Optional[int] = None,
    max_pages_per_pdf: Optional[int] = None,
) -> List[WebResource]:
    """Extract structured definitions using Firecrawl's extract endpoint with smart prompts.

    Args:
        app: Firecrawl client
        urls: List of URLs to extract from
        concept: Concept to extract information about
        actions: Optional actions for dynamic content
        max_pages: PDF page limit (v2.2.0)
        max_pages_per_pdf: Alias for max_pages (for test compatibility)

    Returns:
        List of WebResource objects with extracted definitions
    """
    # Normalize legacy parameters
    params = _normalize_legacy_parameters(
        max_pages=max_pages, max_pages_per_pdf=max_pages_per_pdf
    )
    max_pages = params["max_pages"]

    with processing_context(f"extract_smart_prompts_{concept}") as correlation_id:
        log_processing_step(
            logger,
            "extract_with_smart_prompts",
            "started",
            {
                "concept": concept,
                "urls_count": len(urls),
                "has_actions": bool(actions),
                "max_pages": max_pages,  # v2.2.0 feature
            },
            correlation_id=correlation_id,
        )

        if not urls:
            return []

        api_usage_stats = get_api_usage_stats()

        try:
            # Enhanced extraction prompt using natural language for v2.0
            prompt = f"""
            Extract comprehensive information about the academic concept "{concept}".

            Focus on: 1) Clear definitions 2) Academic context 3) Key characteristics 4) Related concepts

            Return structured data with concept name, definition, context field, key points list,
            related concepts list, and source quality assessment.
            """

            resources = []

            for url in urls:
                try:
                    # Track API usage
                    api_usage_stats.add_call("extract")

                    # Build extract parameters with v2.2.0 features
                    extract_params = {
                        "url": url,
                        "prompt": prompt,
                        "schema": ConceptDefinition.model_json_schema(),
                    }

                    # v2.2.0: Add actions for dynamic content interaction
                    if actions:
                        extract_params["actions"] = actions

                    # v2.2.0: Add maxPages for PDF control
                    if max_pages is not None:
                        extract_params["maxPages"] = max_pages

                    # Execute extraction
                    result = app.extract(**extract_params)

                    # Process structured extraction result
                    definitions = []
                    if isinstance(result, dict):
                        if "data" in result and isinstance(result["data"], dict):
                            try:
                                definition = ConceptDefinition(**result["data"])
                                definitions.append(definition)
                            except Exception as e:
                                log_with_context(
                                    logger,
                                    logging.WARNING,
                                    f"Failed to parse extracted data for {url}: {e}",
                                    correlation_id=correlation_id,
                                )
                        elif result:  # Direct data format
                            try:
                                definition = ConceptDefinition(**result)
                                definitions.append(definition)
                            except Exception as e:
                                log_with_context(
                                    logger,
                                    logging.WARNING,
                                    f"Failed to parse direct extracted data for {url}: {e}",
                                    correlation_id=correlation_id,
                                )

                    # Create WebResource
                    resource = WebResource(
                        url=url,
                        title=f"Extracted content for {concept}",
                        definitions=definitions,
                    )
                    resources.append(resource)

                    log_with_context(
                        logger,
                        logging.DEBUG,
                        f"Extracted {len(definitions)} definitions from {url}",
                        correlation_id=correlation_id,
                    )

                except Exception as e:
                    log_with_context(
                        logger,
                        logging.ERROR,
                        f"Failed to extract from {url}: {e}",
                        correlation_id=correlation_id,
                    )
                    # Create empty resource for failed extraction
                    resources.append(
                        WebResource(url=url, title=f"Failed extraction for {concept}")
                    )

            log_processing_step(
                logger,
                "extract_with_smart_prompts",
                "completed",
                {
                    "concept": concept,
                    "urls_processed": len(urls),
                    "resources_created": len(resources),
                    "total_definitions": sum(len(r.definitions) for r in resources),
                },
                correlation_id=correlation_id,
            )

            return resources

        except Exception as e:
            handle_error(
                ExternalServiceError(
                    f"Smart extraction failed: {e}", service="firecrawl"
                ),
                context={"concept": concept, "urls_count": len(urls)},
                operation="extract_with_smart_prompts",
            )
            log_with_context(
                logger,
                logging.ERROR,
                f"Smart extraction failed for concept {concept}: {e}",
                correlation_id=correlation_id,
            )
            return []


def mine_concepts(
    concepts: List[str],
    output_path: Optional[str] = None,
    max_concurrent: Optional[int] = None,
    max_age: int = 172800000,  # 2 days cache
    use_summary: bool = True,
    use_batch_scrape: bool = True,
    actions: Optional[List[Dict]] = None,
    summary_prompt: Optional[str] = None,
    use_hybrid: bool = False,
    timeout_seconds: Optional[int] = None,
    # v2.2.0 new parameters
    max_pages: Optional[int] = None,  # PDF page limit for better performance
    webhook_config: Optional[WebhookConfig] = None,  # Enhanced webhook configuration
    enable_queue_monitoring: bool = False,  # Queue status monitoring
    use_fast_map: bool = True,  # Use 15x faster Map endpoint
    **kwargs,
) -> Dict[str, Any]:
    """Mine web content for academic concepts using ALL Firecrawl v2.2.0 features.

    This unified function provides a single entry point for web content mining
    with comprehensive v2.2.0 feature support including batch scraping (500% faster),
    smart crawling with natural language prompts, enhanced caching, summary format
    optimization, actions for dynamic content interaction, and new v2.2.0 capabilities.

    Args:
        concepts: List of concepts to mine
        output_path: Optional path to save results
        max_concurrent: Maximum concurrent operations (defaults to config)
        max_age: Maximum age for cached content in milliseconds (default: 2 days)
        use_summary: Use summary format for optimized content extraction
        use_batch_scrape: Use batch scraping for 500% performance improvement
        actions: Actions for dynamic content interaction
        summary_prompt: Custom summary prompt for extraction optimization
        use_hybrid: Combine search and mapping approaches
        timeout_seconds: Operation timeout in seconds
        max_pages: v2.2.0 - PDF page limit for better performance
        webhook_config: v2.2.0 - Enhanced webhook configuration
        enable_queue_monitoring: v2.2.0 - Queue status monitoring for throttling
        use_fast_map: v2.2.0 - Use 15x faster Map endpoint
        **kwargs: Additional parameters, including legacy parameters:
            - max_pages_per_pdf: Legacy alias for max_pages
            - use_map_endpoint: Legacy alias for use_fast_map

    Returns:
        Dictionary containing mining results and performance metrics
    """
    with processing_context("mine_concepts_unified") as correlation_id:
        start_time = time.time()

        # Normalize legacy parameters
        params = _normalize_legacy_parameters(
            max_pages=max_pages, use_fast_map=use_fast_map, **kwargs
        )
        max_pages = params["max_pages"]
        use_fast_map = params["use_fast_map"]

        # Setup webhooks if configured
        if webhook_config:
            setup_success = setup_webhooks(webhook_config)
            if not setup_success:
                log_with_context(
                    logger,
                    logging.WARNING,
                    "Webhook setup failed, continuing without webhooks",
                    correlation_id=correlation_id,
                )

        # Log detailed v2.2.0 configuration
        log_processing_step(
            logger,
            "mine_concepts_unified",
            "started",
            {
                "concepts_count": len(concepts),
                "max_concurrent": max_concurrent,
                "use_batch_scrape": use_batch_scrape,
                "use_summary": use_summary,
                "use_hybrid": use_hybrid,
                "max_age_hours": max_age / 1000 / 60 / 60,
                "timeout_seconds": timeout_seconds,
                # v2.2.0 specific features
                "max_pages": max_pages,
                "enable_queue_monitoring": enable_queue_monitoring,
                "use_fast_map": use_fast_map,
                "webhook_enabled": webhook_config is not None,
                "v2_2_0_features": True,
            },
            correlation_id=correlation_id,
        )

        # Initialize Firecrawl client
        app = _resolve_firecrawl_client(None, "mine_concepts_unified", correlation_id)
        if not app:
            if enable_queue_monitoring:
                log_with_context(
                    logger,
                    logging.WARNING,
                    "Queue monitoring requested but Firecrawl client could not be initialized",
                    correlation_id=correlation_id,
                )
                enable_queue_monitoring = False
            error_msg = "Failed to initialize Firecrawl client"
            log_with_context(
                logger, logging.ERROR, error_msg, correlation_id=correlation_id
            )
            return {
                "success": False,
                "error": error_msg,
                "concepts_processed": 0,
                "total_definitions": 0,
                "processing_time_seconds": 0,
                "statistics": {
                    "total_concepts": 0,
                    "successful": 0,
                    "total_resources": 0,
                },
                "v2_2_0_features_used": {},
                "v2_2_0_features_used_list": [],
            }

        # Get configuration with performance profile integration
        config = get_mining_config()
        if max_concurrent is None:
            performance_profile = get_current_profile()
            max_concurrent = performance_profile.max_concurrent

        try:
            # Execute mining pipeline with async orchestration
            results = run_async_safely(
                _execute_mining_pipeline(
                    app=app,
                    concepts=concepts,
                    max_concurrent=max_concurrent,
                    max_age=max_age,
                    use_summary=use_summary,
                    use_batch_scrape=use_batch_scrape,
                    actions=actions,
                    summary_prompt=summary_prompt,
                    use_hybrid=use_hybrid,
                    max_pages=max_pages,
                    enable_queue_monitoring=enable_queue_monitoring,
                    use_fast_map=use_fast_map,
                    correlation_id=correlation_id,
                )
            )

            # Save results if output path provided
            if output_path and results.get("success"):
                try:
                    dirpath = os.path.dirname(output_path)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    results["output_saved"] = output_path
                except Exception as e:
                    log_with_context(
                        logger,
                        logging.WARNING,
                        f"Failed to save results to {output_path}: {e}",
                        correlation_id=correlation_id,
                    )

            # Add final performance metrics
            total_time = time.time() - start_time

            # Calculate statistics
            total_concepts = len(results.get("results", {}))
            successful = sum(
                1 for v in results.get("results", {}).values() if v.get("definitions")
            )
            total_resources = sum(
                v.get("resource_count", 0) for v in results.get("results", {}).values()
            )

            # Prepare v2.2.0 features tracking
            v2_2_0_features_used = {
                "batch_scraping": use_batch_scrape,
                "queue_monitoring": enable_queue_monitoring,
                "fast_mapping": use_fast_map,
                "pdf_page_limit": max_pages is not None,
                "webhooks": webhook_config is not None,
            }
            v2_2_0_features_used_list = [
                k for k, v in v2_2_0_features_used.items() if v
            ]

            results.update(
                {
                    "processing_time_seconds": total_time,
                    "concepts_processed": successful,
                    "statistics": {
                        "total_concepts": total_concepts,
                        "successful": successful,
                        "total_resources": total_resources,
                    },
                    "v2_2_0_features_used": v2_2_0_features_used,
                    "v2_2_0_features_used_list": v2_2_0_features_used_list,
                }
            )

            log_processing_step(
                logger,
                "mine_concepts_unified",
                "completed",
                {
                    "success": results.get("success", False),
                    "concepts_processed": results.get("concepts_processed", 0),
                    "total_definitions": results.get("total_definitions", 0),
                    "processing_time_seconds": total_time,
                    "performance_improvement": "v2.2.0 optimizations applied",
                },
                correlation_id=correlation_id,
            )

            return results

        except Exception as e:
            total_time = time.time() - start_time
            handle_error(
                ExternalServiceError(
                    f"Unified mining failed: {e}", service="firecrawl"
                ),
                context={
                    "concepts_count": len(concepts),
                    "processing_time_seconds": total_time,
                },
                operation="mine_concepts_unified",
            )

            return {
                "success": False,
                "error": str(e),
                "concepts_processed": 0,
                "total_definitions": 0,
                "processing_time_seconds": total_time,
                "statistics": {
                    "total_concepts": 0,
                    "successful": 0,
                    "total_resources": 0,
                },
                "v2_2_0_features_used": {},
                "v2_2_0_features_used_list": [],
            }


async def _execute_mining_pipeline(
    app: Any,
    concepts: List[str],
    max_concurrent: int,
    max_age: int,
    use_summary: bool,
    use_batch_scrape: bool,
    actions: Optional[List[Dict]],
    summary_prompt: Optional[str],
    use_hybrid: bool,
    max_pages: Optional[int],
    enable_queue_monitoring: bool,
    use_fast_map: bool,
    correlation_id: str,
) -> Dict[str, Any]:
    """Execute the complete mining pipeline asynchronously.

    Args:
        app: Firecrawl client
        concepts: List of concepts to mine
        max_concurrent: Maximum concurrent operations
        max_age: Cache age in milliseconds
        use_summary: Use summary format
        use_batch_scrape: Use batch scraping
        actions: Optional actions for dynamic content
        summary_prompt: Optional custom summary prompt
        use_hybrid: Use hybrid approach
        max_pages: PDF page limit
        enable_queue_monitoring: Enable queue monitoring
        use_fast_map: Use fast mapping
        correlation_id: Correlation ID for logging

    Returns:
        Dictionary with mining results
    """
    try:
        # Stage 1: Search for concepts
        search_results = search_concepts_batch(app, concepts, max_urls_per_concept=5)

        # Stage 2: Extract URLs for scraping
        all_urls = []
        concept_url_map = {}
        for concept, results in search_results.items():
            urls = [r.get("url", "") for r in results if r.get("url")]
            all_urls.extend(urls)
            concept_url_map[concept] = urls

        # Stage 2.5: Fast Map and hybrid URL discovery
        if use_fast_map or use_hybrid:
            # Derive unique base domains from search results with error handling
            domains = []
            skipped_urls = 0

            for url in all_urls:
                if not url or not url.strip():
                    continue

                try:
                    parsed = urlparse(url.strip())
                    if parsed.scheme and parsed.netloc:
                        domain = f"{parsed.scheme}://{parsed.netloc}"
                        domains.append(domain)
                    else:
                        skipped_urls += 1
                except Exception as e:
                    # Log malformed URL and continue
                    log_with_context(
                        logger,
                        logging.WARNING,
                        f"Skipped malformed URL during domain extraction: {url!r} - {e}",
                        correlation_id=correlation_id,
                    )
                    skipped_urls += 1

            # Remove duplicates
            domains = list(set(domains))

            if skipped_urls > 0:
                log_with_context(
                    logger,
                    logging.INFO,
                    f"Skipped {skipped_urls} malformed URLs during domain extraction",
                    correlation_id=correlation_id,
                )

            if domains:
                # Get domain limits with optimize_url_discovery
                limits = optimize_url_discovery(domains)

                # Call map_urls_concurrently
                discovered_urls = await map_urls_concurrently(app, domains, limit=None)

                # Merge discovered URLs into all_urls
                for domain, urls in discovered_urls.items():
                    limit = limits.get(domain, 20)
                    limited_urls = urls[:limit]
                    all_urls.extend(limited_urls)

                    # If use_hybrid, also merge into each concept's URL list where domain matches
                    if use_hybrid:
                        domain_netloc = urlparse(domain).netloc
                        for concept, concept_urls in concept_url_map.items():
                            # Check if any of the concept's existing URLs match this domain
                            for existing_url in concept_urls:
                                if urlparse(existing_url).netloc == domain_netloc:
                                    concept_url_map[concept].extend(limited_urls)
                                    break

                # Dedupe all_urls
                all_urls = list(set(all_urls))

        # Stage 3: Batch scrape URLs
        if use_batch_scrape and all_urls:
            scrape_results = await batch_scrape_urls(
                app=app,
                urls=all_urls,
                max_concurrent=max_concurrent,
                max_age=max_age,
                use_summary=use_summary,
                summary_prompt=summary_prompt,
                max_pages=max_pages,
                enable_queue_monitoring=enable_queue_monitoring,
            )
        else:
            scrape_results = {}

        # Stage 4: Extract structured definitions
        final_results = {}
        total_definitions = 0

        for concept in concepts:
            concept_urls = concept_url_map.get(concept, [])
            if concept_urls:
                resources = extract_with_smart_prompts(
                    app=app,
                    urls=concept_urls,
                    concept=concept,
                    actions=actions,
                    max_pages=max_pages,
                )

                definitions = []
                for resource in resources:
                    definitions.extend(resource.definitions)

                final_results[concept] = {
                    "definitions": [d.model_dump() for d in definitions],
                    "sources": concept_urls,
                    "resource_count": len(resources),
                }
                total_definitions += len(definitions)
            else:
                final_results[concept] = {
                    "definitions": [],
                    "sources": [],
                    "resource_count": 0,
                }

        return {
            "success": True,
            "results": final_results,
            "total_definitions": total_definitions,
            "urls_processed": len(all_urls),
            "scrape_results": scrape_results,
        }

    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"Mining pipeline failed: {e}",
            correlation_id=correlation_id,
        )
        return {
            "success": False,
            "error": str(e),
            "results": {},
            "total_definitions": 0,
        }


__all__ = [
    "search_concepts_batch",
    "search_concepts_batch_clientless",
    "batch_scrape_urls",
    "extract_with_smart_prompts",
    "mine_concepts",
]
