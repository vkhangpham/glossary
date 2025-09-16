"""Unified mining module with Firecrawl v2.2.0 integration for academic glossary extraction.

This module provides a single, clean mine_concepts() function that leverages ALL
Firecrawl v2.2.0 features including batch scraping (500% performance improvement),
smart crawling with natural language prompts, enhanced caching, summary format
optimization, actions for dynamic content interaction, and new v2.2.0 capabilities.

Key Features:
- Batch scraping for 500% performance improvement over sequential scraping
- Smart crawling with natural language prompts for academic content extraction
- Enhanced caching with maxAge parameter for faster repeated requests
- Summary format for optimized content extraction and reduced token usage
- Actions support for dynamic content interaction when needed
- Research category filtering for academic-focused content
- JSON schema extraction for structured data
- Comprehensive error handling and logging with correlation IDs
- Async/sync compatibility with event loop management

NEW v2.2.0 Features:
- Queue Status Endpoint: Monitor job queues with /v2/team/queue-status
- PDF maxPages Parameter: Control PDF page limits for better performance
- 15x Faster Map Endpoint: Improved URL processing and discovery
- API Key Usage Tracking: Monitor API consumption and feature usage
- Enhanced Webhooks: Signatures and event failure handling for real-time notifications

Usage:
    from generate_glossary.mining import mine_concepts

    results = mine_concepts(
        ["machine learning", "neural networks"],
        use_batch_scrape=True,
        use_summary=True,
        max_age=172800000,  # 2 days cache
        max_pages=5,  # v2.2.0: Limit PDF parsing
        webhook_config=webhook_config  # v2.2.0: Enable webhook notifications
    )
"""

# Re-export all models
from .models import (
    ConceptDefinition,
    WebResource,
    QueueStatus,
    ApiUsageStats,
    WebhookConfig,
    PerformanceProfile,
    QueuePredictor
)

# Re-export client functions
from .client import (
    initialize_firecrawl,
    get_client,
    get_firecrawl_api_key,
    validate_api_key,
    check_client_health,
    reset_client,
    get_client_info
)

# Re-export performance functions
from .performance import (
    configure_performance_profile,
    auto_tune_performance,
    get_performance_status,
    get_current_profile,
    reset_performance_state
)

# Re-export queue management functions
from .queue_management import (
    get_queue_status,
    get_queue_status_async,
    generate_queue_insights,
    check_queue_health,
    poll_job_with_adaptive_strategy,
    apply_intelligent_throttling,
    get_queue_predictor,
    set_performance_profile,
    reset_queue_state
)

# Re-export URL processing functions
from .url_processing import (
    map_urls_concurrently,
    map_urls_fast_enhanced_bulk,
    classify_domain_type,
    filter_academic_urls,
    deduplicate_and_score_urls,
    cache_mapping_results,
    get_cached_mapping,
    clear_mapping_cache,
    get_cache_stats,
    optimize_url_discovery
)

# Re-export API tracking functions
from .api_tracking import (
    track_api_usage,
    analyze_usage_patterns,
    calculate_feature_impact,
    generate_optimization_recommendations,
    generate_v220_benchmarks,
    estimate_api_costs,
    calculate_usage_efficiency,
    get_api_usage_stats,
    reset_api_tracking,
    get_usage_trends
)

# Re-export webhook functions
from .webhooks import (
    setup_webhooks,
    verify_webhook_signature,
    handle_webhook_event,
    get_webhook_stats,
    get_recent_events,
    list_active_webhooks,
    remove_webhook,
    test_webhook_connectivity,
    reset_webhook_state
)

# Re-export async processing functions
from .async_processing import (
    ConcurrencyManager,
    AsyncResultAggregator,
    execute_with_resource_management,
    process_with_streaming,
    execute_parallel_pipeline,
    parallel_map,
    throttled_execution,
    get_concurrency_manager,
    reset_concurrency_state
)

# Re-export core mining functions - this is the main API
from .core_mining import (
    search_concepts_batch,
    batch_scrape_urls,
    extract_with_smart_prompts,
    mine_concepts  # Main entry point
)

# Maintain backward compatibility aliases
_search_concepts_batch = search_concepts_batch
_batch_scrape_urls = batch_scrape_urls
_extract_with_smart_prompts = extract_with_smart_prompts

# Legacy function aliases for compatibility
def _map_urls_concurrently(domains, limit=None, concurrency=5):
    """Legacy alias for map_urls_concurrently that injects client and remaps args."""
    import asyncio

    # Resolve client
    app = get_client()

    # Build the coroutine with injected client
    coro = map_urls_concurrently(app, domains, limit=limit, concurrency=concurrency)

    # Run it with appropriate event loop handling
    try:
        return asyncio.get_event_loop().run_until_complete(coro)
    except RuntimeError:
        # Fall back to asyncio.run if no event loop
        return asyncio.run(coro)

def _map_urls_fast_enhanced(*args, **kwargs):
    """Legacy alias for map_urls_fast_enhanced."""
    return map_urls_fast_enhanced(*args, **kwargs)

def map_urls_fast_enhanced(*args, **kwargs):
    """Wrapper that handles both single-domain and bulk domain processing.

    If 'domains' is in kwargs, call bulk function.
    Otherwise, forward to the original single-domain function.
    """
    if 'domains' in kwargs:
        return map_urls_fast_enhanced_bulk(**kwargs)
    else:
        # Import the original function from url_processing
        from .url_processing import map_urls_fast_enhanced as _original_func
        return _original_func(*args, **kwargs)

def _classify_domain(*args, **kwargs):
    """Legacy alias for classify_domain_type."""
    return classify_domain_type(*args, **kwargs)

def _filter_academic_urls(*args, **kwargs):
    """Legacy alias for filter_academic_urls."""
    return filter_academic_urls(*args, **kwargs)

def _deduplicate_and_score_urls(*args, **kwargs):
    """Legacy alias for deduplicate_and_score_urls."""
    return deduplicate_and_score_urls(*args, **kwargs)

def _generate_queue_insights(*args, **kwargs):
    """Legacy alias for generate_queue_insights."""
    return generate_queue_insights(*args, **kwargs)

def _check_queue_health(*args, **kwargs):
    """Legacy alias for check_queue_health."""
    return check_queue_health(*args, **kwargs)

def _analyze_usage_patterns(*args, **kwargs):
    """Legacy alias for analyze_usage_patterns."""
    return analyze_usage_patterns(*args, **kwargs)

def _calculate_feature_impact(*args, **kwargs):
    """Legacy alias for calculate_feature_impact."""
    return calculate_feature_impact(*args, **kwargs)

def _generate_optimization_recommendations(*args, **kwargs):
    """Legacy alias for generate_optimization_recommendations."""
    return generate_optimization_recommendations(*args, **kwargs)

def _generate_v220_benchmarks(*args, **kwargs):
    """Legacy alias for generate_v220_benchmarks."""
    return generate_v220_benchmarks(*args, **kwargs)

def _estimate_api_costs(*args, **kwargs):
    """Legacy alias for estimate_api_costs."""
    return estimate_api_costs(*args, **kwargs)

# Global state management - create instances from the modules
_api_usage_stats = get_api_usage_stats()
_queue_predictor = get_queue_predictor()
_performance_profile = get_current_profile()
_concurrency_manager = get_concurrency_manager()

# Logger setup
from generate_glossary.utils.logger import get_logger
logger = get_logger(__name__)

# Note: Environment setup and API key validation moved to client.py to avoid import-time side effects

# Import failure tracker fallback
try:
    from generate_glossary.utils.failure_tracker import save_failure
except ImportError:
    def save_failure(module, function, error_type, error_message, context=None, failure_dir=None):
        """Fallback implementation that just logs."""
        logger.warning(f"Failure in {module}.{function}: {error_type}: {error_message}")

# Main API - the mine_concepts function is already imported from core_mining
# This serves as the primary entry point for the entire mining system

__all__ = [
    # Core API
    'mine_concepts',  # Main entry point

    # Models
    'ConceptDefinition',
    'WebResource',
    'QueueStatus',
    'ApiUsageStats',
    'WebhookConfig',
    'PerformanceProfile',
    'QueuePredictor',

    # Client management
    'initialize_firecrawl',
    'get_client',
    'get_firecrawl_api_key',
    'validate_api_key',
    'check_client_health',
    'reset_client',
    'get_client_info',

    # Performance management
    'configure_performance_profile',
    'auto_tune_performance',
    'get_performance_status',
    'get_current_profile',
    'reset_performance_state',

    # Queue management
    'get_queue_status',
    'get_queue_status_async',
    'generate_queue_insights',
    'check_queue_health',
    'poll_job_with_adaptive_strategy',
    'apply_intelligent_throttling',
    'get_queue_predictor',
    'set_performance_profile',
    'reset_queue_state',

    # URL processing
    'map_urls_concurrently',
    'map_urls_fast_enhanced',
    'classify_domain_type',
    'filter_academic_urls',
    'deduplicate_and_score_urls',
    'cache_mapping_results',
    'get_cached_mapping',
    'clear_mapping_cache',
    'get_cache_stats',
    'optimize_url_discovery',

    # API tracking
    'track_api_usage',
    'analyze_usage_patterns',
    'calculate_feature_impact',
    'generate_optimization_recommendations',
    'generate_v220_benchmarks',
    'estimate_api_costs',
    'calculate_usage_efficiency',
    'get_api_usage_stats',
    'reset_api_tracking',
    'get_usage_trends',

    # Webhook management
    'setup_webhooks',
    'verify_webhook_signature',
    'handle_webhook_event',
    'get_webhook_stats',
    'get_recent_events',
    'list_active_webhooks',
    'remove_webhook',
    'test_webhook_connectivity',
    'reset_webhook_state',

    # Async processing
    'ConcurrencyManager',
    'AsyncResultAggregator',
    'execute_with_resource_management',
    'process_with_streaming',
    'execute_parallel_pipeline',
    'parallel_map',
    'throttled_execution',
    'get_concurrency_manager',
    'reset_concurrency_state',

    # Core mining functions
    'search_concepts_batch',
    'batch_scrape_urls',
    'extract_with_smart_prompts',

    # Legacy compatibility
    '_search_concepts_batch',
    '_batch_scrape_urls',
    '_extract_with_smart_prompts',
    '_map_urls_concurrently',
    '_map_urls_fast_enhanced',
    '_classify_domain',
    '_filter_academic_urls',
    '_deduplicate_and_score_urls',
    '_generate_queue_insights',
    '_check_queue_health',
    '_analyze_usage_patterns',
    '_calculate_feature_impact',
    '_generate_optimization_recommendations',
    '_generate_v220_benchmarks',
    '_estimate_api_costs',

    # Global state (for compatibility)
    '_api_usage_stats',
    '_queue_predictor',
    '_performance_profile',
    '_concurrency_manager'
]