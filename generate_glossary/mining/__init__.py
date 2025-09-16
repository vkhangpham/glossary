"""
Unified web mining interface with Firecrawl v2.2.0 integration.

This module provides a clean, unified interface for web content extraction
using Firecrawl v2.2.0 features including batch scraping (500% performance improvement),
smart crawling with natural language prompts, enhanced caching, summary format
optimization, actions for dynamic content interaction, and new v2.2.0 capabilities.

Key Features:
- Batch scraping for 500% performance improvement
- Smart crawling with natural language prompts
- Enhanced caching with maxAge parameter
- Summary format for optimized content extraction
- Actions support for dynamic content interaction
- Research category filtering for academic content
- JSON schema extraction for structured data
- Comprehensive error handling and logging
- Queue status monitoring and predictive management
- 15x faster Map endpoint for URL discovery
- Enhanced webhooks with signature verification

Usage:
    from generate_glossary.mining import mine_concepts

    results = mine_concepts(
        ["machine learning", "neural networks"],
        use_batch_scrape=True,
        use_summary=True,
        max_age=172800000,  # 2 days cache
        max_pages=5,  # v2.2.0: Limit PDF parsing
        enable_queue_monitoring=True  # v2.2.0: Queue monitoring
    )
"""

# Import everything from mining module
from .mining import *

# Backward compatibility aliases
mine_concepts_with_firecrawl = mine_concepts

# Static __all__ list - explicitly names all public symbols
# IMPORTANT: Keep this list synchronized when adding/removing public exports!
# Run tests to validate: python -m pytest tests/test_mining_all.py
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

    # Backward compatibility alias
    'mine_concepts_with_firecrawl',
]