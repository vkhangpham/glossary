# Current Mining Module Structure Analysis

## Files and Complexity
- **core_mining.py** (1052 lines): Main mining functions with complex Firecrawl integration, legacy parameter normalization, complex error handling
- **mining.py** (358 lines): Unified interface that re-exports everything from other modules
- **__init__.py** (142 lines): Re-exports everything from mining.py creating triple-layer export chain
- **models.py** (282 lines): Pydantic models including complex queue prediction and performance profiling
- **client.py** (349 lines): Singleton client management with health checking and async method handling
- **performance.py**: Custom performance profiling and tuning
- **queue_management.py**: Custom queue status monitoring and throttling
- **url_processing.py**: Custom URL mapping and processing
- **api_tracking.py**: Custom API usage tracking and analytics
- **webhooks.py**: Custom webhook management
- **async_processing.py**: Custom async processing utilities

## Key Problems Identified

1. **Over-complicated abstractions**: Custom implementations of features Firecrawl v2.2.0 provides natively
2. **Legacy compatibility layers**: Extensive backward compatibility code (`_normalize_legacy_parameters`, legacy aliases)
3. **Complex re-export chains**: `__init__.py` → `mining.py` → `core_mining.py` + 8 other modules
4. **Centralized dependencies**: Heavy reliance on centralized config (1000+ lines) and utils
5. **Unnecessary complexity**: Custom queue management, performance profiling, API tracking that Firecrawl handles

## Firecrawl v2.2.0 Native Capabilities

### Core Methods Available
- `scrape(url, formats=['markdown', 'html'])` – Direct URL scraping
- `crawl(url, limit=100, scrape_options={})` – Website crawling with built-in options
- `map(url)` – URL mapping and discovery
- `search(query, limit=10)` – Web search functionality
- `batch_scrape(urls, formats=[])` – Native batch processing (500% faster)

### Advanced Features
- **AsyncFirecrawl**: Native async client support
- **Queue Management**: v2.2.0 exposes `/v2/team/queue-status`; newer SDK helpers may wrap it, but integrations must be ready to call the endpoint directly.
- **Webhooks**: Native webhook support with signature verification
- **Structured Extraction**: Built-in JSON schema extraction with Pydantic models
- **Caching**: Native caching with `maxAge` parameter
- **PDF Processing**: Built-in `maxPages` parameter for PDF control

## Essential Features to Preserve

### Models to Keep
- `ConceptDefinition`: Core data structure for extracted concepts
- `WebResource`: Container for web resources with definitions

### Models to Remove
- `QueueStatus`: Use Firecrawl's native queue status
- `ApiUsageStats`: Use Firecrawl's native usage tracking
- `WebhookConfig`: Use Firecrawl's native webhook configuration
- `PerformanceProfile`: Use Firecrawl's native performance management
- `QueuePredictor`: Use Firecrawl's native queue management

### Functions to Simplify
- `mine_concepts()` – Simplify to direct Firecrawl SDK calls
- `search_concepts_batch()` – Use Firecrawl's native `search()` method
- `batch_scrape_urls()` – Use Firecrawl's native `batch_scrape()` method
- `extract_with_smart_prompts()` – Use Firecrawl's native structured extraction

### Functions to Remove
- All queue management functions (replace with direct `GET /v2/team/queue-status` usage; keep a helper only if the SDK version supplies one)
- All performance profiling functions (use Firecrawl native)
- All API tracking functions (use Firecrawl native)
- All webhook management functions (use Firecrawl native)
- All async processing utilities (use AsyncFirecrawl)
- All URL processing utilities (use Firecrawl native)

## Migration Strategy

### Target Structure
```
mining/
├── config.yml              # Simple YAML configuration
├── config.py               # Load YAML config with CLI overrides
├── utils.py                # Mining-specific utilities only
├── main.py                 # CLI interface
└── core/
    └── firecrawl_client.py  # Simple Firecrawl wrapper
```

### Complexity Reduction
- **From**: 9+ files, ~3000+ lines
- **To**: 5 files, ~500–600 lines
- **Reduction**: ~80% complexity reduction

### Key Simplifications
1. **Direct SDK Usage**: Replace all custom abstractions with direct Firecrawl v2.2.0 calls
2. **Remove Legacy Code**: Eliminate all backward compatibility layers
3. **Native Features**: Use Firecrawl's built-in queue management, webhooks, batch processing
4. **Simplified Models**: Keep only essential data structures
5. **Module-specific Config**: Replace centralized config dependency with simple YAML

## Benefits of Simplification

1. **Maintainability**: 80% less code to maintain
2. **Reliability**: Use battle-tested Firecrawl features instead of custom implementations
3. **Performance**: Leverage Firecrawl's optimized batch processing and caching
4. **Future-proof**: Automatic access to new Firecrawl features
5. **Simplicity**: Clear, direct API without complex abstractions

## Dependencies

- **External Libraries**: `firecrawl` (Python SDK; uses the `FirecrawlApp` client), `pydantic` v2 (models), `requests` (HTTP fallbacks and direct endpoint calls), `httpx` (async HTTP error handling), `tenacity` (retry policies), `python-dotenv` (optional environment loading)
- **Internal Modules**: `generate_glossary.mining.core_mining`, `generate_glossary.mining.models`, `generate_glossary.mining.client`, `generate_glossary.utils.failure_tracker`, `generate_glossary.config.get_mining_config`, `generate_glossary.utils.logger`, `generate_glossary.utils.error_handler`, `generate_glossary.llm.helpers.run_async_safely`
- **Environment Variables**: `FIRECRAWL_API_KEY` (mandatory for SDK initialization and direct API calls)

## Current API Contracts

The following reflects the live signatures and payloads for the functions that MIGRATION_PLAN.md designates as the “Target Public API”. Use these details when coordinating any simplification work so we do not break downstream callers.

### mine_concepts

- **Parameters:** `concepts: List[str]` (required); `output_path: Optional[str] = None` (writes JSON dump when provided); `max_concurrent: Optional[int] = None` (defaults to `get_current_profile().max_concurrent`); `max_age: int = 172800000`; `use_summary: bool = True`; `use_batch_scrape: bool = True`; `actions: Optional[List[Dict]] = None`; `summary_prompt: Optional[str] = None`; `use_hybrid: bool = False`; `timeout_seconds: Optional[int] = None`; `max_pages: Optional[int] = None`; `webhook_config: Optional[WebhookConfig] = None`; `enable_queue_monitoring: bool = False`; `use_fast_map: bool = True`; `**kwargs` currently accepts the legacy aliases `max_pages_per_pdf` → `max_pages` and `use_map_endpoint` → `use_fast_map`.
- **Key behaviors:** Resolves a Firecrawl client through `get_client()` and aborts with `success=False` when unavailable; normalizes legacy aliases via `_normalize_legacy_parameters`; optionally registers webhooks through `setup_webhooks`; orchestrates `_execute_mining_pipeline()` (search → URL discovery → batch scrape → extraction) while logging v2.2.0 feature usage; persists JSON to `output_path` when supplied and appends telemetry such as processing time and feature flags.
- **Return schema:** On success returns a dict with `success: True`, `results: Dict[str, Dict[str, Any]]` where each concept entry exposes `definitions: List[Dict]`, `sources: List[str]`, and `resource_count: int`, plus aggregate keys `total_definitions: int`, `urls_processed: int`, `scrape_results: Dict[str, Any]`, `processing_time_seconds: float`, `concepts_processed: int`, `statistics: {'total_concepts': int, 'successful': int, 'total_resources': int}`, `v2_2_0_features_used: Dict[str, bool]`, `v2_2_0_features_used_list: List[str]`, and optional `output_saved: str` when an export succeeds. On failure returns `success: False` with `error: str`, zeroed counters, empty `results`, and empty feature tracking dicts.
- **Migration note:** Remains one of the three public exports listed under “Target Public API” in MIGRATION_PLAN.md.

### search_concepts_batch

- **Parameters:** `app: FirecrawlApp` instance; `concepts: List[str]`; `max_urls_per_concept: int = 3`.
- **Key behaviors:** Builds academic-focused queries with quoted multi-word concepts, invokes `app.search()` with v2.2.0 category filtering (falling back when unsupported), tracks usage via `get_api_usage_stats()`, and normalizes heterogeneous SDK responses into plain dictionaries (coercing objects and strings). Errors are routed to `handle_error()` and yield an empty mapping.
- **Return schema:** Dict that maps each concept to a list of Firecrawl search result dicts. Each item preserves the SDK payload (`url`, `title`, `snippet`/`description`, and any additional metadata); non-dict responses are wrapped as `{'url': <str>, 'title': '', 'snippet': ''}`. Returns `{}` on batch failure.
- **Migration note:** Listed in MIGRATION_PLAN.md as a simplification target but retained in the public surface during the bridge period.

### batch_scrape_urls

- **Parameters:** `app: FirecrawlApp`; `urls: List[str]`; `max_concurrent: int = 10`; `max_age: int = 172800000`; `use_summary: bool = True`; `summary_prompt: Optional[str] = None`; `poll_interval: int = 2`; `wait_timeout: int = 120`; `max_pages: Optional[int] = None`; `max_pages_per_pdf: Optional[int] = None` (alias); `enable_queue_monitoring: bool = False`; `enable_api_tracking: bool = False`.
- **Key behaviors:** Normalizes `max_pages_per_pdf` to `max_pages`; optionally sleeps based on queue telemetry; builds Firecrawl batch `formats` (summary or markdown) and attaches `maxPages` both at the top level and under `pageOptions['pdf']`; captures job identifiers and, when present, polls completion via `poll_job_with_adaptive_strategy`; falls back to per-URL `app.scrape()` calls when the batch endpoint is unavailable. Any exception triggers `handle_error()` and an empty dict return.
- **Return schema:** On job submission returns the Firecrawl status dict from `poll_job_with_adaptive_strategy()`—typically including `success`, `data` or `results` lists (each item exposing the requested formats such as `summary`/`markdown` plus `url`, `status`, `error`). When the SDK returns a synchronous payload, that dict is forwarded verbatim. Sequential fallback yields `{'data': {url: scrape_result_dict}}`. Errors return `{}`.
- **Migration note:** Documented here to guide the planned collapse into direct `firecrawl.batch_scrape()` usage defined in MIGRATION_PLAN.md.

### extract_with_smart_prompts

- **Parameters:** `app: FirecrawlApp`; `urls: List[str]`; `concept: str`; `actions: Optional[List[Dict]] = None`; `max_pages: Optional[int] = None`; `max_pages_per_pdf: Optional[int] = None` (alias).
- **Key behaviors:** Generates a natural-language prompt tailored to the concept, records usage via `get_api_usage_stats()`, calls `app.extract()` with the Pydantic `ConceptDefinition.model_json_schema()` schema, attaches optional actions and `maxPages`, and converts each result into a `WebResource` (populated with parsed `ConceptDefinition` instances). Failed extractions append placeholder `WebResource` entries with empty definition lists.
- **Return schema:** List of `WebResource` models (`url: str`, `title: str`, `domain: str`, `definitions: List[ConceptDefinition]`). Each `ConceptDefinition` carries `concept`, `definition`, `context`, `key_points`, `related_concepts`, and `source_quality`. Returns an empty list when the extraction process raises an exception.
- **Migration note:** Complements the simplified structured extraction strategy described in MIGRATION_PLAN.md Phase 1.
