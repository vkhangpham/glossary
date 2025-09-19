# Mining Module Simplification Migration Plan

## Phase 1: Essential Features Analysis

### Functions to Keep and Simplify

**`mine_concepts()` – Main Entry Point**
- **Current**: 200+ lines with complex pipeline orchestration, legacy parameter handling, webhook setup
- **Target**: 50–80 lines using direct Firecrawl SDK calls
- **Simplification**: Remove legacy parameters, use native `batch_scrape()`, eliminate custom queue management

**`search_concepts_batch()` – Concept Search**
- **Current**: 150+ lines with complex result normalization, v2.2.0 compatibility layers
- **Target**: 20–30 lines using direct `firecrawl.search()` calls
- **Simplification**: Remove compatibility fallbacks, use native search result format

**`batch_scrape_urls()` – URL Scraping**
- **Current**: 200+ lines with custom job polling, queue monitoring, fallback logic
- **Target**: 10–20 lines using direct `firecrawl.batch_scrape()` calls
- **Simplification**: Remove custom polling, use native batch processing

**`extract_with_smart_prompts()` – Structured Extraction**
- **Current**: 150+ lines with custom schema handling, error recovery
- **Target**: 30–40 lines using Firecrawl's native structured extraction
- **Simplification**: Use native JSON schema extraction, remove custom error handling

### Models to Keep

**`ConceptDefinition`**
- Keep as-is: Essential data structure for extracted academic concepts
- Used by: Firecrawl's native structured extraction

**`WebResource`**
- Keep as-is: Container for web resources with definitions
- Used by: Result aggregation and output formatting

### Functions/Modules to Remove Completely

**Queue Management (`queue_management.py`)**
- **Reason**: Firecrawl v2.2.0 provides native `get_queue_status()` and queue monitoring
- **Replacement**: Direct `firecrawl.get_queue_status()` calls when needed

**Performance Profiling (`performance.py`)**
- **Reason**: Firecrawl handles performance optimization internally
- **Replacement**: Use Firecrawl's native performance features

**API Tracking (`api_tracking.py`)**
- **Reason**: Firecrawl provides native usage tracking and analytics
- **Replacement**: Use Firecrawl's built-in usage monitoring

**Webhook Management (`webhooks.py`)**
- **Reason**: Firecrawl v2.2.0 has native webhook support with signature verification
- **Replacement**: Use Firecrawl's native webhook configuration

**Async Processing (`async_processing.py`)**
- **Reason**: Firecrawl provides `AsyncFirecrawl` client for async operations
- **Replacement**: Use `AsyncFirecrawl` directly

**URL Processing (`url_processing.py`)**
- **Reason**: Firecrawl's native `map()` method handles URL discovery and processing
- **Replacement**: Direct `firecrawl.map()` calls

**Client Management (`client.py`)**
- **Reason**: Firecrawl SDK handles client initialization and management
- **Replacement**: Direct `Firecrawl(api_key=...)` initialization

## Phase 2: New Simplified Structure

### Target File Structure

**`config.yml`** (New)
```yaml
firecrawl:
  api_key: ${FIRECRAWL_API_KEY}
  timeout: 30
  max_retries: 3

mining:
  max_urls_per_concept: 5
  batch_size: 20
  use_summary: true
  max_age: 172800000  # 2 days cache
  max_pages: 10  # PDF page limit

output:
  format: json
  save_metadata: true
```

**`config.py`** (New – ~50 lines)
- Load YAML configuration
- Handle CLI argument overrides
- Validate required settings
- No environment variable overrides (except credentials)

**`utils.py`** (New – ~100 lines)
- Mining-specific utility functions only
- URL validation and processing helpers
- Result formatting and aggregation
- Error handling specific to mining operations

**`main.py`** (New – ~150 lines)
- CLI interface with argparse
- Command: `mine <concepts> [options]`
- Short aliases for common arguments
- Integration with config system

**`core/firecrawl_client.py`** (New – ~200 lines)
- Simple wrapper around Firecrawl SDK
- Direct method calls without abstractions
- Basic error handling and logging
- Support for both sync and async operations

## Phase 3: Implementation Strategy

### Step 1: Create New Structure
1. Create new `config.yml` with essential Firecrawl settings
2. Create new `config.py` to load YAML and handle CLI overrides
3. Create new `utils.py` with mining-specific utilities only
4. Create new `main.py` with clean CLI interface

### Step 2: Implement Simplified Core
1. Create `core/firecrawl_client.py` with direct SDK usage
2. Implement simplified `mine_concepts()` function
3. Use Firecrawl's native methods: `search()`, `batch_scrape()`, structured extraction
4. Remove all legacy compatibility and custom abstractions

### Step 3: Preserve Essential Models
1. Keep `ConceptDefinition` and `WebResource` models
2. Remove all other models (`QueueStatus`, `ApiUsageStats`, etc.)
3. Update imports to use only essential models

### Step 4: Update API
1. Simplify `mine_concepts()` to use direct Firecrawl calls
2. Remove all legacy function aliases and compatibility layers
3. Update `__init__.py` to export only essential functions

## API Deprecation Mapping

- **Queue & polling**:
  - `get_queue_status`, `get_queue_status_async` → call `GET /v2/team/queue-status` directly (wrap a helper only if the SDK version exposes one).
  - `generate_queue_insights`, `poll_job_with_adaptive_strategy`, `apply_intelligent_throttling`, `get_queue_predictor`, `reset_queue_state` → rely on Firecrawl queue telemetry and remove bespoke predictors.
- **URL mapping**:
  - `map_urls_concurrently`, `map_urls_fast_enhanced`, `_map_urls_concurrently`, `_map_urls_fast_enhanced`, `optimize_url_discovery`, `deduplicate_and_score_urls`, `cache_mapping_results`, `get_cached_mapping`, `clear_mapping_cache`, `get_cache_stats` → migrate to `firecrawl.map()` and Firecrawl's deduplication output.
- **Webhook utilities**:
  - `setup_webhooks`, `verify_webhook_signature`, `handle_webhook_event`, `get_webhook_stats`, `get_recent_events`, `list_active_webhooks`, `remove_webhook`, `test_webhook_connectivity`, `reset_webhook_state` → replace with per-call `webhook={...}` payloads and Firecrawl signature headers.
- **API usage & performance**:
  - `track_api_usage`, `analyze_usage_patterns`, `calculate_feature_impact`, `generate_optimization_recommendations`, `generate_v220_benchmarks`, `estimate_api_costs`, `calculate_usage_efficiency`, `get_api_usage_stats`, `reset_api_tracking`, `get_usage_trends` → consume Firecrawl usage dashboards or endpoints instead of local analytics.
  - `configure_performance_profile`, `auto_tune_performance`, `get_performance_status`, `get_current_profile`, `set_performance_profile`, `reset_performance_state`, `PerformanceProfile` → rely on Firecrawl-managed throttling and remove local tuners.
- **Legacy aliases**:
  - `_search_concepts_batch`, `_batch_scrape_urls`, `_extract_with_smart_prompts`, `_map_urls_*` → delete once the canonical functions exist; avoid exporting duplicates from `__init__.py`.

**Deprecation Timeline**
- Sprint 0 (design): Mark deprecated symbols in documentation and surface warnings when imported.
- Sprint 1 (bridge release): Ship wrappers that forward to Firecrawl while logging warnings; publish beta replacements.
- Sprint 2 (removal): Delete wrappers, shrink `__all__`, and clean downstream imports.

**Communication Plan**
- Document changes in release notes and the mining module README.
- Send direct notifications to downstream teams with sample replacement snippets and deadlines.
- Add CI lint rules to detect deprecated imports and link to this mapping.

## Webhook Transition Strategy

`WebhookConfig` and the `webhook_config` parameter on `mine_concepts()` will be deprecated in favor of per-call `webhook` payloads.

```python
# Legacy
mine_concepts(
    concepts,
    webhook_config=WebhookConfig(
        url="https://hooks.internal/mining",
        events=["started", "page", "completed", "failed"],
        metadata={"pipeline": "mining"},
    ),
)

# Post-migration
mine_concepts(
    concepts,
    webhook={
        "url": "https://hooks.internal/mining",
        "events": ["started", "page", "completed", "failed"],
        "metadata": {"pipeline": "mining"},
    },
)
```

During the bridge release we will translate any provided `WebhookConfig` into the new payload and emit a deprecation warning. Firecrawl signs each delivery; downstream services must continue verifying signatures before acknowledging events.

## Target Public API

Post-migration, `mining/__init__.py` and `mining/mining.py` export only:
- `mine_concepts`
- `ConceptDefinition`
- `WebResource`

All other helpers remain internal or CLI-specific. Collapsing the re-export chain keeps the public surface minimal and prevents new aliases from creeping back into `__all__`.

## Edge Cases & Compatibility

- **Legacy PDF limits**: Convert `max_pages_per_pdf` (and related knobs) into Firecrawl's `maxPages` parameter and clamp to the API's max. Warn when user-supplied values exceed the allowed range.
- **Map endpoint flags**: Treat `use_map_endpoint` as advisory; when `firecrawl.map()` is unavailable, fall back to sequential `firecrawl.scrape()` with throttling and log the downgrade.
- **Batch fallback**: If `firecrawl.batch_scrape()` is disabled or returns a non-success status, revert to sequential scraping to preserve functionality and emit telemetry so operators know a fallback triggered.
- **Missing SDK features**: When the active Firecrawl SDK build lacks `categories` support, omit the parameter, rely on query filters, and surface a warning so teams know category filtering was skipped.


## Phase 4: Benefits and Validation

### Complexity Reduction
- **Before**: 9+ files, ~3000+ lines, complex abstractions
- **After**: 5 files, ~500–600 lines, direct SDK usage
- **Reduction**: ~80% code reduction

### Performance Improvements
- Use Firecrawl's native batch processing (500% faster)
- Leverage built-in caching and optimization
- Remove overhead from custom abstractions

### Maintainability Improvements
- Direct SDK usage – easier to understand and debug
- No legacy compatibility code to maintain
- Automatic access to new Firecrawl features
- Clear separation of concerns

### Testing Strategy
- Test direct Firecrawl SDK integration
- Validate essential functionality preservation
- Ensure CLI interface works correctly
- Test configuration loading and overrides

### Migration Timeline
- **Phase 1 – Analysis & Design (2–3 weeks)**: Complete feature inventory, finalize target architecture, secure stakeholder sign-off.
- **Phase 2 – Implementation (3–4 weeks)**: Build new Firecrawl wrappers, port essential functions, deprecate legacy modules.
- **Phase 3 – Testing & Validation (1–2 weeks)**: Run automated suites, execute staging dry-runs, validate data parity.
- **Cutover Milestone**: Feature-flagged production rollout with acceptance criteria met across functionality, performance, and observability benchmarks.

### Resource Estimates
- **Engineering**: 2–3 backend developers driving refactor, 1 QA engineer for validation, 1 SRE for deployment and observability.
- **Tooling & Infrastructure**: CI runners for regression suites, dedicated staging environment mirroring production, feature-flag service for controlled enablement, automated backup pipelines.
- **Support**: Product owner for acceptance, tech writer for documentation updates.

### Rollback Plan
- Maintain parallel Git branches for legacy and new pipelines until post-cutover validation completes.
- Wrap new mining code paths behind feature flags to allow gradual rollout and rapid disable.
- Schedule automated database and schema backups ahead of migration windows and keep snapshots for 7 days.
- Document quick revert procedures (git revert scripts, infrastructure toggles) and ensure on-call coverage during rollout.
- Configure monitoring and alerting on mining throughput, error rates, and Firecrawl usage; trigger rollback if thresholds breach agreed rollback criteria.

## Phase 5: Migration Checklist

### Pre-Migration
- [ ] Document current API usage patterns
- [ ] Identify all external dependencies on mining module
- [ ] Create test cases for essential functionality
- [ ] Perform risk assessment and define rollback criteria with stakeholders
- [ ] Establish monitoring/alerting baselines for existing mining metrics

### During Migration
- [ ] Create new simplified structure
- [ ] Implement direct Firecrawl SDK usage
- [ ] Remove all legacy code and abstractions
- [ ] Update imports and dependencies
- [ ] Execute canary or blue-green rollout plan using feature flags
- [ ] Enable detailed logging and metrics for new Firecrawl-backed paths
- [ ] Publish operational runbook covering failure scenarios and rollback triggers

### Post-Migration
- [ ] Validate all functionality works correctly
- [ ] Update documentation and examples
- [ ] Remove old files completely
- [ ] Update tests to match new structure
- [ ] Confirm monitoring alerts and dashboards reflect expected baselines
- [ ] Conduct rollback drill or tabletop exercise if rollback not triggered in production
- [ ] Review incident metrics and collect lessons learned with action items
- [ ] Document contingency plans for follow-up iterations
