# Current vs Firecrawl v2.2.0 Native Features Mapping

## Core Mining Functions

### `search_concepts_batch()` → `firecrawl.search()`

**Current Implementation (150+ lines):**
- Custom query building with academic focus
- Complex result normalization and structure handling
- v2.2.0 compatibility layers and fallbacks
- Custom error handling and retry logic
- API usage tracking integration

**Firecrawl v2.2.0 Native:**
```python
results = firecrawl.search(
    query=f"{concept} (definition OR explanation OR academic)",
    limit=max_urls_per_concept,
    categories=["research"]  # Built-in academic filtering
)
```

**Simplification Benefits:**
- 90% code reduction (150+ lines → 10–15 lines)
- Native academic content filtering
- Built-in result structure consistency
- Automatic error handling and retries

### `batch_scrape_urls()` → `firecrawl.batch_scrape()`

**Current Implementation (200+ lines):**
- Custom job submission and polling logic
- Complex queue monitoring and throttling
- Custom performance profiling integration
- Fallback to sequential scraping
- Custom timeout and retry handling

**Firecrawl v2.2.0 Native:**
```python
result = firecrawl.batch_scrape(
    urls=urls,
    formats=["markdown", "summary"],
    maxAge=max_age,
    maxPages=max_pages
)
```

**Simplification Benefits:**
- 95% code reduction (200+ lines → 10 lines)
- Native 500% performance improvement
- Built-in queue management and throttling
- Automatic job polling and status tracking
- Native timeout and error handling

### `extract_with_smart_prompts()` → Firecrawl Structured Extraction

**Current Implementation (150+ lines):**
- Custom prompt engineering for academic content
- Manual schema validation and error recovery
- Complex result processing and normalization
- Custom API usage tracking

**Firecrawl v2.2.0 Native:**
```python
result = firecrawl.scrape(
    url=url,
    formats=[{"type": "json", "schema": ConceptDefinition.model_json_schema()}]
)
```

**Simplification Benefits:**
- 85% code reduction (150+ lines → 20–25 lines)
- Native Pydantic schema integration
- Built-in prompt optimization for extraction
- Automatic result validation

## Advanced Features

### Queue Management → Native Queue Status

**Current Implementation (`queue_management.py` – 300+ lines):**
- Custom `QueueStatus` model with prediction logic
- Complex queue health monitoring
- Custom throttling and adaptive polling strategies
- Performance profiling integration

**Firecrawl v2.2.0 Native:**
```python
queue_status = firecrawl.get_queue_status()
# Returns native queue information with built-in monitoring
```

**Simplification Benefits:**
- 100% code elimination (300+ lines → 0 lines)
- Native queue monitoring and insights
- Built-in throttling and load balancing
- Real-time queue status updates

### Performance Profiling → Native Performance Management

**Current Implementation (`performance.py` – 200+ lines):**
- Custom `PerformanceProfile` classes
- Manual performance tuning and optimization
- Custom concurrency management
- Complex performance metrics tracking

**Firecrawl v2.2.0 Native:**
- Built-in performance optimization
- Automatic concurrency management
- Native performance metrics and monitoring
- Intelligent resource allocation

**Simplification Benefits:**
- 100% code elimination (200+ lines → 0 lines)
- Superior performance through native optimization
- Automatic scaling and resource management
- Built-in performance analytics

### API Tracking → Native Usage Monitoring

**Current Implementation (`api_tracking.py` – 250+ lines):**
- Custom `ApiUsageStats` model with detailed tracking
- Manual usage pattern analysis
- Custom cost estimation and optimization recommendations
- Complex performance metrics calculation

**Firecrawl v2.2.0 Native:**
- Built-in API usage tracking and analytics
- Native cost monitoring and optimization
- Automatic usage pattern analysis
- Real-time usage insights

**Simplification Benefits:**
- 100% code elimination (250+ lines → 0 lines)
- More accurate usage tracking
- Built-in cost optimization
- Real-time usage analytics

### Webhook Management → Native Webhook Support

**Current Implementation (`webhooks.py` – 200+ lines):**
- Custom `WebhookConfig` model
- Manual webhook signature verification
- Custom event handling and retry logic
- Complex webhook state management

**Firecrawl v2.2.0 Native:**
```python
payload = {
    "url": "https://example.edu/research",
    "webhook": {
        "url": "https://hooks.internal/mining",
        "metadata": {"pipeline": "mining", "batch_id": batch_id},
        "events": [
            "crawl.started",
            "crawl.page",
            "crawl.completed",
            "crawl.failed"
        ],
    },
}
firecrawl.crawl(**payload)
```

> Events are namespaced per job (`crawl.started`, `crawl.page`, `crawl.completed`, `crawl.failed`, etc.) and Firecrawl v2.2.0 signs every webhook delivery by default, so consumers must verify signatures before processing payloads.

If you prefer the SDK helper, the same configuration can be expressed as:
```python
firecrawl.setup_webhook(
    url="https://hooks.internal/mining",
    metadata={"pipeline": "mining"},
    events=["crawl.started", "crawl.page", "crawl.completed", "crawl.failed"],
)
# which produces the equivalent job payload as the inline `webhook` block above.
```

**Simplification Benefits:**
- 100% code elimination (200+ lines → 0 lines)
- Native signature verification
- Built-in event handling and retries
- Automatic webhook management

### Async Processing → AsyncFirecrawl

**Current Implementation (`async_processing.py` – 300+ lines):**
- Custom `ConcurrencyManager` and `AsyncResultAggregator`
- Manual async orchestration and resource management
- Complex parallel processing pipelines
- Custom throttling and execution control

**Firecrawl v2.2.0 Native:**
```python
from firecrawl import AsyncFirecrawl

async_firecrawl = AsyncFirecrawl(api_key="your-key")
result = await async_firecrawl.scrape(url)
```

**Simplification Benefits:**
- 100% code elimination (300+ lines → 0 lines)
- Native async support with optimal performance
- Built-in resource management and throttling
- Automatic concurrency optimization

### URL Processing → Native URL Discovery

**Current Implementation (`url_processing.py` – 400+ lines):**
- Custom URL mapping and discovery logic
- Manual domain classification and filtering
- Complex caching and deduplication
- Custom academic URL filtering

**Firecrawl v2.2.0 Native:**
```python
url_map = firecrawl.map(url)  # 15x faster native mapping
```

**Simplification Benefits:**
- 95% code reduction (400+ lines → 20 lines)
- 15x faster URL discovery
- Built-in domain intelligence
- Native deduplication and filtering

### Client Management → Direct SDK Usage

**Current Implementation (`client.py` – 349 lines):**
- Singleton pattern with complex state management
- Custom health checking and connection monitoring
- Manual API key validation and client initialization
- Complex async method handling

**Firecrawl v2.2.0 Native:**
```python
from firecrawl import Firecrawl

firecrawl = Firecrawl(api_key="your-key")
# Client ready to use immediately
```

**Simplification Benefits:**
- 95% code reduction (349 lines → 15–20 lines)
- Built-in health checking and monitoring
- Automatic connection management
- Native async support

## Models Simplification

### Keep Essential Models

**`ConceptDefinition`** – Core data structure
```python
class ConceptDefinition(BaseModel):
    concept: str
    definition: str
    context: str
    key_points: List[str] = []
    related_concepts: List[str] = []
    source_quality: float = 0.0
```

**`WebResource`** – Result container
```python
class WebResource(BaseModel):
    url: str
    title: str = ""
    definitions: List[ConceptDefinition] = []
```

### Remove Complex Models

- `QueueStatus` → Use Firecrawl's native queue status
- `ApiUsageStats` → Use Firecrawl's native usage tracking
- `WebhookConfig` → Use Firecrawl's native webhook config
- `PerformanceProfile` → Use Firecrawl's native performance management
- `QueuePredictor` → Use Firecrawl's native queue intelligence

## Summary of Simplification

### Total Code Reduction
- **Before**: 9+ files, ~3000+ lines
- **After**: 5 files, ~500–600 lines
- **Reduction**: ~80% code elimination

### Feature Improvements
- **Performance**: 500% faster batch processing, 15x faster URL mapping
- **Reliability**: Battle-tested Firecrawl features vs custom implementations
- **Maintainability**: Direct SDK usage vs complex abstractions
- **Future-proof**: Automatic access to new Firecrawl features

### Preserved Functionality
- All core mining capabilities maintained
- Academic concept extraction preserved
- CLI interface and configuration system
- Essential data models and result formats
