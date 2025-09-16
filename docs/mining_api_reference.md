# Mining API Reference

## Core Functions

### mine_concepts()
Main mining function with comprehensive v2.2.0 feature support:

```python
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
    **kwargs
) -> Dict[str, Any]
```

**Returns:**
Dictionary containing:
- `results`: Dictionary of extracted concepts and definitions
- `statistics`: Performance metrics dict with `total_concepts`, `successful`, and `total_resources` counts
- `v2_2_0_features_used`: Dictionary of v2.2.0 feature flags that were enabled
- `v2_2_0_features_used_list`: List of v2.2.0 features that were utilized
- `processing_time_seconds`: Total processing time in seconds
- `concepts_processed`: Number of successfully processed concepts

### search_concepts_batch()
Batch search with Firecrawl's search endpoint:

```python
def search_concepts_batch(
    app: FirecrawlApp,
    concepts: List[str],
    max_urls_per_concept: int = 3
) -> Dict[str, List[Dict[str, Any]]]
```

**Returns:**
Dictionary mapping concept names to lists of search results, where each result contains:
- `url`: The URL of the found resource
- `title`: Page title
- `snippet`: Text snippet containing the concept

### batch_scrape_urls()
High-performance batch scraping with v2.2.0 enhancements:

```python
async def batch_scrape_urls(
    app: FirecrawlApp,
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
    enable_api_tracking: bool = False
) -> Dict[str, Any]
```

**Returns:**
Dictionary containing either:
- Job-based result: `{"data": {...}}` or `{"results": [...]}`
- Fallback structure: `{"data": {url: result}}`

## Client Management

### get_client()
Get configured Firecrawl client:

```python
def get_client() -> Optional[FirecrawlApp]
```

### validate_api_key()
Validate Firecrawl API key format:

```python
def validate_api_key(api_key: Optional[str] = None) -> bool
```

## Queue Management

### get_queue_status()
Get real-time Firecrawl queue status:

```python
def get_queue_status() -> Optional[Dict[str, Any]]
```

**Returns:**
Dictionary or QueueStatus object with queue metrics, or None if unavailable.

### apply_intelligent_throttling()
Apply adaptive throttling based on queue status:

```python
async def apply_intelligent_throttling(app: FirecrawlApp) -> float
```

**Returns:**
Recommended delay in seconds before next operation.

## Data Models

### ConceptDefinition
Schema for extracted academic concept definitions:

```python
class ConceptDefinition(BaseModel):
    concept: str = Field(description="The academic term or concept")
    definition: str = Field(description="Clear, comprehensive definition")
    context: str = Field(description="Academic field or domain")
    key_points: List[str] = Field(default=[], description="Key characteristics")
    related_concepts: List[str] = Field(default=[], description="Related terms")
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="0..1 source quality score")
```

### WebResource
Schema for web resources containing definitions:

```python
class WebResource(BaseModel):
    url: str
    title: str = ""
    definitions: List[ConceptDefinition] = Field(default_factory=list)
    domain: str = ""  # Auto-populated from URL
```

### QueueStatus
Schema for Firecrawl v2.2.0 queue status response:

```python
class QueueStatus(BaseModel):
    jobs_in_queue: int = Field(description="Number of jobs currently in queue")
    active_jobs: int = Field(description="Number of active jobs being processed")
    queued_jobs: int = Field(description="Number of jobs waiting to be processed")
    waiting_jobs: int = Field(description="Number of jobs waiting to be processed")
    completed_jobs: int = Field(default=0, description="Number of completed jobs")
    failed_jobs: int = Field(default=0, description="Number of failed jobs")
    max_concurrency: int = Field(description="Maximum concurrent jobs allowed")
    processing_time_avg: float = Field(default=0.0, description="Average processing time")
    queue_utilization: float = Field(default=0.0, description="Queue utilization ratio")
    estimated_wait_time: int = Field(default=0, description="Estimated wait time in seconds")
    most_recent_success: Optional[str] = Field(default=None, description="Timestamp of most recent successful job")
```

### WebhookConfig
Configuration for webhook notifications:

```python
class WebhookConfig(BaseModel):
    url: str = Field(description="Webhook endpoint URL")
    events: List[str] = Field(default=["started", "page", "completed", "failed"], description="Events to subscribe to")
    secret: Optional[str] = Field(default=None, description="Secret for webhook signature verification")
    verify_signature: bool = Field(default=True, description="Whether to verify webhook signatures")
    retry_on_failure: bool = Field(default=True, description="Whether to retry failed webhook calls")
```

### PerformanceProfile
Performance tuning profiles for different use cases:

```python
class PerformanceProfile(BaseModel):
    name: str = Field(description="Profile name")
    max_concurrent: int = Field(description="Maximum concurrent operations")
    polling_strategy: str = Field(default="adaptive", description="Polling strategy: adaptive, exponential, linear")
    queue_threshold: float = Field(default=0.8, description="Queue load threshold for throttling")
    cache_priority: str = Field(default="balanced", description="Cache priority: speed, accuracy, balanced")
    timeout_multiplier: float = Field(default=1.0, description="Timeout adjustment multiplier")
    retry_strategy: str = Field(default="intelligent", description="Retry strategy: aggressive, conservative, intelligent")

    @classmethod
    def speed_optimized(cls) -> 'PerformanceProfile'

    @classmethod
    def accuracy_focused(cls) -> 'PerformanceProfile'

    @classmethod
    def balanced(cls) -> 'PerformanceProfile'
```

### ApiUsageStats
Schema for comprehensive API usage tracking:

```python
class ApiUsageStats(BaseModel):
    search_calls: int = Field(default=0, description="Number of search API calls")
    scrape_calls: int = Field(default=0, description="Number of scrape API calls")
    extract_calls: int = Field(default=0, description="Number of extract API calls")
    map_calls: int = Field(default=0, description="Number of map API calls")
    batch_scrape_calls: int = Field(default=0, description="Number of batch scrape API calls")
    queue_status_calls: int = Field(default=0, description="Number of queue status API calls")
    total_calls: int = Field(default=0, description="Total API calls made")

    # Performance metrics
    call_durations: Dict[str, List[float]] = Field(default_factory=dict, description="API call duration tracking")
    error_counts: Dict[str, int] = Field(default_factory=dict, description="Error count by call type")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    throttle_events: int = Field(default=0, description="Number of throttling events")
    concurrent_operations: int = Field(default=0, description="Peak concurrent operations")
    call_history: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed call history")

    def add_call(self, call_type: str, duration: float = 0.0, success: bool = True, cached: bool = False, error: bool = False)
    def get_performance_metrics(self) -> Dict[str, Any]
```

## Usage Examples

### Basic Usage
```python
from generate_glossary.mining import mine_concepts

# Simple concept mining
results = mine_concepts(["machine learning", "artificial intelligence"])

# Access results
concepts_found = results.get("results", {})
stats = results.get("statistics", {})
features_used = results.get("v2_2_0_features_used_list", [])

print(f"Found {len(concepts_found)} concepts")
print(f"Features used: {features_used}")
print(f"Processing time: {results.get('processing_time_seconds', 0):.2f}s")
```

### Advanced Usage with v2.2.0 Features
```python
from generate_glossary.mining import mine_concepts
from generate_glossary.mining.models import WebhookConfig, PerformanceProfile

# Advanced configuration
webhook_config = WebhookConfig(
    url="https://your-webhook.com/endpoint",
    events=["completed", "failed"],
    verify_signature=True
)

performance_profile = PerformanceProfile.speed_optimized()

results = mine_concepts(
    concepts=["deep learning", "neural networks"],
    max_pages=15,  # v2.2.0: PDF page limit
    enable_queue_monitoring=True,  # v2.2.0: Queue monitoring
    use_fast_map=True,  # v2.2.0: 15x faster Map endpoint
    webhook_config=webhook_config,  # v2.2.0: Enhanced webhooks
    use_batch_scrape=True,  # 500% performance improvement
    use_summary=True,  # Optimized content extraction
    max_concurrent=15,
    output_path="results.json"
)

# Check comprehensive results
print(f"Processing time: {results['processing_time_seconds']:.2f}s")
print(f"Concepts processed: {results['concepts_processed']}")
print(f"v2.2.0 features: {', '.join(results['v2_2_0_features_used_list'])}")
print(f"Statistics: {results['statistics']}")
```

### Queue Monitoring and Performance
```python
from generate_glossary.mining.queue_management import get_queue_status
from generate_glossary.mining.models import QueueStatus

# Monitor queue status
status = get_queue_status()
if status and isinstance(status, dict):
    queue_status = QueueStatus.model_validate(status)
    print(f"Queue utilization: {queue_status.queue_utilization:.1%}")
    print(f"Active jobs: {queue_status.active_jobs}")
    print(f"Estimated wait: {queue_status.estimated_wait_time}s")
```

### Error Handling
```python
try:
    results = mine_concepts(["complex topic"])

    if "error" in results:
        print(f"Mining failed: {results['error']}")
    else:
        # Process successful results
        concepts = results.get("results", {})
        for concept_name, concept_data in concepts.items():
            print(f"Found: {concept_name}")
            print(f"  Definitions: {len(concept_data.get('definitions', []))}")
            print(f"  Sources: {len(concept_data.get('sources', []))}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Return Value Details

The `mine_concepts()` function returns a dictionary with the following structure:

```python
{
    "success": true,               # Whether the operation succeeded
    "results": {
        # Dict mapping concept strings to result objects
        "machine learning": {
            "definitions": [
                {
                    "concept": "machine learning",
                    "definition": "A method of data analysis...",
                    "context": "Computer Science",
                    "source_quality": 0.95
                }
            ],
            "sources": ["https://example.com/ml-guide"],
            "resource_count": 15
        }
    },
    "statistics": {
        "total_concepts": 10,      # Total concepts processed
        "successful": 5,          # Successful extractions
        "total_resources": 45      # Total web resources found
    },
    "v2_2_0_features_used": {
        "batch_scraping": true,    # Batch scraping was enabled
        "queue_monitoring": true,  # Queue monitoring was enabled
        "fast_mapping": true,      # Fast Map endpoint was used
        "pdf_page_limit": true,    # PDF page limiting was used
        "webhooks": false          # Webhooks were not configured
    },
    "v2_2_0_features_used_list": [
        "batch_scraping",          # Features that were actually used
        "queue_monitoring",
        "fast_mapping",
        "pdf_page_limit"
    ],
    "processing_time_seconds": 120.5,  # Total processing time
    "concepts_processed": 5            # Successfully processed concepts
}
```

If an error occurs, the return structure will include:
```python
{
    "error": "Description of what went wrong",
    "statistics": {
        "total_concepts": 0,
        "successful": 0,
        "total_resources": 0
    },
    "v2_2_0_features_used": {},
    "v2_2_0_features_used_list": [],
    "processing_time_seconds": 0,
    "concepts_processed": 0
}
```