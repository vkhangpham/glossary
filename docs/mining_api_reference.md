# Mining API Reference

## Core Functions

### mine_concepts()
Main mining function with v2.2.0 features:
```python
def mine_concepts(
    concepts: List[str],
    max_pages: Optional[int] = None,
    use_queue_monitoring: bool = False,
    use_map_endpoint: bool = False,
    performance_profile: Optional[str] = None
) -> List[ConceptDefinition]
```

### search_concepts_batch()
Batch search with academic filtering:
```python
def search_concepts_batch(
    concepts: List[str],
    research_categories: Optional[List[str]] = None,
    academic_focus: bool = True
) -> Dict[str, List[WebResource]]
```

### batch_scrape_urls()
High-performance batch scraping:
```python
def batch_scrape_urls(
    urls: List[str],
    intelligent_throttling: bool = True,
    max_pages_per_url: Optional[int] = None
) -> List[WebResource]
```

## Client Management

### get_client()
Get singleton Firecrawl client:
```python
def get_client() -> Optional[FirecrawlClient]
```

### validate_api_key()
Validate API key:
```python
def validate_api_key(api_key: Optional[str] = None) -> bool
```

## Performance Management

### configure_performance_profile()
Apply performance profile:
```python
def configure_performance_profile(profile_name: str) -> PerformanceProfile
# profile_name: "speed", "accuracy", "balanced"
```

### get_performance_status()
Get performance metrics:
```python
def get_performance_status() -> Dict[str, Any]
```

## Queue Management

### get_queue_status()
Get real-time queue status:
```python
def get_queue_status() -> QueueStatus
```

### apply_intelligent_throttling()
Apply adaptive throttling:
```python
def apply_intelligent_throttling(queue_status: QueueStatus) -> float
```

## Data Models

### ConceptDefinition
```python
class ConceptDefinition(BaseModel):
    term: str
    definition: str
    sources: List[str]
    confidence: float
    extracted_at: datetime
```

### QueueStatus
```python
class QueueStatus(BaseModel):
    load_percentage: float
    estimated_wait_time: float
    active_jobs: int
    timestamp: datetime
```

### WebResource
```python
class WebResource(BaseModel):
    url: str
    content: str
    metadata: Dict[str, Any]
    extracted_at: datetime
```

## Examples

```python
from generate_glossary.mining import mine_concepts, get_queue_status

# Basic usage
results = mine_concepts(["machine learning"])

# With v2.2.0 features
results = mine_concepts(
    concepts=["AI", "ML"],
    max_pages=10,
    use_queue_monitoring=True,
    performance_profile="speed"
)

# Monitor queue
status = get_queue_status()
print(f"Queue load: {status.load_percentage}%")
```