# Mining Module

Modular web content extraction for academic research with Firecrawl v2.2.0 integration.

## Architecture

Mining module refactored from monolithic file into 9 focused modules:

```
generate_glossary/mining/
├── models.py              # Pydantic data models
├── client.py              # Firecrawl client management
├── performance.py         # Performance profiles and auto-tuning
├── queue_management.py    # Queue monitoring and analytics
├── url_processing.py      # URL mapping and optimization
├── api_tracking.py        # Usage analytics
├── webhooks.py            # Webhook configuration
├── async_processing.py    # Concurrency management
├── core_mining.py         # Main mining functions
├── mining.py              # Unified facade (backward compatibility)
└── cli.py                 # Command-line interface
```

## Firecrawl v2.2.0 Features

- **Queue Status Monitoring**: Real-time analytics with intelligent throttling
- **PDF Page Limits**: Performance optimization with maxPages parameter
- **15x Faster Map Endpoint**: Enhanced URL discovery
- **Enhanced Webhooks**: Signature verification and secure handling
- **Structured Extraction**: Pydantic schema-based validation
- **API Usage Tracking**: Analytics and cost optimization
- **Performance Profiles**: Speed/accuracy/balanced configurations

## Usage

### Basic Usage
```python
from generate_glossary.mining import mine_concepts

# Existing code works unchanged
results = mine_concepts(["machine learning", "AI"])

# Add v2.2.0 features
results = mine_concepts(
    concepts=["machine learning", "AI"],
    max_pages=10,
    use_queue_monitoring=True,
    use_map_endpoint=True,
    performance_profile="speed"
)
```

### CLI Usage
```bash
# Basic usage
uv run mine-web concepts.txt --output results.json

# With v2.2.0 features
uv run mine-web concepts.txt --output results.json \
  --max-pages 10 \
  --queue-status \
  --use-map-endpoint
```

## Module Documentation

### models.py
Pydantic models for type-safe data handling:
```python
from generate_glossary.mining.models import ConceptDefinition, WebResource

concept = ConceptDefinition(
    term="machine learning",
    definition="Method of data analysis",
    sources=["https://example.com"],
    confidence=0.95
)
```

### client.py
Firecrawl client management:
```python
from generate_glossary.mining.client import get_client, validate_api_key

client = get_client()
is_valid = validate_api_key("fc-your-key")
```

### performance.py
Performance optimization:
```python
from generate_glossary.mining.performance import configure_performance_profile

configure_performance_profile("speed")  # speed, accuracy, balanced
```

### queue_management.py
Queue monitoring:
```python
from generate_glossary.mining.queue_management import get_queue_status

status = get_queue_status()
print(f"Queue load: {status.load_percentage}%")
```

## Testing

```bash
# Run all tests
uv run pytest tests/mining/ -v

# Run validation script
python test_execution_script.py
```

## Migration

All existing code works without changes. New v2.2.0 features are opt-in:

```python
# Before (still works)
results = mine_concepts(["AI"])

# After (optional enhancements)
results = mine_concepts(["AI"], max_pages=10, use_queue_monitoring=True)
```

See [Migration Guide](../../docs/v2_2_0_migration_guide.md) for details.