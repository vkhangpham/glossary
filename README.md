# Academic Glossary Analysis

This repository contains tools for generating, validating, analyzing, and visualizing an academic glossary across different hierarchy levels.

## Hierarchy Structure

The academic glossary is structured in 5 levels:

- **Level 0**: Broad academic domains - corresponds to Colleges of a University (99% F1 score)
- **Level 1**: Academic fields - corresponds to Departments of a College (99% F1 score)
- **Level 2**: Specialized topics - corresponds to Research Areas of a Department (~90-95% F1 score)
- **Level 3**: Conference/journal topics - corresponds to specialized topics discussed in academic conferences and journals (~90-95% F1 score)
- **Level 4**: Individual academic papers - corresponds to specific research publications

## Project Components

1. **Generation Pipeline**: Multi-stage pipeline to extract academic concepts from various sources
2. **Validation System**: Rule-based, web-based, and LLM-based validation of extracted concepts
3. **Deduplication Framework**: Graph-based, rule-based, and LLM-based methods for removing duplicates
4. **Hierarchy Builder**: Construction of parent-child relationships between terms
5. **Visualization Tools**: Interactive interfaces for exploring the hierarchy
6. **Evaluation Framework**: Analysis of the hierarchy quality with metrics and visualizations
7. **Prompt Management System**: Centralized, versioned prompt registry with GEPA optimization support

## Setup

```bash
git clone https://github.com/yourusername/glossary-analysis.git
cd glossary-analysis
uv sync

export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export FIRECRAWL_API_KEY="your-firecrawl-key"
```

## Web Mining with Firecrawl v2.2.0

Firecrawl v2.2.0 integration with modular architecture provides 15x performance improvement through:

- 15x faster URL discovery with Map endpoint
- 500% faster batch scraping with queue monitoring
- Real-time queue analytics and intelligent throttling
- PDF page limits for performance optimization
- Enhanced webhooks with signature verification
- Structured extraction with Pydantic schemas
- Performance profiles (speed/accuracy/balanced)

### Modular Architecture
Mining module refactored into 9 focused modules:
- `models.py` - Data models and validation
- `client.py` - Firecrawl client management
- `performance.py` - Performance profiles and auto-tuning
- `queue_management.py` - Queue monitoring and analytics
- `url_processing.py` - URL mapping and optimization
- `api_tracking.py` - Usage analytics and cost estimation
- `webhooks.py` - Webhook configuration and handling
- `async_processing.py` - Concurrency management
- `core_mining.py` - Main mining functions
- `mining.py` - Unified facade (backward compatibility)

### Usage
```bash
# Basic usage (existing code unchanged)
uv run mine-web concepts.txt --output results.json

# With v2.2.0 features
uv run mine-web concepts.txt --output results.json \
  --max-pages 10 --queue-status --use-map-endpoint
```

## Usage

### Running the Full Pipeline

```bash
# Level 0 Generation Pipeline
python -m generate_glossary.generation.lv0.lv0_s0_get_college_names
python -m generate_glossary.generation.lv0.lv0_s1_extract_concepts --provider openai
python -m generate_glossary.generation.lv0.lv0_s2_filter_by_institution_freq
python -m generate_glossary.generation.lv0.lv0_s3_verify_single_token --provider openai

# Similar patterns for Level 1, 2, and 3...
```

### Building and Visualizing the Hierarchy

```bash
# Build the hierarchy
python -m hierarchy.hierarchy_builder -o data/hierarchy.json --verbose

# Evaluate the hierarchy
python -m hierarchy.hierarchy_evaluator_cli --save-all --verbose

# Start the visualization server
python -m hierarchy.hierarchy_visualizer -p 5000
```

### Validation System

The project features a functional validation system with immutable data structures:

```python
from generate_glossary.validation.core import validate_terms_functional, ValidationConfig
from generate_glossary.validation.config import get_profile

config = get_profile("academic")
results = validate_terms_functional(
    ["machine learning", "artificial intelligence"],
    config=config
)

for term, result in results.items():
    print(f"{term}: {'✓' if result.is_valid else '✗'} (confidence: {result.confidence:.2f})")
```

## Documentation

For detailed documentation on specific components:

- [Generation Documentation](generate_glossary/generation/README.md)
- [Validation Documentation](generate_glossary/validation/README.md)
- [Deduplication Documentation](generate_glossary/deduplication/README.md)
- [Hierarchy Documentation](hierarchy/README.md)
- [Mining Module Documentation](generate_glossary/mining/README.md)
- [Changelog](CHANGELOG.md)

---

This project focuses on academic concept extraction and hierarchy building, follows functional programming principles throughout, and is designed for processing university-level academic structure.