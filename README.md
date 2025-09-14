# Academic Glossary Analysis

This repository contains tools for generating, validating, analyzing, and visualizing an academic glossary across different hierarchy levels.

## Hierarchy Structure

## Hierarchy Structure

The academic glossary is structured in 5 levels:

- **Level 0**: Broad academic domains - corresponds to Colleges of a University (99% F1 score)
- **Level 1**: Academic fields - corresponds to Departments of a College (99% F1 score)
- **Level 2**: Specialized topics - corresponds to Research Areas of a Department (~90-95% F1 score)
- **Level 3**: Conference/journal topics - corresponds to specialized topics discussed in academic conferences and journals (~90-95% F1 score)
- **Level 4**: Individual academic papers - corresponds to specific research publications

## Project Components

The project consists of several integrated components:

1. **Generation Pipeline**: Multi-stage pipeline to extract academic concepts from various sources
2. **Validation System**: Rule-based, web-based, and LLM-based validation of extracted concepts
3. **Deduplication Framework**: Graph-based, rule-based, and LLM-based methods for removing duplicates
4. **Hierarchy Builder**: Construction of parent-child relationships between terms
5. **Visualization Tools**: Interactive interfaces for exploring the hierarchy
6. **Evaluation Framework**: Analysis of the hierarchy quality with metrics and visualizations
7. **Prompt Management System**: Centralized, versioned prompt registry with GEPA optimization support

## Directory Structure

```
generate_glossary/
├── generation/             # Components for initial term extraction
│   ├── lv0/                # Level 0 generation scripts (colleges)
│   ├── lv1/                # Level 1 generation scripts (departments) 
│   ├── lv2/                # Level 2 generation scripts (research areas)
│   └── lv3/                # Level 3 generation scripts (conference topics)
├── validation/             # Term validation components
├── deduplication/          # Graph-based deduplication module
├── utils/                  # Shared utilities
│   ├── web_search/         # Web search and content extraction utilities
│   └── ...                 # Other utilities
prompts/
├── __init__.py             # Public API exports
├── registry.py             # Pure functional prompt access
├── storage.py              # JSON-based versioned storage
├── optimizer/              # GEPA-based prompt optimization
│   ├── concept_extraction_adapter.py  # Level 0-3 extraction adapters
│   └── optimizer.py        # High-level optimization API
└── data/library/           # Versioned prompt library
    ├── extraction/         # All extraction prompts by level
    ├── validation/         # Validation prompts
    └── deduplication/      # Deduplication prompts

data/
├── lv0/                    # Level 0 data
│   ├── raw/                # Generated terms
│   ├── postprocessed/      # Validated and deduplicated terms
│   ├── metadata.json       # Term metadata
│   └── lv0_resources.json  # Web content for terms
├── lv1/                    # Level 1 data (similar structure)
├── lv2/                    # Level 2 data (similar structure)
├── lv3/                    # Level 3 data (similar structure)
├── final/                  # Final processed data
│   ├── lv0/                # Level 0 final terms
│   ├── lv1/                # Level 1 final terms
│   ├── lv2/                # Level 2 final terms
│   └── lv3/                # Level 3 final terms
├── hierarchy.json          # Complete hierarchy structure
└── evaluation/             # Hierarchy evaluation results
```

## Glossary Pipeline Overview

The glossary is built level-by-level through a multi-stage pipeline:

### 1. Generation Phase

This phase extracts and initially refines potential terms:

1. **Step 0: Initial Term Extraction**: Extracts candidate terms from appropriate sources
   - **L0**: College/School names from faculty data
   - **L1**: Department names from web searches using L0 terms
   - **L2**: Research areas from web searches using L1 terms
   - **L3**: Conference topics from web searches using L2 terms

2. **Step 1: Concept Extraction**: Uses LLMs to extract standardized academic concepts
   - Example: "Department of Electrical and Computer Engineering" → "electrical engineering", "computer engineering"

3. **Step 2: Frequency Filtering**: Filters concepts based on occurrence frequency
   - Keeps concepts that appear in a minimum percentage of sources

4. **Step 3: Verification**: Uses LLMs to verify single-word concepts
   - Ensures single words are valid academic concepts (e.g., "arts", "law")
   - Multi-word concepts typically bypass this verification

### 2. Processing Phase

After generation, concepts undergo further processing:

1. **Web Content Mining**: Searches the web to gather content for each concept
   - Collects definitions, descriptions, and related information

2. **Validation**: Applies multiple validation methods using the functional validation system
   - **Rule-based**: Basic structural checks (length, format, blacklist)
   - **Web-based**: Validates against web content with relevance scoring
   - **LLM-based**: Final validation with language models
   - **Functional Architecture**: Pure functional design with immutable data structures
   - **Configuration Profiles**: Pre-configured validation profiles (academic, strict, fast, etc.)
   - **Caching**: Persistent caching for performance optimization

3. **Deduplication**: Graph-based identification and grouping of term variations
   - **Graph-first architecture**: Terms are nodes, duplicates are connected components
   - **Three edge creation methods**:
     - **Rule-based**: Text similarity, compound terms, acronyms
     - **Web-based**: URL overlap, domain patterns, content similarity
     - **LLM-based**: Semantic analysis of terms with minimal URL overlap

4. **Metadata Collection**: Consolidates information about each term
   - Sources, parent-child relationships, variations, resources

### 3. Hierarchy Building

After processing all levels, a complete hierarchy is built:

1. **Parent-Child Relationships**: Establishes connections between terms
2. **Variation Consolidation**: Merges metadata from variations
3. **Resource Transfer**: Transfers content between related terms

### 4. Hierarchy Evaluation & Visualization

The completed hierarchy can be analyzed and visualized:

1. **Quality Metrics**: Calculates structural and connectivity metrics
2. **Issue Detection**: Identifies problems in the hierarchy
3. **Interactive Visualization**: Provides a web interface for exploration

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/glossary-analysis.git
cd glossary-analysis

# Install requirements
uv sync

# Set up API keys (for LLM-based operations)
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

## Web Mining with Firecrawl (NEW - 4x Faster)

The project now supports Firecrawl SDK for web content mining, providing:
- **4x faster extraction** compared to the original approach
- **Single API call** instead of complex HTML parsing pipeline
- **Built-in AI extraction** with structured schemas
- **Automatic JavaScript rendering** for dynamic sites
- **80% cost reduction** compared to separate search + LLM calls

### Setup Firecrawl:
1. Get your API key from [firecrawl.dev](https://www.firecrawl.dev) ($83/month for 10,000 concepts)
2. Add to environment: `export FIRECRAWL_API_KEY='fc-your-key'`
3. Test integration: `python test_firecrawl.py`

### Web Mining:
```bash
# Modern web mining with Firecrawl (recommended)
export FIRECRAWL_API_KEY='fc-your-key'
uv run mine-web -i data/lv0/raw/lv0_s3_verified_concepts.txt -o data/lv0/lv0_resources

# Using the consolidated mining module (new in cd3799d)
python -m generate_glossary.mining.cli -i data/lv0/raw/lv0_s3_verified_concepts.txt -o data/lv0/lv0_resources
```

## Usage

### Prompt Management System

The project now includes a centralized prompt management system with GEPA optimization support:

```python
from prompts import get_prompt, register_prompt

# Get a prompt from the registry
system_prompt = get_prompt("extraction.level0_system")
user_prompt = get_prompt("extraction.level0_user", keyword="engineering")

# Register a new prompt programmatically
register_prompt(
    key="validation.custom",
    prompt="Your custom validation prompt here",
    variables=["term"]
)

# Run prompt optimization directly
python prompt_optimization/optimizers/lv0_s1.py  # For concept extraction
python prompt_optimization/optimizers/lv0_s3.py  # For discipline verification

# See prompt_optimization/README_SIMPLIFIED.md for detailed instructions

# Test prompt optimization setup
python test_prompt_optimization.py
```

The prompt system provides:
- **Centralized Storage**: All prompts in `prompts/data/library/` as versioned JSON
- **SHA256 Versioning**: Automatic version tracking on prompt changes
- **Template Variables**: Support for `{variable}` substitution
- **Pure Functional API**: No OOP, follows project patterns
- **GEPA Optimization**: Evolutionary optimization for better prompts via DSPy 3.0+
- **Performance**: <0.1ms prompt loading with LRU caching
- **Batch Training**: Matches production usage (batches of 20 institutions)

See `prompt_optimization/README.md` for detailed usage and `prompt_optimization/TROUBLESHOOTING.md` for common issues.

### Running the Full Pipeline

The generation pipeline is run level-by-level using individual scripts. Each level follows the same 4-step process:

```bash
# Level 0 Generation Pipeline
python -m generate_glossary.generation.lv0.lv0_s0_get_college_names
python -m generate_glossary.generation.lv0.lv0_s1_extract_concepts --provider openai
python -m generate_glossary.generation.lv0.lv0_s2_filter_by_institution_freq
python -m generate_glossary.generation.lv0.lv0_s3_verify_single_token --provider openai

# Level 1 Generation Pipeline (uses Level 0 results)
python -m generate_glossary.generation.lv1.lv1_s0_get_dept_names --input data/lv0/lv0_final.txt
python -m generate_glossary.generation.lv1.lv1_s1_extract_concepts --provider openai
python -m generate_glossary.generation.lv1.lv1_s2_filter_by_institution_freq
python -m generate_glossary.generation.lv1.lv1_s3_verify_single_token --provider openai

# Similar patterns for Level 2 and Level 3...
```

### Building and Visualizing the Hierarchy

After processing all levels:

```bash
# Build the hierarchy
python -m hierarchy.hierarchy_builder -o data/hierarchy.json --verbose

# Evaluate the hierarchy
python -m hierarchy.hierarchy_evaluator_cli --save-all --verbose

# Start the visualization server
python -m hierarchy.hierarchy_visualizer -p 5000
```

Then access the visualization at: `http://localhost:5000`

### Duplicate Analysis

To analyze potential duplicates:

```bash
# Analyze Level 2 duplicates
python analyze_duplicates.py -l 2 -s 0.7 -c 0.3 -m 0.1 -v

# Generate review spreadsheet
python duplicate_analyzer.py
```

## Documentation

## Validation System

The project features a modern **functional validation system** with immutable data structures and pure functions:

### Functional Validation API

```python
from generate_glossary.validation.core import (
    validate_terms_functional, ValidationConfig, ValidationResult
)
from generate_glossary.validation.config import get_profile

# Use pre-configured profiles
config = get_profile("academic")  # or "strict", "fast", "technical", etc.

# Functional validation with immutable results
results = validate_terms_functional(
    ["machine learning", "artificial intelligence", "deep learning"],
    config=config,
    web_content=web_content  # Optional web content for web validation
)

# Results are immutable ValidationResult objects
for term, result in results.items():
    print(f"{term}: {'✓' if result.is_valid else '✗'} "
          f"(confidence: {result.confidence:.2f})")
```

### Configuration Profiles

The system provides pre-configured profiles for different use cases:

```python
from generate_glossary.validation.config import (
    get_profile, get_recommended_profile, list_profiles
)

# Available profiles
profiles = list_profiles()  # ['academic', 'strict', 'fast', 'technical', ...]

# Get profile for specific use case
academic_config = get_profile("academic")      # Balanced for academic terms
strict_config = get_profile("strict")          # High-quality, comprehensive validation
fast_config = get_profile("fast")              # Speed-optimized, rule-based only
technical_config = get_profile("technical")    # Optimized for technical terms

# Smart profile selection
config = get_recommended_profile("quality")    # Returns comprehensive_profile
config = get_recommended_profile("speed")      # Returns fast_profile
```

### Custom Configuration

```python
from generate_glossary.validation.config import (
    create_validation_config, create_rule_config, create_web_config
)

# Create custom configuration
custom_config = create_validation_config(
    modes=("rule", "web"),
    min_confidence=0.8,
    parallel=True,
    rule_config=create_rule_config(
        min_term_length=3,
        max_term_length=50
    ),
    web_config=create_web_config(
        min_relevant_sources=2,
        min_score=0.7
    )
)
```

### Caching and Performance

```python
from generate_glossary.validation.cache import load_cache_from_disk
from generate_glossary.validation.core import validate_terms_with_cache

# Load cache for performance
cache_state = load_cache_from_disk()

# Validation with caching
results, updated_cache = validate_terms_with_cache(
    terms=["cached_term", "new_term"],
    config=config,
    cache_state=cache_state,
    auto_save=True  # Automatically save cache updates
)
```

### Legacy Compatibility

The functional system maintains full backward compatibility:

```python
from generate_glossary.validation import validate_terms, ValidationModes

# Legacy API still works
results = validate_terms(
    ["term1", "term2"],
    modes=[ValidationModes.RULE, ValidationModes.WEB],
    min_confidence=0.6
)
```

### Migration Guide

For detailed migration instructions from the legacy API to the functional system, see:
- [Migration Guide](docs/validation-migration-guide.md)

### Performance Improvements

The functional validation system provides:

- **Immutable Data Structures**: Thread-safe, memory-efficient validation results
- **Pure Functions**: Predictable, testable, composable validation logic
- **Persistent Caching**: Disk-based caching with automatic expiry management
- **Parallel Processing**: Optimized concurrent validation across multiple modes
- **Configuration Profiles**: Pre-tuned settings for different validation scenarios
- **Functional Composition**: Higher-order functions for custom validation pipelines

## Documentation

For detailed documentation on specific components:

- [Generation Documentation](generate_glossary/generation/README.md)
- [Validation Documentation](generate_glossary/validation/README.md)
- [Validation Migration Guide](docs/validation-migration-guide.md)
- [Deduplication Documentation](generate_glossary/deduplication/README.md)
- [Hierarchy Documentation](hierarchy/README.md)
- [Disambiguation Documentation](generate_glossary/disambiguation/README.md)
- [Changelog](CHANGELOG.md)


## Web Mining

Web mining uses the consolidated `generate_glossary.mining` module with Firecrawl exclusively. The new architecture provides:
- Unified `mining.py` core module (1,142 lines) replacing old multi-file structure
- Dedicated CLI interface (`cli.py`) for streamlined operations
- Backward compatibility with existing commands
- Enhanced error handling and type hints
- Single source of truth for all web mining functionality

Tavily is currently not supported via the CLI.
