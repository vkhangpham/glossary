# Academic Concept Generation

This package provides tools for generating and extracting academic concepts across multiple hierarchy levels, from broad academic domains to specialized conference topics. The pipeline uses DSPy-based structured generation and supports prompt optimization through GEPA (Genetic-Pareto Evolutionary Algorithm).

## Overview

The concept generation pipeline is designed to build a hierarchical academic glossary through a level-by-level approach:

1. **Level 0**: Colleges/Schools - Broad academic domains (e.g., "College of Engineering")
2. **Level 1**: Departments - Academic units under colleges (e.g., "Department of Computer Science")
3. **Level 2**: Research Areas/Venues - Specialized areas under departments, also treated as venues
4. **Level 3**: Venue Areas - Specific topics covered in venues from Level 2

Each level builds upon the previous one, creating a parent-child relationship where terms from the level above serve as the context for generating terms at the current level.

### Test Mode

All generation scripts support a test mode for development and debugging:

- **Test mode behavior**: Uses smaller datasets and saves output to `data/test/` directory
- **Path handling in test mode**: 
  - Paths under `data/` are automatically redirected to `data/test/`
  - Custom input files outside `data/` directory are used as-is without redirection
  - This allows testing with custom datasets while preserving the isolation of test outputs
- **Running in test mode**: Add `--test` flag to any command:
  ```bash
  uv run generate -l 0 -s 0 --test
  # Or directly:
  python -m generate_glossary.generation.lv0.lv0_s0_get_college_names --test
  ```

## Hierarchical Relationships

The glossary follows a strict hierarchical structure:

```
Level 0: Colleges/Schools
  └── Level 1: Departments (under colleges)
       └── Level 2: Research Areas/Venues (under departments)
            └── Level 3: Venue Areas (under venues)
```

For example:
- **Level 0**: College of Engineering
  - **Level 1**: Department of Computer Science
    - **Level 2**: Machine Learning (research area/venue)
      - **Level 3**: Deep Learning, Reinforcement Learning (venue areas)

## Common Architecture

Each level follows a consistent four-step pipeline:

1. **Step 0 (`_s0_`)**: Initial term extraction
   - Extracts candidate terms from appropriate sources
   - Uses web search, data mining, or external data

2. **Step 1 (`_s1_`)**: Concept extraction
   - Extracts standardized academic concepts from raw terms
   - Uses LLMs to normalize and extract canonical terms

3. **Step 2 (`_s2_`)**: Frequency filtering
   - Filters concepts based on frequency across sources
   - Removes rare or potentially spurious concepts

4. **Step 3 (`_s3_`)**: Single-token verification
   - Verifies single-word concepts using LLMs
   - Ensures common single words are valid academic concepts
   - Multi-word concepts typically bypass this verification

## Level 0: Colleges/Schools

Level 0 represents the highest level in the concept hierarchy, focusing on broad academic institutions like "College of Arts and Sciences", "School of Engineering", etc.

### Methodology

1. **College Name Extraction** (`lv0_s0_get_college_names.py`):
   - Extracts college/school names from faculty data
   - Selects top institutions with the most colleges

2. **Concept Extraction** (`lv0_s1_extract_concepts.py`):
   - Uses LLMs to extract disciplines from college names
   - Example: "College of Arts and Sciences" → "arts", "sciences"

3. **Institution Frequency Filtering** (`lv0_s2_filter_by_institution_freq.py`):
   - Requires concepts to appear in at least 60% of institutions

4. **Single-Token Verification** (`lv0_s3_verify_single_token.py`):
   - Verifies concepts like "arts", "law" are valid academic domains

### Usage

```bash
# Using unified CLI
uv run generate -l 0 -s 0  # Extract college names
uv run generate -l 0 -s 1 --provider openai  # Extract concepts
uv run generate -l 0 -s 2  # Filter by frequency
uv run generate -l 0 -s 3 --provider openai  # Verify tokens

# Or using direct scripts
python -m generate_glossary.generation.lv0.lv0_s0_get_college_names

# Step 1: Extract concepts from college names
python -m generate_glossary.generation.lv0.lv0_s1_extract_concepts [--provider openai|gemini]

# Step 2: Filter concepts by institution frequency
python -m generate_glossary.generation.lv0.lv0_s2_filter_by_institution_freq

# Step 3: Verify single-token concepts
python -m generate_glossary.generation.lv0.lv0_s3_verify_single_token [--provider openai|gemini]
```

### Output

- `data/lv0/raw/lv0_s3_verified_concepts.txt`: Verified Level 0 concepts
- Metadata stored in corresponding JSON files

## Level 1: Departments

Level 1 represents academic departments that exist under the umbrella of colleges/schools identified in Level 0.

### Methodology

1. **Department Name Extraction** (`lv1_s0_get_dept_names.py`):
   - Uses Level 0 college terms to perform web searches for associated departments
   - Searches university websites for departments belonging to specific colleges
   - Validates department names with LLMs

2. **Concept Extraction** (`lv1_s1_extract_concepts.py`):
   - Extracts standardized fields from department names
   - Example: "Department of Electrical and Computer Engineering" → "electrical engineering", "computer engineering"

3. **Institution Frequency Filtering** (`lv1_s2_filter_by_institution_freq.py`):
   - Filters concepts based on appearance across institutions

4. **Single-Token Verification** (`lv1_s3_verify_single_token.py`):
   - Verifies single-word field names using LLMs

### Usage

```bash
# Using unified CLI
uv run generate -l 1 -s 0  # Extract department names (default: data/lv0/lv0_final.txt)
uv run generate -l 1 -s 1 --provider openai  # Extract concepts
uv run generate -l 1 -s 2  # Filter by frequency
uv run generate -l 1 -s 3 --provider openai  # Verify tokens

# Or using direct scripts
python -m generate_glossary.generation.lv1.lv1_s0_get_dept_names [--parent-level-file data/lv0/lv0_final.txt]

# Step 1: Extract concepts from department names
python -m generate_glossary.generation.lv1.lv1_s1_extract_concepts [--provider openai|gemini]

# Step 2: Filter concepts by institution frequency
python -m generate_glossary.generation.lv1.lv1_s2_filter_by_institution_freq

# Step 3: Verify single-token concepts
python -m generate_glossary.generation.lv1.lv1_s3_verify_single_token [--provider openai|gemini]
```

### Output

- `data/lv1/raw/lv1_s3_verified_concepts.txt`: Verified Level 1 concepts
- Metadata stored in corresponding JSON files

## Level 2: Research Areas/Venues

Level 2 represents specialized research areas under departments from Level 1. These research areas also serve as venues for academic discourse.

### Methodology

1. **Research Area Extraction** (`lv2_s0_get_research_areas.py`):
   - Uses Level 1 department terms to perform web searches for research areas
   - Searches department websites for research groups, labs, and specializations
   - Extracts research area lists using specialized web mining
   - Validates research areas with LLMs

2. **Concept Extraction** (`lv2_s1_extract_concepts.py`):
   - Extracts standardized research topics from raw research area names
   - Normalizes terminology across different departments

3. **Frequency Filtering** (`lv2_s2_filter_by_freq.py`):
   - Filters concepts based on frequency across departments
   - Ensures research areas are representative

4. **Single-Token Verification** (`lv2_s3_verify_single_token.py`):
   - Verifies single-word research areas using LLMs

### Key Features

- Research-specific keywords and patterns for filtering
- Customized scoring function for research area lists
- Enhanced web content mining with LLM validation

### Usage

```bash
# Using unified CLI
uv run generate -l 2 -s 0  # Extract research areas (default: data/lv1/lv1_final.txt)
uv run generate -l 2 -s 1 --provider openai  # Extract concepts
uv run generate -l 2 -s 2  # Filter by frequency
uv run generate -l 2 -s 3 --provider openai  # Verify tokens

# Or using direct scripts
python -m generate_glossary.generation.lv2.lv2_s0_get_research_areas [--input-file data/lv1/lv1_final.txt]

# Step 1: Extract concepts from research areas
python -m generate_glossary.generation.lv2.lv2_s1_extract_concepts [--provider openai|gemini]

# Step 2: Filter concepts by department frequency
python -m generate_glossary.generation.lv2.lv2_s2_filter_by_freq

# Step 3: Verify single-token concepts
python -m generate_glossary.generation.lv2.lv2_s3_verify_single_token [--provider openai|gemini]
```

### Output

- `data/lv2/raw/lv2_s3_verified_concepts.txt`: Verified Level 2 concepts
- Metadata stored in corresponding JSON files

## Level 3: Venue Areas

Level 3 represents the most specialized level, extracting specific topic areas covered within the venues (research areas) identified in Level 2.

### Methodology

1. **Venue Area Extraction** (`lv3_s0_get_conference_topics.py`):
   - Uses Level 2 venue terms to perform web searches for specialized topics
   - Searches for "call for papers", "special issues", and conference tracks
   - Identifies specific topics covered within these broader research areas
   - Extracts topic lists using specialized web mining
   - Validates topics with LLMs

2. **Concept Extraction** (`lv3_s1_extract_concepts.py`):
   - Extracts standardized venue-specific topics from raw topic names
   - Normalizes terminology across different venues

3. **Frequency Filtering** (`lv3_s2_filter_by_freq.py`):
   - Filters concepts based on frequency across venues
   - Ensures venue areas are representative

4. **Single-Token Verification** (`lv3_s3_verify_single_token.py`):
   - Verifies single-word venue areas using LLMs

### Key Features

- Venue-specific keywords and patterns for filtering
- Markdown and code block cleaning for LLM responses
- Enhanced preprocessing of HTML content

### Usage

```bash
# Using unified CLI
uv run generate -l 3 -s 0  # Extract conference topics (default: data/lv2/lv2_final.txt)
uv run generate -l 3 -s 1 --provider openai  # Extract concepts
uv run generate -l 3 -s 2  # Filter by frequency
uv run generate -l 3 -s 3 --provider openai  # Verify tokens

# Or using direct scripts
python -m generate_glossary.generation.lv3.lv3_s0_get_conference_topics [--input-file data/lv2/lv2_final.txt]

# Step 1: Extract concepts from venue areas
python -m generate_glossary.generation.lv3.lv3_s1_extract_concepts [--provider openai|gemini]

# Step 2: Filter concepts by venue frequency
python -m generate_glossary.generation.lv3.lv3_s2_filter_by_freq

# Step 3: Verify single-token concepts
python -m generate_glossary.generation.lv3.lv3_s3_verify_single_token [--provider openai|gemini]
```

### Output

- `data/lv3/raw/lv3_s3_verified_concepts.txt`: Verified Level 3 concepts
- Metadata stored in corresponding JSON files

## Shared Utilities

The generation pipeline uses several shared utilities:

### Web Search Utilities

- **Consolidated Mining Module** (`generate_glossary/mining/`): Unified web content extraction (new architecture)
  - `mining.py`: Core mining functionality with Firecrawl v2.0 integration
  - `cli.py`: Dedicated CLI interface for mining operations
  - Replaces old multi-file structure with single source of truth
- **Web Extraction** (`generate_glossary/generation/web_extraction_firecrawl.py`): Generation-specific extraction utilities

### LLM Integration

- **Provider Interface** (`utils/llm/`): Abstract interface for different LLM providers
- Support for OpenAI, Gemini, and other providers

## Running the Complete Pipeline

The generation pipeline can be run using either the unified CLI interface or individual scripts. Each level follows the same 4-step process:

### Using the Unified CLI (Recommended)

```bash
# Example: Complete Level 0 Pipeline
uv run generate -l 0 -s 0  # Get college names from Excel
uv run generate -l 0 -s 1 --provider openai  # Extract concepts via LLM
uv run generate -l 0 -s 2  # Filter by institution frequency  
uv run generate -l 0 -s 3 --provider openai  # Verify single tokens

# Example: Complete Level 1 Pipeline (uses Level 0 results)
uv run generate -l 1 -s 0  # Web extraction for departments (default: data/lv0/lv0_final.txt)
uv run generate -l 1 -s 1 --provider openai  # Extract department concepts
uv run generate -l 1 -s 2  # Frequency filtering
uv run generate -l 1 -s 3 --provider openai  # Token verification

# Example: Level 2 with custom input and Gemini provider
uv run generate -l 2 -s 0 --input-file data/lv1/custom.txt
uv run generate -l 2 -s 1 --provider gemini
uv run generate -l 2 -s 2
uv run generate -l 2 -s 3 --provider gemini

# Test mode (processes 10% sample)
uv run generate -l 1 -s 1 --provider openai --test
```

### Using Direct Script Execution

For advanced users who prefer direct script execution:

```bash
# Example: Complete Level 0 Pipeline
python -m generate_glossary.generation.lv0.lv0_s0_get_college_names
python -m generate_glossary.generation.lv0.lv0_s1_extract_concepts --provider openai
python -m generate_glossary.generation.lv0.lv0_s2_filter_by_institution_freq
python -m generate_glossary.generation.lv0.lv0_s3_verify_single_token --provider openai

# Example: Complete Level 1 Pipeline (uses Level 0 results)
python -m generate_glossary.generation.lv1.lv1_s0_get_dept_names --parent-level-file data/lv0/lv0_final.txt
python -m generate_glossary.generation.lv1.lv1_s1_extract_concepts --provider openai
python -m generate_glossary.generation.lv1.lv1_s2_filter_by_freq
python -m generate_glossary.generation.lv1.lv1_s3_verify_tokens --provider openai
```

## Integration with Validation and Deduplication

After generating concepts, the terms should go through validation and deduplication:

1. **Validation**: Apply rule-based, web-based, and LLM-based validation
2. **Deduplication**: Remove duplicates using graph-based or other methods
3. **Metadata Collection**: Consolidate metadata for each term

See the main [README.md](../../README.md) for details on the complete process.

## Requirements

- Python 3.6+
- aiohttp (for asynchronous web requests)
- LLM API access (OpenAI, Gemini, etc.)
- HTML parsing libraries (BeautifulSoup, etc.)
- Dotenv for environment configuration

## Module API Reference

### Level 0 Modules

#### `lv0_s0_get_college_names.py`
Extracts college/school names from faculty data files.

**Key Components:**
- `load_source_data()`: Loads faculty extraction report (Excel file)
- `extract_college_names()`: Extracts unique college names from faculty data
- Outputs to: `data/generation/lv0/lv0_s0_output.txt`

#### `lv0_s1_extract_concepts.py`
Extracts academic concepts from college/school names using DSPy-based structured generation with automatic prompt optimization support.

**Key Components:**
- `ConceptExtraction`: Pydantic model for concept extraction
- `extract_concepts_structured()`: Uses DSPy-based structured completion
- `process_source_chunk()`: Processes batches of sources
- **Prompt Optimization Support**: 
  - Automatically loads optimized prompts from `data/prompts/lv0_s1_*_latest.json`
  - Falls back to default prompts if optimized versions not found
  - Supports GEPA-optimized prompts with 21% performance improvement
- **Configuration:**
  - `BATCH_SIZE`: 20 institutions per LLM request
  - `MAX_RETRIES`: 3 structured completion attempts
  - `FREQUENCY_THRESHOLD`: 2 minimum occurrences
  - `TEMPERATURE`: 1.0 for GPT-5 models
  - `PROVIDER`: Configurable via `--provider` flag (openai/gemini)

#### `lv0_s2_filter_by_institution_freq.py`
Filters concepts by frequency across institutions.

**Key Components:**
- Frequency-based filtering using configurable thresholds
- Removes rare or spurious concepts
- Outputs filtered concepts to next stage

#### `lv0_s3_verify_single_token.py`
Verifies single-word concepts are valid academic terms using LLM-based binary classification.

**Key Components:**
- LLM-based verification of single tokens
- Multi-word concepts bypass verification
- **Prompt Optimization Support**:
  - Automatically loads optimized prompts from `data/prompts/lv0_s3_*_latest.json`
  - Falls back to default prompts if optimized versions not found
  - GEPA-optimized for high-precision binary classification
- Binary classification approach with evidence-based reasoning
- Final output: `data/generation/lv0/lv0_s3_output.txt`

### Shared Modules

#### `shared/checkpoint.py`
Provides checkpoint and recovery functionality for long-running processes.

**Functions:**
- `process_with_checkpoint()`: Wrapper for processing with automatic checkpointing
- `load_checkpoint()`: Loads previous checkpoint state
- `save_checkpoint()`: Saves current processing state

**Usage:**
```python
from generate_glossary.generation.shared import process_with_checkpoint

results = process_with_checkpoint(
    items=sources,
    batch_size=100,
    checkpoint_file="checkpoint.json",
    process_batch_func=process_batch
)
```

#### `shared/concept_extraction.py`
Common concept extraction utilities shared across levels.

**Functions:**
- `normalize_concept()`: Normalizes academic concepts (lowercase, trim)
- `split_compound_concepts()`: Handles "and", "&", "/" separators
- `validate_concept()`: Basic validation rules

#### `shared/frequency_filtering.py`
Shared frequency filtering logic.

**Functions:**
- `filter_by_frequency()`: Filters concepts by occurrence count
- `calculate_frequency_threshold()`: Dynamic threshold calculation

#### `shared/token_verification.py`
Single-token verification utilities.

**Functions:**
- `is_single_token()`: Checks if concept is single word
- `verify_single_tokens()`: Batch verification using LLMs

#### `shared/level_config.py`
Configuration management for different hierarchy levels.

**Classes:**
- `LevelConfig`: Configuration dataclass for each level
  - `level_number`: 0-3
  - `batch_size`: Processing batch size
  - `frequency_threshold`: Minimum frequency
  - `consensus_attempts`: Consensus attempts (configured via `generate_glossary/config.py`)

**Note:** Consensus is implemented in `generate_glossary/generation/concept_extraction.py` and configured via `generate_glossary/config.py`, not in `utils/llm/`.

### Integration with LLM Utils & Prompt Optimization

The generation modules integrate with centralized LLM utilities and prompt optimization:

#### `utils/llm/` Integration

**Key Functions Used:**
- `completion()`: Standard text completion with DSPy-based signatures
- `structured_completion()`: Structured completion with Pydantic models
- `async_completion()`: Asynchronous text completion
- `async_structured_completion()`: Asynchronous structured completion
- `_try_load_optimized_prompt()`: Loads optimized prompts from registry

**DSPy-Based Completion:**
```python
result = await async_structured_completion(
    messages=messages,
    response_model=ConceptExtractionList,
    model_provider="openai",  # Provider-based model selection
    temperature=1.0,
    max_tokens=4000
)
```

#### Prompt Optimization Integration

**Automatic Loading of Optimized Prompts:**
```python
# Generation scripts automatically check for optimized prompts
from prompt_optimization.core import load_prompt

# Auto-loads optimized prompts via prompt_optimization.core.load_prompt
system_prompt = load_prompt("lv0_s1_system")  # Loads latest optimized version
user_prompt = load_prompt("lv0_s1_user")      # Falls back to defaults if not found

# Alternative: Use llm utility function directly
from generate_glossary.llm import _try_load_optimized_prompt
system_prompt, user_prompt, metadata = _try_load_optimized_prompt("lv0_s1")
```

**Optimization Workflow:**
1. Run GEPA optimizer: `uv run optimize-prompt --name lv0_s1`
2. Prompts saved to: `data/prompts/lv0_s1_*_latest.json`
3. Generation scripts automatically use optimized versions
4. Falls back to defaults if optimization not available

## Environment Configuration

Required environment variables:

```bash
# LLM API Keys
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key

# Optional: Override default models
GLOSSARY_LLM_PROVIDER=openai  # or gemini, anthropic
GLOSSARY_LLM_MODEL=gpt-4      # specific model

# Web Mining with Firecrawl
FIRECRAWL_API_KEY=fc-your-key
```

## Testing Support

All generation modules support test mode:

```bash
# Run in test mode (uses test data directory)
uv run generate -l 0 -s 1 --test

# Test mode features:
# - Reads from data/generation/tests/
# - Writes to data/generation/tests/
# - Preserves production data
# - Smaller datasets for quick iteration
```

## Performance Optimization

### Parallel Processing
- Multiple LLM calls executed in parallel for structured completion
- Batch processing with configurable chunk sizes
- Asynchronous web requests for search operations

### Caching Strategy
- LLM responses cached for 1 hour (configurable)
- Web content cached indefinitely
- Checkpoint files for recovery from failures

### Resource Management
- `MAX_WORKERS`: Limits concurrent operations
- `CHUNK_SIZE`: Controls memory usage
- `BATCH_SIZE`: Optimizes LLM API calls

## Error Handling

### Checkpoint Recovery
All long-running operations support checkpoint recovery:
```python
# Automatic recovery from last checkpoint
checkpoint_file = Path(".checkpoints/lv0_s1_checkpoint.json")
if checkpoint_file.exists():
    logger.info("Resuming from checkpoint")
```

### LLM Error Handling
- Network errors: Automatic retry with exponential backoff
- API errors: Graceful degradation with logging
- Validation errors: Continue with partial results

### Data Validation
- Input validation at each step
- Output verification before proceeding
- Metadata tracking for audit trail

## Best Practices

1. **LLM Provider Selection**:
   - For critical steps, use more capable models (OpenAI)
   - For faster processing, use efficient models (Gemini)
   - Use tier-based selection for automatic model choice

2. **Web Search Configuration**:
   - Adjust search patterns based on level
   - Use domain-specific keywords for better results
   - Enable Firecrawl for 4x faster extraction

3. **Caching**:
   - All web content and search results are cached
   - Use the cached results when possible to avoid redundant requests
   - Clear cache periodically for fresh data

4. **Prompt Optimization**:
   - Run GEPA optimization for critical prompts: `uv run optimize-prompt --name lv0_s1`
   - Use `auto="heavy"` for production optimization
   - Separate task model (gpt-5-nano) from reflection model (gpt-5)
   - Monitor optimization reports in `data/optimization_reports/`
   - Optimized prompts automatically loaded by generation scripts

5. **Monitoring**:
   - Check metadata files for processing statistics
   - Monitor checkpoint files for recovery points
   - Review logs for error patterns 