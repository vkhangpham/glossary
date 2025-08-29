# Practical Utils Cleanup Plan

## Current Utils Analysis

### What's Actually Being Used:
```
15x  logger.py             → setup_logger()
8x   llm_simple.py         → infer_text(), infer_structured(), get_random_llm_config()
6x   resilient_processing  → ConceptExtractionProcessor, KeywordVerificationProcessor
2x   firecrawl_web_miner   → Firecrawl integration
2x   checkpoint.py         → CheckpointManager, with_checkpoint
1x   metadata_collector    → Various collection functions
1x   web_miner_runner      → run_web_mining()
1x   secure_config         → (Used internally by other utils)
```

### What's the Problem:
1. **metadata_collector.py (1,256 lines)** - Doing WAY too much
2. **Mixed responsibilities** - Some utils are infrastructure, some are business logic
3. **Circular dependencies** - Utils importing from each other
4. **Unused code** - web_miner.py seems replaced by firecrawl but still referenced

---

## Simple, Practical Restructuring

### Keep Utils Minimal - Move Business Logic Out

```
generate_glossary/
├── utils/                     # ONLY true utilities (300 lines max total)
│   ├── logger.py              # ✓ Keep as-is (58 lines)
│   ├── llm.py                 # ✓ Rename from llm_simple.py
│   └── config.py              # ✓ Extract from secure_config.py
│
├── processing/                # Move processing logic here
│   ├── checkpoint.py          # Move from utils
│   ├── resilient.py           # Move from utils  
│   └── batch.py               # Extract batch processing patterns
│
├── mining/                    # All web mining in one place
│   ├── firecrawl.py           # Move from utils
│   ├── runner.py              # Move from utils
│   └── __init__.py            # Single interface
│
├── metadata/                  # Break up metadata_collector.py
│   ├── collector.py           # Core collection logic (200 lines)
│   ├── file_discovery.py     # find_step_file, find_final_file (150 lines)
│   ├── extractors.py          # extract_parent_from_college, etc (200 lines)
│   └── consolidator.py       # Consolidation logic (200 lines)
│
└── security/                  # Security as separate concern
    ├── api_keys.py            # API key management
    └── sanitizer.py           # Log sanitization
```

---

## Specific Changes

### 1. Fix metadata_collector.py (PRIORITY 1)
**Current**: 1,256 lines doing everything  
**Action**: Split into 4 focused modules

```python
# metadata/file_discovery.py (~150 lines)
def find_step_file(level_dir, level, step, extension=None)
def find_final_file(level_dir, level)
def find_step_metadata(raw_dir, level, step)

# metadata/extractors.py (~200 lines)
def extract_parent_from_college(college_name)
def extract_concepts_from_file(file_path)
def extract_metadata_from_json(json_path)

# metadata/collector.py (~200 lines)
def collect_metadata(level, data_dir)
def collect_resources(level, data_dir)

# metadata/consolidator.py (~200 lines)
def consolidate_metadata(metadata_list)
def merge_resources(resource_list)
```

### 2. Simplify LLM Interface (PRIORITY 2)
**Current**: llm_simple.py with provider-specific functions  
**Action**: Single interface, provider config

```python
# utils/llm.py
def call_llm(prompt, response_model=None, provider=None):
    """Single entry point for all LLM calls"""
    provider = provider or get_default_provider()
    return provider.call(prompt, response_model)

# No more openai_text(), anthropic_structured(), etc.
```

### 3. Clean Up Web Mining (PRIORITY 3)
**Current**: Mixed between utils and scattered files  
**Action**: Consolidate in mining/ directory

```python
# mining/__init__.py
from .runner import mine_web_content

# Single public interface
def mine_web_content(terms, output_file, use_firecrawl=True):
    """Mine web content for terms"""
    if use_firecrawl:
        return firecrawl.mine(terms, output_file)
    else:
        # Fallback to old method if needed
        pass
```

### 4. Extract Processing Patterns (PRIORITY 4)
**Current**: resilient_processing.py with class inheritance  
**Action**: Functional approach with composition

```python
# processing/batch.py
def process_in_batches(items, batch_fn, batch_size=10, checkpoint_mgr=None):
    """Generic batch processing with optional checkpointing"""
    for batch in chunks(items, batch_size):
        if checkpoint_mgr:
            checkpoint_mgr.save_batch(batch)
        results = batch_fn(batch)
        yield results
```

---

## Migration Steps (Can Do Today)

### Step 1: Create New Structure (30 min)
```bash
mkdir -p generate_glossary/{processing,mining,metadata,security}
touch generate_glossary/metadata/__init__.py
touch generate_glossary/mining/__init__.py
# etc.
```

### Step 2: Move Without Breaking (1 hour)
```python
# Keep backwards compatibility temporarily
# utils/metadata_collector.py
from ..metadata.collector import *
from ..metadata.file_discovery import *
# This file becomes a shim that imports from new locations
```

### Step 3: Update Imports Gradually (2 hours)
```python
# Old
from generate_glossary.utils.metadata_collector import collect_metadata

# New
from generate_glossary.metadata import collect_metadata
```

### Step 4: Remove Shims (30 min)
Once all imports updated, delete the compatibility shims

---

## What NOT to Do

1. **Don't over-engineer** - No abstract base classes, no complex hierarchies
2. **Don't break everything at once** - Keep backwards compatibility during migration
3. **Don't create more utils** - If it doesn't fit the 4 categories, it shouldn't be in utils
4. **Don't mix concerns** - Utils should know nothing about glossaries, levels, or concepts

---

## Success Criteria

### Before:
- utils/ has 8 files, 3,191 lines
- metadata_collector.py is 1,256 lines
- Mixed responsibilities everywhere

### After:
- utils/ has 3 files, <300 lines total
- No file > 400 lines
- Clear separation: utils, processing, mining, metadata, security
- All imports work without changes to calling code

---

## Why This Approach Works

1. **Gradual** - Can be done incrementally without breaking prod
2. **Simple** - No fancy patterns, just moving files and functions
3. **Practical** - Addresses real pain points (metadata_collector size)
4. **Maintainable** - Clear where everything belongs

---

## Timeline

- **Today**: Create structure, move metadata_collector
- **Tomorrow**: Move web mining, consolidate LLM interface  
- **Day 3**: Extract processing patterns, clean up imports
- **Day 4**: Remove compatibility shims, update docs

Total: 4 days of part-time work, fully backwards compatible throughout.