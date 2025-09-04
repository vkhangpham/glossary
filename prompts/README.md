# Prompt Management System

A centralized, versioned prompt management system with built-in GEPA (Genetic-Pareto Evolutionary Algorithm) optimization support for the Academic Glossary project.

## Features

- **Centralized Storage**: All prompts stored as versioned JSON files
- **Pure Functional API**: Clean, functional programming interface
- **SHA256 Versioning**: Automatic version tracking on prompt changes
- **Template Variables**: Dynamic substitution with `{variable}` syntax
- **GEPA Optimization**: Evolutionary prompt optimization for better performance
- **High Performance**: <0.1ms prompt loading with LRU caching
- **Type Safety**: Full Pydantic model validation

## Installation

The prompt system is included in the main project. No separate installation needed.

```bash
# Install project dependencies (if not already done)
uv sync
```

## Quick Start

```python
from prompts import get_prompt, register_prompt

# Get a prompt from the registry
system_prompt = get_prompt("extraction.level0_system")

# Get a prompt with variable substitution
user_prompt = get_prompt("extraction.level0_user", 
                         keyword="engineering",
                         colleges_str="College of Engineering, School of Engineering")

# Register a new prompt (usually done via JSON files)
register_prompt(
    key="custom.my_prompt",
    prompt="Analyze {term} for academic relevance",
    variables=["term"]
)
```

## Directory Structure

```
prompts/
├── __init__.py             # Public API exports
├── registry.py             # Pure functional prompt access
├── storage.py              # JSON-based versioned storage
├── optimizer/              # GEPA optimization module
│   ├── __init__.py
│   ├── concept_extraction_adapter.py  # Extraction task adapter
│   └── optimizer.py        # High-level optimization API
└── data/library/           # Versioned prompt library
    ├── extraction/         # Extraction prompts by level
    │   ├── level0_system.json
    │   └── level0_user_template.json
    ├── token_verification/ # Verification prompts
    │   └── level0.json
    └── metrics.json        # Performance metrics
```

## API Reference

### Core Functions

#### `get_prompt(key: str, version: Optional[str] = None, **kwargs) -> str`

Retrieve a prompt from the registry with optional variable substitution.

```python
# Simple retrieval
prompt = get_prompt("extraction.level0_system")

# With variables
prompt = get_prompt("validation.term", term="computer science")

# Specific version
prompt = get_prompt("extraction.level0_system", 
                   version="abc123...")
```

#### `register_prompt(key: str, prompt: str, variables: List[str] = None)`

Register a new prompt programmatically (prefer JSON files for production).

```python
register_prompt(
    key="analysis.domain",
    prompt="Classify {domain} into academic categories",
    variables=["domain"]
)
```

#### `list_prompts() -> Dict[str, Any]`

List all available prompts with their metadata.

```python
prompts = list_prompts()
for key, info in prompts.items():
    print(f"{key}: v{info['version'][:8]}")
```

## Prompt File Format

Prompts are stored as JSON files with the following structure:

```json
{
  "prompt_key": {
    "content": "The actual prompt text with {variables}",
    "version": "1.0.0",
    "variables": ["variable1", "variable2"],
    "metadata": {
      "level": 0,
      "task": "extraction",
      "created": "2025-09-04",
      "source": "module/that/uses/it.py"
    }
  }
}
```

## GEPA Optimization

The system includes built-in support for evolutionary prompt optimization using GEPA.

### Running Optimization

```python
from prompts.optimizer import optimize_prompt

# Optimize a prompt with training data
optimized, metrics = optimize_prompt(
    prompt_key="extraction.level0",
    training_data=training_examples,
    validation_data=validation_examples,
    level=0,
    task_model="openai/gpt-4o-mini",
    reflection_model="openai/gpt-4",
    max_metric_calls=100
)

print(f"Improvement: {metrics['improvement']:.2%}")
```

### Testing Optimization

```bash
# Run the test optimization script (requires OpenAI API key)
python test_prompt_optimization.py
```

## Best Practices

### 1. Use JSON Files for Prompts

Store prompts in JSON files rather than registering programmatically:

```json
// prompts/data/library/extraction/level1.json
{
  "level1_system": {
    "content": "Your prompt here...",
    "version": "1.0.0",
    "variables": []
  }
}
```

### 2. Namespace Your Keys

Use dot notation to organize prompts:

- `extraction.level0_system` - Extraction system prompts
- `validation.term` - Validation prompts
- `deduplication.llm` - Deduplication prompts

### 3. Document Variables

Always specify which variables a prompt expects:

```json
{
  "variables": ["term", "context", "examples"],
  "content": "Analyze {term} in the context of {context}..."
}
```

### 4. Version Important Changes

The system automatically tracks versions using SHA256 hashes. Check metrics.json for version history:

```json
{
  "extraction.level0_system": {
    "versions": ["abc123...", "def456..."],
    "current_version": "def456...",
    "last_modified": "2025-09-04T10:30:00"
  }
}
```

## Performance

The prompt system is designed for high performance:

- **Loading**: <0.1ms per prompt (with caching)
- **Cold Start**: ~1ms for first prompt load
- **Memory**: Minimal overhead with lazy loading
- **Caching**: LRU cache with automatic invalidation

## Testing

```bash
# Run prompt registry tests
python test_prompt_registry.py

# Run optimization tests
python test_prompt_optimization.py
```

## Migration Guide

To migrate existing inline prompts to the registry:

1. Extract prompt to JSON file
2. Add appropriate metadata
3. Replace inline string with `get_prompt()` call

Example migration:

```python
# Before
SYSTEM_PROMPT = """You are an expert..."""

# After
from prompts import get_prompt
SYSTEM_PROMPT = get_prompt("extraction.level0_system")
```

## Future Enhancements

- [ ] Prompt versioning UI
- [ ] A/B testing framework
- [ ] Automatic prompt migration tools
- [ ] Prompt performance analytics
- [ ] Multi-language prompt support
- [ ] Prompt composition/chaining

## Contributing

When adding new prompts:

1. Create JSON file in appropriate directory
2. Use consistent key naming
3. Document all variables
4. Add tests for variable substitution
5. Consider optimization potential

## License

Part of the Academic Glossary Analysis project.