# Prompt Optimization API Reference

This document provides a complete API reference for the prompt optimization module, which uses DSPy's GEPA (Genetic-Pareto Evolutionary Algorithm) to optimize prompts for the glossary generation pipeline.

## Table of Contents

- [Core Module](#core-module)
- [CLI Module](#cli-module)
- [Optimizers](#optimizers)
  - [Base Template](#base-template)
  - [Level 0 Step 1 Optimizer](#level-0-step-1-optimizer)

## Core Module

### `prompt_optimization.core`

Core utilities for saving and loading optimized prompts with versioning support.

#### Functions

##### `save_prompt(prompt_key: str, prompt_content: str, metadata: Dict[str, Any], output_dir: Path = Path("data/prompts")) -> Path`

Saves an optimized prompt with SHA256-based versioning and metadata.

**Parameters:**
- `prompt_key` (str): Unique identifier for the prompt (e.g., "lv0_s1_system")
- `prompt_content` (str): The actual prompt text or DSPy signature
- `metadata` (Dict[str, Any]): Optimization metadata including:
  - `train_size`: Number of training examples
  - `val_size`: Number of validation examples
  - `task_model`: Model used for task execution
  - `reflection_model`: Model used for reflection
  - `optimization_mode`: Optimization level (light/medium/heavy)
  - `final_score`: Final optimization score
- `output_dir` (Path): Directory to save prompts (default: "data/prompts")

**Returns:**
- `Path`: Path to the saved versioned prompt file

**Example:**
```python
from prompt_optimization.core import save_prompt

metadata = {
    "train_size": 52,
    "val_size": 23,
    "task_model": "gpt-4o-mini",
    "reflection_model": "gpt-4o",
    "optimization_mode": "medium"
}

path = save_prompt(
    prompt_key="lv0_s1_system",
    prompt_content="Extract academic concepts...",
    metadata=metadata
)
```

##### `load_prompt(prompt_key: str, version: Optional[str] = None) -> Dict[str, Any]`

Loads a prompt by key and optionally by specific version.

**Parameters:**
- `prompt_key` (str): Prompt identifier to load
- `version` (Optional[str]): Specific version hash (default: loads latest)

**Returns:**
- `Dict[str, Any]`: Prompt data including content, version, and metadata

**Example:**
```python
from prompt_optimization.core import load_prompt

# Load latest version
prompt_data = load_prompt("lv0_s1_system")

# Load specific version
prompt_data = load_prompt("lv0_s1_system", version="17598ae3")
```

## CLI Module

### `prompt_optimization.cli`

Command-line interface for running prompt optimization with GEPA.

#### Command: `optimize_prompt`

Main CLI command for optimizing prompts.

**Usage:**
```bash
uv run optimize-prompt --prompt PROMPT_ID [OPTIONS]
```

**Options:**
- `--prompt` (required): Prompt to optimize (e.g., lv0_s1, lv1_s2)
- `--auto`: Optimization level (choices: light, medium, heavy; default: medium)
  - `light`: ~5 minutes, basic optimization
  - `medium`: ~15 minutes, balanced optimization
  - `heavy`: ~30+ minutes, thorough optimization
- `--max-calls`: Maximum LLM calls budget limit (optional)
- `--num-threads`: Number of parallel threads (default: 8)
- `--task-model`: Model for task execution (default: openai/gpt-5-nano)
- `--reflection-model`: Model for reflection/optimization (default: openai/gpt-5)
- `--train-split`: Train/validation split ratio (default: 0.7)
- `--batch-size`: Training batch size (default: 5, production uses 20)
- `--verbose`: Enable verbose output

**Example:**
```bash
# Basic optimization
uv run optimize-prompt --prompt lv0_s1 --auto light

# Production-like optimization
uv run optimize-prompt --prompt lv0_s1 --auto heavy --batch-size 20

# Budget-limited optimization
uv run optimize-prompt --prompt lv0_s1 --max-calls 100
```

## Optimizers

### Base Template

#### `prompt_optimization.optimizers.template`

Template for creating new prompt optimizers. Provides the structure and required methods.

##### Required Functions

###### `create_training_data(batch_size: int = 5) -> List[Dict]`

Creates or loads training data for optimization.

**Parameters:**
- `batch_size` (int): Size of batches for training (default: 5)

**Returns:**
- `List[Dict]`: Training examples with input/output pairs

###### `prepare_dspy_examples(training_data: List[Dict]) -> List[Example]`

Converts training data to DSPy Example format.

**Parameters:**
- `training_data` (List[Dict]): Raw training data

**Returns:**
- `List[Example]`: DSPy-formatted examples

###### `metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction`

Evaluation metric with rich textual feedback for GEPA.

**Parameters:**
- `gold`: Ground truth example
- `pred`: Predicted output
- `trace`: Execution trace (optional)
- `pred_name`: Predictor name (optional)
- `pred_trace`: Prediction trace (optional)

**Returns:**
- `dspy.Prediction`: Score and feedback

**Requirements:**
- Must return score between 0 and 1
- Must provide actionable feedback text
- Last 3 parameters must be optional

###### `get_program() -> dspy.Module`

Returns the DSPy program to optimize.

**Returns:**
- `dspy.Module`: Program instance (e.g., ConceptExtractor)

###### `save_optimized_prompts(optimized_program, trainset, valset, task_model, reflection_model, optimization_mode) -> List[Path]`

Saves the optimized prompts and program state.

**Parameters:**
- `optimized_program`: Optimized DSPy program
- `trainset`: Training examples
- `valset`: Validation examples
- `task_model`: Model used for tasks
- `reflection_model`: Model used for reflection
- `optimization_mode`: Optimization level used

**Returns:**
- `List[Path]`: Paths to saved files

### Level 0 Step 1 Optimizer

#### `prompt_optimization.optimizers.lv0_s1`

Optimizer for Level 0 Step 1: Concept extraction from college/school names.

##### Classes

###### `ConceptExtraction`

Pydantic model for a single concept extraction.

**Fields:**
- `source` (str): Source text being processed
- `concepts` (List[str]): List of extracted concepts

###### `ConceptExtractionList`

Container for multiple concept extractions.

**Fields:**
- `extractions` (List[ConceptExtraction]): List of concept extractions

###### `ExtractConceptsSignature`

DSPy signature for concept extraction task.

**Fields:**
- `sources` (str): Input field - list of college/school names
- `extractions` (ConceptExtractionList): Output field - extracted concepts

###### `ConceptExtractor`

DSPy module implementing the concept extraction logic.

**Methods:**
- `__init__()`: Initializes with ChainOfThought predictor
- `forward(sources)`: Executes extraction on input sources

##### Functions

###### `create_training_data(batch_size: int = 5) -> List[Dict]`

Loads and batches training data from `data/prompts_training_data/lv0_s1.json`.

**Training Data Format:**
```json
[
  {
    "input": "College of Engineering",
    "expected": ["engineering"]
  },
  {
    "input": "School of Medicine",
    "expected": ["medicine"]
  }
]
```

**Returns:**
- Mix of batched examples (80%) and single examples (20%)

###### `calculate_f1(gold_concepts: set, pred_concepts: set) -> float`

Calculates F1 score between two sets of concepts.

**Parameters:**
- `gold_concepts` (set): Ground truth concepts
- `pred_concepts` (set): Predicted concepts

**Returns:**
- `float`: F1 score between 0 and 1

###### `metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction`

Evaluates extraction quality with detailed feedback.

**Evaluation:**
- Calculates per-source F1 scores
- Identifies missing and extra concepts
- Provides specific feedback for improvements

**Returns:**
- `dspy.Prediction` with:
  - `score`: Average F1 score across all sources
  - `feedback`: Descriptive feedback (e.g., "F1: 0.85. College of Engineering missing: civil engineering")

###### `load_optimized_program(program_path: Path = None) -> ConceptExtractor`

Loads a previously optimized program from disk.

**Parameters:**
- `program_path` (Path): Path to saved program (default: "data/prompts/lv0_s1_program.json")

**Returns:**
- `ConceptExtractor`: Loaded optimized program

**Example Usage:**
```python
from prompt_optimization.optimizers.lv0_s1 import load_optimized_program

# Load and use optimized program
program = load_optimized_program()
result = program.forward(sources="- College of Engineering\n- School of Medicine")
```

## Integration with Generation Pipeline

The optimized prompts are automatically loaded by the generation scripts:

```python
# In generate_glossary/generation/lv0/lv0_s1_extract_concepts.py
from generate_glossary.utils.llm import load_prompt_from_file

optimized_system = load_prompt_from_file("data/prompts/lv0_s1_system_latest.json")
if optimized_system:
    SYSTEM_PROMPT = optimized_system
    logger.info("Loaded optimized system prompt from file")
```

The `load_prompt_from_file` function automatically handles DSPy format extraction, extracting the instructions from complex DSPy signatures.

## File Structure

```
prompt_optimization/
├── __init__.py
├── core.py                    # Core utilities
├── cli.py                     # CLI interface
├── optimizers/
│   ├── __init__.py
│   ├── template.py           # Template for new optimizers
│   └── lv0_s1.py            # Level 0 Step 1 optimizer
├── README.md                  # User guide
├── API_REFERENCE.md          # This file
├── TROUBLESHOOTING.md        # Common issues and solutions
└── OPTIMIZATION_RESULTS.md   # Results from optimization runs
```

## Output Files

Optimization creates the following files:

```
data/
├── prompts/                   # Optimized prompts
│   ├── lv0_s1_system_latest.json     # Latest system prompt
│   ├── lv0_s1_system_<hash>.json     # Versioned system prompt
│   ├── lv0_s1_user_latest.json       # Latest user template
│   ├── lv0_s1_user_<hash>.json       # Versioned user template
│   └── lv0_s1_program.json           # Full optimized program
└── prompts_training_data/     # Training data
    └── lv0_s1.json           # Training examples
```

## Best Practices

1. **Match Production Usage**: Use the same batch size in optimization as in production (20)
2. **Use Appropriate Models**: Budget models for tasks, better models for reflection
3. **Provide Rich Feedback**: Detailed feedback in metrics improves optimization
4. **Save Everything**: All optimizations are versioned for comparison
5. **Test Before Production**: Verify optimized prompts on test data first

## Error Handling

The module includes comprehensive error handling:

- **Training Data Not Found**: Clear error message with expected path
- **Missing Required Functions**: Validates optimizer has all required methods
- **API Key Issues**: Checks for environment variables
- **DSPy Version**: Requires DSPy 3.0.3+ for GEPA support
- **SQLite Cache Issues**: Automatically disables problematic disk cache

See `TROUBLESHOOTING.md` for detailed error resolution.