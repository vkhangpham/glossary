# Prompt Optimization with GEPA

A practical system for optimizing prompts using the GEPA (Genetic-Pareto Evolutionary Algorithm) framework from DSPy.

## Directory Structure

```
prompt_optimization/
├── core.py                    # Shared utilities for saving/loading
├── optimizers/                # Individual optimization scripts
│   ├── template.py           # Template for new optimizers
│   ├── lv0_s1.py            # Level 0 Step 1: College concept extraction
│   ├── lv0_s2.py            # (future) Level 0 Step 2: Filtering
│   ├── lv1_s1.py            # (future) Level 1 Step 1: Department extraction
│   └── ...                  # More optimizers as needed
└── README.md

data/
├── prompts/                   # Optimized prompts (output)
│   ├── lv0_s1_system_latest.json
│   └── lv0_s1_<hash>.json
└── prompts_training_data/     # Training data for optimization
    └── lv0_s1.json
```

## Naming Convention

Optimizer scripts follow a consistent naming pattern:
- `lv{level}_s{step}.py` (e.g., `lv0_s1.py`, `lv1_s2.py`)
- Corresponds to the generation pipeline steps

## Quick Start

### 1. Run Optimization via CLI

```bash
# Set your API key
export OPENAI_API_KEY="your-key"

# Light optimization (~5 min)
uv run optimize-prompt --prompt lv0_s1 --auto light

# Medium optimization with custom batch size (recommended)
uv run optimize-prompt --prompt lv0_s1 --auto medium --batch-size 5

# Heavy optimization matching production batch size
uv run optimize-prompt --prompt lv0_s1 --auto heavy --batch-size 20

# Custom models (GPT-5 requires special handling)
uv run optimize-prompt --prompt lv0_s1 --task-model openai/gpt-4o-mini --reflection-model openai/gpt-4o
```

This will:
1. Load training data from `data/prompts_training_data/`
2. Create batched examples matching production usage
3. Run GEPA optimization with rich feedback
4. Save optimized prompts to `data/prompts/`
5. Save full program state for recovery

### 2. Usage in Application Code

Prompts are automatically loaded at module initialization:

```python
# In lv0_s1_extract_concepts.py
from generate_glossary.utils.llm import load_prompt_from_file

# Load optimized prompt if available
optimized = load_prompt_from_file("data/prompts/lv0_s1_system_latest.json")
if optimized:
    SYSTEM_PROMPT = optimized
    logger.info("Loaded optimized prompt")
```

## Creating New Optimizers

1. **Copy the template**:
```bash
cp prompt_optimization/optimizers/template.py prompt_optimization/optimizers/lv2_s1.py
```

2. **Modify the template**:
   - Define your Pydantic models
   - Create training examples
   - Implement evaluation metric
   - Update prompt keys

3. **Run optimization**:
```bash
uv run python prompt_optimization/optimizers/lv2_s1.py
```

## How GEPA Works

GEPA is a reflective prompt evolution algorithm that:

1. **Reflects** on execution traces (inputs, outputs, errors)
2. **Identifies** what worked well and what didn't
3. **Proposes** new prompt variations
4. **Evaluates** candidates on validation data
5. **Maintains** a Pareto frontier of best performers

## Training Data Format

Training data should be JSON with input/output pairs:

```json
[
  {
    "input": {
      "sources": "College of Engineering"
    },
    "output": {
      "extractions": [
        {"source": "College of Engineering", "concepts": ["engineering"]}
      ]
    }
  }
]
```

## Configuration Options

### Optimization Levels

- `auto="light"`: Fast optimization (~5 min)
- `auto="medium"`: Balanced (~15 min) **[Recommended]**
- `auto="heavy"`: Thorough (~30+ min)

### Model Selection

```python
# Budget model for task execution
task_lm = dspy.LM("openai/gpt-4o-mini")

# Better model for reflection
reflection_lm = dspy.LM("openai/gpt-4o")
```

## Best Practices

1. **Quality over Quantity**: 5-10 high-quality training examples > 100 poor ones
2. **Meaningful Feedback**: Provide specific feedback in metrics, not just scores
3. **Version Control**: Prompts auto-versioned by content hash
4. **One-Time Run**: Optimize once per major version
5. **Fallback Strategy**: Always provide defaults for robustness

## File Output Structure

Each optimization creates:
- `{key}_latest.json`: Latest version for easy access
- `{key}_{hash}.json`: Versioned file with SHA hash
- `{key}_program.dspy`: Full optimized DSPy program

Example output file:
```json
{
  "key": "lv0_s1_system",
  "content": "You are an expert at extracting...",
  "version": "a3f2b891",
  "metadata": {
    "created_at": "2024-01-20T10:30:00",
    "optimization_method": "GEPA",
    "train_size": 10,
    "val_size": 5,
    "task_model": "gpt-4o-mini",
    "reflection_model": "gpt-4o"
  }
}
```

## Tips for Writing Optimizers

1. **Start Simple**: Begin with basic training examples
2. **Test Metrics**: Ensure your metric accurately measures quality
3. **Iterate**: Run optimization multiple times with different settings
4. **Monitor Costs**: GEPA makes many LLM calls - use budget models
5. **Save Everything**: Keep all versions for comparison

## CRITICAL GOTCHAS AND ISSUES

### 1. Training/Production Batch Size Mismatch
**Problem**: Training on single institutions when production uses batches of 20.
**Solution**: Use `--batch-size` parameter to match production usage.
```bash
# Training with batches (matches production)
uv run optimize-prompt --prompt lv0_s1 --batch-size 20
```

### 2. DSPy Version Compatibility
**Problem**: GEPA not available in DSPy < 3.0
**Solution**: Must use DSPy 3.0.3+ which includes GEPA 0.0.7
```toml
dspy = "^3.0.3"
```

### 3. GPT-5 Model Requirements
**Problem**: GPT-5 models require `temperature=1.0` and `max_tokens=16000`
**Solution**: CLI automatically handles this, but manual usage needs:
```python
dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=16000)
```

### 4. Pydantic Validation Errors
**Problem**: LLM returns list instead of dict with 'extractions' key
**Solution**: Simplified models without complex validation:
```python
class ConceptExtractionList(BaseModel):
    extractions: List[ConceptExtraction] = Field(default_factory=list)
```

### 5. Lost Optimization Results
**Problem**: Expensive optimization crashes and loses all progress
**Solution**: Three-layer save strategy:
- System prompt: `data/prompts/lv0_s1_system_[hash].json`
- User template: `data/prompts/lv0_s1_user_[hash].json`
- Full program: `data/prompts/lv0_s1_program.json` (MOST CRITICAL)

**Recovery**: Load saved program without re-optimization:
```python
from prompt_optimization.optimizers.lv0_s1 import load_optimized_program
program = load_optimized_program()
```

### 6. Metric Signature Requirements
**Problem**: GEPA requires specific 5-argument metric signature
**Solution**: Last 3 arguments must be optional:
```python
def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
```

### 7. Training Data Format Mismatch
**Problem**: Optimizer expects different format than existing data
**Solution**: Check `data/prompts_training_data/lv0_s1.json` format:
```json
[{"input": "College of Engineering", "expected": ["engineering"]}]
```

### 8. Import Path Issues
**Problem**: sys.path hacking breaks when run from different directories
**Solution**: Use UV package management with proper entry points:
```toml
[project.scripts]
optimize-prompt = "prompt_optimization.cli:optimize_prompt"
```

### 9. File Save Verification
**Problem**: Program saves but file is empty or corrupted
**Solution**: Always verify file size after saving:
```python
if program_path.exists():
    size = program_path.stat().st_size
    if size < 100:
        raise ValueError("Program file too small, likely corrupted")
```

### 10. Individual vs Batch Examples
**Problem**: GEPA best practice is individual examples, but production uses batches
**Solution**: Mix both for robustness (80% batched, 20% single)

## Future Enhancements

- [ ] Batch optimization runner for multiple prompts
- [ ] Automatic training data generation from existing outputs
- [ ] Metric library for common evaluation patterns
- [ ] Cost tracking and optimization budget limits
- [ ] A/B testing framework for comparing optimized prompts