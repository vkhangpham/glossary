# Prompt Optimization System

## Overview

This is a streamlined prompt optimization system using DSPy GEPA (Gradient-Enhanced Performance Augmentation) for automatic prompt optimization. The system provides both CLI and direct execution methods, with automatic environment variable loading from `.env` files.

## Quick Start

### 1. Setup Environment

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-key-here
GEPA_GEN_MODEL=gpt-5-nano        # Optional: defaults to gpt-5-nano (task model)
GEPA_REFLECTION_MODEL=gpt-5      # Optional: defaults to gpt-5 (strong reflection model - best practice)
GEPA_AUTO=light                  # Optional: light/medium/heavy (defaults to light for testing)
```

### 2. Run Optimization

```bash
# Using the CLI (recommended - auto-loads .env)
uv run optimize-prompt --name lv0_s1
uv run optimize-prompt --name lv0_s3

# With specific budget controls
uv run optimize-prompt --name lv0_s1 --max-full-evals 10
uv run optimize-prompt --name lv0_s1 --auto medium

# Direct execution (also auto-loads .env)
uv run python prompt_optimization/optimizers/lv0_s1.py
uv run python prompt_optimization/optimizers/lv0_s3.py
```

Optimized prompts are automatically saved to `data/prompts/` after optimization completes.

### 3. Automatic Loading in Generation Scripts

Generation scripts automatically load optimized prompts:

```python
# In lv0_s1_extract_concepts.py
optimized_prompt = load_prompt_from_file("data/prompts/lv0_s1_system_latest.json")
if optimized_prompt:
    system_prompt = optimized_prompt  # Use optimized version
else:
    system_prompt = DEFAULT_PROMPT    # Fall back to default
```

## File Structure

### Project Organization
```
prompt_optimization/
├── cli.py                   # CLI interface with automatic .env loading
├── core.py                  # Core save/load functionality
├── optimizers/
│   ├── lv0_s1.py           # Level 0 Step 1 optimizer
│   ├── lv0_s3.py           # Level 0 Step 3 optimizer
│   ├── template.py         # Template for new optimizers
│   └── common.py           # Shared utilities (LLM config, data loading)

data/prompts/                        # Auto-generated optimized prompts
├── lv0_s1_system_latest.json    
├── lv0_s1_user_latest.json      
├── lv0_s3_system_latest.json    
└── lv0_s3_user_latest.json      

data/prompts_training_data/          # Training data for optimization
├── lv0_s1.json                  # Training examples for concept extraction
└── lv0_s3.json                  # Training examples for discipline verification
```

### JSON Format
```json
{
  "content": "The actual prompt text goes here...",
  "metadata": {
    "created_at": "2024-01-15T10:30:00",
    "optimization_method": "GEPA",
    "prompt_key": "lv0_s1_system",
    "prompt_type": "system",
    "content_length": 523,
    "version": "1.0"
  }
}
```

## CLI Usage

### List Available Optimizers
```bash
uv run optimize-prompt --list
```

### Run with Different Budget Controls
```bash
# Light optimization (fast, ~5 minutes)
uv run optimize-prompt --name lv0_s1 --auto light

# Medium optimization (balanced, ~15 minutes)  
uv run optimize-prompt --name lv0_s1 --auto medium

# Heavy optimization (thorough, ~30+ minutes)
uv run optimize-prompt --name lv0_s1 --auto heavy

# Specific number of full evaluations
uv run optimize-prompt --name lv0_s1 --max-full-evals 10

# Specific number of metric calls
uv run optimize-prompt --name lv0_s1 --max-metric-calls 500
```

### Override Models
```bash
# Use different generation model
uv run optimize-prompt --name lv0_s1 --model gpt-4o

# Use different reflection model
uv run optimize-prompt --name lv0_s1 --reflection-model gpt-4o

# Verbose output for debugging
uv run optimize-prompt --name lv0_s1 --verbose
```

### Creating New Optimizers
```python
# Copy the template
cp prompt_optimization/optimizers/template.py \
   prompt_optimization/optimizers/lv1_s2.py

# Edit to define your task
# Run directly
python prompt_optimization/optimizers/lv1_s2.py
```

## Integration with Generation Scripts

### How It Works

1. **Generation scripts check for optimized prompts first**:
   - Look for `data/prompts/{key}_latest.json`
   - If found, use the optimized version
   - If not found, use the default prompt

2. **No code changes needed**:
   - Generation scripts already have this logic built in
   - Just save optimized prompts to the right location

3. **Fallback behavior**:
   - Missing files return `None` → use defaults
   - Malformed JSON returns `None` → use defaults
   - Empty content returns `None` → use defaults

### Supported Scripts

Currently integrated with:
- `lv0_s1_extract_concepts.py` - Concept extraction from colleges
- `lv0_s3_verify_single_token.py` - Single-token verification

## Benefits of Simplified System

### 1. Direct Execution
- No CLI framework overhead
- Run optimizers like any Python script
- Simple `python optimizer.py` command

### 2. Self-Contained Optimizers
- Each optimizer is a complete standalone script
- Includes its own `if __name__ == "__main__"` block
- Can be run directly or imported

### 3. Simplified Save Mechanism
- Single `save_prompt()` function
- Automatic JSON formatting
- Consistent file naming

### 4. Seamless Integration
- Generation scripts auto-load optimized prompts
- Fallback to defaults if not found
- No configuration needed

## Creating Custom Optimizers

### Using the Template
```python
# 1. Copy template.py to your new optimizer
cp prompt_optimization/optimizers/template.py \
   prompt_optimization/optimizers/lv1_s2.py

# 2. Update the following in your new file:
#    - Pydantic models for your task
#    - DSPy signature definition
#    - Training data creation
#    - Metric function
#    - Save prompt keys

# 3. Run directly
python prompt_optimization/optimizers/lv1_s2.py
```

### Example Structure
```python
# Define your task models
class YourOutputModel(BaseModel):
    field1: str
    field2: List[str]

# Define DSPy signature
class YourTaskSignature(dspy.Signature):
    input_field: str = dspy.InputField()
    output_field: YourOutputModel = dspy.OutputField()

# Implement metric
def metric_with_feedback(gold, pred, trace=None):
    score = calculate_score(gold, pred)
    feedback = generate_feedback(score)
    return score, feedback

# Run with direct execution
if __name__ == "__main__":
    optimize_prompts()
```

## Testing

Run the integration test suite:

```bash
python test_prompt_integration.py
```

This will:
- Test save/load round-trip compatibility
- Verify path matching with generation scripts
- Test fallback behavior for missing files
- Validate JSON format compatibility
- Create sample optimized prompts

## Troubleshooting

### Issue: Optimized prompts not being used

**Check file location**:
```bash
ls -la data/prompts/
```

**Verify JSON format**:
```bash
cat data/prompts/lv0_s1_system_latest.json | jq .
```

**Check logs**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Issue: JSON decode errors

**Validate JSON**:
```bash
python -m json.tool data/prompts/lv0_s1_system_latest.json
```

**Check for BOM or encoding issues**:
```bash
file data/prompts/lv0_s1_system_latest.json
```

### Issue: Empty content warnings

Make sure the prompt content is not empty:
```python
# This will fail
save_prompt("key", "")  # Empty string

# This will work
save_prompt("key", "Valid prompt content")
```

## Environment Setup

### Using .env File (Recommended)
Create a `.env` file in the project root:
```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional model overrides (following best practices)
GEPA_GEN_MODEL=gpt-5-nano        # Default: gpt-5-nano (efficient task model)
GEPA_REFLECTION_MODEL=gpt-5      # Default: gpt-5 (strong reflection model)
GEPA_AUTO=heavy                  # Default: light (use heavy for production)
```

The `.env` file is **automatically loaded** by both the CLI and direct execution methods.

### Manual Export (Alternative)
```bash
export OPENAI_API_KEY="your-key"
export GEPA_GEN_MODEL="gpt-4o-mini"
export GEPA_REFLECTION_MODEL="gpt-4o"
```

### Special Model Requirements
**Note**: OpenAI reasoning models (gpt-5-nano, gpt-5) require:
- `temperature=1.0`
- `max_tokens >= 16000` (32000 for reflection models - best practice)

These are automatically configured when detected.

### Training Data Format

Training data is stored in `data/prompts_training_data/`:

```json
// data/prompts_training_data/lv0_s1.json
[
  {
    "input": "College of Engineering",
    "expected": ["engineering", "computer science", "mechanical engineering"]
  }
]

// data/prompts_training_data/lv0_s3.json  
[
  {
    "keyword": "engineering",
    "expected": true,
    "evidence_colleges": ["College of Engineering", "School of Engineering"]
  }
]
```

## GEPA Best Practices

### Core Configuration Principles

#### 1. Budget Configuration (Critical)
- **For production**: Use `auto="heavy"` for optimal performance
- **For testing**: Use smaller budgets like `max_full_evals=1-10`
- **Budget scaling**: GEPA scales with higher budget - more budget = better results
- **Efficiency**: GEPA uses 35x fewer rollouts than GRPO while achieving better results

#### 2. Reflection Language Model Selection
- **Use a powerful reflection LM**: Use `gpt-4` or stronger for reflection
- **Separate from task LM**: Use a stronger model for reflection than the one being optimized
- **High temperature for reflection**: `temperature=1.0` for diverse reflection
- **Large context**: `max_tokens=32000` to capture full reflection content

#### 3. Essential Tracking Parameters
```python
optimizer = GEPA(
    track_stats=True,           # CRITICAL - enables detailed_results for reporting
    track_best_outputs=True,    # Recommended for analysis
    # ... other params
)
```

#### 4. Threading and Performance
- **Use parallel processing**: `num_threads=16-32` for faster optimization
- **Balance threads with API limits**: Don't exceed your API rate limits
- **Reflection minibatch**: `reflection_minibatch_size=3` for efficiency

#### 5. Metric Design (Crucial)
```python
def metric_with_feedback(example, prediction, trace=None):
    # Must return dspy.Prediction with score AND feedback
    return dspy.Prediction(
        score=calculated_score,      # Numeric score
        feedback="Detailed feedback" # Natural language guidance
    )
```

### Recommended Configuration Template
```python
from dspy import GEPA

# Best practices configuration based on research
optimizer = GEPA(
    metric=your_metric_with_feedback,           # Must provide rich feedback
    reflection_lm=dspy.LM(
        model="openai/gpt-4",                   # Strong model for reflection
        temperature=1.0,                        # High temperature for diversity
        max_tokens=32000,                       # Large context for reflection
        api_key=api_key
    ),
    num_threads=16,                             # Parallel processing
    track_stats=True,                           # Essential for detailed analysis & reporting
    track_best_outputs=True,                    # Helpful for debugging
    auto="heavy",                               # Use for production (vs max_full_evals for testing)
    reflection_minibatch_size=3,                # Efficiency setting
    candidate_selection_strategy="pareto",      # Diversity in candidate selection
)
```

### Data Preparation
- **Quality over quantity**: GEPA works with few examples due to rich feedback
- **Balanced splits**: Use 80/20 train/validation split
- **Representative examples**: Ensure training data covers key task aspects

### Anti-Patterns to Avoid
- **Don't use scalar-only metrics** - GEPA needs rich natural language feedback
- **Don't use weak reflection models** - The reflection LM should be stronger than task LM  
- **Don't skip tracking** - Always use `track_stats=True` for analysis and reporting
- **Don't over-constrain budget** - GEPA works efficiently, allow reasonable budget

### Validation and Testing
- **Start with small budget** for testing (`max_full_evals=1-5`)
- **Scale up gradually** to `auto="heavy"` for production
- **Monitor detailed_results** for insights into optimization process
- **Compare against baselines** to validate improvements

### Key Research Insights
- GEPA outperforms GRPO by 10% average, up to 20% on some tasks
- Uses natural language reflection instead of scalar rewards
- Works with as few as 1-10 evaluations due to rich feedback
- Builds Pareto frontier of candidates for robust optimization
- Particularly effective when domain-specific feedback is available

### Optimization Reports
The system automatically generates comprehensive optimization reports:
- **TXT reports**: Human-readable summary with key insights
- **JSON reports**: Detailed data for further analysis
- **Performance comparisons**: Before vs after optimization metrics
- **GEPA analysis**: Insights from `detailed_results` with `track_stats=True`

Reports are saved to `data/optimization_reports/` with timestamps.

### General Best Practices
1. **Always test optimized prompts** before deploying
2. **Keep backups** of working prompts
3. **Use descriptive keys** that match the generation script expectations
4. **Document optimization parameters** in metadata
5. **Version control** optimized prompts in git
6. **Monitor optimization reports** for insights and improvements
7. **Use domain-specific feedback** in metrics for better results

## Next Steps

1. **Run existing optimizers**:
   ```bash
   python prompt_optimization/optimizers/lv0_s1.py
   python prompt_optimization/optimizers/lv0_s3.py
   ```

2. **Create new optimizers** using the template:
   ```bash
   cp prompt_optimization/optimizers/template.py \
      prompt_optimization/optimizers/your_task.py
   ```

3. **Verify integration** with generation scripts:
   ```bash
   ls -la data/prompts/
   python test_prompt_integration.py
   ```

4. **Monitor results** in generation logs to confirm optimized prompt usage