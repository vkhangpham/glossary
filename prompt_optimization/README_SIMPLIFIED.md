# Simplified Prompt Optimization System

## Overview

This is a simplified prompt optimization workflow using direct Python execution instead of complex CLI commands. The system uses DSPy GEPA for optimization and a straightforward JSON save/load mechanism that integrates seamlessly with generation scripts.

## Quick Start

### 1. Run Optimization

```bash
# Direct execution of optimizer scripts
python prompt_optimization/optimizers/lv0_s1.py
python prompt_optimization/optimizers/lv0_s3.py

# Or use runner scripts
python run_lv0_s1_optimization.py
python run_lv0_s3_optimization.py
```

### 2. Save Optimized Prompt

```python
from prompt_optimization.save import save_prompt

# Save the optimized prompt
save_prompt("lv0_s1_system", optimized_prompt)
# Creates: data/prompts/lv0_s1_system_latest.json
```

### 3. Automatic Loading

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
├── optimizers/
│   ├── lv0_s1.py            # Level 0 Step 1 optimizer (direct execution)
│   ├── lv0_s3.py            # Level 0 Step 3 optimizer (direct execution)
│   ├── template.py          # Template for new optimizers
│   └── common.py            # Shared utilities
├── core.py                  # Core GEPA optimization logic
└── save.py                  # Simplified save mechanism

data/prompts/
├── lv0_s1_system_latest.json    # Level 0 Step 1 system prompt
├── lv0_s1_user_latest.json      # Level 0 Step 1 user prompt
├── lv0_s3_system_latest.json    # Level 0 Step 3 system prompt
└── lv0_s3_user_latest.json      # Level 0 Step 3 user prompt
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

## Direct Execution Workflow

### Running Optimizers
```bash
# Set up environment
export OPENAI_API_KEY="your-key"

# Run optimizers directly - no CLI needed!
python prompt_optimization/optimizers/lv0_s1.py
python prompt_optimization/optimizers/lv0_s3.py
```

The optimizers:
1. Load training data from `data/training/`
2. Run GEPA optimization with DSPy
3. Save optimized prompts to `data/prompts/`
4. Generation scripts automatically use the optimized versions

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

### Required Environment Variables
```bash
# Required for optimization
export OPENAI_API_KEY="your-key"

# Optional: Override default models
export GEPA_GEN_MODEL="gpt-4o-mini"      # Generation model
export GEPA_REFLECTION_MODEL="gpt-4"     # Reflection model (defaults to gen model)
```

### Training Data Format
```json
// data/training/lv0_s1.json
[
  {
    "input": "College of Engineering",
    "expected_output": [
      {
        "source": "College of Engineering",
        "concepts": ["engineering", "computer science", "mechanical engineering"],
        "reasoning": "Core academic disciplines in engineering"
      }
    ]
  }
]
```

## Best Practices

1. **Always test optimized prompts** before deploying
2. **Keep backups** of working prompts
3. **Use descriptive keys** that match the generation script expectations
4. **Document optimization parameters** in metadata
5. **Version control** optimized prompts in git

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