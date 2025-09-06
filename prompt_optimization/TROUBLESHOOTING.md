# Prompt Optimization Troubleshooting Guide

## Common Errors and Solutions

### 1. "ValueError: OpenAI's reasoning models require passing temperature=1.0"

**Error**:
```
ValueError: OpenAI's reasoning models require passing temperature=1.0 and max_tokens >= 16000
```

**Cause**: GPT-5 models have specific parameter requirements.

**Solution**: The CLI handles this automatically, but for manual usage:
```python
dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=16000)
```

### 2. "WARNING dspy.adapters.json_adapter: Failed to use structured output format"

**Error**:
```
WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode
ValidationError: Input should be a valid dictionary or instance of ConceptExtractionList
```

**Cause**: DSPy struggles with complex Pydantic models when LLM returns unexpected formats.

**Solutions**:
1. Simplify Pydantic models (remove custom validators)
2. Use clearer signature descriptions
3. Consider using SimpleExtractSignature with JSON strings

### 3. "Training data not found"

**Error**:
```
FileNotFoundError: Training data not found at data/prompts_training_data/lv0_s1.json
```

**Solution**: Create training data file with correct format:
```json
[
  {"input": "College of Engineering", "expected": ["engineering"]},
  {"input": "School of Medicine", "expected": ["medicine"]}
]
```

### 4. Lost Optimization Progress

**Symptom**: Optimization crashes after running for 30+ minutes

**Prevention**:
1. Use lighter optimization modes first
2. Save intermediate results
3. Monitor with logging

**Recovery**:
```python
# Load saved program without re-optimization
from prompt_optimization.optimizers.lv0_s1 import load_optimized_program
program = load_optimized_program()  # Loads from data/prompts/lv0_s1_program.json
```

### 5. Batch Size Confusion

**Symptom**: "Using 40 training examples" when you have 160 data points

**Explanation**: 
- 160 items → batched into groups
- With batch_size=5: creates ~32 batch examples + 32 single examples
- 70/30 split → 40 training, 18 validation

**Fix**: Adjust batch size to control example count:
```bash
# Smaller batches = more examples
uv run optimize-prompt --prompt lv0_s1 --batch-size 2

# Larger batches = fewer examples (but more realistic)
uv run optimize-prompt --prompt lv0_s1 --batch-size 20
```

### 6. Module Import Errors

**Error**:
```
ImportError: No module named 'prompt_optimization'
```

**Solutions**:
1. Use UV to run: `uv run optimize-prompt`
2. Ensure pyproject.toml has entry point:
```toml
[project.scripts]
optimize-prompt = "prompt_optimization.cli:optimize_prompt"
```

### 7. GEPA Not Found

**Error**:
```
AttributeError: module 'dspy' has no attribute 'GEPA'
```

**Cause**: Old DSPy version

**Solution**: Update to DSPy 3.0.3+:
```bash
uv add "dspy>=3.0.3"
```

### 8. Metric Function Errors

**Error**:
```
TypeError: metric_with_feedback() missing 3 required positional arguments
```

**Cause**: GEPA requires specific signature

**Solution**: Ensure last 3 args are optional:
```python
def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
```

### 9. Empty or Corrupted Save Files

**Symptom**: Saved files exist but are empty or very small

**Detection**:
```python
if program_path.stat().st_size < 100:
    print("WARNING: File seems corrupted!")
```

**Prevention**: Add verification after saves
**Recovery**: Check for backup files with hash versions

### 10. Training/Production Mismatch

**Symptom**: Optimized prompts perform poorly in production

**Cause**: Training on single items, production uses batches

**Solution**: Match production batch size:
```bash
# Production uses BATCH_SIZE=20
uv run optimize-prompt --prompt lv0_s1 --batch-size 20
```

## Debug Commands

### Check DSPy Version
```python
import dspy
print(dspy.__version__)  # Should be 3.0.3+
```

### Verify Training Data
```python
from prompt_optimization.optimizers.lv0_s1 import create_training_data
data = create_training_data(batch_size=5)
print(f"Created {len(data)} examples")
print(f"First example: {data[0]}")
```

### Test Metric Function
```python
from prompt_optimization.optimizers.lv0_s1 import metric_with_feedback, ConceptExtractionList, ConceptExtraction

gold = type('obj', (object,), {
    'extractions': ConceptExtractionList(extractions=[
        ConceptExtraction(source="College of Engineering", concepts=["engineering"])
    ])
})()

result = metric_with_feedback(gold, gold)
print(f"Score: {result.score}, Feedback: {result.feedback}")
```

### Load Saved Program
```python
from pathlib import Path
path = Path("data/prompts/lv0_s1_program.json")
if path.exists():
    print(f"Program exists, size: {path.stat().st_size} bytes")
else:
    print("No saved program found")
```

## Best Practices to Avoid Issues

1. **Start with Light Mode**: Test with `--auto light` first
2. **Use Budget Models**: Start with gpt-4o-mini for tasks
3. **Match Production**: Use same batch_size as production
4. **Verify Saves**: Check file sizes after optimization
5. **Keep Logs**: Use `--verbose` flag for debugging
6. **Test Locally**: Verify metric function before optimization
7. **Incremental Changes**: Don't change everything at once
8. **Version Control**: Commit before major optimizations
9. **Monitor Costs**: GEPA makes many LLM calls
10. **Document Issues**: Add new gotchas to this file!

## Getting Help

If you encounter issues not covered here:
1. Check the main README for updates
2. Review the optimizer source code comments
3. Look at git history for similar issues
4. Test with minimal examples first