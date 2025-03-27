# Glossary Generation Pipeline

This document explains how to use the pipeline scripts to generate and process glossary terms for different levels.

## Overview

The glossary generation pipeline consists of two main phases:

1. **Generation Phase**: Extract raw concepts from academic sources
2. **Processing Phase**: Validate and deduplicate concepts

Each level (0-4) represents a different source of academic concepts:

- **Level 0**: Broad academic disciplines from college/school names
- **Level 1**: Major academic fields from department names
- **Level 2**: Research areas from journal names
- **Level 3**: Specific research topics from paper titles
- **Level 4**: Detailed research concepts from paper abstracts

## Pipeline Scripts

Two scripts are provided to run the pipeline:

1. `run_pipeline.py`: Command-line script with full control over all options
2. `run_interactive.py`: Interactive script that prompts for options

### Using the Command-Line Script

The command-line script provides full control over all pipeline options:

```bash
# Basic usage (run all steps for level 0)
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 0

# Specify a different LLM provider
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 0 --provider openai

# Skip certain phases
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 0 --skip-generation --skip-web-mining

# Use a different deduplication mode
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 0 --dedup-mode rule
```

#### Command-Line Options

```
--level, -l            Level to process (0-4)
--provider, -p         LLM provider to use (default: gemini)
--skip-generation, -sg Skip the generation phase
--skip-web-mining, -sw Skip the web mining step
--skip-validation, -sv Skip the validation steps
--skip-deduplication, -sd Skip the deduplication step
--dedup-mode, -dm      Deduplication mode to use (default: graph)
--cooldown, -c         Cooldown period in seconds between steps (default: 5)
```

### Using the Interactive Script

The interactive script provides a user-friendly interface to run the pipeline:

```bash
# Run the interactive script
PYTHONPATH=. python -m generate_glossary.run_interactive
```

The script will prompt for:

1. The level to process (0-4)
2. The LLM provider to use
3. The cooldown period between steps
4. The deduplication mode to use
5. Which phases to skip (if any)

After confirming the configuration, the script will run the pipeline with the specified options.

## Pipeline Phases

### Generation Phase

The generation phase consists of four steps for each level:

1. **Step 0**: Extract source data (college names, department names, etc.)
2. **Step 1**: Extract concepts from the source data
3. **Step 2**: Filter concepts by frequency
4. **Step 3**: Verify concepts

### Web Mining

After generating concepts, the pipeline mines web content for each concept to gather additional information.

### Validation

The validation phase consists of three steps:

1. **Rule-based validation**: Basic validation based on term structure
2. **Web-based validation**: Validation using web content
3. **LLM-based validation**: Validation using LLM analysis

### Deduplication

The deduplication phase removes duplicate concepts using one of four modes:

1. **Graph-based deduplication** (recommended): Combines rule-based and web-based approaches
2. **Rule-based deduplication**: Basic deduplication using academic variations
3. **Web-based deduplication**: Deduplication using web content similarity
4. **LLM-based deduplication**: Deduplication using LLM analysis

## Output Files

The pipeline generates the following output files for each level:

```
data/
├── lv{X}/
│   ├── raw/                 # Raw generated data
│   │   ├── lv{X}_s0_*.txt
│   │   ├── lv{X}_s1_*.txt
│   │   ├── lv{X}_s2_*.txt
│   │   └── lv{X}_s3_verified_concepts.txt
│   ├── postprocessed/       # Validated and deduplicated data
│   │   ├── lv{X}_rv.txt     # Rule-validated
│   │   ├── lv{X}_rv.json
│   │   ├── lv{X}_wv.txt     # Web-validated
│   │   ├── lv{X}_wv.json
│   │   ├── lv{X}_lv.txt     # LLM-validated
│   │   ├── lv{X}_lv.json
│   │   ├── lv{X}_final.txt  # Final deduplicated terms
│   │   └── lv{X}_final.json
│   └── lv{X}_resources.json # Web content for this level
```

## Troubleshooting

### Common Issues

1. **Missing NLTK resources**: If you encounter errors related to NLTK resources, try manually downloading them:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

2. **API key errors**: Ensure that the appropriate API keys are set for the LLM provider you're using:
   - OpenAI: `OPENAI_API_KEY`
   - Gemini: `GEMINI_API_KEY`
   - Anthropic: `ANTHROPIC_API_KEY`

3. **Path errors**: Make sure to run the scripts with `PYTHONPATH=.` to ensure proper module imports.

4. **Memory errors**: For large datasets, you may need to adjust batch sizes or run on a machine with more memory.

### Resuming a Failed Pipeline

If a pipeline fails at a specific phase, you can resume it by skipping the completed phases:

```bash
# Example: Resume from validation after generation and web mining completed
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 0 --skip-generation --skip-web-mining
```

## Advanced Usage

### Cross-Level Deduplication

The deduplication phase automatically handles cross-level deduplication when processing levels 1-4. It ensures that terms from higher levels (smaller numbers) are prioritized over terms from lower levels.

### Custom LLM Providers

The pipeline supports multiple LLM providers:
- `openai`: OpenAI (GPT-4o)
- `gemini`: Google Gemini
- `anthropic`: Anthropic Claude
- `deepseek`: DeepSeek AI

### Performance Tuning

For large datasets, you can adjust the cooldown period and batch sizes to optimize performance:

```bash
# Example: Set a longer cooldown period
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 0 --cooldown 10
``` 