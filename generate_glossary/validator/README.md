# Concept Validator

This module provides tools for validating technical concepts using different validation modes.

## Features

- Multiple validation modes:
  - **Rule-based** validation (basic term structure)
  - **Web-based** validation (content verification)
  - **LLM-based** validation (concept verification)
- Automatically skip terms rejected in previous levels
- Track which level rejected each term
- Parallel processing with progress bar
- Customizable score thresholds

## Usage

### Basic Usage

```bash
# Validate terms using rule-based validation
python -m generate_glossary.validator.cli terms.txt -m rule -o results/rule_valid

# Validate terms using web-based validation
python -m generate_glossary.validator.cli terms.txt -m web -w web_content.json -o results/web_valid

# Validate terms using LLM-based validation
python -m generate_glossary.validator.cli terms.txt -m llm -p gemini -o results/llm_valid
```

### Using Level-Based Validation

When validating terms across multiple levels, you can use the `-l` or `--level` flag to automatically skip terms that were rejected in previous levels:

```bash
0# Level 1 validation
python -m generate_glossary.validator.cli terms.txt -m rule -l 1 -o data/lv1/lv1_rule_valid

# Level 2 validation (will skip terms rejected in level 0 and 1)
python -m generate_glossary.validator.cli terms.txt -m web -w web_content.json -l 2 -o data/lv2/lv2_web_valid

# Level 3 validation (will skip terms rejected in levels 0, 1 and 2)
python -m generate_glossary.validator.cli terms.txt -m llm -p gemini -l 3 -o data/lv3/lv3_llm_valid
```

The validator will automatically look for previous validation results in the `data/lvX/lvX*_valid.json` files for levels below the specified level.

### Additional Options

```
usage: cli.py [-h] [-m {rule,web,llm}] [-l LEVEL] [-w WEB_CONTENT] [-s MIN_SCORE]
              [-r MIN_RELEVANCE_SCORE] [-p PROVIDER] [-o OUTPUT] [-n]
              [--save-web-content SAVE_WEB_CONTENT] [--update-web-content]
              terms

positional arguments:
  terms                 Term to validate or path to file containing terms (one per line)

options:
  -h, --help            show this help message and exit
  -m {rule,web,llm}, --mode {rule,web,llm}
                        Validation mode to use (default: rule)
  -l LEVEL, --level LEVEL
                        Current validation level (if specified, terms rejected in levels 1 to level-1 will be automatically skipped)
  -w WEB_CONTENT, --web-content WEB_CONTENT
                        Path to web content JSON file (required for web mode)
  -s MIN_SCORE, --min-score MIN_SCORE
                        Minimum score for web content validation (default: 0.7)
  -r MIN_RELEVANCE_SCORE, --min-relevance-score MIN_RELEVANCE_SCORE
                        Minimum relevance score for web content to be considered relevant to the term (default: 0.77)
  -p PROVIDER, --provider PROVIDER
                        LLM provider for validation (default: gemini)
  -o OUTPUT, --output OUTPUT
                        Base path for output files (will create .txt and .json files)
  -n, --no-progress     Disable progress bar
  --save-web-content SAVE_WEB_CONTENT
                        Path to save updated web content with relevance scores (JSON file)
  --update-web-content  Update the input web content file in-place with relevance scores (only for web mode)
```

## Output Format

The validator outputs two files:

- `*.json`: Full validation results including details
- `*.txt`: List of valid terms only

### JSON Format Example

```json
{
  "machine learning": {
    "term": "machine learning",
    "is_valid": true,
    "mode": "web",
    "details": {
      "num_sources": 5,
      "verified_sources": [
        {
          "url": "https://example.com/ml",
          "title": "Machine Learning Basics",
          "score": 0.92,
          "relevance_score": 0.89
        }
      ],
      "unverified_sources": [],
      "relevant_sources": [
        {
          "url": "https://example.com/ml",
          "title": "Machine Learning Basics",
          "score": 0.92,
          "relevance_score": 0.89
        }
      ],
      "has_relevant_sources": true,
      "highest_relevance_score": 0.89
    }
  },
  "invalid term": {
    "term": "invalid term",
    "is_valid": false,
    "mode": "web",
    "details": {
      "reason": "Rejected in level 1",
      "level_rejected": 1
    }
  }
}
```
