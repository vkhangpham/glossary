# Web Content Validation with Relevance Scoring

This module provides functionality for validating technical concepts using web content, with a focus on assessing the relevance of web content to the concept being validated.

## Overview

The validator module supports multiple validation modes:

1. **Rule-based validation**: Basic validation based on term structure and format
2. **Wikipedia-based validation**: Validation using Wikipedia data
3. **LLM-based validation**: Validation using LLM analysis
4. **Web-based validation**: Validation using web content with relevance scoring

## Web Content Relevance Scoring

The web-based validation now includes a relevance scoring mechanism that assesses how relevant a web content is to the concept being validated. This is important because a web content might be high quality (high score) but not directly relevant to the concept.

### How Relevance Scoring Works

The relevance score is calculated using multiple strategies:

1. **Title Relevance (50% weight)**:
   - Exact match of term in title
   - Fuzzy matching for term variations
   - Word overlap between term and title

2. **Snippet Relevance (30% weight)**:
   - Exact match of term in snippet
   - Fuzzy matching for term variations
   - Word overlap between term and snippet

3. **Content Relevance (20% weight)**:
   - Term frequency in processed content
   - Normalized by content length

The final relevance score is a weighted combination of these components, resulting in a score between 0 and 1.

### Usage

When using web-based validation, you can now specify a minimum relevance score:

```python
from generate_glossary.validator import validate

results = validate(
    terms=["arts", "computer science"],
    mode="web",
    web_content=web_content_data,
    min_score=0.7,  # Minimum content quality score
    min_relevance_score=0.3  # Minimum relevance score
)
```

From the command line:

```bash
python -m generate_glossary.validator.cli terms.txt -m web -w web_content.json --min-score 0.7 --min-relevance-score 0.3 -o output
```

### Testing Relevance Scores

You can use the `test_relevance.py` script to test relevance scoring for a specific term:

```bash
python -m generate_glossary.validator.test_relevance "arts" web_content.json
```

This will display the relevance scores for each web content associated with the term "arts".

To run a full validation with relevance scoring:

```bash
python -m generate_glossary.validator.test_relevance "arts" web_content.json --validate
```

## Validation Result Format

The validation result now includes relevance information:

```json
{
  "term": "arts",
  "is_valid": true,
  "mode": "web",
  "details": {
    "num_sources": 5,
    "verified_sources": [
      {
        "url": "https://en.wikipedia.org/wiki/The_arts",
        "title": "The arts",
        "score": 3.98,
        "relevance_score": 0.95
      },
      ...
    ],
    "unverified_sources": [...],
    "relevant_sources": [
      {
        "url": "https://en.wikipedia.org/wiki/The_arts",
        "title": "The arts",
        "score": 3.98,
        "relevance_score": 0.95
      },
      ...
    ],
    "has_relevant_sources": true,
    "highest_relevance_score": 0.95
  }
}
```

## Configuration

The default thresholds are:
- `DEFAULT_MIN_SCORE = 0.7`: Minimum score for content quality
- `DEFAULT_MIN_RELEVANCE_SCORE = 0.3`: Minimum score for content relevance

These can be adjusted based on your specific needs. 