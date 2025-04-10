# Academic Glossary Deduplicator

This module provides tools for identifying and consolidating duplicate or highly similar academic concepts across the hierarchy.

## Overview

The deduplicator identifies semantically similar academic terms that represent the same concept but may vary in wording, spelling, or specificity. It establishes canonical names for groups of related terms while preserving their relationships.

## Features

- Multiple deduplication strategies:
  - **Graph-based** (Recommended): Combines linguistic variations and content similarity in a graph structure
  - **Rule-based**: Identifies basic linguistic variations (plural/singular, spelling)
  - **Web-based**: Finds terms with similar associated web content
  - **LLM-based**: Uses language models to determine conceptual relationships
- Cross-level deduplication support
- Relationship detection between terms in different hierarchy levels
- Metadata preservation during consolidation
- Parallel processing with customizable batch size

## Deduplication Methodology

The deduplicator uses a multi-dimensional approach:

### 1. Semantic Similarity Analysis

- Uses transformer embeddings to detect terms with similar meanings
- Compares content associated with terms through cosine similarity
- Higher values (closer to 1.0) indicate semantically similar terms

### 2. Co-occurrence Statistics

- Analyzes how frequently terms appear together in the same contexts
- Calculates conditional probability P(term1|term2)
- Low co-occurrence between similar terms suggests potential duplication

### 3. Graph-based Relationship Detection

- Builds a graph of terms connected by similarity edges
- Finds transitive relationships between terms
- Identifies clusters of related terms

## Types of Duplicates Detected

### 1. Exact Synonyms
Terms that refer to the same concept with different wording
- Example: "computer applications" / "business applications"

### 2. Hierarchical Relationships
Terms where one is a broader/narrower concept of the other
- Example: "artificial intelligence" / "applied artificial intelligence"

### 3. Spelling Variations
Terms that differ only in spelling conventions
- Example: "behavior analysis" / "behaviour analysis"

### 4. Terminological Variations
Different terminology for the same concept across disciplines
- Example: "machine learning" / "statistical learning"

### 5. Abbreviated Forms
A term and its abbreviation/acronym
- Example: "AI" / "artificial intelligence"

## Usage

### Basic Usage

```bash
# Run graph-based deduplication on level 0
PYTHONPATH=. python -m generate_glossary.deduplicator.cli \
  data/lv0/postprocessed/lv0_lv.txt \
  -m graph \
  -w data/lv0/lv0_resources.json \
  -o data/lv0/postprocessed/lv0_final
```

### Cross-Level Deduplication

For levels 1 and above, include references to higher-level terms:

```bash
# Run graph-based deduplication on level 2 with references to levels 0 and 1
PYTHONPATH=. python -m generate_glossary.deduplicator.cli \
  data/lv2/postprocessed/lv2_lv.txt \
  -m graph \
  -w data/lv2/lv2_resources.json \
  -t 0:data/lv0/postprocessed/lv0_final.txt 1:data/lv1/postprocessed/lv1_final.txt \
  -c 0:data/lv0/lv0_resources.json 1:data/lv1/lv1_resources.json \
  -o data/lv2/postprocessed/lv2_final
```

### Command-line Options

```
-m, --mode            Deduplication mode: graph, rule, web, llm (default: graph)
-w, --web-content     Path to web content JSON file
-t, --higher-level-terms  Paths to higher level term files (format: level:path)
-c, --higher-level-web-content  Paths to higher level web content files
-o, --output          Base path for output files
-p, --provider        LLM provider for llm mode (default: gemini)
-s, --min-score       Minimum score threshold for web content (default: 0.7)
-r, --min-relevance-score  Minimum relevance score for web content (default: 0.3)
-b, --batch-size      Batch size for parallel processing (default: 100)
-x, --max-workers     Maximum number of worker processes (default: auto)
-e, --use-enhanced-linguistics  Use enhanced linguistic analysis
```

## Deduplication Modes

### Graph Mode (Recommended)

The graph mode combines rule-based and web-based approaches:

1. Builds a similarity graph based on linguistic variations
2. Enriches the graph with web content similarity edges
3. Detects connected components to find term clusters
4. Identifies the canonical term for each cluster
5. Preserves variation relationships in the output

```bash
# Example with custom thresholds
python -m generate_glossary.deduplicator.cli terms.txt \
  -m graph \
  -w web_content.json \
  -s 0.75 \
  -r 0.4 \
  -e \
  -o output/deduplicated
```

### Rule Mode

The rule mode uses linguistic analysis to detect variations:

1. Normalizes terms (lowercase, stemming)
2. Detects plural/singular forms
3. Identifies common spelling variations
4. Groups terms based on normalized forms

```bash
# Example with enhanced linguistics
python -m generate_glossary.deduplicator.cli terms.txt \
  -m rule \
  -e \
  -o output/deduplicated
```

### Web Mode

The web mode compares terms based on their associated web content:

1. Calculates content similarity between terms
2. Groups terms with highly similar content
3. Uses content relevance scores to improve accuracy

```bash
# Example with custom thresholds
python -m generate_glossary.deduplicator.cli terms.txt \
  -m web \
  -w web_content.json \
  -s 0.8 \
  -o output/deduplicated
```

### LLM Mode

The LLM mode uses language models to determine term relationships:

1. Sends pairs of potentially related terms to the LLM
2. Asks the LLM to classify the relationship
3. Groups terms based on the LLM's judgment

```bash
# Example with OpenAI
python -m generate_glossary.deduplicator.cli terms.txt \
  -m llm \
  -p openai \
  -o output/deduplicated
```

## Output Format

The deduplicator produces two output files:

1. `<output>.txt`: List of deduplicated terms (canonical forms only)
2. `<output>.json`: Detailed information about term relationships

### JSON Format Example

```json
{
  "artificial intelligence": {
    "term": "artificial intelligence",
    "variations": ["AI", "artificial intelligence systems"],
    "canonical": "artificial intelligence",
    "is_canonical": true,
    "variation_of": null,
    "similarity_scores": {
      "AI": 0.92,
      "artificial intelligence systems": 0.85
    }
  },
  "AI": {
    "term": "AI",
    "variations": [],
    "canonical": "artificial intelligence",
    "is_canonical": false,
    "variation_of": "artificial intelligence",
    "similarity_scores": {
      "artificial intelligence": 0.92
    }
  }
}
```

## Integration with the Pipeline

The deduplicator is typically run after validation in the pipeline:

1. Generate terms → 2. Mine web content → 3. Validate terms → **4. Deduplicate terms** → 5. Collect metadata

After deduplication, the `metadata_collector_cli.py` script will consolidate information about the deduplicated terms.

## Best Practices

1. **Use graph mode for most cases**
   - It combines the strengths of rule-based and web-based approaches
   - Works well for both within-level and cross-level deduplication

2. **Include higher-level terms for context**
   - Always include terms from levels above when deduplicating levels 1+
   - This ensures proper hierarchical relationships

3. **Adjust thresholds for different levels**
   - Lower levels (0-1) often need higher thresholds (0.75-0.8)
   - Higher levels (2-3) may work better with lower thresholds (0.6-0.7)

4. **Enable enhanced linguistics**
   - The `-e` flag improves detection of morphological variations
   - Especially useful for technical terms with affixes 