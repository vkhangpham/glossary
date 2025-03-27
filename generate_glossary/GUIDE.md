# Glossary Generation Pipeline Guide

This guide outlines the complete process for generating, validating, and deduplicating academic concepts across different levels (0-4).

## Directory Structure

Each level follows this structure:
```
data/
├── lv0/
│   ├── raw/                 # Raw generated data
│   │   ├── lv0_s0_college_names.txt
│   │   ├── lv0_s0_metadata.json
│   │   ├── lv0_s1_extracted_concepts.txt
│   │   ├── lv0_s1_metadata.json
│   │   ├── lv0_s2_filtered_concepts.txt
│   │   ├── lv0_s2_metadata.json
│   │   ├── lv0_s3_verified_concepts.txt
│   │   └── lv0_s3_metadata.json
│   ├── postprocessed/       # Validated and deduplicated data
│   │   ├── lv0_rv.txt       # Rule-validated
│   │   ├── lv0_rv.json
│   │   ├── lv0_wv.txt       # Web-validated
│   │   ├── lv0_wv.json
│   │   ├── lv0_lv.txt       # LLM-validated
│   │   ├── lv0_lv.json
│   │   └── lv0_final.txt    # Final deduplicated terms
│   ├── metadata.json        # Collected metadata about terms
│   └── lv0_resources.json   # Web content for this level
├── lv1/
│   ├── raw/
│   ├── postprocessed/
│   ├── metadata.json
│   └── lv1_resources.json
└── ...
```

## Pipeline Overview

The pipeline consists of two main phases:
1. **Generation Phase**: Extract raw concepts from academic sources
2. **Processing Phase**: Validate and deduplicate concepts
3. **Metadata Collection**: Collect comprehensive metadata about each term

## Generation Phase

Each level has its own generation scripts:

### Level 0 (Broad Academic Disciplines)
```bash
# Step 0: Extract college/school names
python -m generate_glossary.generation.lv0.lv0_s0_get_college_names

# Step 1: Extract concepts from college names
python -m generate_glossary.generation.lv0.lv0_s1_extract_concepts

# Step 2: Filter by institution frequency
python -m generate_glossary.generation.lv0.lv0_s2_filter_by_institution_freq

# Step 3: Verify single-token concepts
python -m generate_glossary.generation.lv0.lv0_s3_verify_single_token
```

### Level 1 (Major Academic Fields)
```bash
# Similar steps for level 1
python -m generate_glossary.generation.lv1.lv1_s0_...
# ...and so on
```

### Web Content Mining
After generating concepts, mine web content for each level:
```bash
python -m generate_glossary.web_miner_cli --input data/lvX/raw/lvX_s3_verified_concepts.txt --output data/lvX/lvX_resources
```

## Processing Phase

After generating concepts and mining web content, run these steps in order:

### 1. Rule-based Validation
```bash
PYTHONPATH=. python -m generate_glossary.validator.cli data/lvX/raw/lvX_s3_verified_concepts.txt -m rule -o data/lvX/postprocessed/lvX_rv
```

### 2. Web-based Validation with Relevance Scores
```bash
# Run web validation and update web content with relevance scores in-place
PYTHONPATH=. python -m generate_glossary.validator.cli data/lvX/postprocessed/lvX_rv.txt -m web -w data/lvX/lvX_resources.json -o data/lvX/postprocessed/lvX_wv --update-web-content
```

### 3. LLM-based Validation
```bash
PYTHONPATH=. python -m generate_glossary.validator.cli data/lvX/postprocessed/lvX_wv.txt -m llm -p openai -o data/lvX/postprocessed/lvX_lv
```

### 4. Deduplication (Choose ONE Method)

You only need to choose ONE deduplication method. Graph-based deduplication is recommended as it combines the benefits of rule-based and web-based approaches.

#### Option A: Graph-based Deduplication (Recommended)
For level 0:
```bash
# Use the web content with relevance scores for better deduplication
PYTHONPATH=. python -m generate_glossary.deduplicator.cli data/lvX/postprocessed/lvX_lv.txt -m graph -w data/lvX/lvX_resources.json -o data/lvX/postprocessed/lvX_final
```

For levels 1+:
```bash
# Use the web content with relevance scores for better deduplication
PYTHONPATH=. python -m generate_glossary.deduplicator.cli data/lvX/postprocessed/lvX_lv.txt -t 0:data/lv0/postprocessed/lv0_final.txt ... -m graph -w data/lvX/lvX_resources.json -c 0:data/lv0/lv0_resources.json ... -o data/lvX/postprocessed/lvX_final
```

#### Option B: Rule-based Deduplication
For level 0:
```bash
PYTHONPATH=. python -m generate_glossary.deduplicator.cli data/lvX/postprocessed/lvX_lv.txt -m rule -o data/lvX/postprocessed/lvX_final
```

For levels 1+:
```bash
PYTHONPATH=. python -m generate_glossary.deduplicator.cli data/lvX/postprocessed/lvX_lv.txt -t 0:data/lv0/postprocessed/lv0_final.txt ... -m rule -o data/lvX/postprocessed/lvX_final
```

#### Option C: LLM-based Deduplication
```bash
PYTHONPATH=. python -m generate_glossary.deduplicator.cli data/lvX/postprocessed/lvX_lv.txt -m llm -p openai -o data/lvX/postprocessed/lvX_final
```

The final output file `data/lvX/postprocessed/lvX_final.txt` contains the fully processed terms for that level.

### 5. Metadata Collection

After completing all validation and deduplication steps, collect comprehensive metadata about each term:

```bash
PYTHONPATH=. python -m generate_glossary.metadata_collector_cli -l X
```

This generates a `metadata.json` file in the level directory that contains metadata **only** for terms that appear in the final output file (`lv{X}_final.txt`). For each term, the metadata includes:
- Raw sources for each term (from Step 1 output)
- Direct parent terms (extracted from college/category information)
- Term variations (collected from deduplication JSON files)

The metadata collector searches for information from multiple sources:

1. **Step 1 CSV file**: The primary source for term origin information
2. **Metadata JSON files**: Additional metadata from each generation step in the `raw` directory
3. **Deduplication JSON files**: Term variation information from the `postprocessed` directory

The collector follows a robust data collection approach:
- First identifies all terms in the final output file
- Creates metadata entries for all final terms
- Looks for source information in both CSV and JSON files
- Extracts parent information from college names and explicit parent fields
- Collects all term variations from deduplication files
- Handles issues gracefully when files are missing or in different formats

If any of the normal source files are missing, the collector will still work with whatever data is available, making it resilient to different pipeline configurations.

The metadata is useful for:
- Understanding term relationships
- Tracing term origins
- Building hierarchical knowledge graphs
- Supporting search and filtering in applications

## Validation and Deduplication Modes

### Validation Modes
- **rule**: Basic validation based on term structure and format
- **web**: Validation using web content (Wikipedia, academic resources)
- **llm**: Validation using LLM analysis

### Deduplication Modes
- **graph** (Recommended): Graph-based deduplication using both academic variations and web content overlap
- **rule**: Basic deduplication using academic variations (plural/singular, spelling)
- **llm**: LLM-based deduplication for complex cases
- **web**: Web-based deduplication using content similarity

## Cross-Level Deduplication

When running deduplication for levels 1+, always include references to higher levels:

```bash
# For level 1
-t 0:data/lv0/postprocessed/lv0_final.txt

# For level 2
-t 0:data/lv0/postprocessed/lv0_final.txt 1:data/lv1/postprocessed/lv1_final.txt
```

This ensures proper hierarchical relationships between levels.

### Output Filtering

When using cross-level deduplication, the deduplicator will only include terms from the input level in the output. Terms from higher levels (reference levels) are used for relationship detection but won't appear in the output files.

The output will include:
1. Original input terms that aren't variations of other terms
2. Canonical terms for any input terms that are variations

This ensures that your output files contain only relevant terms for the current level, while still preserving all the variation relationships.

## Common CLI Arguments

### Validator CLI
```
-m, --mode            Validation mode: rule, web, llm (default: rule)
-w, --web-content     Path to web content JSON file (required for web mode)
-s, --min-score       Minimum score for web content validation (default: 0.7)
-r, --min-relevance-score  Minimum relevance score for web content (default: 0.3)
-p, --provider        LLM provider for validation (default: gemini)
-o, --output          Base path for output files
-n, --no-progress     Disable progress bar
--save-web-content    Path to save updated web content with relevance scores to a new file
--update-web-content  Update the input web content file in-place with relevance scores
```

When using web validation, you can add relevance scores to the web content in two ways:
1. Use `--update-web-content` to update the input web content file in-place (recommended to avoid redundancy)
2. Use `--save-web-content` to save the updated web content to a new file

These relevance scores are used by the deduplicator to improve term relationship detection by filtering out irrelevant web content.

### Deduplicator CLI
```
-m, --mode            Deduplication mode: rule, web, llm, graph (default: graph)
-w, --web-content     Path to web content JSON file
-t, --higher-level-terms  Paths to higher level term files (format: level:path)
-c, --higher-level-web-content  Paths to higher level web content files
-o, --output          Base path for output files
-p, --provider        LLM provider for llm mode (default: gemini)
-s, --min-score       Minimum score threshold for web content (default: 0.7)
-r, --min-relevance-score  Minimum relevance score for web content to be considered relevant to a term (default: 0.3)
-b, --batch-size      Batch size for parallel processing (default: 100)
-x, --max-workers     Maximum number of worker processes (default: auto)
-e, --use-enhanced-linguistics  Use enhanced linguistic analysis
```

### Metadata Collector CLI
```
-l, --level           Level number (0, 1, 2, etc.)
-o, --output          Output file path (default: data/lvX/metadata.json)
-v, --verbose         Enable verbose logging
```

## Troubleshooting

### API Keys
- For LLM validation/deduplication, ensure API keys are set:
  - OpenAI: `OPENAI_API_KEY`
  - Gemini: `GEMINI_API_KEY`

### Common Errors
- **Missing web content**: Ensure web content is mined before validation
- **Parameter errors**: Check CLI parameters match function signatures
- **Import errors**: Use `PYTHONPATH=.` prefix for all commands

## Output Files

Each step produces two files:
- `.txt`: Contains the list of terms (one per line)
- `.json`: Contains detailed metadata about the validation/deduplication process

The JSON files are useful for debugging and analysis.

Additionally, the metadata collection step produces:
- `metadata.json`: Comprehensive term metadata including sources, parents, and variations