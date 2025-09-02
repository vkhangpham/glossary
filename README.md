# Academic Glossary Analysis

This repository contains tools for generating, validating, analyzing, and visualizing an academic glossary across different hierarchy levels.

## Hierarchy Structure

## Hierarchy Structure

The academic glossary is structured in 5 levels:

- **Level 0**: Broad academic domains - corresponds to Colleges of a University (99% F1 score)
- **Level 1**: Academic fields - corresponds to Departments of a College (99% F1 score)
- **Level 2**: Specialized topics - corresponds to Research Areas of a Department (~90-95% F1 score)
- **Level 3**: Conference/journal topics - corresponds to specialized topics discussed in academic conferences and journals (~90-95% F1 score)
- **Level 4**: Individual academic papers - corresponds to specific research publications

## Project Components

The project consists of several integrated components:

1. **Generation Pipeline**: Multi-stage pipeline to extract academic concepts from various sources
2. **Validation System**: Rule-based, web-based, and LLM-based validation of extracted concepts
3. **Deduplication Framework**: Graph-based, rule-based, and LLM-based methods for removing duplicates
4. **Hierarchy Builder**: Construction of parent-child relationships between terms
5. **Visualization Tools**: Interactive interfaces for exploring the hierarchy
6. **Evaluation Framework**: Analysis of the hierarchy quality with metrics and visualizations

## Directory Structure

```
generate_glossary/
├── generation/             # Components for initial term extraction
│   ├── lv0/                # Level 0 generation scripts (colleges)
│   ├── lv1/                # Level 1 generation scripts (departments) 
│   ├── lv2/                # Level 2 generation scripts (research areas)
│   └── lv3/                # Level 3 generation scripts (conference topics)
├── validator/              # Term validation components
├── deduplication/          # Graph-based deduplication module
├── utils/                  # Shared utilities
│   ├── web_search/         # Web search and content extraction utilities
│   └── ...                 # Other utilities
<!-- TODO: [Documentation] Remove references to non-existent pipeline scripts -->
<!-- run_pipeline.py and run_interactive.py do not exist in current codebase -->
<!-- hierarchy_* scripts are in separate hierarchy/ directory, not generate_glossary/ -->
├── CLI scripts for each step # Individual generation, validation, deduplication steps
└── See individual component READMEs for usage examples

data/
├── lv0/                    # Level 0 data
│   ├── raw/                # Generated terms
│   ├── postprocessed/      # Validated and deduplicated terms
│   ├── metadata.json       # Term metadata
│   └── lv0_resources.json  # Web content for terms
├── lv1/                    # Level 1 data (similar structure)
├── lv2/                    # Level 2 data (similar structure)
├── lv3/                    # Level 3 data (similar structure)
├── final/                  # Final processed data
│   ├── lv0/                # Level 0 final terms
│   ├── lv1/                # Level 1 final terms
│   ├── lv2/                # Level 2 final terms
│   └── lv3/                # Level 3 final terms
├── hierarchy.json          # Complete hierarchy structure
└── evaluation/             # Hierarchy evaluation results
```

## Glossary Pipeline Overview

The glossary is built level-by-level through a multi-stage pipeline:

### 1. Generation Phase

This phase extracts and initially refines potential terms:

1. **Step 0: Initial Term Extraction**: Extracts candidate terms from appropriate sources
   - **L0**: College/School names from faculty data
   - **L1**: Department names from web searches using L0 terms
   - **L2**: Research areas from web searches using L1 terms
   - **L3**: Conference topics from web searches using L2 terms

2. **Step 1: Concept Extraction**: Uses LLMs to extract standardized academic concepts
   - Example: "Department of Electrical and Computer Engineering" → "electrical engineering", "computer engineering"

3. **Step 2: Frequency Filtering**: Filters concepts based on occurrence frequency
   - Keeps concepts that appear in a minimum percentage of sources

4. **Step 3: Verification**: Uses LLMs to verify single-word concepts
   - Ensures single words are valid academic concepts (e.g., "arts", "law")
   - Multi-word concepts typically bypass this verification

### 2. Processing Phase

After generation, concepts undergo further processing:

1. **Web Content Mining**: Searches the web to gather content for each concept
   - Collects definitions, descriptions, and related information

2. **Validation**: Applies multiple validation methods
   - **Rule-based**: Basic structural checks
   - **Web-based**: Validates against web content with relevance scoring
   - **LLM-based**: Final validation with language models

3. **Deduplication**: Graph-based identification and grouping of term variations
   - **Graph-first architecture**: Terms are nodes, duplicates are connected components
   - **Three edge creation methods**:
     - **Rule-based**: Text similarity, compound terms, acronyms
     - **Web-based**: URL overlap, domain patterns, content similarity
     - **LLM-based**: Semantic analysis of terms with minimal URL overlap

4. **Metadata Collection**: Consolidates information about each term
   - Sources, parent-child relationships, variations, resources

### 3. Hierarchy Building

After processing all levels, a complete hierarchy is built:

1. **Parent-Child Relationships**: Establishes connections between terms
2. **Variation Consolidation**: Merges metadata from variations
3. **Resource Transfer**: Transfers content between related terms

### 4. Hierarchy Evaluation & Visualization

The completed hierarchy can be analyzed and visualized:

1. **Quality Metrics**: Calculates structural and connectivity metrics
2. **Issue Detection**: Identifies problems in the hierarchy
3. **Interactive Visualization**: Provides a web interface for exploration

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/glossary-analysis.git
cd glossary-analysis

# Install requirements
uv sync

# Set up API keys (for LLM-based operations)
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

## Web Mining with Firecrawl (NEW - 4x Faster)

The project now supports Firecrawl SDK for web content mining, providing:
- **4x faster extraction** compared to the original approach
- **Single API call** instead of complex HTML parsing pipeline
- **Built-in AI extraction** with structured schemas
- **Automatic JavaScript rendering** for dynamic sites
- **80% cost reduction** compared to separate search + LLM calls

### Setup Firecrawl:
1. Get your API key from [firecrawl.dev](https://www.firecrawl.dev) ($83/month for 10,000 concepts)
2. Add to environment: `export FIRECRAWL_API_KEY='fc-your-key'`
3. Test integration: `python test_firecrawl.py`

### Web Mining Options:
```bash
# Use Firecrawl SDK (recommended - 4x faster)
export USE_FIRECRAWL=true
uv run glossary-web-miner -i data/lv0/raw/lv0_s3_verified_concepts.txt -o data/lv0/lv0_resources

# Use original approach (slower but no API key needed)
export USE_FIRECRAWL=false
uv run glossary-web-miner -i data/lv0/raw/lv0_s3_verified_concepts.txt -o data/lv0/lv0_resources
```

## Usage

### Running the Full Pipeline

The generation pipeline is run level-by-level using individual scripts. Each level follows the same 4-step process:

```bash
# Level 0 Generation Pipeline
python -m generate_glossary.generation.lv0.lv0_s0_get_college_names
python -m generate_glossary.generation.lv0.lv0_s1_extract_concepts --provider openai
python -m generate_glossary.generation.lv0.lv0_s2_filter_by_institution_freq
python -m generate_glossary.generation.lv0.lv0_s3_verify_single_token --provider openai

# Level 1 Generation Pipeline (uses Level 0 results)
python -m generate_glossary.generation.lv1.lv1_s0_get_dept_names --input data/lv0/lv0_final.txt
python -m generate_glossary.generation.lv1.lv1_s1_extract_concepts --provider openai
python -m generate_glossary.generation.lv1.lv1_s2_filter_by_institution_freq
python -m generate_glossary.generation.lv1.lv1_s3_verify_single_token --provider openai

# Similar patterns for Level 2 and Level 3...
```

### Building and Visualizing the Hierarchy

After processing all levels:

```bash
# Build the hierarchy
python -m hierarchy.hierarchy_builder -o data/hierarchy.json --verbose

# Evaluate the hierarchy
python -m hierarchy.hierarchy_evaluator_cli --save-all --verbose

# Start the visualization server
python -m hierarchy.hierarchy_visualizer -p 5000
```

Then access the visualization at: `http://localhost:5000`

### Duplicate Analysis

To analyze potential duplicates:

```bash
# Analyze Level 2 duplicates
python analyze_duplicates.py -l 2 -s 0.7 -c 0.3 -m 0.1 -v

# Generate review spreadsheet
python duplicate_analyzer.py
```

## Documentation

For detailed documentation on specific components:

- [Generation Documentation](generate_glossary/generation/README.md)
- [Validation Documentation](generate_glossary/validation/README.md)
- [Deduplication Documentation](generate_glossary/deduplication/README.md)
- [Hierarchy Documentation](hierarchy/README.md)
- [Disambiguation Documentation](generate_glossary/disambiguation/README.md)
- [Changelog](CHANGELOG.md)


## Tavily Search Integration

The web content mining tool now supports using Tavily as an alternative search provider. To use this functionality:

1. Install the required dependency: `pip install tavily-python>=0.1.4`
2. Get a Tavily API key from [Tavily's website](https://tavily.com)
3. Set the API key as an environment variable: `export TAVILY_API_KEY=tvly-YOUR_API_KEY` or add it to your `.env` file

To use Tavily for mining content:

```bash
python -m generate_glossary.web_miner_cli -i input_terms.txt -o output_file --search-provider tavily
```

See the detailed documentation in `generate_glossary/utils/README_TAVILY.md` for more information.
