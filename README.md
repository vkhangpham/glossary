# Academic Glossary Analysis

This repository contains tools for generating, validating, analyzing, and visualizing an academic glossary across different hierarchy levels.

## Hierarchy Structure

The academic glossary is structured in multiple levels:

- **Level 0**: Broad academic domains - corresponds to Colleges of a University
- **Level 1**: Academic fields - corresponds to Departments of a College
- **Level 2**: Specialized topics - corresponds to Research Areas of a Department
- **Level 3**: Conference/journal topics - corresponds to specialized topics discussed in academic conferences and journals
- **Level 4**: Detailed research concepts - corresponds to concepts from research paper abstracts

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
├── deduplicator/           # Term deduplication components
├── utils/                  # Shared utilities
│   ├── web_search/         # Web search and content extraction utilities
│   └── ...                 # Other utilities
├── run_pipeline.py         # Command-line pipeline runner
├── run_interactive.py      # Interactive pipeline runner
├── hierarchy_builder.py    # Builds term hierarchy
├── hierarchy_evaluator.py  # Evaluates hierarchy quality
└── hierarchy_visualizer.py # Visualizes term hierarchy

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

3. **Deduplication**: Identifies and groups term variations
   - **Graph** (Recommended): Combines rule-based and web content analysis
   - **Rule**: Basic linguistic variations (plural/singular, spelling)
   - **Web**: Uses similarity of web content
   - **LLM**: Uses language models to determine relationships

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
pip install -r requirements.txt

# Set up API keys (for LLM-based operations)
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

## Usage

### Running the Full Pipeline

For a guided, interactive experience:

```bash
# Run the interactive pipeline
PYTHONPATH=. python -m generate_glossary.run_interactive
```

For command-line control:

```bash
# Run the pipeline for level 0
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 0

# Run with custom options
PYTHONPATH=. python -m generate_glossary.run_pipeline --level 1 --provider openai --dedup-mode graph
```

### Building and Visualizing the Hierarchy

After processing all levels:

```bash
# Build the hierarchy
python -m generate_glossary.hierarchy_builder -o data/hierarchy.json --verbose

# Evaluate the hierarchy
python -m generate_glossary.hierarchy_evaluator --save-all --verbose

# Start the visualization server
python -m generate_glossary.hierarchy_visualizer -p 5000
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

- [Validator Documentation](generate_glossary/validator/README.md)
- [Deduplicator Documentation](generate_glossary/deduplicator/README.md)
- [Hierarchy Documentation](generate_glossary/hierarchy/README.md)
- [Pipeline Guide](generate_glossary/GUIDE.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
