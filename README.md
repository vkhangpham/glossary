# Academic Glossary Analysis

This repository contains tools for analyzing an academic glossary across different hierarchy levels.

## Overview

The academic glossary is structured in three levels:

- **Level 0**: Broad academic domains - corresponds to Colleges of a University
- **Level 1**: Academic fields - corresponds to Departments of a College
- **Level 2**: Specialized topics - corresponds to Research Areas of a Department
- **Level 3**: Conference/journal topics - corresponds to specialized topics discussed in academic conferences and journals

The analysis tools help identify relationships between terms, detect potential duplicates, and visualize term clusters.

## Glossary Pipeline Overview

The glossary is built level-by-level through a multi-stage pipeline involving generation and processing phases for each level (L0, L1, L2, L3).

### I. Generation Phase (`generate_glossary/generation/lvX/`)

This phase extracts and initially refines potential terms for a specific level, often using the finalized terms from the level above as input or context.

1.  **Step 0: Initial Term Extraction (`_s0_`)**
    *   **Purpose:** Extracts candidate terms for the current level based on the previous level's output or external data.
    *   **L0 (`lv0_s0_get_college_names.py`):** Extracts College/School names (L0 terms) from external faculty data.
    *   **L1 (`lv1_s0_get_dept_names.py`):** Uses L0 terms (Colleges) to perform web searches and LLM validation to find associated Department names (L1 terms).
    *   **L2 (`lv2_s0_get_research_areas.py`):** Uses L1 terms (Departments) to perform web searches and LLM validation to find associated Research Area/Course names (L2 terms).
    *   **L3 (`lv3_s0_get_conference_topics.py`):** Uses L2 terms (Research Areas) to perform web searches and LLM validation to find associated Conference Topics and Journal Special Issues (L3 terms).

2.  **Step 1: Concept Extraction (`_s1_`)**
    *   **Purpose:** Processes the terms generated in Step 0 using an LLM to extract relevant, standardized academic concepts or keywords associated with those terms.
    *   **Example (`lv1_s1_extract_concepts.py`):** Takes L1 Department names and extracts concepts like "electrical engineering", "computer engineering" from "Department of Electrical and Computer Engineering".

3.  **Step 2: Frequency Filtering (`_s2_`)**
    *   **Purpose:** Filters the concepts extracted in Step 1 based on how frequently they appear across their sources (e.g., institutions for L0, departments for L1, topics for L2). This removes less common or potentially spurious concepts.
    *   **Example (`lv0_s2_filter_by_institution_freq.py`):** Keeps L0 concepts that appear in a minimum percentage of the source institutions.

4.  **Step 3: Single-Token Verification (`_s3_`)**
    *   **Purpose:** Uses an LLM to specifically verify single-word concepts generated in the previous steps. This step ensures that common single words accepted are indeed valid academic concepts in the given context (e.g., "arts", "law") and filters out generic words. Multi-word concepts typically bypass this verification.

### II. Processing Phase (Using `validator`, `deduplicator`, `web_miner`, `metadata_collector`)

After the initial generation, the verified concepts undergo further processing to ensure quality and uniqueness.

1.  **Web Content Mining (`web_miner.py`, `web_miner_cli.py`)**
    *   **Purpose:** For each verified concept from the generation phase, searches the web (general search, Wikipedia) to gather relevant text content (definitions, descriptions). This content is used in subsequent validation and deduplication steps.

2.  **Validation (`validator/`)**
    *   **Purpose:** Applies multiple validation layers to the concept list.
    *   **Modes:**
        *   `rule`: Basic structural checks.
        *   `web`: Validates concepts against the mined web content, calculating relevance scores for each piece of content relative to the concept.
        *   `llm`: Uses an LLM for a final validation check.

3.  **Deduplication (`deduplicator/`)**
    *   **Purpose:** Identifies and groups variations of the same concept. It aims to establish a canonical name for each group of related terms.
    *   **Modes:**
        *   `rule`: Basic linguistic variations (plural/singular, spelling).
        *   `web`: Uses similarity of associated web content.
        *   `llm`: Uses an LLM to determine relationships.
        *   `graph` (Recommended): Combines rule-based variations and web content overlap in a graph structure to find relationships, including transitive ones. Can use higher-level terms (`-t`) and their web content (`-c`) for cross-level analysis.

4.  **Metadata Collection (`metadata_collector_cli.py`)**
    *   **Purpose:** Consolidates metadata for each *final*, deduplicated term. It gathers information about the term's origin (raw sources), its variations (from deduplication), and parent relationships (based on source context like College/Department).

The output of the deduplication step (`lvX_final.txt`) represents the finalized set of terms for that level, ready for analysis or use in generating the next level.

### Level Relationships

The glossary levels are designed to be hierarchical:

*   **L0 (Colleges/Domains)** -> **L1 (Departments/Fields)** -> **L2 (Research Areas/Topics)** -> **L3 (Conference/Journal Topics)**

This hierarchy is established and maintained throughout the pipeline:

1.  **Generation:** As described in *Generation Phase - Step 0*, the finalized terms from a higher level (e.g., L0 `lv0_final.txt`) are used as the primary input or context to discover and validate terms for the next lower level (e.g., L1). Level 3 specifically uses Level 2 research areas as search queries to identify relevant conference topics and journal special issues.

2.  **Deduplication:** The graph-based deduplicator (`-m graph`) can explicitly use finalized terms from higher levels (`-t` argument) and their web content (`-c` argument). This allows it to identify and correctly handle relationships *between* levels, such as when a term at L1 is a direct variation or parent of a term at L2, or when a conference topic at L3 is a more specialized variation of a research area at L2.

3.  **Metadata:** The metadata collection step attempts to link terms to their parent concepts from higher levels, reinforcing the hierarchical structure. For Level 3, this means linking conference/journal topics back to their source research areas from Level 2.

## Key Features

- **Co-occurrence Analysis**: Detect terms that frequently appear together
- **Conditional Co-occurrence**: Identify potential duplicate terms based on statistical metrics
- **Visualization**: Generate visual representations of term clusters
- **Duplicate Detection**: Multi-metric approach to identify potential duplicate terms
- **Review Framework**: Tools to facilitate expert review of potential duplicates

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/glossary-analysis.git
cd glossary-analysis

# Install requirements
pip install -r requirements.txt
```

## Usage

### Conditional Co-occurrence Analysis

```bash
# Analyze Level 2 terms with a conditional ratio of 8.0
python visualize_conditional.py data/lv2_conditional_duplicates.json -r 8.0 -s 2 -o data/lv2_conditional_viz_top.png
```

### Duplicate Detection

```bash
# Detect potential duplicates in Level 2
python duplicate_detector.py -l 2 -s 2 -r 5.0

# Generate consolidated review spreadsheet
python duplicate_analyzer.py

# View analysis across all levels
cat all_levels_analysis.md
```

### Review Process

See `review_process.md` for a detailed guide on reviewing potential duplicates.

## File Structure

```
data/
  ├── lv0_metadata.json   # Level 0 term metadata
  ├── lv1_metadata.json   # Level 1 term metadata
  ├── lv2_metadata.json   # Level 2 term metadata
  └── ...                 # Analysis outputs
scripts/
  ├── duplicate_detector.py   # Detect potential duplicates
  ├── duplicate_analyzer.py   # Generate review spreadsheets
  └── visualize_conditional.py # Visualize conditional co-occurrences
docs/
  ├── review_process.md   # Guide for reviewing duplicates
  └── all_levels_analysis.md  # Analysis across all levels
```

## Reports

- `all_levels_analysis.md`: Comprehensive analysis of conditional co-occurrences across all three levels
- `review_process.md`: Detailed guide for the duplicate review process

## Key Findings

- Level 0 terms show broad interconnections between academic domains
- Level 1 reveals 34 distinct clusters of related fields
- Level 2 contains 83 clusters with highly specialized relationships
- Different threshold values are needed for each level to identify true duplicates

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
