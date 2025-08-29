# Deduplication Module

Graph-based deduplication where **the graph is everything**.
- Terms are nodes
- Duplicates are connected components
- Everything else is just querying the graph

## Core Philosophy

"Deduplication" is just a term we use to communicate. Under the hood:
- Every concept is a node in the graph
- Pairs of duplicates have edges between them
- Connected components represent groups of duplicates
- Canonical selection is just picking one representative from each component

## Architecture

### `main.py`
Primary entry point for building the graph:
- `build_graph()` - Build or extend the graph with terms and edges
- That's it. One function to build everything.

### `api.py`
Functions to query the graph without modifying it:
- `get_deduplicated_terms()` - Get canonical terms (clean list)
- `get_all_variations()` - Get terms with their variations
- `get_duplicate_clusters()` - Get connected components
- `is_duplicate_pair()` - Check if two terms are connected
- And more query functions...

### Supporting Modules

**Edge Creation (3 Methods):**
- `rule_based_dedup.py` - Text similarity, compound terms, acronyms
- `web_based_dedup.py` - URL overlap, domain patterns, content similarity  
- `llm_based_dedup.py` - Semantic understanding using language models

**Core Operations:**
- `graph_builder.py` - Graph construction utilities
- `canonical_selector.py` - Canonical term selection logic
- `graph_io.py` - Save/load graphs


## Edge Creation Methods

The graph uses 3 methods to identify duplicates:

1. **Rule-based** (fast, deterministic):
   - Text similarity (SequenceMatcher)
   - Compound term relationships ("Machine Learning" ↔ "Learning")
   - Acronym detection ("ML" ↔ "Machine Learning")
   - Known synonym patterns

2. **Web-based** (requires web content):
   - URL overlap (terms appearing on same pages)
   - Domain-specific patterns (arxiv.org, acm.org)
   - Content similarity from web snippets

3. **LLM-based** (semantic understanding):
   - Identifies non-obvious duplicates ("CV" ↔ "Computer Vision")
   - Understands context and domain
   - Groups conceptually identical terms
   - Handles ambiguous acronyms

## Progressive Flow

```
Level 0 terms → Rule-based → (Optional: LLM) → Web resources → Web-based → (Optional: LLM)
                                                                                    ↓
Level 1 terms → Rule-based → (Optional: LLM) → Web resources → Web-based → (Optional: LLM)
                                                                                    ↓
                                                                            Final Graph
                                                                                    ↓
                                                                        Canonical Selection
```

## Usage

```python
from generate_glossary.deduplication import (
    build_graph, 
    save_graph, 
    load_graph,
    get_deduplicated_terms,
    get_all_variations
)

# Build the graph (this is all you need)
graph = build_graph(
    terms_by_level={0: lv0_terms, 1: lv1_terms},
    web_content=all_web_content,
    existing_graph_path="previous_graph.pkl"  # Optional - extend existing
)

# Save it
save_graph(graph, "my_graph.pkl")

# Later, load and query it
graph = load_graph("my_graph.pkl")

# Get clean list (no duplicates)
canonical_terms = get_deduplicated_terms(graph)

# Get terms with their variations
variations = get_all_variations(graph)
```

## CLI Usage

```bash
# Build initial graph
uv run glossary-deduplicator terms.txt -l 0 -o output/lv0

# Extend with web content
uv run glossary-deduplicator terms.txt -l 0 -w web_content.json -g output/lv0.graph.pkl -o output/lv0_web

# Add next level
uv run glossary-deduplicator lv1_terms.txt -l 1 -g output/lv0_web.graph.pkl -o output/lv0_lv1

# With LLM for semantic understanding
uv run glossary-deduplicator terms.txt -l 0 --use-llm --llm-provider gemini -o output/lv0_llm

# Or run directly
uv run python -m generate_glossary.deduplication.main terms.txt -l 0 -o output/graph
```

## Graph as Output

The deduplication graph is a genuine output artifact that:
- Can be saved and loaded for incremental processing
- Can be visualized and analyzed
- Can be re-processed with different canonical selection strategies
- Preserves all relationship information for transparency