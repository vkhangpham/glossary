# Academic Hierarchy Builder

This script builds a hierarchical data structure from the metadata collected across multiple levels. It maps all terms from levels 0, 1, and 2 into a connected hierarchy based on parent-child relationships.

## Features

- Builds a comprehensive hierarchical structure from metadata across levels 0, 1, and 2
- Establishes parent-child relationships between terms in different levels
- Includes term variations from the deduplication process
- Transfers sources and resources from variations to canonical terms
- Provides various export formats for the hierarchy (JSON, GML, GraphML)
- Creates visualizations of the hierarchy

## Requirements

- Python 3.6+
- NetworkX (for graph operations)
- Matplotlib (for visualization)

You can install the required packages using:

```bash
pip install networkx matplotlib
```

## Usage

### Basic Usage

To build the hierarchy and save it to the default location (`data/hierarchy.json`):

```bash
python hierarchy_builder.py
```

### Advanced Options

The script supports several command-line options:

```bash
python hierarchy_builder.py --output data/custom_hierarchy.json --verbose
```

### Command-line Arguments

- `-o, --output`: Output file path for the hierarchy (default: `data/hierarchy.json`)
- `-g, --graph`: Export graph to file (specified by path)
- `-f, --format`: Format for graph export (choices: `gml`, `graphml`, `json`, default: `gml`)
- `-v, --visualize`: Create a visualization and save to the specified path
- `-m, --max-nodes`: Maximum number of nodes to include in visualization (default: 100)
- `--no-transfer-sources`: Disable transferring sources from variations to canonical terms
- `--no-transfer-resources`: Disable transferring resources from variations to canonical terms
- `--verbose`: Enable verbose output

### Examples

Export as a graph in GraphML format:

```bash
python hierarchy_builder.py --graph data/hierarchy.graphml --format graphml --verbose
```

Create a visualization with 200 nodes:

```bash
python hierarchy_builder.py --visualize data/hierarchy_viz.png --max-nodes 200 --verbose
```

Disable source and resource transfer:

```bash
python hierarchy_builder.py --no-transfer-sources --no-transfer-resources
```

## Metadata Consolidation

By default, the hierarchy builder will transfer sources and resources from variations to their canonical terms. This means:

1. When a term is identified as a variation of a canonical term, any sources associated with the variation will be added to the canonical term's sources (if not already present).
2. Similarly, any resources (URLs, titles, content) associated with variations are transferred to the canonical term, avoiding duplicates based on URL.

This consolidation ensures that canonical terms have comprehensive information aggregated from all their variations.

## Hierarchy Structure

The generated hierarchy has the following structure:

```json
{
  "levels": {
    "0": ["term1", "term2", ...],
    "1": ["term3", "term4", ...],
    "2": ["term5", "term6", ...]
  },
  "relationships": {
    "parent_child": [["parent_term", "child_term", level], ...],
    "variations": [["canonical_term", "variation"], ...]
  },
  "terms": {
    "term1": {
      "level": 0,
      "sources": ["source1", "source2", ...],
      "parents": ["parent1", "parent2", ...],
      "variations": ["var1", "var2", ...],
      "children": ["child1", "child2", ...],
      "resources": [
        {
          "url": "https://example.com/resource1",
          "title": "Resource Title",
          "processed_content": "Content excerpt...",
          "score": 0.95,
          "relevance_score": 0.85
        },
        ...
      ]
    },
    ...
  },
  "stats": {
    "total_terms": 1000,
    "terms_by_level": {"0": 10, "1": 100, "2": 890},
    "total_relationships": 950,
    "total_variations": 500
  }
}
```

## Integration with the Glossary Pipeline

This hierarchy builder is designed to work with the metadata collected by the `metadata_collector.py` script. After running the metadata collection for all levels, use this script to build the complete hierarchy.

## Visualization

The visualization feature creates a graph representation of the hierarchy with:

- Color-coded nodes by level
- Directed edges showing parent-child relationships
- Node labels for term names

Due to the potentially large size of the full hierarchy, the visualization by default samples a maximum of 100 nodes. You can adjust this with the `--max-nodes` parameter. 