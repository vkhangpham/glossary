# Academic Hierarchy System

This package provides tools for building, evaluating, and visualizing the hierarchical structure of academic terms across multiple levels.

## Components

The hierarchy system consists of three main components:

1. **Hierarchy Builder**: Constructs a hierarchical data structure from metadata
2. **Hierarchy Evaluator**: Analyzes hierarchy quality with metrics and visualizations
3. **Hierarchy Visualizer**: Provides an interactive web interface for exploring the hierarchy

## Hierarchy Builder

The hierarchy builder creates a comprehensive hierarchical data structure by:

- Establishing parent-child relationships between terms across different levels
- Including term variations from the deduplication process
- Consolidating metadata and resources from variations to canonical terms
- Supporting various export formats (JSON, GML, GraphML)

### Basic Usage

```bash
# Build hierarchy with default settings
python -m hierarchy.hierarchy_builder

# Build with custom output path and verbose output
python -m hierarchy.hierarchy_builder -o data/custom_hierarchy.json --verbose
```

### Command-line Arguments

```
-o, --output          Output file path for the hierarchy (default: data/hierarchy.json)
-g, --graph           Export graph to file (specified by path)
-f, --format          Format for graph export (choices: gml, graphml, json, default: gml)
-v, --visualize       Create a visualization and save to the specified path
-m, --max-nodes       Maximum number of nodes to include in visualization (default: 100)
--no-transfer-sources Disable transferring sources from variations to canonical terms
--no-transfer-resources Disable transferring resources from variations to canonical terms
--verbose             Enable verbose output
```

### Examples

```bash
# Export as GraphML
python -m hierarchy.hierarchy_builder --graph data/hierarchy.graphml --format graphml

# Create visualization with more nodes
python -m hierarchy.hierarchy_builder --visualize data/hierarchy_viz.png --max-nodes 200
```

## Hierarchy Evaluator

The hierarchy evaluator analyzes the quality of the constructed hierarchy by calculating metrics and generating visualizations. It helps identify issues and assess the overall structure.

### Basic Usage

```bash
# Generate a quick summary
python -m hierarchy.hierarchy_evaluator_cli -q

# Generate a detailed HTML report with visualizations
python -m hierarchy.hierarchy_evaluator_cli -r -v

# Evaluate a specific hierarchy file
python -m hierarchy.hierarchy_evaluator_cli -i path/to/hierarchy.json -r
```

### Key Metrics Evaluated

1. **Structural Metrics**:
   - Term distribution across levels
   - Orphaned terms (terms without parents)
   - Branching factors (average children per term)
   - Terminal terms (terms without children)

2. **Connectivity Analysis**:
   - Level-to-level connectivity
   - Parent-child relationship coverage

3. **Variation Consolidation**:
   - Distribution of term variations
   - Variation consolidation effectiveness

4. **Issue Detection**:
   - Orphaned terms at inappropriate levels
   - Terms with excessive children (potential over-branching)
   - Terms with too many parents (potential ambiguity)
   - Disconnected subgraphs in the hierarchy

### Generated Visualizations

The detailed report includes several visualizations:

- Term distribution chart
- Level connectivity heatmap
- Network structure visualization
- Branching factor analysis
- Variation distribution analysis

## Hierarchy Visualizer

The hierarchy visualizer provides an interactive web interface for exploring the hierarchy, viewing term metadata, and analyzing quality metrics.

### Basic Usage

```bash
# Start the visualization server on default port
python -m hierarchy.hierarchy_visualizer

# Specify a custom port
python -m hierarchy.hierarchy_visualizer -p 8080
```

### Features

- Interactive graph visualization with zooming and filtering
- Term search functionality
- Metadata viewing panel
- Quality analysis dashboard
- Parent-child relationship exploration

### Quality Dashboard

Access the quality dashboard at:
```
http://localhost:5000/quality
```

It provides:
- Visual summaries of hierarchy metrics
- Interactive charts
- Issue detection and recommendations

## Hierarchy Structure

The generated hierarchy has the following JSON structure:

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

## Three-Step Workflow

The recommended workflow for building, evaluating, and visualizing the hierarchy is:

### Step 1: Build the Hierarchy

```bash
python -m hierarchy.hierarchy_builder -o data/hierarchy.json --verbose
```

This will:
- Load metadata from all levels
- Build parent-child relationships
- Consolidate variations
- Save the hierarchy to `data/hierarchy.json`

### Step 2: Evaluate the Hierarchy

```bash
python -m hierarchy.hierarchy_evaluator_cli --save-all --verbose
```

This will:
- Load the hierarchy from `data/hierarchy.json`
- Calculate quality metrics
- Generate visualizations
- Save evaluation results to `data/evaluation/`
- Generate an HTML report at `data/reports/hierarchy_quality_report.html`

### Step 3: Visualize the Hierarchy

```bash
python -m hierarchy.hierarchy_visualizer -p 5000
```

This will:
- Start a web server on port 5000
- Load the hierarchy and evaluation data
- Serve the web interface at http://localhost:5000

## Important Notes

1. The three steps must be performed in order.
2. If you make changes to the hierarchy, you need to re-run steps 1 and 2.
3. The visualization server displays pre-computed data; no calculations happen on-demand.
4. To update visualizations or metrics, re-run the hierarchy evaluator.

## Requirements

- Python 3.6+
- NetworkX (for graph operations)
- Matplotlib (for static visualizations)
- Flask (for visualization server)
- Plotly (for interactive visualizations)

You can install the required packages using:

```bash
pip install networkx matplotlib flask plotly
``` 