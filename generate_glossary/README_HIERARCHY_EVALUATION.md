# Hierarchy Evaluation Workflow

This document describes the three-step workflow for building, evaluating, and visualizing the academic term hierarchy.

## Step 1: Build the Hierarchy

First, run the hierarchy builder to create the hierarchy data structure:

```bash
python -m generate_glossary.hierarchy_builder -o data/hierarchy.json --verbose
```

This will:
- Load metadata from all levels
- Build parent-child relationships
- Consolidate variations
- Save the hierarchy to `data/hierarchy.json`

## Step 2: Evaluate the Hierarchy (Offline Processing)

Next, run the hierarchy evaluator to analyze the hierarchy and save evaluation results:

```bash
python -m generate_glossary.hierarchy_evaluator --save-all --verbose
```

This will:
- Load the hierarchy from `data/hierarchy.json`
- Calculate metrics like terms per level, orphans, branching factors, etc.
- Generate visualizations (bar charts, heatmaps, etc.)
- Save all evaluation results to `data/evaluation/` directory
- Generate an HTML report at `data/reports/hierarchy_quality_report.html`

The following files will be created:
- `data/evaluation/metrics.json` - Basic hierarchy metrics
- `data/evaluation/issues.json` - Detected hierarchy issues
- `data/evaluation/connectivity.json` - Level connectivity analysis
- `data/evaluation/summary.json` - Summary statistics for each level
- `data/evaluation/visualizations/*.json` - Pre-rendered visualizations

## Step 3: Run the Visualization Server

Finally, run the visualization server to display the hierarchy and evaluation results:

```bash
python -m generate_glossary.hierarchy_visualizer -p 5000
```

This will:
- Start a web server on port 5000
- Load the hierarchy and resources data
- Load pre-computed evaluation results (no calculations are performed)
- Serve the web interface at http://localhost:5000

Access the quality dashboard at http://localhost:5000/quality

## Important Notes

1. The three steps must be performed in order.
2. If you make changes to the hierarchy, you need to re-run steps 1 and 2 before step 3.
3. The visualization server only displays pre-computed data; no processing happens on-demand.
4. To update visualizations or metrics, re-run the hierarchy evaluator (step 2).

## Typical Workflow

For development or testing changes:

1. Build the hierarchy: `python -m generate_glossary.hierarchy_builder -o data/hierarchy.json --verbose`
2. Evaluate (offline): `python -m generate_glossary.hierarchy_evaluator --save-all --verbose`
3. View the results: `python -m generate_glossary.hierarchy_visualizer -p 5000`

For quick updates after minor changes to the hierarchy:

1. Build the hierarchy: `python -m generate_glossary.hierarchy_builder -o data/hierarchy.json`
2. Re-evaluate: `python -m generate_glossary.hierarchy_evaluator --save-all`
3. Restart the visualization server 