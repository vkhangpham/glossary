# Metadata Collection & Term Promotion

## Metadata Collection

The `metadata_collector.py` script collects information about terms after a level is completed, including:

- Source information (where the term was found)
- Parent-child relationships (hierarchical connections)
- Term variations (alternative forms of the same term)
- Resources (related articles, definitions, etc.)

All outputs are saved to both the original location and a dedicated `data/final` directory.

## Basic Usage

```bash
# Collect metadata for level 3 (with default resource collection and term promotion)
python -m generate_glossary.metadata_collector 3 -v

# Skip resource collection
python -m generate_glossary.metadata_collector 3 -v -r

# Skip term promotion
python -m generate_glossary.metadata_collector 3 -v -p
```

## Output Locations

All outputs are stored in two locations:

1. **Original location**: `data/lvX/` (for backward compatibility)
2. **Final directory**: `data/final/lvX/` (for use with hierarchy builder)

The final directory structure contains:
```
data/final/
├── promotion_log.json
├── lv0/
│   ├── lv0_final.txt
│   ├── lv0_metadata.json
│   └── lv0_filtered_resources.json
├── lv1/
│   ├── lv1_final.txt
│   ├── lv1_metadata.json
│   └── lv1_filtered_resources.json
├── lv2/
│   ├── lv2_final.txt
│   ├── lv2_metadata.json
│   └── lv2_filtered_resources.json
└── lv3/
    ├── lv3_final.txt
    ├── lv3_metadata.json
    └── lv3_filtered_resources.json
```

## Integration with Hierarchy Builder

The metadata collector is designed to work seamlessly with the hierarchy builder. The hierarchy builder will automatically detect and use data from the `data/final` directory if it exists, which is where the metadata collector stores the processed data with term promotion applied.

### Smart Data Source Selection

The hierarchy builder prioritizes loading data from the final directory (`data/final/lvX/`) before falling back to the original location if those files don't exist. This ensures that the hierarchy is built using the promoted terms when available.

### Warning Messages and Directory Detection

If the final directory is missing or empty, the hierarchy builder will display a warning message indicating that it will use the original data files, which may not have proper term promotion. This helps users understand which data source is being used for building the hierarchy.

### Transparent Operation

The hierarchy builder provides clear feedback about which directory is being used for loading data, especially when run with the verbose flag. This transparency helps users track the data flow through the system.

## Hierarchy Evaluation

To help assess the quality of the generated academic hierarchy, several evaluation tools are provided:

### Command-Line Evaluation

You can quickly evaluate the hierarchy quality using the evaluation tool:

```bash
# Generate a quick summary report
python -m generate_glossary.hierarchy_evaluator_cli -q

# Generate a detailed HTML report with visualizations
python -m generate_glossary.hierarchy_evaluator_cli -r -v

# Specify a different hierarchy file to evaluate
python -m generate_glossary.hierarchy_evaluator_cli -i path/to/hierarchy.json -r
```

### Interactive Quality Dashboard

The hierarchy visualizer includes an integrated quality analysis dashboard:

1. Start the visualizer:
   ```bash
   python -m generate_glossary.hierarchy_visualizer
   ```

2. Click the "Quality Analysis" button in the sidebar or navigate to:
   ```
   http://localhost:5000/quality
   ```

### Key Metrics Evaluated

The evaluation tools analyze several aspects of the hierarchy:

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

### Visualizations

The detailed report includes several visualizations:

- Term distribution chart
- Level connectivity heatmap
- Network structure visualization
- Branching factor analysis
- Variation distribution analysis

These tools help ensure that the academic hierarchy maintains a logical structure and consistent parent-child relationships across all four levels.

## Term Promotion (Enabled by Default)

The metadata collector automatically promotes terms between levels based on their parent-child relationships to ensure a consistent hierarchical structure when processing level 3. This feature can be disabled with the `-p` flag.

When terms are promoted, the original files remain untouched, while the `data/final` directory contains the updated hierarchy with promoted terms.

### What Term Promotion Does

The term promotion feature:

1. Identifies terms with hierarchical inconsistencies:
   - Level 3 terms with only level 0/1 parents are promoted to level 2
   - Level 2 terms with only level 0 parents are promoted to level 1
2. Promotes these terms to the appropriate level in the final directory
3. Updates all relevant files in the final directory:
   - Updates metadata.json files
   - Updates final.txt term lists
   - Moves resources between levels

### Example

If a level 3 term "quantum entropy" has a level 1 parent "quantum mechanics" but no level 2 parents, it will be promoted to level 2 to maintain hierarchical consistency. This change is reflected in the `data/final` directory.

Term promotion ensures that the hierarchical structure follows a clean parent-child relationship pattern, making the resulting glossary more coherent and better organized.

## Enhanced Variation Handling

The script now fully merges metadata from variations into their canonical terms:

1. **Complete Source Sharing**: All sources from variations are added to the canonical term
2. **Parent Inheritance**: Both canonical terms and variations inherit each other's parent relationships
3. **Comprehensive Metadata**: Ensures canonical terms have the complete set of metadata from all variations

For example, if "artificial intelligence" (canonical term) has a variation "AI", all metadata from "AI" (sources, parents, etc.) is automatically merged into the canonical term.

## Command-Line Options

```
usage: metadata_collector.py [-h] [-o OUTPUT] [-r] [-v] [--no-variations] [-p] level

Collect metadata for terms after level completion

positional arguments:
  level                 Level number (0, 1, 2, 3, etc.)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: data/lvX/metadata.json)
  -r, --resources       Skip collecting resources (default: collect resources)
  -v, --verbose         Enable verbose output
  --no-variations       Do not include metadata for variations
  -p, --promote         Skip promoting terms based on parent relationships (default: promote terms)
```

## Workflow Integration

The typical workflow for all four levels is:

1. Process level 0 terms: `python -m generate_glossary.metadata_collector 0 -v`
2. Process level 1 terms: `python -m generate_glossary.metadata_collector 1 -v` 
3. Process level 2 terms: `python -m generate_glossary.metadata_collector 2 -v`
4. Process level 3 terms: `python -m generate_glossary.metadata_collector 3 -v`
   - This automatically collects resources and performs term promotion
5. Build the hierarchy (using data from the final directory): `python -m generate_glossary.hierarchy_builder` 