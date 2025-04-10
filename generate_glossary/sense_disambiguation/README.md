# Sense Disambiguation Package

This package provides tools to identify and resolve ambiguity within the academic glossary hierarchy.

## Purpose

Scientific terms can be ambiguous, meaning the same word (e.g., "stress", "model") can have different meanings depending on the academic context (e.g., Psychology vs. Materials Science). The goal of this package is to detect terms in the generated `hierarchy.json` that likely represent merged, ambiguous concepts and split them into distinct senses with appropriate contextual tags.

This is crucial for ensuring the accuracy and utility of the glossary, especially for downstream tasks like annotation or analysis where precise meaning is important.

## Components

The package consists of two main components:

1. **Detectors** (`detector.py`): Responsible for identifying potentially ambiguous terms in the hierarchy
2. **Splitter** (`splitter.py`): Analyzes detected terms and proposes splitting them into distinct senses with domain-specific tags

## 1. Ambiguity Detection (`detector.py`)

### 1.1 Parent Context Detector

The `ParentContextDetector` identifies ambiguous terms by analyzing the hierarchical context of a term's parents.

**Logic:**

1. **Identify Canonical Terms:** Only considers terms marked as canonical (primary representations, not variations) based on the `lv*_final.txt` files.
2. **Load Hierarchy:** Reads the consolidated `hierarchy.json` file produced by the `hierarchy_builder`.
3. **Analyze Parents:** For each canonical term with multiple parents listed in `hierarchy.json`:
   * It traces the ancestry of each parent up to Level 0 (College/Domain) and Level 1 (Department/Field).
   * It collects the set of unique `(Level 0 Ancestor, Level 1 Ancestor)` pairs for all parents.
4. **Flag Ambiguity:** A canonical term is flagged as potentially ambiguous if:
   * Its parents trace back to ancestors in *different* Level 0 domains.
   * Its parents trace back to the *same* Level 0 domain but *different* Level 1 domains.

**Input:**

* Path to the `hierarchy.json` file.
* A glob pattern matching the `lv*_final.txt` files containing canonical terms.

**Output:**

* A list of canonical term strings identified as potentially ambiguous based on parent context divergence.

### 1.2 Resource Cluster Detector

The `ResourceClusterDetector` takes a content-based approach, using sentence embeddings and clustering to identify terms whose resources appear to cover distinct topics.

**Logic:**

1. **Extract Informative Content:** For each resource, the system extracts the most informative portions from beginning, middle, and end sections, optimizing for semantic richness.
2. **Embed Content:** Resource snippets are embedded using a sentence transformer model (default: `all-MiniLM-L6-v2`).
3. **Cluster Embeddings:** Density-based clustering is applied to identify semantic clusters:
   * **DBSCAN:** Standard density-based clustering with explicit distance threshold (default)
   * **HDBSCAN:** Alternative clustering that handles varying density clusters (if installed)
4. **Flag Ambiguity:** Terms are flagged as potentially ambiguous when their resources form multiple distinct clusters.
5. **Calculate Metrics:** For each term, detailed metrics including cluster sizes, inter-cluster distances, and average separation are computed.

**Important Note:** The detector processes terms from all hierarchy levels together to ensure accurate semantic clustering. Results are filtered by level afterward. This provides better clustering quality by considering relationships across the entire hierarchy.

**Parameters:**

- `dbscan_eps`: 0.45 (moderate threshold that works for most terms)
- `dbscan_min_samples`: 2 (minimum samples to form a cluster)

**Content Extraction:**
The `_extract_informative_content` method employs a balanced approach to extract content:

* Takes portions from beginning (often containing definitions)
* Includes middle sections (core concepts)
* Adds ending portions (conclusions/summaries)
* Handles different content types (strings, lists, etc.)

**Input:**

* Path to the `hierarchy.json` file.
* A glob pattern matching the `lv*_final.txt` files containing canonical terms.

**Output:**

* A list of potentially ambiguous terms based on resource clustering.
* A detailed JSON containing cluster assignments and metrics for each term.

### 1.3 Hybrid Ambiguity Detector

The `HybridAmbiguityDetector` combines multiple detection approaches to provide higher confidence ambiguity detection with meaningful scoring.

**Logic:**

1. **Multiple Detection Methods:** Runs several detection strategies:
   * Parent Context Detection (hierarchy-based)
   * DBSCAN Clustering (embedding-based)
   * HDBSCAN Clustering (when available)
2. **Cross-Level Processing:** Terms from all hierarchy levels are clustered together to ensure accurate semantic grouping, then filtered by level.
3. **Result Aggregation:** Combines results from all detectors while tracking which detector found each term.
4. **Confidence Scoring:** Assigns confidence scores (0.0-1.0) to each ambiguous term based on:
   * How many detectors identified the term
   * Cluster separation metrics when available
   * Resource quality and quantity
5. **Confidence Classification:** Categorizes terms as high (≥0.8), medium (≥0.5), or low (<0.5) confidence.

**Output:**

* Comprehensive JSON results with detailed metrics and confidence scores
* Filtered term lists by confidence level
* Detailed per-term information showing which detection methods agreed
* Combined results file containing data for all processed levels

**Advantages:**

* More reliable detection by combining complementary signals
* Better semantic clustering by analyzing terms across all hierarchy levels together
* Prioritization through confidence scoring
* Detailed diagnostics through integrated metrics

### 1.4 Persistent Vector Storage

The codebase includes a persistent vector storage solution based on FAISS that provides several advantages:

1. **Persistent Storage:** Embeddings are stored on disk using FAISS indices, allowing them to be reused across multiple program runs.
2. **Tiered Caching:** The system uses a two-tier caching approach:
   * In-memory cache (`EmbeddingCache`) for fastest access during a session
   * Persistent storage (`PersistentVectorStore`) for cross-session reuse
3. **Metadata Storage:** Each embedding is stored with metadata including the source text and additional contextual information.
4. **Efficient Retrieval:** The system automatically checks both caches when embeddings are requested, optimizing for speed while ensuring persistence.
5. **API Compatibility:** The vector store implementation provides methods compatible with the existing caching system, making integration seamless.

**Key Features:**

- Automatic saving of embeddings at regular intervals
- Robust error handling for loading and saving operations
- Support for batch operations to improve efficiency
- Built-in statistics tracking for performance monitoring
- Vector similarity search capabilities for advanced use cases

**Storage Structure:**

```
data/
  vector_store/
    embeddings.faiss       # FAISS index containing the actual vectors
    metadata.pickle        # Metadata mapping between hashes and original content
```

**Requirements:**

- The `faiss-cpu` package is required for this functionality
- For GPU acceleration, `faiss-gpu` can be installed instead

## 2. Sense Splitting (`splitter.py`)

The `SenseSplitter` is responsible for analyzing ambiguous terms detected by the detectors and determining whether and how to split them into distinct senses.

### 2.1 Core Functionality

**Process:**

1. **Load Cluster Results:** Takes pre-computed cluster results from one of the detectors.
2. **Group Resources by Cluster:** For each ambiguous term, resources are grouped according to their cluster assignment.
3. **Generate Sense Tags:** LLM-based tagging assigns meaningful domain-specific tags to each cluster (with fallbacks to TF-IDF and context-based approaches).
4. **Validate Splits:** Each potential split is validated through multiple signals:
   * LLM-based field distinctness check to determine if the sense tags represent truly different concepts
   * Embedding-based cluster separation calculation
   * Keyword-based content analysis
5. **Generate Proposals:** Creates split proposals for terms that pass validation, organizing resources into distinct sense nodes.

### 2.2 Key Methods

#### Sense Tag Generation

The `_generate_sense_tags` method uses a multi-layered approach to assign meaningful domain tags to each cluster:

1. **LLM-based Tagging (Primary):** Uses a language model to analyze resource content and generate appropriate academic field tags. Includes level-specific context to ensure appropriate granularity.
2. **TF-IDF Tagging (Fallback 1):** Extracts domain-specific keywords using TF-IDF to identify distinctive terms in each cluster.
3. **Parent Context Tagging (Fallback 2):** Analyzes which parent terms the resources in a cluster are connected to.
4. **Generic Tagging (Final Fallback):** Uses "sense_X" format if all other methods fail.

#### Field Distinctness Validation

The `_check_field_distinctness_with_llm` method uses a language model to determine if two academic fields represent truly distinct concepts or merely different aspects of the same idea:

```
FIELD 1: "image_processing"
FIELD 2: "market_segmentation"
```

The LLM analyzes whether these represent fundamentally different core meanings (valid split) or just different aspects/applications of the same concept (invalid split).

#### Cluster Separation Calculation

The `_calculate_cluster_separation` method determines how well-separated clusters are in embedding space:

1. **Embedding-based Distance:** Computes the cosine distance between cluster centroids.
2. **Keyword Analysis:** Analyzes shared vocabulary between clusters to detect semantic differences.
3. **Level-specific Thresholds:** Applies different thresholds based on hierarchy level (stricter for higher levels).
4. **Score Boosting:** Enhances separation scores for clusters with high keyword distinctness but lower embedding distance.

### 2.3 Level-Specific Processing

The splitter applies different parameters based on the hierarchy level being processed:

| Level | Description      | Threshold | DBSCAN Parameters      |
| ----- | ---------------- | --------- | ---------------------- |
| 0     | College/School   | 0.7       | eps=0.6, min_samples=3 |
| 1     | Department       | 0.6       | eps=0.5, min_samples=2 |
| 2     | Research Area    | 0.5       | eps=0.4, min_samples=2 |
| 3     | Conference Topic | 0.25      | eps=0.3, min_samples=2 |

## Usage Examples

### Running via CLI

The simplest way to use the sense disambiguation system is through the command-line interface:

#### 1. Ambiguity Detection

```bash
# Run hybrid detector (recommended)
python -m generate_glossary.sense_disambiguation.cli detect --detector hybrid --level 2 --min-confidence 0.6

# Run specific detectors if needed
python -m generate_glossary.sense_disambiguation.cli detect --detector parent-context --level 2
python -m generate_glossary.sense_disambiguation.cli detect --detector resource-cluster --level 2 --clustering dbscan
```

**Important Note:** The detection process clusters terms from all hierarchy levels together to ensure accurate semantic grouping, but the results are filtered to only include terms from the specified level. This provides better clustering quality while maintaining level-specific output.

**Key Parameters:**

- `--detector`: Detection method (`hybrid`, `parent-context`, or `resource-cluster`)
- `--level`: Hierarchy level to filter results for (0-3)
- `--min-confidence`: Confidence threshold for hybrid detector (default: 0.5)
- `--model`: Sentence transformer model to use (default: `all-MiniLM-L6-v2`)
- `--clustering`: Clustering algorithm (`dbscan` or `hdbscan`)
- `--min-resources`: Minimum number of resources for a term to be analyzed (default: 5)

#### 2. Sense Splitting

```bash
# Run splitting on detection results
python -m generate_glossary.sense_disambiguation.cli split --level 2 --input-file data/ambiguity_detection_results/hybrid_detection_results_level2_[TIMESTAMP].json --use-llm
```

**Key Parameters:**

- `--level`: Hierarchy level being processed (0-3)
- `--input-file`: Path to the detection results JSON file
- `--use-llm`: Enable LLM for generating sense tags (recommended)
- `--output-dir`: Directory to save split proposals (optional)

### Running Programmatically

For more advanced usage or integration with other systems, you can also use the classes directly in Python:

#### Running the ResourceClusterDetector

```python
from generate_glossary.sense_disambiguation.detector import ResourceClusterDetector

# Define paths
hierarchy_file = "data/hierarchy.json"
final_terms_pattern = "data/final/lv*/lv*_final.txt"
output_dir = "data/ambiguity_detection_results"

# Initialize detector with DBSCAN (default)
detector = ResourceClusterDetector(
    hierarchy_file_path=hierarchy_file,
    final_term_files_pattern=final_terms_pattern,
    model_name='all-MiniLM-L6-v2',
    min_resources=5,
    clustering_algorithm='dbscan',  # or 'hdbscan'
    level=2,  # Specify level for filtering results
    output_dir=output_dir
)

# Run detection
ambiguous_terms = detector.detect_ambiguous_terms()

# Save detailed results including cluster assignments
result_path = detector.save_detailed_results("cluster_results.json")

print(f"Found {len(ambiguous_terms)} potentially ambiguous terms.")
print(f"Detailed results saved to: {result_path}")
```

#### Running the HybridAmbiguityDetector

```python
from generate_glossary.sense_disambiguation.detector import HybridAmbiguityDetector

# Define paths
hierarchy_file = "data/hierarchy.json"
final_terms_pattern = "data/final/lv*/lv*_final.txt"
output_dir = "data/ambiguity_detection_results"

# Initialize hybrid detector
hybrid_detector = HybridAmbiguityDetector(
    hierarchy_file_path=hierarchy_file,
    final_term_files_pattern=final_terms_pattern,
    model_name='all-MiniLM-L6-v2',
    min_resources=5,
    level=2,  # Optional: filter results to specific level
    output_dir=output_dir
)

# Run hybrid detection
results = hybrid_detector.detect_ambiguous_terms()

# Get high and medium confidence results
confidence_results = hybrid_detector.get_results_by_confidence(min_confidence=0.5)
high_confidence_terms = confidence_results["high"]
medium_confidence_terms = confidence_results["medium"]

# Save detailed results
output_path = hybrid_detector.save_results()

print(f"Found {len(results)} potentially ambiguous terms")
print(f"High confidence: {len(high_confidence_terms)}")
print(f"Medium confidence: {len(medium_confidence_terms)}")
print(f"Results saved to: {output_path}")
```

#### Running the SenseSplitter

```python
from generate_glossary.sense_disambiguation.splitter import SenseSplitter

# Define paths
hierarchy_file = "data/hierarchy.json"
cluster_results_file = "data/ambiguity_detection_results/hybrid_detection_results_level2_[TIMESTAMP].json"
output_dir = "data/sense_disambiguation_results"

# Initialize splitter
splitter = SenseSplitter(
    hierarchy_file_path=hierarchy_file,
    candidate_terms_list=[],  # Will be populated from the cluster results file
    cluster_results={},       # Will be populated from the cluster results file
    use_llm_for_tags=True,
    level=2,
    output_dir=output_dir
)

# Load pre-computed cluster results
splitter._load_cluster_results_from_file(cluster_results_file)

# Generate and save split proposals
accepted, rejected, output_path = splitter.run(save_output=True)

print(f"Generated {len(accepted)} accepted and {len(rejected)} rejected proposals")
print(f"Results saved to: {output_path}")
```

## Frequently Asked Questions (FAQ)

**Q: How does the `ResourceClusterDetector` represent a term for clustering?**

A: The detector doesn't represent the term itself as a single vector. Instead, it focuses on the term's associated **resources** (text snippets like definitions, abstracts).

1. It extracts informative text snippets from each resource associated with the term.
2. Each text snippet is converted into a high-dimensional **embedding vector** using a sentence transformer model (e.g., `all-MiniLM-L6-v2`).
3. Therefore, a term is effectively represented by the **set of embedding vectors** derived from its associated resource content.

**Q: How does clustering work in the `ResourceClusterDetector`?**

A: Clustering is performed on the **set of embedding vectors** for a term's resources, not on the term name itself.

1. The goal is to see if these vectors group together in the embedding space, indicating semantic similarity.
2. A density-based algorithm (like DBSCAN or HDBSCAN) is applied to the embeddings.
3. The algorithm groups vectors that are close to each other (semantically similar).
4. The output is a **cluster label** for each resource embedding, indicating which semantic group it belongs to.

**Q: What makes the `ResourceClusterDetector` flag a term as potentially ambiguous?**

A: A term is flagged if its associated resource embeddings form **two or more distinct clusters** (ignoring noise points labeled `-1`).

* The rationale is that if the textual resources linked to a single term naturally separate into multiple semantic groups, the term is likely being used with different meanings or in different contexts within those resources. For example, resources for "stress" might form one cluster related to psychology and another related to materials science.

**Q: What do the `-1` cluster labels mean?**

A: In DBSCAN and HDBSCAN, a label of `-1` signifies a **noise point**. This means the corresponding resource embedding was not close enough to any core group of points to be assigned to a cluster according to the algorithm's parameters (e.g., `eps` and `min_samples` for DBSCAN). Terms are only flagged as ambiguous if there are at least two *non-noise* clusters (labels >= 0).

**Q: Why cluster based on resources instead of the term name or hierarchy?**

A: Clustering resources directly analyzes the *semantic content* associated with a term. This helps uncover ambiguity even if the term appears in similar hierarchical positions. The `ParentContextDetector` specifically looks at hierarchical differences, while the `ResourceClusterDetector` looks at content differences. The `HybridAmbiguityDetector` combines both signals.

**Q: How does the `HybridAmbiguityDetector` combine results?**

A: It runs multiple detectors (Parent Context, DBSCAN Resource Cluster, HDBSCAN Resource Cluster if available) and combines their findings:

1. It gathers all unique terms flagged by *any* detector.
2. For each term, it calculates a **confidence score** based on how many detectors flagged it and metrics like cluster separation (if applicable).
3. Terms flagged by multiple detectors or showing strong cluster separation receive higher confidence scores. This helps prioritize the most likely ambiguous terms.

## Interpretation of Results

The `split_proposals_levelX_*.json` files contain detailed information about each proposed split:

```json
{
  "original_term": "segmentation",
  "level": 3,
  "cluster_count": 2,
  "proposed_senses": [
    {
      "sense_tag": "image_segmentation",
      "cluster_id": 0,
      "resource_count": 3,
      "sample_resources": [...]
    },
    {
      "sense_tag": "market_segmentation",
      "cluster_id": 1,
      "resource_count": 2,
      "sample_resources": [...]
    }
  ],
  "split_reason": "Valid split confirmed by semantic analysis: Tags represent distinct academic fields (separation: 0.69)"
}
```

Each proposal includes:

- The original term and its level
- Number of clusters found
- Proposed senses with their tags and sample resources
- Reasoning for accepting or rejecting the split
