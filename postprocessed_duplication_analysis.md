# Academic Glossary Duplicate Analysis & Consolidation Guide

## 1. Overview

This document describes the methodology, results, and recommendations for duplicate analysis performed on the academic glossary. The analysis identifies potentially redundant terms within the hierarchy by detecting semantically similar terms with low co-occurrence patterns.

The academic glossary spans multiple levels of hierarchy:
- **Level 0**: Broad academic domains (Colleges)
- **Level 1**: Academic fields (Departments)
- **Level 2**: Specialized topics (Research Areas)
- **Level 3**: Conference/journal topics

## 2. Methodology

### 2.1 Approach

The duplicate analysis uses a multi-dimensional approach combining:

1. **Semantic Similarity**: Transformer embeddings detect terms with similar meanings
2. **Co-occurrence Statistics**: Analysis of how often terms appear together 
3. **Information Theory Metrics**: Measurement of mutual information between terms

This combination identifies semantically similar terms (high embedding similarity) that rarely co-occur (low conditional probability and mutual information) - a strong indicator of potential duplication.

### 2.2 Key Metrics

#### Embedding Similarity
- **Implementation**: Transformer-based embeddings (SentenceTransformer model 'all-MiniLM-L6-v2')
- **Input**: Concatenated processed content from all resources associated with a term
- **Calculation**: Cosine similarity between term embeddings
- **Interpretation**: Higher values (closer to 1.0) indicate terms with similar semantic meanings

#### Conditional Probability
- **Definition**: P(term1|term2) = probability of term1 occurring given term2 has occurred
- **Data sources**: Term occurrences in sources (departments/institutions) and resources (URLs)
- **Calculation**: `P(term1|term2) = co_occurrences / count_of_term2_occurrences`
- **Interpretation**: Lower values indicate terms rarely appear *together in the same context* (e.g., same source or resource URL). If terms are interchangeable synonyms, the presence of one makes the simultaneous presence of the other in the same context less likely.

#### Mutual Information
- **Definition**: Measures the statistical dependency between the occurrences of two terms across the entire dataset.
- **Formula**: `I(X;Y) = Σ_x Σ_y p(x,y) * log2(p(x,y)/(p(x)p(y)))`
- **Interpretation**: Lower values indicate the terms' usage patterns are relatively *independent* across the corpus. Knowing one term appears doesn't strongly predict whether the other term appears elsewhere. This helps identify synonyms preferred in different sub-domains or contexts.

### 2.3 Detection Criteria

Terms are flagged as potential duplicates when they meet all of the following criteria:
- **High semantic similarity**: Embedding similarity ≥ 0.7
- **Low co-occurrence**: Conditional probability ≤ 0.3 (in at least one direction)
- **Low information sharing**: Mutual information ≤ 0.1 

### 2.4 Data Sources

The analysis utilizes two primary data sources:
1. **Metadata files** (`lvX_metadata.json`): Contains term relationships, variations, and source information
2. **Resource files** (`lvX_filtered_resources.json`): Contains processed content and URLs associated with terms

## 3. Results Analysis

### 3.1 Summary Statistics

- **Total potential duplicates identified**: 312 pairs
- **Number of parent categories with duplicates**: ~70 (based on network visualizations)
- **Average similarity score**: ~0.78 (estimated from samples)

### 3.2 Key Findings

1. **Cross-field terminology variations**: Many duplicate pairs represent the same concept expressed differently across academic fields (e.g., "business applications" vs "computer applications")

2. **Specificity differences**: General terms often paired with more specific variants (e.g., "artificial intelligence" vs "applied artificial intelligence")

3. **Regional/linguistic variations**: Some duplicates reflect American vs British spelling or terminology differences

4. **Hierarchical relationships**: Some potential duplicates actually represent hierarchical relationships (broader vs narrower terms)

### 3.3 Notable Duplicate Patterns

Based on the sample of results reviewed, several patterns emerge:

1. **Component-Whole Relations**: Terms where one is a component or specific type of the other
   - Example: "global supply chain management" and "supply chain"

2. **Applied vs Theoretical Terms**: General concepts paired with their applied counterparts
   - Example: "artificial intelligence" and "applied artificial intelligence"

3. **Technology-Process Pairs**: Where a technology and its associated process are distinct terms
   - Example: "simulation modeling" and "modeling"

### 3.4 Visualization Insights

The scatter plot visualization (`lv2_duplicates_scatter.png`) shows the relationship between:
- Embedding similarity (x-axis)
- Conditional probability (y-axis)
- Mutual information (color intensity)

The network visualizations by parent category help identify clusters of potentially interchangeable terms within specific academic domains.

## 4. Duplicate Types and Consolidation Strategies

### 4.1 Exact Synonyms

**Description**: Terms that refer to the exact same concept but use different wording.

**Examples**:
- "computer applications" / "business applications"
- "programming languages" / "computer languages"

**Recommendation**:
- Select the more commonly used term as canonical
- Prefer domain-agnostic terms unless the context is clearly domain-specific
- Maintain both terms in the resources to preserve domain-specific connections
- Add redirects from the non-canonical term to the canonical term

### 4.2 Hierarchical Relationships Miscategorized as Duplicates

**Description**: Terms where one is actually a broader or narrower concept of the other.

**Examples**:
- "global supply chain management" / "supply chain"
- "artificial intelligence" / "applied artificial intelligence"

**Recommendation**:
- Keep both terms as separate entries
- Establish explicit hierarchical relationships in the glossary
- Add "broader term" and "narrower term" links
- Ensure the parent-child relationship is properly reflected in the metadata

### 4.3 Spelling Variations

**Description**: Terms that differ only in spelling conventions (British vs. American, etc.).

**Examples**:
- "behaviour analysis" / "behavior analysis"
- "modelling" / "modeling"

**Recommendation**:
- Standardize on one spelling form (generally American English for consistency)
- Keep the non-standard spelling as a variation rather than a separate term
- Add automatic mappings for search and retrieval

### 4.4 Terminological Variations

**Description**: Different terminology for the same concept across disciplines.

**Examples**:
- "machine learning" / "statistical learning"
- "computational linguistics" / "natural language processing"

**Recommendation**:
- Consider the audience and primary discipline when selecting the canonical term
- Preserve discipline-specific terminology in the metadata
- Add cross-references to facilitate interdisciplinary understanding
- Consider adding scope notes to explain disciplinary contexts

### 4.5 Abbreviated/Full Form Pairs

**Description**: A term and its abbreviation or acronym form.

**Examples**:
- "ML" / "machine learning"
- "NLP" / "natural language processing"
- "AI" / "artificial intelligence"

**Recommendation**:
- Keep the full form as the canonical term
- Add the abbreviation as a variation
- When the abbreviation is extremely common, consider dual entries with bidirectional links

## 5. Consolidation Implementation Plan

### 5.1 Decision Framework

When deciding whether to consolidate terms, apply this sequential decision framework:

1. **Semantic equivalence test**: Are the terms truly conceptually equivalent? If not, maintain separate entries.

2. **Domain specificity check**: Are the terms specific to different domains? If yes, consider keeping both with cross-references.

3. **Usage dominance assessment**: Is one term significantly more common in the literature? If yes, prefer it as canonical.

4. **Precision evaluation**: Does one term more precisely represent the concept? If yes, prefer the more precise term.

5. **Consistency alignment**: Which term better aligns with the glossary's existing terminology patterns?

### 5.2 Prioritization Strategy

1. **High-priority consolidation candidates**:
   - Term pairs with similarity > 0.8 and conditional probability < 0.1
   - Terms with identical or near-identical processed content
   - Terms that differ only in spelling variations or word order

2. **Medium-priority review candidates**:
   - Term pairs with similarity between 0.7-0.8 and conditional probability < 0.2
   - Terms that may have hierarchical relationships

3. **Low-priority or exclude**:
   - Term pairs with significant domain-specific differences
   - Term pairs with high mutual information despite meeting other criteria

### 5.3 Implementation Process

1. **Export duplicates to spreadsheet**:
   - Copy potential duplicates from `lv2_potential_duplicates.json` to a structured format
   - Add columns for: Decision (Consolidate/Keep Separate), Canonical Term, Relationship Type

2. **Review and annotate**:
   - Apply the decision framework to each pair
   - Document the rationale for each decision
   - Identify canonical terms for those to be consolidated

3. **Update data files**:
   - Modify `lvX_final.txt` to include only canonical terms
   - Update metadata to include relationships between consolidated terms
   - Merge resources of consolidated terms under the canonical term

4. **Validation check**:
   - Verify no key information is lost in consolidation
   - Check for consistency in the updated glossary
   - Test search and retrieval with both consolidated and non-canonical terms

### 5.4 Example Consolidation Record

```
Original terms: "machine learning", "statistical learning"
Decision: Consolidate
Canonical term: "machine learning"
Rationale: More widely used across disciplines, higher resource count
Relationship type: Exact synonym
Implementation: 
- Remove "statistical learning" from lv2_final.txt
- Add "statistical learning" to variations for "machine learning" in metadata
- Merge resources from both terms
- Add note: "Also known as statistical learning in statistics and mathematics disciplines"
```

## 6. Technical Implementation

The duplicate analysis was implemented in Python using:
- Transformer-based embeddings (SentenceTransformer)
- Cosine similarity for semantic comparison
- Custom information theory metrics for co-occurrence analysis
- NetworkX for visualization of term relationships

The implementation is available in the `analyze_duplicates.py` script with these key features:
- Customizable thresholds for similarity, conditional probability, and mutual information
- Visualization generation for analysis and presentation
- Support for different levels of the glossary hierarchy

### 6.1 Usage Instructions

To run the analysis:

```bash
# Basic usage
python analyze_duplicates.py -l <level>

# With custom thresholds
python analyze_duplicates.py -l <level> -s <similarity_threshold> -c <conditional_prob_threshold> -m <mutual_info_threshold>

# Example with specific thresholds
python analyze_duplicates.py -l 2 -s 0.7 -c 0.3 -m 0.2 -v
```

Parameters:
- `-l, --level`: Hierarchy level (0, 1, 2, 3)
- `-s, --similarity`: Similarity threshold (default: 0.7)
- `-c, --conditional-prob`: Conditional probability threshold (default: 0.3)
- `-m, --mutual-info`: Mutual information threshold (default: 0.1)
- `-v, --verbose`: Enable verbose logging
- `--no-visualize`: Disable visualization generation

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Content-dependent quality**: Analysis quality depends on the richness of processed content
2. **Threshold sensitivity**: Results are sensitive to threshold parameter settings
3. **Limited cross-level detection**: Current implementation primarily analyzes within-level duplicates

### 7.2 Future Improvements

1. **Cross-level analysis**: Extend to detect duplicates across different hierarchy levels
2. **Contextual embeddings**: Use context-aware embeddings to better distinguish domain-specific usages
3. **Interactive review interface**: Develop a tool for expert review of potential duplicates
4. **Feedback incorporation**: Add mechanism to improve detection based on expert feedback
5. **Automatic threshold optimization**: Dynamically adjust thresholds based on data characteristics

## 8. Conclusion

The duplicate analysis methodology effectively identifies semantically similar terms with low co-occurrence patterns, providing a systematic approach to glossary consolidation. The combination of transformer-based embeddings with statistical co-occurrence metrics offers a robust framework for detecting terminological redundancies while respecting domain-specific variations.

By applying the consolidation strategies outlined in this document, the academic glossary can achieve greater consistency, improved searchability, and enhanced user experience while preserving important disciplinary distinctions and relationships between terms. 