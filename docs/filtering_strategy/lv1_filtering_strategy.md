# Level 1 Department Filtering Strategy

## Overview

The Level 1 Department Filtering Strategy is a comprehensive approach for extracting high-quality academic department names from web sources. It uses level 0 terms (academic disciplines) as seeds to discover university departments, employing a multi-stage filtering process with binary LLM-based decisions as the final validation step.

## Architecture

The system follows a pipeline architecture:

1. **Search Query Construction**: Uses level 0 terms to create targeted search queries
2. **Web Page Fetching**: Retrieves and processes HTML content from search results
3. **List Extraction**: Identifies potential department lists from various HTML structures
4. **Pre-filtering**: Applies heuristic-based filtering to reduce LLM API calls
5. **LLM Validation**: Uses binary (yes/no) LLM decisions for final validation
6. **Post-processing**: Consolidates and normalizes department names

## Key Components

### 1. Enhanced Aggressive Pre-filtering

Pre-filtering reduces unnecessary LLM API calls by filtering out obviously non-department lists:

- **Keyword Matching**: Uses a comprehensive dictionary of department-related keywords
- **Pattern Detection**: Applies regex patterns to identify department naming conventions
- **HTML Structure Analysis**: Identifies and filters out navigation elements, footers, and sidebars
- **Length Consistency Detection**: Ensures lists have consistent formatting
- **Non-Department Terms Threshold**: Rejects lists with too many irrelevant terms

Configuration Parameters:
- `DEPT_KEYWORD_THRESHOLD = 0.3` - Minimum percentage of items containing department keywords
- `LENGTH_VARIANCE_THRESHOLD = 0.6` - Maximum allowed variance in item lengths
- `PRE_FILTER_THRESHOLD = 0.55` - Threshold for heuristic-based pre-filtering
- `NON_DEPT_TERMS_THRESHOLD = 0.25` - Maximum percentage of non-department terms allowed

### 2. Statistical Filtering Integration

Statistical methods enhance filtering accuracy:

- **Jaccard Similarity Detection**: Identifies related department names
- **Domain-specific Weighting**: Prioritizes .edu domains with a 1.5x boost
- **Hierarchical Scoring**: Applies weighted scoring across multiple factors
- **Metadata Tracking**: Records comprehensive quality assessment information

Configuration Parameters:
- `JACCARD_SIMILARITY_THRESHOLD = 0.7` - Threshold for considering items similar
- `MIN_EDU_DOMAIN_WEIGHT = 1.5` - Weight multiplier for .edu domains

### 3. Binary LLM Decision Process

LLM-based validation uses binary decisions:

- **Specialized System Prompt**: Enforces clear yes/no decisions
- **JSON Response Format**: Structured output for easy parsing
- **Fallback Parsing**: Handles JSON formatting issues in LLM responses
- **Robust Error Handling**: Manages API call failures gracefully

Configuration Parameters:
- `LLM_VALIDATION_THRESHOLD = 0.6` - Minimum quality score for LLM validation
- `LLM_DECISIONS_THRESHOLD = 0.75` - Percentage of lists needing LLM approval
- `QUALITY_THRESHOLD = 0.7` - Minimum quality score to consider a list valid

### 4. Post-processing Validation

Final validation steps ensure high-quality output:

- **Cross-validation**: Prioritizes departments appearing in multiple lists
- **Filtering Rate Tracking**: Measures pre-filtering effectiveness
- **Educational Domain Weighting**: Gives higher priority to .edu sources

## Performance Metrics

Testing with all 13 level 0 terms demonstrated the following performance:

- **Efficiency**: 
  - Pre-filtering reduced candidate lists by approximately 60%
  - Estimated LLM API call savings: ~500 calls across all level 0 terms

- **Quality**: 
  - Extracted 665 unique departments from 13 level 0 terms
  - Binary LLM decisions effectively filtered out non-departmental lists

- **Speed**:
  - Total runtime: ~16.5 minutes for processing all 13 terms
  - Processing rate: 1.3 terms/minute with 3 concurrent terms

## Tuning Recommendations

For optimal performance, consider the following parameter adjustments:

1. **For More Conservative Results** (higher quality, fewer departments):
   - Increase `QUALITY_THRESHOLD` to 0.75-0.8
   - Increase `PRE_FILTER_THRESHOLD` to 0.6-0.65
   - Decrease `NON_DEPT_TERMS_THRESHOLD` to 0.2

2. **For More Liberal Results** (more departments, potentially lower quality):
   - Decrease `QUALITY_THRESHOLD` to 0.65
   - Decrease `PRE_FILTER_THRESHOLD` to 0.5
   - Increase `NON_DEPT_TERMS_THRESHOLD` to 0.3

## Integration with Pipeline

The department filtering strategy integrates with the broader concept processing pipeline:

1. **Input**: Level 0 terms from `data/lv0/postprocessed/lv0_final.txt`
2. **Output**: Department names saved to `data/lv1/lv1_s0_department_names.txt`
3. **Metadata**: Detailed execution information saved to `data/lv1/lv1_s0_metadata.json`

## Error Handling and Resilience

The implementation includes robust error handling:

- **Connection Issues**: Handles 403 Forbidden, timeouts, SSL errors
- **Encoding Detection**: Automatically identifies correct encoding or falls back to alternatives
- **LLM Response Parsing**: Deals with non-standard JSON formats in LLM responses
- **Caching Mechanism**: Avoids redundant processing of previously fetched URLs

## Limitations and Future Work

Current limitations and opportunities for future improvement:

1. **Domain-specific Challenges**: Some academic areas (e.g., nursing) may produce fewer results
2. **API Rate Limits**: Large-scale processing may require rate limit management
3. **Parameter Tuning**: Ongoing experimentation with threshold parameters could further optimize results
4. **Feedback Loop**: Implementing a mechanism to use validated data to improve future filtering
5. **Alternative LLM Providers**: Testing with different LLM providers could improve parsing reliability

## Conclusion

The Level 1 Department Filtering Strategy successfully addresses the project's goals by:

1. Making LLM the final binary decision-maker for filtering
2. Implementing aggressive pre-filtering to reduce unnecessary API calls
3. Providing robust end-to-end processing from level 0 terms to level 1 departments

This approach significantly improves both the quality of extracted departments and the efficiency of the extraction process.
