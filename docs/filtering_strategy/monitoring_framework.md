# Monitoring Framework for Level 1 Filtering Strategy

## Overview

This document outlines the monitoring framework for assessing and improving the Level 1 Department Filtering Strategy. The framework establishes key performance indicators (KPIs), data collection methods, and feedback mechanisms to ensure continuous improvement of the filtering process.

## Key Performance Indicators (KPIs)

### 1. Efficiency Metrics

- **Pre-filtering Rate**: Percentage of lists eliminated by pre-filtering
  - Target: >50% reduction in candidate lists
  - Calculation: `(total_lists - post_filtered_lists) / total_lists * 100`

- **LLM API Call Savings**:
  - Target: >40% reduction compared to non-pre-filtered approach
  - Calculation: `(potential_api_calls - actual_api_calls) / potential_api_calls * 100`

- **Processing Time**:
  - Target: Average <90 seconds per level 0 term
  - Calculation: `total_execution_time / number_of_level0_terms`

### 2. Quality Metrics

- **Department Yield**:
  - Target: Average >40 departments per level 0 term
  - Calculation: `total_extracted_departments / number_of_level0_terms`

- **Validation Success Rate**:
  - Target: >70% of LLM-validated lists are approved
  - Calculation: `approved_lists / total_validated_lists * 100`

- **Education Domain Rate**:
  - Target: >80% of departments sourced from .edu domains
  - Calculation: `edu_sourced_departments / total_departments * 100`

### 3. Reliability Metrics

- **Error Rate**:
  - Target: <5% of extraction attempts result in errors
  - Calculation: `error_count / total_extraction_attempts * 100`

- **LLM Response Parsing Success**:
  - Target: >95% successful parsing of LLM responses
  - Calculation: `successful_parses / total_responses * 100`

- **Connection Success Rate**:
  - Target: >80% of URL fetches successful
  - Calculation: `successful_fetches / total_fetches * 100`

## Data Collection

The monitoring framework uses the following data sources:

### 1. Execution Logs

- **Location**: Standard output and log files
- **Data Points**:
  - Pre-filtering success rates per level 0 term
  - API call counts and savings
  - Execution time per term
  - Error occurrences and types

### 2. Metadata JSON

- **Location**: `data/lv1/lv1_s0_metadata.json`
- **Data Points**:
  - Department counts by level 0 term
  - URL sources for departments
  - Department quality scores
  - Level 0 term to department mappings

### 3. Raw Search Results

- **Location**: `data/lv1/raw_search_results/`
- **Data Points**:
  - Search results by level 0 term
  - Extracted lists with verification status
  - URLs processed for each term

## Monitoring Implementation

### 1. Automated Metrics Collection

```python
def collect_metrics(metadata_path, log_path):
    """Collect metrics from metadata and logs"""
    metrics = {
        "efficiency": {},
        "quality": {},
        "reliability": {}
    }
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Calculate efficiency metrics
    metrics["efficiency"]["pre_filtering_rate"] = (
        (metadata.get("total_lists", 0) - metadata.get("filtered_lists", 0)) / 
        max(1, metadata.get("total_lists", 0)) * 100
    )
    
    # Calculate API call savings
    metrics["efficiency"]["api_call_savings"] = (
        (metadata.get("potential_api_calls", 0) - metadata.get("actual_api_calls", 0)) / 
        max(1, metadata.get("potential_api_calls", 0)) * 100
    )
    
    # Other metrics calculations...
    
    return metrics
```

### 2. Reporting Dashboard

Create a simple web dashboard using tools like Dash or Streamlit to visualize:

- Pre-filtering effectiveness over time
- Department yield by academic discipline
- Error rates and types
- LLM validation success rates

### 3. Alerting System

Implement alerts for:

- Abnormally low extraction rates
- High error rates
- API rate limit approaches
- Unusually long processing times

## Feedback Loop

### 1. Iterative Improvement Process

1. **Collect Data**: Gather metrics from pipeline runs
2. **Analyze Trends**: Identify patterns in successes and failures
3. **Prioritize Issues**: Focus on metrics furthest from targets
4. **Implement Changes**: Adjust thresholds or algorithms
5. **Test Changes**: Run comparison tests
6. **Measure Impact**: Compare pre and post-change metrics
7. **Document Findings**: Update documentation with insights

### 2. Parameter Tuning Framework

For automated parameter tuning:

```python
def tune_parameters(test_terms, parameter_ranges):
    """Find optimal parameters via grid search"""
    best_params = {}
    best_score = 0
    
    # Generate parameter combinations
    param_combinations = list(itertools.product(*parameter_ranges.values()))
    
    for params in param_combinations:
        param_dict = dict(zip(parameter_ranges.keys(), params))
        
        # Run test with these parameters
        score = test_parameters(test_terms, param_dict)
        
        if score > best_score:
            best_score = score
            best_params = param_dict
    
    return best_params, best_score
```

### 3. A/B Testing

For comparing filtering variations:

1. Split level 0 terms into two equal groups
2. Apply different filtering strategies to each group
3. Compare metrics between groups
4. Select the better-performing strategy

## Periodic Reviews

Schedule regular review sessions:

1. **Weekly Check**: Quick review of logs and metrics
2. **Monthly Review**: Detailed analysis and parameter tuning
3. **Quarterly Assessment**: Comprehensive evaluation and strategy adjustments

## Conclusion

This monitoring framework provides a structured approach to measuring the performance of the Level 1 Department Filtering Strategy. By consistently tracking these metrics and implementing the feedback mechanisms, we can ensure the filtering strategy continues to improve in both efficiency and effectiveness over time.
