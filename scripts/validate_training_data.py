#!/usr/bin/env python3
"""
Comprehensive validation script for expanded training data quality and coverage.
Analyzes pattern distribution, linguistic correctness, and academic terminology standards.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def categorize_example(example: Dict[str, Any]) -> str:
    """
    Categorize an example as simple or complex based on patterns.
    Returns the specific pattern type if complex.
    """
    input_text = example['input'].lower()
    expected = example['expected']
    
    # Empty expected array (special case)
    if not expected:
        return 'empty_expected'
    
    # Check for complex patterns
    patterns = {
        'shared_noun_distribution': [
            'development, regeneration, and stem cell',
            'east asian languages and civilizations',
            'environment and sustainability',
            'environment and natural resources',
            'earth and climate sciences',
            'earth and ocean sciences'
        ],
        'morphological_transformation': [
            'development, regeneration',
            'biophysics and structural biology',
            'cell and molecular biology',
            'immunobiology and microbiology'
        ],
        'multi_field_parsing': [
            'education, society, and human development',
            'public affairs and policy',
            'public health and health policy',
            'public administration and policy'
        ],
        'gender_identity_studies': [
            'women\'s, gender, and sexuality studies',
            'gender and sexuality studies'
        ],
        'arts_combinations': [
            'media arts and sciences',
            'cinema and media arts'
        ]
    }
    
    for pattern_type, keywords in patterns.items():
        for keyword in keywords:
            if keyword in input_text:
                return pattern_type
    
    # Check if it's a simple extraction (single concept)
    if len(expected) == 1:
        return 'simple_extraction'
    
    # Multiple concepts but not a recognized complex pattern
    return 'multiple_concepts'

def analyze_pattern_coverage(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze pattern coverage and distribution."""
    pattern_counts = Counter()
    for example in data:
        pattern_type = categorize_example(example)
        pattern_counts[pattern_type] += 1
    
    total = len(data)
    complex_patterns = ['shared_noun_distribution', 'morphological_transformation', 
                       'multi_field_parsing', 'gender_identity_studies', 'arts_combinations']
    
    complex_count = sum(pattern_counts[p] for p in complex_patterns)
    simple_count = pattern_counts.get('simple_extraction', 0)
    
    return {
        'total_examples': total,
        'simple_examples': simple_count,
        'complex_examples': complex_count,
        'complex_percentage': (complex_count / total * 100) if total > 0 else 0,
        'pattern_distribution': dict(pattern_counts),
        'target_percentage': 30.0
    }

def validate_linguistic_correctness(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate linguistic correctness of transformations."""
    issues = []
    
    # Common morphological transformations in academic fields
    valid_transformations = {
        'development': 'developmental',
        'regeneration': 'regenerative',
        'computation': 'computational',
        'evolution': 'evolutionary',
        'inflammation': 'inflammatory',
        'infection': 'infectious'
    }
    
    for i, example in enumerate(data):
        input_text = example['input'].lower()
        expected = example['expected']
        
        # Check morphological transformations
        for base, adjective in valid_transformations.items():
            if base in input_text:
                # Check if the transformation is applied correctly in expected
                for concept in expected:
                    if adjective in concept.lower() and 'biology' in concept.lower():
                        # Valid transformation
                        pass
                    elif base in concept.lower() and 'biology' not in concept.lower():
                        # Base form used where adjective might be expected
                        issues.append({
                            'example_index': i,
                            'type': 'morphological',
                            'input': example['input'],
                            'expected': expected,
                            'issue': f"Possible missing transformation: {base} → {adjective}"
                        })
    
    return issues

def validate_json_structure(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate JSON structure and format."""
    issues = []
    seen_inputs = set()
    
    for i, example in enumerate(data):
        # Check required fields
        if 'input' not in example or 'expected' not in example:
            issues.append({
                'example_index': i,
                'type': 'structure',
                'issue': 'Missing required fields (input/expected)'
            })
            continue
        
        # Check for duplicates
        input_text = example['input']
        if input_text in seen_inputs:
            issues.append({
                'example_index': i,
                'type': 'duplicate',
                'input': input_text,
                'issue': 'Duplicate input found'
            })
        seen_inputs.add(input_text)
        
        # Check expected array structure
        if not isinstance(example['expected'], list):
            issues.append({
                'example_index': i,
                'type': 'structure',
                'issue': 'Expected field is not a list'
            })
            continue
        
        # Check each expected concept
        for j, concept in enumerate(example['expected']):
            if not isinstance(concept, str):
                issues.append({
                    'example_index': i,
                    'type': 'structure',
                    'issue': f'Expected concept at index {j} is not a string'
                })
    
    return issues

def check_academic_terminology(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check academic terminology standards."""
    issues = []
    
    # Common academic field suffixes and patterns
    valid_suffixes = ['studies', 'science', 'sciences', 'engineering', 'biology', 
                     'chemistry', 'physics', 'mathematics', 'arts', 'education',
                     'policy', 'affairs', 'administration', 'health', 'medicine']
    
    # Known valid academic fields (partial list for validation)
    known_fields = {
        'computer science', 'electrical engineering', 'mechanical engineering',
        'biology', 'chemistry', 'physics', 'mathematics', 'statistics',
        'economics', 'psychology', 'sociology', 'anthropology', 'political science',
        'history', 'philosophy', 'literature', 'linguistics', 'art history',
        'public health', 'public policy', 'business administration', 'education'
    }
    
    for i, example in enumerate(data):
        for concept in example['expected']:
            concept_lower = concept.lower().strip()
            
            # Check if concept has a valid suffix
            has_valid_suffix = any(concept_lower.endswith(suffix) for suffix in valid_suffixes)
            
            # Check if it's a known field or a reasonable variation
            is_known = concept_lower in known_fields
            
            # Check for reasonable compound fields
            is_compound = any(field in concept_lower for field in known_fields)
            
            if not has_valid_suffix and not is_known and not is_compound:
                issues.append({
                    'example_index': i,
                    'type': 'terminology',
                    'concept': concept,
                    'input': example['input'],
                    'issue': 'Potentially non-standard academic field name'
                })
    
    return issues

def generate_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistical analysis of training data."""
    concept_counts = [len(example['expected']) for example in data]
    
    # Distribution of concept counts
    count_distribution = Counter(concept_counts)
    
    # Average concepts per example
    avg_concepts = sum(concept_counts) / len(concept_counts) if concept_counts else 0
    
    # Find examples with empty expected arrays
    empty_expected = [i for i, example in enumerate(data) if not example['expected']]
    
    # Academic domain distribution (simplified categorization)
    domain_counts = defaultdict(int)
    for example in data:
        input_lower = example['input'].lower()
        if any(term in input_lower for term in ['engineering', 'computer', 'technology']):
            domain_counts['STEM'] += 1
        elif any(term in input_lower for term in ['arts', 'humanities', 'history', 'literature']):
            domain_counts['Humanities'] += 1
        elif any(term in input_lower for term in ['social', 'psychology', 'sociology', 'economics']):
            domain_counts['Social Sciences'] += 1
        elif any(term in input_lower for term in ['medicine', 'health', 'biology', 'chemistry']):
            domain_counts['Life Sciences'] += 1
        elif any(term in input_lower for term in ['business', 'management', 'administration']):
            domain_counts['Business'] += 1
        else:
            domain_counts['Other'] += 1
    
    return {
        'average_concepts_per_example': avg_concepts,
        'concept_count_distribution': dict(count_distribution),
        'empty_expected_indices': empty_expected,
        'domain_distribution': dict(domain_counts),
        'max_concepts': max(concept_counts) if concept_counts else 0,
        'min_concepts': min(concept_counts) if concept_counts else 0
    }

def generate_report(coverage: Dict[str, Any], linguistic_issues: List[Dict[str, Any]],
                   structure_issues: List[Dict[str, Any]], terminology_issues: List[Dict[str, Any]],
                   statistics: Dict[str, Any]) -> str:
    """Generate comprehensive validation report."""
    report = []
    report.append("=" * 80)
    report.append("TRAINING DATA VALIDATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Pattern Coverage Analysis
    report.append("## PATTERN COVERAGE ANALYSIS")
    report.append("-" * 40)
    report.append(f"Total Examples: {coverage['total_examples']}")
    report.append(f"Simple Examples: {coverage['simple_examples']}")
    report.append(f"Complex Examples: {coverage['complex_examples']}")
    report.append(f"Complex Percentage: {coverage['complex_percentage']:.1f}%")
    report.append(f"Target Percentage: {coverage['target_percentage']:.1f}%")
    report.append("")
    report.append("Pattern Distribution:")
    for pattern, count in sorted(coverage['pattern_distribution'].items()):
        report.append(f"  - {pattern}: {count}")
    report.append("")
    
    # Statistical Analysis
    report.append("## STATISTICAL ANALYSIS")
    report.append("-" * 40)
    report.append(f"Average Concepts per Example: {statistics['average_concepts_per_example']:.2f}")
    report.append(f"Max Concepts: {statistics['max_concepts']}")
    report.append(f"Min Concepts: {statistics['min_concepts']}")
    report.append("")
    report.append("Concept Count Distribution:")
    for count, freq in sorted(statistics['concept_count_distribution'].items()):
        report.append(f"  - {count} concept(s): {freq} examples")
    report.append("")
    report.append("Domain Distribution:")
    for domain, count in sorted(statistics['domain_distribution'].items()):
        report.append(f"  - {domain}: {count}")
    if statistics['empty_expected_indices']:
        report.append(f"\nExamples with empty expected arrays: {statistics['empty_expected_indices']}")
    report.append("")
    
    # JSON Structure Validation
    report.append("## JSON STRUCTURE VALIDATION")
    report.append("-" * 40)
    if structure_issues:
        report.append(f"Found {len(structure_issues)} structure issues:")
        for issue in structure_issues[:5]:  # Show first 5
            report.append(f"  - Index {issue['example_index']}: {issue['issue']}")
        if len(structure_issues) > 5:
            report.append(f"  ... and {len(structure_issues) - 5} more")
    else:
        report.append("✓ All examples have valid JSON structure")
    report.append("")
    
    # Linguistic Correctness
    report.append("## LINGUISTIC CORRECTNESS")
    report.append("-" * 40)
    if linguistic_issues:
        report.append(f"Found {len(linguistic_issues)} potential linguistic issues:")
        for issue in linguistic_issues[:5]:  # Show first 5
            report.append(f"  - Index {issue['example_index']}: {issue['issue']}")
        if len(linguistic_issues) > 5:
            report.append(f"  ... and {len(linguistic_issues) - 5} more")
    else:
        report.append("✓ No linguistic issues detected")
    report.append("")
    
    # Academic Terminology
    report.append("## ACADEMIC TERMINOLOGY")
    report.append("-" * 40)
    if terminology_issues:
        report.append(f"Found {len(terminology_issues)} potential terminology issues:")
        for issue in terminology_issues[:5]:  # Show first 5
            report.append(f"  - Index {issue['example_index']}: {issue['concept']} - {issue['issue']}")
        if len(terminology_issues) > 5:
            report.append(f"  ... and {len(terminology_issues) - 5} more")
    else:
        report.append("✓ All academic terminology appears valid")
    report.append("")
    
    # Summary and Recommendations
    report.append("## SUMMARY & RECOMMENDATIONS")
    report.append("-" * 40)
    
    # Overall quality assessment
    total_issues = len(structure_issues) + len(linguistic_issues) + len(terminology_issues)
    quality_score = max(0, 100 - (total_issues * 2))  # Deduct 2 points per issue
    report.append(f"Overall Quality Score: {quality_score}/100")
    report.append("")
    
    # Recommendations
    report.append("Recommendations:")
    if coverage['complex_percentage'] < coverage['target_percentage']:
        deficit = coverage['target_percentage'] - coverage['complex_percentage']
        additional_needed = int((deficit / 100) * coverage['total_examples'])
        report.append(f"  1. Add {additional_needed} more complex pattern examples to reach 30% target")
    else:
        report.append("  1. ✓ Complex pattern coverage meets target")
    
    if linguistic_issues:
        report.append("  2. Review and correct morphological transformations")
    
    if terminology_issues:
        report.append("  3. Validate non-standard academic field names")
    
    if not (linguistic_issues or terminology_issues or structure_issues):
        report.append("  2. Training data quality is excellent - ready for GEPA optimization")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Main validation function."""
    # Load training data
    training_file = Path(__file__).parent.parent / "data" / "prompts_training_data" / "lv0_s1.json"
    
    if not training_file.exists():
        print(f"Error: Training data file not found at {training_file}")
        sys.exit(1)
    
    print(f"Loading training data from {training_file}...")
    data = load_training_data(training_file)
    print(f"Loaded {len(data)} training examples")
    print()
    
    # Run all validations
    print("Running pattern coverage analysis...")
    coverage = analyze_pattern_coverage(data)
    
    print("Validating linguistic correctness...")
    linguistic_issues = validate_linguistic_correctness(data)
    
    print("Validating JSON structure...")
    structure_issues = validate_json_structure(data)
    
    print("Checking academic terminology...")
    terminology_issues = check_academic_terminology(data)
    
    print("Generating statistics...")
    statistics = generate_statistics(data)
    
    print()
    # Generate and print report
    report = generate_report(coverage, linguistic_issues, structure_issues, 
                           terminology_issues, statistics)
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / "training_data_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_file}")
    
    # Return success if quality is acceptable
    quality_score = max(0, 100 - (len(structure_issues) + len(linguistic_issues) + len(terminology_issues)) * 2)
    if quality_score >= 80:
        print("\n✓ Training data meets quality standards for GEPA optimization")
        return 0
    else:
        print(f"\n⚠ Training data quality score ({quality_score}/100) below threshold (80/100)")
        return 1

if __name__ == "__main__":
    sys.exit(main())