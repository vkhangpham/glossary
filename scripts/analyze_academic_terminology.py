#!/usr/bin/env python3
"""Analyze academic terminology quality in training data.

This script cross-references expected concepts against standard academic terminology
and flags non-standard formations and inconsistencies.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Set, List, Dict, Tuple


# Curated list of standard academic terminology patterns and fields
STANDARD_ACADEMIC_FIELDS = {
    # Sciences
    "biology", "chemistry", "physics", "mathematics", "statistics",
    "computer science", "data science", "environmental science",
    "earth science", "astronomy", "geology", "geography",
    
    # Engineering
    "engineering", "mechanical engineering", "electrical engineering",
    "civil engineering", "chemical engineering", "biomedical engineering",
    "software engineering", "aerospace engineering", "industrial engineering",
    
    # Medicine and Health
    "medicine", "nursing", "public health", "pharmacy", "dentistry",
    "veterinary medicine", "health sciences", "clinical sciences",
    "biomedical sciences", "neuroscience", "genetics",
    
    # Social Sciences
    "psychology", "sociology", "anthropology", "political science",
    "economics", "international relations", "social work", "criminology",
    
    # Humanities
    "history", "philosophy", "literature", "linguistics", "languages",
    "english", "classics", "religious studies", "theology", "ethics",
    
    # Arts
    "art", "music", "theater", "dance", "film", "media studies",
    "visual arts", "performing arts", "fine arts", "design",
    
    # Business
    "business", "management", "marketing", "finance", "accounting",
    "entrepreneurship", "business administration", "economics",
    "international business", "operations management",
    
    # Education
    "education", "teaching", "curriculum", "educational psychology",
    "special education", "early childhood education",
    
    # Professional Fields
    "law", "architecture", "urban planning", "journalism",
    "communications", "information science", "library science",
    
    # Interdisciplinary
    "bioinformatics", "computational biology", "digital humanities",
    "cognitive science", "environmental studies", "gender studies",
    "ethnic studies", "area studies", "development studies"
}

# Common academic term patterns
VALID_PATTERNS = [
    r"^\w+ology$",  # biology, psychology, etc.
    r"^\w+ics$",    # physics, mathematics, economics, etc.
    r"^\w+istry$",  # chemistry, dentistry, etc.
    r"^\w+ing$",    # engineering, nursing, teaching, etc.
    r"^\w+ment$",   # management, development, etc.
    r"^\w+tion$",   # education, communication, etc.
    r"^\w+ence$",   # science, etc.
    r"^\w+ure$",    # agriculture, architecture, etc.
]

# Terms that look academic but might be incorrectly extracted
SUSPICIOUS_TERMS = {
    "school", "college", "university", "institute", "center", "department",
    "program", "division", "faculty", "academy", "institution",
    "studies", "research", "advanced", "graduate", "undergraduate",
    "professional", "academic", "interdisciplinary", "international"
}


def normalize_term(term: str) -> str:
    """Normalize a term for comparison."""
    return term.lower().strip()


def is_standard_field(term: str) -> bool:
    """Check if term matches a standard academic field."""
    normalized = normalize_term(term)
    return normalized in STANDARD_ACADEMIC_FIELDS


def matches_valid_pattern(term: str) -> bool:
    """Check if term matches common academic patterns."""
    normalized = normalize_term(term)
    return any(re.match(pattern, normalized) for pattern in VALID_PATTERNS)


def is_suspicious(term: str) -> bool:
    """Check if term might be incorrectly extracted."""
    words = normalize_term(term).split()
    return len(words) == 1 and words[0] in SUSPICIOUS_TERMS


def analyze_concept_quality(concept: str) -> Dict[str, any]:
    """Analyze a single concept for quality issues."""
    analysis = {
        "concept": concept,
        "is_standard": is_standard_field(concept),
        "matches_pattern": matches_valid_pattern(concept),
        "is_suspicious": is_suspicious(concept),
        "issues": []
    }
    
    # Check for various issues
    if not concept.strip():
        analysis["issues"].append("empty")
    
    if concept.isupper():
        analysis["issues"].append("all_caps")
    
    if any(char.isdigit() for char in concept):
        analysis["issues"].append("contains_numbers")
    
    if len(concept) < 3:
        analysis["issues"].append("too_short")
    
    if len(concept) > 50:
        analysis["issues"].append("too_long")
    
    # Check for non-academic patterns
    if "of" in concept.lower() or "and" in concept.lower():
        if not is_standard_field(concept):
            analysis["issues"].append("possible_phrase_not_field")
    
    # Flag if not standard and doesn't match patterns
    if not analysis["is_standard"] and not analysis["matches_pattern"]:
        if not analysis["is_suspicious"]:  # Suspicious terms are expected to not match
            analysis["issues"].append("non_standard_formation")
    
    return analysis


def find_similar_concepts(concepts: List[str], threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """Find concepts that are similar but not identical."""
    similar_pairs = []
    normalized = [(c, normalize_term(c)) for c in concepts]
    
    for i, (c1, n1) in enumerate(normalized):
        for c2, n2 in normalized[i+1:]:
            if n1 != n2:
                # Calculate simple similarity
                longer = max(len(n1), len(n2))
                if longer > 0:
                    # Check for substring relationship
                    if n1 in n2 or n2 in n1:
                        similarity = min(len(n1), len(n2)) / longer
                        if similarity >= threshold:
                            similar_pairs.append((c1, c2, similarity))
    
    return similar_pairs


def analyze_terminology(file_path: Path) -> Dict:
    """Analyze terminology quality in training data."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    all_concepts = []
    concept_sources = defaultdict(list)
    
    # Collect all concepts
    for item in data:
        source = item["input"]
        concepts = item["expected"]
        for concept in concepts:
            all_concepts.append(concept)
            concept_sources[concept].append(source)
    
    # Analyze each unique concept
    unique_concepts = list(set(all_concepts))
    concept_analyses = [analyze_concept_quality(c) for c in unique_concepts]
    
    # Categorize results
    results = {
        "total_concepts": len(all_concepts),
        "unique_concepts": len(unique_concepts),
        "standard_fields": sum(1 for a in concept_analyses if a["is_standard"]),
        "pattern_matches": sum(1 for a in concept_analyses if a["matches_pattern"]),
        "suspicious_terms": sum(1 for a in concept_analyses if a["is_suspicious"]),
        "non_standard": [],
        "issues_by_type": defaultdict(list),
        "similar_concepts": find_similar_concepts(unique_concepts),
        "frequency_analysis": Counter(all_concepts).most_common(20)
    }
    
    # Collect non-standard and issues
    for analysis in concept_analyses:
        if not analysis["is_standard"] and not analysis["matches_pattern"]:
            if not analysis["is_suspicious"]:
                results["non_standard"].append(analysis["concept"])
        
        for issue in analysis["issues"]:
            results["issues_by_type"][issue].append(analysis["concept"])
    
    return results


def generate_report(results: Dict) -> str:
    """Generate human-readable report."""
    report = []
    report.append("=" * 60)
    report.append("ACADEMIC TERMINOLOGY ANALYSIS")
    report.append("=" * 60)
    report.append("")
    
    # Overview
    report.append("OVERVIEW:")
    report.append(f"  Total concepts: {results['total_concepts']}")
    report.append(f"  Unique concepts: {results['unique_concepts']}")
    report.append(f"  Duplication rate: {1 - results['unique_concepts']/results['total_concepts']:.1%}")
    report.append("")
    
    # Quality Metrics
    report.append("TERMINOLOGY QUALITY:")
    pct_standard = results['standard_fields'] / results['unique_concepts'] * 100
    pct_pattern = results['pattern_matches'] / results['unique_concepts'] * 100
    pct_suspicious = results['suspicious_terms'] / results['unique_concepts'] * 100
    
    report.append(f"  Standard academic fields: {results['standard_fields']} ({pct_standard:.1f}%)")
    report.append(f"  Match valid patterns: {results['pattern_matches']} ({pct_pattern:.1f}%)")
    report.append(f"  Suspicious terms: {results['suspicious_terms']} ({pct_suspicious:.1f}%)")
    report.append("")
    
    # Non-standard formations
    if results['non_standard']:
        report.append("NON-STANDARD FORMATIONS:")
        report.append(f"  Found {len(results['non_standard'])} non-standard terms:")
        for term in results['non_standard'][:10]:
            report.append(f"    - {term}")
        if len(results['non_standard']) > 10:
            report.append(f"    ... and {len(results['non_standard']) - 10} more")
        report.append("")
    
    # Issues by type
    if results['issues_by_type']:
        report.append("TERMINOLOGY ISSUES:")
        for issue_type, terms in results['issues_by_type'].items():
            report.append(f"  {issue_type}: {len(terms)} terms")
            for term in terms[:3]:
                report.append(f"    - {term}")
            if len(terms) > 3:
                report.append(f"    ... and {len(terms) - 3} more")
        report.append("")
    
    # Similar concepts
    if results['similar_concepts']:
        report.append("SIMILAR CONCEPTS (potential duplicates):")
        for c1, c2, sim in results['similar_concepts'][:5]:
            report.append(f"  - '{c1}' ~ '{c2}' (similarity: {sim:.1%})")
        if len(results['similar_concepts']) > 5:
            report.append(f"  ... and {len(results['similar_concepts']) - 5} more pairs")
        report.append("")
    
    # Most frequent concepts
    report.append("TOP 10 MOST FREQUENT CONCEPTS:")
    for concept, freq in results['frequency_analysis'][:10]:
        report.append(f"  {concept:30s}: {freq:3d} occurrences")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS:")
    
    if pct_standard < 50:
        report.append("  ⚠️  Low percentage of standard academic fields. Review extraction logic.")
    
    if results['suspicious_terms'] > results['unique_concepts'] * 0.1:
        report.append("  ⚠️  High number of suspicious terms. May be extracting institution types instead of fields.")
    
    if results['non_standard']:
        report.append(f"  ⚠️  Found {len(results['non_standard'])} non-standard terms. Review for consistency.")
    
    if results['similar_concepts']:
        report.append(f"  ⚠️  Found {len(results['similar_concepts'])} similar concept pairs. Consider deduplication.")
    
    if not any([pct_standard < 50, results['suspicious_terms'] > results['unique_concepts'] * 0.1,
               results['non_standard'], results['similar_concepts']]):
        report.append("  ✓ Terminology quality appears good overall.")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze academic terminology quality")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/prompts_training_data/lv0_s1.json"),
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for report (default: stdout)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw analysis as JSON"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if quality issues found"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Training data file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    try:
        results = analyze_terminology(args.input)
        
        if args.json:
            # Convert for JSON serialization
            json_results = {
                **results,
                "similar_concepts": [
                    {"concept1": c1, "concept2": c2, "similarity": sim}
                    for c1, c2, sim in results["similar_concepts"]
                ],
                "issues_by_type": dict(results["issues_by_type"])
            }
            output = json.dumps(json_results, indent=2)
        else:
            output = generate_report(results)
        
        if args.output:
            args.output.write_text(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)
        
        # Exit with error if strict mode and issues found
        if args.strict:
            has_issues = (
                results['non_standard'] or
                results['suspicious_terms'] > results['unique_concepts'] * 0.1 or
                results['standard_fields'] / results['unique_concepts'] < 0.5
            )
            if has_issues:
                sys.exit(1)
        
    except Exception as e:
        print(f"Error analyzing terminology: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()