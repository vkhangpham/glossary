#!/usr/bin/env python

import argparse
import os
from .hierarchy_evaluator import (
    load_hierarchy, 
    generate_html_report, 
    generate_level_summary,
    calculate_terms_per_level,
    find_orphan_terms,
    calculate_branching_factor,
    check_variation_consolidation,
    analyze_level_connectivity,
    detect_hierarchy_issues,
    visualize_hierarchy_metrics
)

DATA_DIR = 'data'
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')


def generate_quick_report(hierarchy, output_dir=REPORTS_DIR):
    """Generate a quick text report of hierarchy statistics."""
    terms_per_level = calculate_terms_per_level(hierarchy)
    orphans = find_orphan_terms(hierarchy)
    total_orphans = sum(len(terms) for terms in orphans.values())
    branching = calculate_branching_factor(hierarchy)
    variation_stats = check_variation_consolidation(hierarchy)
    issues = detect_hierarchy_issues(hierarchy)
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    print("\n" + "=" * 60)
    print("ACADEMIC HIERARCHY QUALITY REPORT")
    print("=" * 60)
    
    print("\nSUMMARY STATISTICS:")
    print(f"Total Terms: {hierarchy['stats']['total_terms']}")
    print(f"Total Relationships: {hierarchy['stats']['total_relationships']}")
    print(f"Total Variations: {hierarchy['stats']['total_variations']}")
    
    print("\nTERMS PER LEVEL:")
    for level, count in terms_per_level.items():
        print(f"  Level {level}: {count} terms")
    
    print("\nHIERARCHY STRUCTURE:")
    print(f"Orphaned Terms: {total_orphans}")
    for level, terms in orphans.items():
        if terms:
            print(f"  Level {level}: {len(terms)} orphans")
    
    print("\nBRANCHING FACTORS:")
    for level, factor in branching.items():
        print(f"  Level {level}: {factor:.2f} children per term (avg)")
    
    print("\nVARIATION STATS:")
    for level in sorted(variation_stats["terms_with_variations"].keys()):
        terms_with_var = variation_stats["terms_with_variations"].get(level, 0)
        avg_var = variation_stats["avg_variations_per_term"].get(level, 0)
        print(f"  Level {level}: {terms_with_var} terms with variations, {avg_var:.2f} variations per term (avg)")
    
    print("\nISSUES DETECTED:")
    if total_issues > 0:
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"  {issue_type.replace('_', ' ').title()}: {len(issue_list)} issues")
    else:
        print("  No significant issues detected")
    
    print("\nRECOMMENDATIONS:")
    recommendations = []
    
    # Check for high orphan count
    if total_orphans > hierarchy['stats']['total_terms'] * 0.1:
        recommendations.append("- High number of orphaned terms detected. Consider running the metadata collector with term promotion enabled.")
    
    # Check for imbalanced levels
    level_counts = {int(k): v for k, v in terms_per_level.items()}
    if any(level_counts.get(i, 0) < level_counts.get(i+1, 0) for i in range(3)):
        recommendations.append("- Imbalanced hierarchy detected. Lower levels should have more terms than higher levels.")
    
    # Check for connectivity issues
    connectivity = analyze_level_connectivity(hierarchy)
    for key, data in connectivity.items():
        from_level, to_level = key.split('_to_')
        if data['perc_to_connected'] < 80:
            recommendations.append(f"- Poor connectivity from Level {from_level} to Level {to_level}. {data['perc_to_connected']:.1f}% of Level {to_level} terms have parents.")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("  No specific recommendations at this time.")
    
    print("\nFor detailed analysis including visualizations, use the --report option.")
    print("=" * 60)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Evaluate the quality of the academic hierarchy')
    parser.add_argument('-i', '--input', type=str, default='data/hierarchy.json',
                        help='Input hierarchy file (default: data/hierarchy.json)')
    parser.add_argument('-r', '--report', action='store_true',
                        help='Generate a detailed HTML report with visualizations')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for report (default: data/reports)')
    parser.add_argument('-q', '--quick', action='store_true',
                        help='Print a quick summary to the console without generating a report')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set default output if not specified
    if args.output is None:
        args.output = REPORTS_DIR
    
    if args.verbose:
        print(f"Loading hierarchy from {args.input}")
    
    # Load hierarchy data
    hierarchy = load_hierarchy(args.input)
    
    if not hierarchy:
        print("Error: Failed to load hierarchy data")
        return
    
    # Generate report based on options
    if args.quick or not args.report:
        generate_quick_report(hierarchy, args.output)
    
    if args.report:
        if args.verbose:
            print("Generating detailed HTML report...")
        
        output_file = os.path.join(args.output, 'hierarchy_quality_report.html')
        report_file = generate_html_report(hierarchy, output_file)
        
        print(f"\nHierarchy quality report generated: {report_file}")
        
        # Generate visualizations separately to enable more viewing options
        if args.verbose:
            print("Generating additional visualizations...")
        
        visualize_hierarchy_metrics(hierarchy, args.output)


if __name__ == '__main__':
    main() 