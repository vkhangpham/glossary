#!/usr/bin/env python

import os
import json
from flask import Blueprint, render_template, jsonify, request

# Create Blueprint for the evaluator views
evaluator_bp = Blueprint('evaluator', __name__, url_prefix='/evaluator')

# Directories for loaded data
DATA_DIR = 'data'
EVALUATION_DIR = os.path.join(DATA_DIR, 'evaluation')
VISUALIZATION_DIR = os.path.join(EVALUATION_DIR, 'visualizations')

# Files with pre-computed data
METRICS_FILE = os.path.join(EVALUATION_DIR, 'metrics.json')
ISSUES_FILE = os.path.join(EVALUATION_DIR, 'issues.json')
CONNECTIVITY_FILE = os.path.join(EVALUATION_DIR, 'connectivity.json')
SUMMARY_FILE = os.path.join(EVALUATION_DIR, 'summary.json')


def check_evaluation_files():
    """Check if all required evaluation files exist."""
    required_files = [
        METRICS_FILE,
        ISSUES_FILE,
        CONNECTIVITY_FILE,
        SUMMARY_FILE,
        os.path.join(VISUALIZATION_DIR, 'terms_per_level.json'),
        os.path.join(VISUALIZATION_DIR, 'connectivity.json'),
        os.path.join(VISUALIZATION_DIR, 'branching.json'),
        os.path.join(VISUALIZATION_DIR, 'variations.json')
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"WARNING: The following evaluation files are missing:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease run hierarchy_evaluator.py with --save-all flag first:")
        print("python -m generate_glossary.hierarchy_evaluator --save-all --verbose")
        return False
    
    return True


def load_json_file(filepath):
    """Load data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None


@evaluator_bp.route('/metrics')
def get_metrics():
    """API endpoint to get basic hierarchy metrics."""
    metrics = load_json_file(METRICS_FILE)
    if not metrics:
        return jsonify({"error": "Evaluation metrics not available"}), 404
    
    return jsonify(metrics)


@evaluator_bp.route('/issues')
def get_issues():
    """API endpoint to get potential hierarchy issues."""
    issues = load_json_file(ISSUES_FILE)
    if not issues:
        return jsonify({"error": "Evaluation issues not available"}), 404
    
    # Format issues for better display in the frontend
    formatted_issues = {}
    
    for issue_type, issue_list in issues.items():
        formatted_issues[issue_type] = []
        
        for issue in issue_list:
            # Create display-friendly text from issue object
            if issue_type == "orphan_terms":
                description = f"Term '{issue['term']}' (Level {issue['level']}) has no parent terms"
            elif issue_type == "redundant_paths":
                description = f"Term '{issue['term']}' (Level {issue['level']}) has both direct and indirect paths to ancestor '{issue['ancestor']}'"
            elif issue_type == "inconsistent_branching":
                description = f"Term '{issue['term']}' (Level {issue['level']}) has {issue['children_count']} children (avg: {issue['avg_level_children']:.2f})"
            elif issue_type == "unbalanced_subtrees":
                level_info = ', '.join([f"Level {k}: {v}" for k, v in issue.get('children_by_level', {}).items()])
                description = f"Term '{issue['term']}' (Level {issue['level']}) has children across multiple levels: {level_info}"
            else:
                # Fallback for unknown issue types
                description = f"Issue with term '{issue.get('term', 'unknown')}': {issue.get('issue', 'unknown issue')}"
            
            formatted_issues[issue_type].append({
                "id": len(formatted_issues[issue_type]),
                "term": issue.get('term', ''),
                "level": issue.get('level', ''),
                "description": description,
                "details": issue  # Keep original details for reference
            })
    
    return jsonify(formatted_issues)


@evaluator_bp.route('/connectivity')
def get_connectivity():
    """API endpoint to get level connectivity data."""
    connectivity = load_json_file(CONNECTIVITY_FILE)
    if not connectivity:
        return jsonify({"error": "Connectivity data not available"}), 404
    
    return jsonify(connectivity)


@evaluator_bp.route('/summary')
def get_summary():
    """API endpoint to get a summary of hierarchy statistics."""
    summary = load_json_file(SUMMARY_FILE)
    if not summary:
        return jsonify({"error": "Summary data not available"}), 404
    
    return jsonify(summary)


@evaluator_bp.route('/visualization/<vis_type>')
def get_visualization(vis_type):
    """API endpoint to get visualizations of hierarchy metrics."""
    if vis_type not in ['terms_per_level', 'connectivity', 'branching', 'variations']:
        return jsonify({"error": "Unknown visualization type"}), 400
    
    vis_file = os.path.join(VISUALIZATION_DIR, f'{vis_type}.json')
    vis_data = load_json_file(vis_file)
    
    if not vis_data:
        return jsonify({"error": "Requested visualization not available"}), 404
    
    return jsonify(vis_data)


def register_with_app(app):
    """Register the evaluator blueprint with the Flask app."""
    # Check if evaluation files exist
    if not check_evaluation_files():
        print("WARNING: Some evaluation files are missing. The quality dashboard may not function correctly.")
    else:
        print("Evaluation data loaded successfully")
    
    # Register blueprint
    app.register_blueprint(evaluator_bp)
    
    print("Hierarchy evaluator integration registered successfully") 