#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, request, send_from_directory
import time

app = Flask(__name__, template_folder='templates', static_folder='static')

DATA_DIR = 'data'
HIERARCHY_FILE = os.path.join(DATA_DIR, 'hierarchy.json')

# Global variables to store data
hierarchy_data = {}
resources_data = {}


def load_hierarchy() -> Dict[str, Any]:
    """Load hierarchy data from the JSON file."""
    if not os.path.exists(HIERARCHY_FILE):
        print(f"Hierarchy file not found: {HIERARCHY_FILE}")
        return {}
    
    with open(HIERARCHY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_resources(level: int) -> Dict[str, List[Dict[str, Any]]]:
    """Load resources for the specified level."""
    resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
    
    if not os.path.exists(resources_file):
        print(f"Resources file not found: {resources_file}")
        return {}
    
    with open(resources_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def combine_data() -> Dict[str, Any]:
    """Combine hierarchy and resources data for visualization."""
    combined_data = hierarchy_data.copy()
    
    # Add resources to each term
    for term, term_data in combined_data["terms"].items():
        level = term_data["level"]
        if level in resources_data and term in resources_data[level]:
            term_data["resources"] = resources_data[level][term]
        else:
            term_data["resources"] = []
    
    return combined_data


@app.route('/')
def index():
    """Render the main visualization page."""
    return render_template('index.html', timestamp=int(time.time()))


@app.route('/api/hierarchy')
def get_hierarchy():
    """API endpoint to get the combined hierarchy and resources data."""
    return jsonify(combine_data())


@app.route('/api/term/<term>')
def get_term(term):
    """API endpoint to get details for a specific term."""
    combined_data = combine_data()
    if term in combined_data["terms"]:
        return jsonify(combined_data["terms"][term])
    else:
        return jsonify({"error": "Term not found"}), 404


@app.route('/api/search')
def search():
    """API endpoint to search for terms."""
    query = request.args.get('q', '').lower()
    
    if not query:
        return jsonify([])
    
    combined_data = combine_data()
    results = []
    
    # Search in terms
    for term, term_data in combined_data["terms"].items():
        if query in term.lower():
            results.append({
                "term": term,
                "level": term_data["level"],
                "type": "term"
            })
    
    # Search in variations
    for term, variation in combined_data["relationships"]["variations"]:
        if query in variation.lower():
            results.append({
                "term": term,
                "variation": variation,
                "level": combined_data["terms"].get(term, {}).get("level", -1),
                "type": "variation"
            })
    
    return jsonify(results)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Visualize the hierarchy with term resources')
    parser.add_argument('-p', '--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run the server on (default: 127.0.0.1)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Ensure required directories exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    if not os.path.exists(templates_dir) or not os.path.exists(static_dir):
        print(f"Warning: Template or static directories are missing. Please create them before running.")
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
    
    # Load hierarchy data
    global hierarchy_data
    hierarchy_data = load_hierarchy()
    
    if args.verbose:
        print(f"Loaded hierarchy with {hierarchy_data.get('stats', {}).get('total_terms', 0)} terms")
    
    # Load resources data
    global resources_data
    for level in range(3):
        resources_data[level] = load_resources(level)
        if args.verbose:
            terms_with_resources = len(resources_data[level])
            print(f"Loaded {terms_with_resources} terms with resources for level {level}")
    
    # Run the Flask app
    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.verbose)


if __name__ == '__main__':
    main() 