import argparse
import json
import os
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, request
import time
import re

app = Flask(__name__, template_folder='templates', static_folder='static')

# Get the application root directory (one level up from the script location)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FINAL_DIR = os.path.join(DATA_DIR, 'final')
HIERARCHY_FILE = os.path.join(DATA_DIR, 'hierarchy.json')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')

# Print paths for debugging
print(f"Root directory: {ROOT_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Final directory: {FINAL_DIR}")
print(f"Analysis directory: {ANALYSIS_DIR}")

# Global variables to store data
hierarchy_data = {}
resources_data = {}
duplicate_data = {}


def load_hierarchy() -> Dict[str, Any]:
    """Load hierarchy data from the JSON file."""
    if not os.path.exists(HIERARCHY_FILE):
        print(f"Hierarchy file not found: {HIERARCHY_FILE}")
        return {}
    
    with open(HIERARCHY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_resources(level: int) -> Dict[str, List[Dict[str, Any]]]:
    """Load resources for the specified level.
    
    Prioritizes data from the final directory, then falls back to original locations.
    """
    # First try final directory resources
    final_resources_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
    
    # Then try filtered resources in original location
    filtered_resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
    
    # Then fall back to full resources in original location
    resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_resources.json')
    
    # Try each location in order
    if os.path.exists(final_resources_file):
        print(f"Loading resources for level {level} from final directory")
        with open(final_resources_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif os.path.exists(filtered_resources_file):
        print(f"Loading resources for level {level} from filtered resources")
        with open(filtered_resources_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif os.path.exists(resources_file):
        print(f"Loading resources for level {level} from original resources")
        with open(resources_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    print(f"Resources file not found for level {level}")
    return {}


def load_duplicate_analysis(level: int) -> Dict[str, Any]:
    """Load duplicate analysis data for the specified level."""
    duplicates_file = os.path.join(ANALYSIS_DIR, f'lv{level}', f'lv{level}_potential_duplicates.json')
    
    if not os.path.exists(duplicates_file):
        print(f"Duplicate analysis file not found: {duplicates_file}")
        return {}
    
    with open(duplicates_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_source_similarity(term1: str, term2: str, terms_data: Dict[str, Dict[str, Any]]) -> int:
    """Calculate the number of shared sources between two terms."""
    if term1 not in terms_data or term2 not in terms_data:
        return 0
    
    sources1 = set(terms_data[term1].get("sources", []))
    sources2 = set(terms_data[term2].get("sources", []))
    
    return len(sources1.intersection(sources2))


def rank_siblings(term: str, siblings: List[str], terms_data: Dict[str, Dict[str, Any]]) -> List[str]:
    """Rank siblings based on number of shared sources with the given term."""
    if not siblings:
        return []
        
    # Calculate similarity scores for each sibling
    similarity_scores = [(sibling, calculate_source_similarity(term, sibling, terms_data)) 
                         for sibling in siblings]
    
    # Sort siblings by similarity score (descending)
    sorted_siblings = [s[0] for s in sorted(similarity_scores, key=lambda x: x[1], reverse=True)]
    
    return sorted_siblings


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


@app.route('/duplicates')
def duplicates():
    """Render the duplicate analysis visualization page."""
    level = request.args.get('level', '2')
    try:
        level = int(level)
    except ValueError:
        level = 2  # Default to level 2
        
    return render_template('duplicates.html', level=level, timestamp=int(time.time()))


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


@app.route('/api/siblings/<term>')
def get_siblings(term):
    """API endpoint to get ranked siblings for a specific term."""
    combined_data = combine_data()
    
    if term not in combined_data["terms"]:
        return jsonify({"error": "Term not found"}), 404
    
    term_data = combined_data["terms"][term]
    parent_terms = term_data.get("parents", [])
    
    # Find all siblings (terms that share at least one parent with this term)
    siblings = []
    for parent in parent_terms:
        if parent in combined_data["terms"]:
            siblings.extend(combined_data["terms"][parent].get("children", []))
    
    # Remove duplicates and the term itself
    siblings = list(set(siblings))
    if term in siblings:
        siblings.remove(term)
    
    # Rank siblings by source similarity
    ranked_siblings = rank_siblings(term, siblings, combined_data["terms"])
    
    # Limit the number of siblings based on the configuration
    max_siblings = app.config.get('MAX_SIBLINGS', 10)
    ranked_siblings = ranked_siblings[:max_siblings]
    
    # Add similarity score for frontend display
    result = []
    for sibling in ranked_siblings:
        similarity = calculate_source_similarity(term, sibling, combined_data["terms"])
        result.append({
            "term": sibling,
            "similarity_score": similarity,
            "level": combined_data["terms"][sibling]["level"]
        })
    
    return jsonify(result)


@app.route('/api/duplicates')
def get_duplicates():
    """API endpoint to get duplicate analysis data."""
    level = request.args.get('level', '2')
    try:
        level = int(level)
    except ValueError:
        level = 2  # Default to level 2
    
    # Load duplicate analysis data if not already loaded
    if level not in duplicate_data:
        duplicate_data[level] = load_duplicate_analysis(level)
    
    # Group duplicates by parent category
    parent_groups = {}
    for key, data in duplicate_data[level].items():
        parent = data.get("parent", "Unknown")
        if parent not in parent_groups:
            parent_groups[parent] = []
        parent_groups[parent].append(data)
    
    result = {
        "level": level,
        "total_duplicates": len(duplicate_data[level]),
        "parent_groups": parent_groups
    }
    
    return jsonify(result)


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
                "level": combined_data["terms"][term]["level"] if term in combined_data["terms"] else -1,
                "type": "variation"
            })
    
    # Sort by relevance (exact match first, then by level)
    results.sort(key=lambda x: (0 if x.get("term", "").lower() == query or x.get("variation", "").lower() == query else 1, 
                              x.get("level", 999)))
    
    # Limit to 50 results
    return jsonify(results[:50])


@app.route('/quality')
def quality_dashboard():
    """Render the hierarchy quality evaluation dashboard."""
    return render_template('evaluator.html')


@app.route('/api/find_latest_evaluation')
def find_latest_evaluation():
    """Find the most recent evaluation file for a given level."""
    level = request.args.get('level', '2')
    try:
        level = int(level)
    except ValueError:
        level = 2  # Default to level 2
    
    # Define potential locations to search for evaluation files
    search_locations = [
        os.path.join(DATA_DIR, 'evaluation'),
        os.path.join(DATA_DIR, 'analysis', f'lv{level}'),
        os.path.join(DATA_DIR, 'analysis'),
        os.path.join(ROOT_DIR, 'exports')
    ]
    
    # Initialize variables to track the latest file
    latest_file = None
    latest_timestamp = None
    
    # Search pattern for evaluation files
    json_pattern = f"duplicate_evaluations_lv{level}_*.json"
    csv_pattern = f"duplicate_evaluations_lv{level}_*.csv"
    
    # Prioritize JSON files over CSV (since they contain more complete data)
    for location in search_locations:
        if os.path.exists(location):
            # First search for JSON files
            json_files = glob.glob(os.path.join(location, json_pattern))
            for file_path in json_files:
                # Extract date from filename (expected format: duplicate_evaluations_lvX_YYYY-MM-DD.json)
                match = re.search(r'_(\d{4}-\d{2}-\d{2})\.json$', file_path)
                if match:
                    file_date = match.group(1)
                    if latest_timestamp is None or file_date > latest_timestamp:
                        latest_timestamp = file_date
                        latest_file = file_path
                else:
                    # If no date in filename, use file modification time
                    file_mtime = os.path.getmtime(file_path)
                    if latest_timestamp is None or file_mtime > latest_timestamp:
                        latest_timestamp = file_mtime
                        latest_file = file_path
            
            # If no JSON files found, look for CSV files
            if latest_file is None:
                csv_files = glob.glob(os.path.join(location, csv_pattern))
                for file_path in csv_files:
                    match = re.search(r'_(\d{4}-\d{2}-\d{2})\.csv$', file_path)
                    if match:
                        file_date = match.group(1)
                        if latest_timestamp is None or file_date > latest_timestamp:
                            latest_timestamp = file_date
                            latest_file = file_path
                    else:
                        file_mtime = os.path.getmtime(file_path)
                        if latest_timestamp is None or file_mtime > latest_timestamp:
                            latest_timestamp = file_mtime
                            latest_file = file_path
    
    if latest_file:
        # Convert file path to URL path for frontend
        file_url = f"/static/evaluations/{os.path.basename(latest_file)}"
        
        # If the file is not in static directory, copy it there so it can be served
        static_evaluations_dir = os.path.join(app.static_folder, 'evaluations')
        os.makedirs(static_evaluations_dir, exist_ok=True)
        
        static_file_path = os.path.join(static_evaluations_dir, os.path.basename(latest_file))
        
        # Only copy if the file doesn't already exist in static directory
        if not os.path.exists(static_file_path) or os.path.getmtime(latest_file) > os.path.getmtime(static_file_path):
            with open(latest_file, 'rb') as src_file:
                with open(static_file_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
        
        return jsonify({
            "success": True,
            "filePath": file_url,
            "level": level,
            "timestamp": latest_timestamp
        })
    else:
        return jsonify({
            "success": False,
            "message": f"No evaluation files found for level {level}",
            "level": level
        })


@app.route('/api/save_evaluation_export', methods=['POST'])
def save_evaluation_export():
    """Save an evaluation export file to the server."""
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "message": "No file part in the request"
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "success": False,
            "message": "No file selected"
        }), 400
    
    # Define the directory where exports will be saved
    exports_dir = os.path.join(ROOT_DIR, 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    
    # Also save a copy to the static directory for direct access
    static_evaluations_dir = os.path.join(app.static_folder, 'evaluations')
    os.makedirs(static_evaluations_dir, exist_ok=True)
    
    # Save the file to both locations
    file_path = os.path.join(exports_dir, file.filename)
    file.save(file_path)
    
    # Create a copy in the static directory
    static_file_path = os.path.join(static_evaluations_dir, file.filename)
    shutil.copy2(file_path, static_file_path)
    
    return jsonify({
        "success": True,
        "message": "File saved successfully",
        "filePath": file_path,
        "fileUrl": f"/static/evaluations/{file.filename}"
    })


def main():
    """Main function for command-line execution."""
    global hierarchy_data, resources_data
    
    parser = argparse.ArgumentParser(description='Visualize the academic hierarchy')
    parser.add_argument('-p', '--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Load hierarchy data
    print("Loading hierarchy data...")
    hierarchy_data = load_hierarchy()
    
    if not hierarchy_data:
        print("Error: Failed to load hierarchy data. Exiting.")
        return
    
    # Load resources data for each level
    print("Loading resources data...")
    for level in range(4):  # Levels 0, 1, 2, 3
        resources_data[level] = load_resources(level)
        
    # Register the evaluator blueprint
    try:
        from .hierarchy_evaluator_viz import register_with_app
        register_with_app(app)
        print("Hierarchy evaluator integration registered successfully")
    except ImportError:
        print("Hierarchy evaluator not available. Quality analysis functions will be disabled.")
    
    # Run the Flask app
    print(f"Starting server on port {args.port}...")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main() 