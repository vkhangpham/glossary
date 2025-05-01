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
from collections import Counter

app = Flask(__name__, template_folder='templates', static_folder='static')

# Get the application root directory (one level up from the script location)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
FINAL_DIR = os.path.join(DATA_DIR, 'final')
HIERARCHY_FILE = os.path.join(FINAL_DIR, 'hierarchy.json')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')
MANUAL_EVALUATION_FILE = os.path.join(DATA_DIR, 'manual_evaluations.json')

# Print paths for debugging
print(f"Root directory: {ROOT_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Final directory: {FINAL_DIR}")
print(f"Analysis directory: {ANALYSIS_DIR}")

# Global variables to store data
hierarchy_data = {}
resources_data = {}
duplicate_data = {}
manual_evaluations = {}


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


@app.route('/manual_evaluation')
def manual_evaluation_page():
    """Render the manual evaluation page."""
    return render_template('manual_evaluation.html', timestamp=int(time.time()))


@app.route('/evaluation_stats')
def evaluation_stats_page():
    """Render the evaluation statistics page."""
    return render_template('evaluation_stats.html', timestamp=int(time.time()))


@app.route('/api/load_evaluation/<term>')
def load_evaluation(term: str):
    """API endpoint to load manual evaluation for a specific term."""
    evaluation = manual_evaluations.get(term)
    if evaluation:
        return jsonify({"success": True, "evaluation": evaluation})
    else:
        return jsonify({"success": False, "message": "No evaluation found"})


@app.route('/api/save_evaluation', methods=['POST'])
def save_evaluation():
    """API endpoint to save manual evaluation for a specific term."""
    data = request.json
    term = data.get('term')
    evaluation_data = data.get('evaluation')

    if not term or not evaluation_data:
        return jsonify({"success": False, "message": "Missing term or evaluation data"}), 400

    # Store the evaluation
    manual_evaluations[term] = evaluation_data
    save_manual_evaluations() # Persist to file

    return jsonify({"success": True, "message": "Evaluation saved successfully"})


@app.route('/api/evaluation_stats')
def get_evaluation_stats():
    """API endpoint to calculate and return aggregate evaluation statistics."""
    if not manual_evaluations:
        return jsonify({
            "total_evaluated": 0,
            "importance_counts": {},
            "level_correctness_counts": {},
            "variation_correctness_avg": 0,
            "parent_correctness_avg": 0
        })

    total_evaluated = len(manual_evaluations)
    importance_counts = Counter()
    level_correctness_counts = Counter()
    level_correctness_aggregated_counts = Counter()
    total_variations_evaluated = 0
    correct_variations = 0
    total_parents_evaluated = 0
    correct_parents = 0

    for term, evaluation in manual_evaluations.items():
        if evaluation.get('academic_importance'):
            importance_counts[evaluation['academic_importance']] += 1

        # Aggregate Level Correctness
        level_decision = evaluation.get('level_correctness')
        if level_decision:
            if level_decision == 'correct':
                level_correctness_aggregated_counts['Correct'] += 1
            elif level_decision in ['0', '1', '2', '3']:
                actual_level = hierarchy_data.get('terms', {}).get(term, {}).get('level')
                if actual_level is not None:
                    try:
                        suggested_level = int(level_decision)
                        actual_level = int(actual_level)
                        if suggested_level > actual_level:
                            level_correctness_aggregated_counts['Higher'] += 1
                        elif suggested_level < actual_level:
                            level_correctness_aggregated_counts['Lower'] += 1
                        else:
                            # Suggested level is same as actual - treat as Correct
                            level_correctness_aggregated_counts['Correct'] += 1
                    except (ValueError, TypeError):
                        print(f"Warning: Could not compare levels for term '{term}'. Actual: {actual_level}, Suggested: {level_decision}")
                        # Decide how to handle? Maybe add an 'Error' category or ignore?
                        # For now, ignore entries where levels aren't comparable integers.
                else:
                    print(f"Warning: Could not find actual level for term '{term}' in hierarchy data.")
            # else: ignore other potential values

        if 'variation_correctness' in evaluation and isinstance(evaluation['variation_correctness'], dict):
            num_variations = len(evaluation['variation_correctness'])
            if num_variations > 0:
                total_variations_evaluated += num_variations
                correct_variations += sum(1 for v in evaluation['variation_correctness'].values() if v)

        if 'parent_relationships' in evaluation and isinstance(evaluation['parent_relationships'], dict):
            num_parents = len(evaluation['parent_relationships'])
            if num_parents > 0:
                total_parents_evaluated += num_parents
                correct_parents += sum(1 for p in evaluation['parent_relationships'].values() if p)

    variation_correctness_avg = (correct_variations / total_variations_evaluated * 100) if total_variations_evaluated > 0 else 0
    parent_correctness_avg = (correct_parents / total_parents_evaluated * 100) if total_parents_evaluated > 0 else 0

    stats = {
        "total_evaluated": total_evaluated,
        "importance_counts": dict(importance_counts),
        "level_correctness_counts": dict(level_correctness_aggregated_counts),
        "variation_correctness_avg": round(variation_correctness_avg, 2),
        "parent_correctness_avg": round(parent_correctness_avg, 2),
        "total_variations_rated": total_variations_evaluated,
        "correct_variations_count": correct_variations,
        "total_parents_rated": total_parents_evaluated,
        "correct_parents_count": correct_parents
    }

    return jsonify(stats)


@app.route('/api/all_evaluations')
def get_all_evaluated_terms():
    """API endpoint to get the full dictionary of manual evaluations."""
    # Return the whole dictionary, not just the keys
    return jsonify(manual_evaluations)


def load_manual_evaluations() -> Dict[str, Any]:
    """Load manual evaluations from the JSON file."""
    if not os.path.exists(MANUAL_EVALUATION_FILE):
        return {}
    try:
        with open(MANUAL_EVALUATION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {MANUAL_EVALUATION_FILE}. Returning empty evaluations.")
        return {}


def save_manual_evaluations():
    """Save manual evaluations to the JSON file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(MANUAL_EVALUATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(manual_evaluations, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving manual evaluations to {MANUAL_EVALUATION_FILE}: {e}")


def main():
    """Main function for command-line execution."""
    global hierarchy_data, resources_data, manual_evaluations
    
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
        
    # Load existing manual evaluations
    print("Loading manual evaluations...")
    manual_evaluations = load_manual_evaluations()
    print(f"Loaded {len(manual_evaluations)} existing manual evaluations.")
    
    # Register the evaluator blueprint (Commented out as it's part of the old system)
    # try:
    #     from .hierarchy_evaluator_viz import register_with_app
    #     register_with_app(app)
    #     print("Hierarchy evaluator integration registered successfully")
    # except ImportError:
    #     print("Hierarchy evaluator integration not available or disabled.")
    
    # Run the Flask app
    print(f"Starting server on port {args.port}...")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main() 