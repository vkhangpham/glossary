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
import functools

app = Flask(__name__, template_folder='templates', static_folder='static')

# Get the application root directory (correctly finding glossary in the path)
GLOSSARY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(GLOSSARY_DIR)
DATA_DIR = os.path.join(GLOSSARY_DIR, 'data')
FINAL_DIR = os.path.join(DATA_DIR, 'final')
HIERARCHY_FILE = os.path.join(FINAL_DIR, 'hierarchy.json')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')
MANUAL_EVALUATION_FILE = os.path.join(DATA_DIR, 'manual_evaluations.json')

# Print paths for debugging
print(f"Glossary directory: {GLOSSARY_DIR}")
print(f"Root directory: {ROOT_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Final directory: {FINAL_DIR}")
print(f"Hierarchy file: {HIERARCHY_FILE}")
print(f"Analysis directory: {ANALYSIS_DIR}")

# Global variables to store data
hierarchy_data = {}
resources_data = {}
duplicate_data = {}
manual_evaluations = {}

# Add cache decorator for expensive functions
def cache_result(func):
    """Cache decorator for expensive functions"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return wrapper


@cache_result
def load_hierarchy() -> Dict[str, Any]:
    """Load hierarchy data from the JSON file."""
    if not os.path.exists(HIERARCHY_FILE):
        print(f"Hierarchy file not found: {HIERARCHY_FILE}")
        return {}
    
    try:
        with open(HIERARCHY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading hierarchy file: {e}")
        return {}


@cache_result
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
    try:
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
    except Exception as e:
        print(f"Error loading resources for level {level}: {e}")
    
    print(f"Resources file not found for level {level}")
    return {}


@cache_result
def load_duplicate_analysis(level: int) -> Dict[str, Any]:
    """Load duplicate analysis data for the specified level."""
    duplicates_file = os.path.join(ANALYSIS_DIR, f'lv{level}', f'lv{level}_potential_duplicates.json')
    
    if not os.path.exists(duplicates_file):
        print(f"Duplicate analysis file not found: {duplicates_file}")
        return {}
    
    try:
        with open(duplicates_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading duplicate analysis: {e}")
        return {}


# Only load data when actually needed, not all on startup
def get_hierarchy_data():
    """Get hierarchy data, loading it if not already loaded."""
    global hierarchy_data
    if not hierarchy_data:
        hierarchy_data = load_hierarchy()
    return hierarchy_data


def get_resources_data(level):
    """Get resources data for a specific level, loading it if not already loaded."""
    global resources_data
    if level not in resources_data:
        resources_data[level] = load_resources(level)
    return resources_data[level]


def get_duplicate_data(level):
    """Get duplicate data for a specific level, loading it if not already loaded."""
    global duplicate_data
    if level not in duplicate_data:
        duplicate_data[level] = load_duplicate_analysis(level)
    return duplicate_data[level]


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


@cache_result
def combine_data() -> Dict[str, Any]:
    """Combine hierarchy and resources data for visualization."""
    combined_data = get_hierarchy_data().copy()
    
    # Add resources to each term
    if "terms" in combined_data:
        for term, term_data in combined_data["terms"].items():
            level = term_data["level"]
            level_resources = get_resources_data(level)
            if term in level_resources:
                term_data["resources"] = level_resources[term]
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
    duplicate_data_level = get_duplicate_data(level)
    
    # Group duplicates by parent category
    parent_groups = {}
    for key, data in duplicate_data_level.items():
        parent = data.get("parent", "Unknown")
        if parent not in parent_groups:
            parent_groups[parent] = []
        parent_groups[parent].append(data)
    
    result = {
        "level": level,
        "total_duplicates": len(duplicate_data_level),
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
    if "terms" in combined_data:
        for term, term_data in combined_data["terms"].items():
            if query in term.lower():
                results.append({
                    "term": term,
                    "level": term_data["level"],
                    "type": "term"
                })
    
    # Search in variations
    if "relationships" in combined_data and "variations" in combined_data["relationships"]:
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
    # Get term details from the hierarchy data
    hierarchy_data = get_hierarchy_data()
    term_details = {}
    if term in hierarchy_data.get("terms", {}):
        term_details = {
            "level": hierarchy_data["terms"][term].get("level"),
            "parents": hierarchy_data["terms"][term].get("parents", []),
            "children": hierarchy_data["terms"][term].get("children", []),
            "resources": hierarchy_data["terms"][term].get("resources", []),
            "related_concepts": hierarchy_data["terms"][term].get("related_concepts", {}),
            "definition": hierarchy_data["terms"][term].get("definition", "")
        }
        
        # Find variations for the term
        variations = []
        if "relationships" in hierarchy_data and "variations" in hierarchy_data["relationships"]:
            variations = [v[1] for v in hierarchy_data["relationships"]["variations"] if v[0] == term]
        term_details["variations"] = variations
    
    # Check if this term has been evaluated
    evaluation = manual_evaluations.get(term)
    is_evaluated = evaluation is not None
    
    return jsonify({
        "success": True, 
        "is_evaluated": is_evaluated,
        "evaluation": evaluation if is_evaluated else {},
        "term_details": term_details
    })


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
    if not manual_evaluations: # and not hierarchy_data: # Check for hierarchy_data too
        return jsonify({
            "total_evaluated": 0,
            "importance_counts": {},
            "level_correctness_counts": {},
            "variation_correctness_avg": 0,
            "parent_correctness_avg": 0,
            "potential_parents_count": 0,
            "potential_variations_count": 0,
            "orphan_terms": [],
            "leaf_terms_not_level_3": []
        })

    total_evaluated = len(manual_evaluations)
    importance_counts = Counter()
    level_correctness_counts = Counter()
    level_correctness_aggregated_counts = Counter()
    total_variations_evaluated = 0
    correct_variations = 0
    total_parents_evaluated = 0
    correct_parents = 0
    
    # New counters for potential parents and variations
    total_potential_parents = 0
    selected_potential_parents = 0
    total_potential_variations = 0
    selected_potential_variations = 0

    # Lists for new stats
    orphan_terms_list = []
    leaf_terms_not_level_3_list = []

    hierarchy_data = get_hierarchy_data()
    if hierarchy_data and "terms" in hierarchy_data:
        for term, details in hierarchy_data["terms"].items():
            # Check for orphan terms (exclude level 0 terms)
            if not details.get("parents"): # Handles empty list or missing key
                level = details.get("level")
                try:
                    level_int = int(level) if level is not None else -1
                    # Only include orphan terms that are NOT in level 0
                    if level_int != 0:
                        orphan_terms_list.append({"term": term, "level": level_int})
                except ValueError:
                    # If level isn't a valid integer, we'll include it to be safe
                    orphan_terms_list.append({"term": term, "level": level})

            # Check for leaf terms not at level 3
            is_leaf = not details.get("children") # Handles empty list or missing key
            level = details.get("level")
            # Ensure level is an integer for comparison, otherwise skip if level is None or not convertible
            try:
                if level is not None:
                    level_int = int(level)
                    if is_leaf and level_int != 3:
                        leaf_terms_not_level_3_list.append({"term": term, "level": level_int})
            except ValueError:
                print(f"Warning: Could not convert level '{level}' to int for term '{term}' while checking for leaf nodes.")


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
                
        # Process potential parents from related concepts
        if 'potential_parents' in evaluation and isinstance(evaluation['potential_parents'], dict):
            potential_parents = evaluation['potential_parents']
            total_potential_parents += len(potential_parents)
            selected_potential_parents += sum(1 for v in potential_parents.values() if v)
            
        # Process potential variations from related concepts
        if 'potential_variations' in evaluation and isinstance(evaluation['potential_variations'], dict):
            potential_variations = evaluation['potential_variations']
            total_potential_variations += len(potential_variations)
            selected_potential_variations += sum(1 for v in potential_variations.values() if v)

    variation_correctness_avg = (correct_variations / total_variations_evaluated * 100) if total_variations_evaluated > 0 else 0
    parent_correctness_avg = (correct_parents / total_parents_evaluated * 100) if total_parents_evaluated > 0 else 0

    # Sort the new lists for consistent output (optional, but good practice)
    orphan_terms_list.sort(key=lambda x: (x.get("level", 999), x["term"]))
    leaf_terms_not_level_3_list.sort(key=lambda x: (x.get("level", 999), x["term"]))

    stats = {
        "total_evaluated": total_evaluated,
        "importance_counts": dict(importance_counts),
        "level_correctness_counts": dict(level_correctness_aggregated_counts),
        "variation_correctness_avg": round(variation_correctness_avg, 2),
        "parent_correctness_avg": round(parent_correctness_avg, 2),
        "total_variations_rated": total_variations_evaluated,
        "correct_variations_count": correct_variations,
        "total_parents_rated": total_parents_evaluated,
        "correct_parents_count": correct_parents,
        "potential_parents": {
            "total": total_potential_parents,
            "selected": selected_potential_parents,
            "percent": round((selected_potential_parents / total_potential_parents * 100) if total_potential_parents > 0 else 0, 2)
        },
        "potential_variations": {
            "total": total_potential_variations,
            "selected": selected_potential_variations,
            "percent": round((selected_potential_variations / total_potential_variations * 100) if total_potential_variations > 0 else 0, 2)
        },
        "orphan_terms": orphan_terms_list,
        "leaf_terms_not_level_3": leaf_terms_not_level_3_list
    }

    return jsonify(stats)


@app.route('/api/all_evaluations')
def get_all_evaluated_terms():
    """API endpoint to get the full dictionary of manual evaluations."""
    # Return the whole dictionary, not just the keys
    return jsonify(manual_evaluations)


@cache_result
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
    except Exception as e:
        print(f"Error loading manual evaluations: {e}")
        return {}


def save_manual_evaluations():
    """Save manual evaluations to the JSON file."""
    global manual_evaluations
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(MANUAL_EVALUATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(manual_evaluations, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving manual evaluations to {MANUAL_EVALUATION_FILE}: {e}")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Visualize the academic hierarchy')
    parser.add_argument('-p', '--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--preload', action='store_true',
                        help='Preload all data at startup (default: lazy loading)')
    
    args = parser.parse_args()
    
    # Load manual evaluations at startup
    print("Loading manual evaluations...")
    global manual_evaluations
    manual_evaluations = load_manual_evaluations()
    print(f"Loaded {len(manual_evaluations)} existing manual evaluations.")
    
    # Optionally preload data
    if args.preload:
        print("Preloading hierarchy data...")
        get_hierarchy_data()
        
        print("Preloading resources data...")
        for level in range(4):  # Levels 0, 1, 2, 3
            get_resources_data(level)
    else:
        print("Using lazy loading for data. Data will be loaded on first access.")
    
    # Run the Flask app
    print(f"Starting server on port {args.port}...")
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main() 