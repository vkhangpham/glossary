import argparse
import json
import os
import pickle
import logging
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from difflib import SequenceMatcher
import networkx as nx
import sys # Added to redirect print to stderr

# Attempt to import from generate_glossary package
try:
    from generate_glossary.deduplicator.graph_dedup import build_term_graph, add_rule_based_edges, select_canonical_terms
    GRAPH_DEDUP_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import graph_dedup functions from generate_glossary.deduplicator.graph_dedup.", file=sys.stderr)
    print("Step 2 (match via deduplication pipeline) will be skipped.", file=sys.stderr)
    print("Please ensure 'generate_glossary' is in PYTHONPATH or the script is run from the correct parent directory.", file=sys.stderr)
    GRAPH_DEDUP_AVAILABLE = False

# --- Configuration ---
# Load environment variables and ensure they're properly set
load_dotenv()

# Explicitly set GOOGLE_API_KEY in current process environment if it exists in .env
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    print(f"GOOGLE_API_KEY found and set in environment", file=sys.stderr)
else:
    print(f"WARNING: GOOGLE_API_KEY not found in environment variables", file=sys.stderr)

# --- Configure Logging for Cleaner Output ---
# Suppress INFO and DEBUG messages from root logger (affects graph_dedup)
logging.getLogger().setLevel(logging.WARNING)
# Suppress INFO and DEBUG messages from Google API client libraries
logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING)
logging.getLogger("google.auth.transport.requests").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING) # Often source of HTTP logs
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

MODEL_NAME_EMBEDDING = "text-embedding-005"
BASE_DATA_PATH = "data/final"
EMBEDDINGS_DB_PATH = "data/embeddings/resource_embeddings.pkl"
DIRECT_MATCH_SIMILARITY_THRESHOLD = 0.95
EMBEDDING_QUERY_SIMILARITY_THRESHOLD = 0.65
TOP_N_EMBEDDING_RESULTS = 3

# Define cache paths for Step 2 based on EMBEDDINGS_DB_PATH directory
EMBEDDINGS_DIR = os.path.dirname(EMBEDDINGS_DB_PATH)
GRAPH_CACHE_PATH_S2 = '.dedup_cache/graph_cache.pickle'
EMBEDDINGS_CACHE_PATH_S2 = '.dedup_cache/embeddings_cache.pickle'

# --- Helper Functions ---
def normalize_string(s):
    if not isinstance(s, str):
        return ""
    return s.lower().strip()

def get_string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    similarity_val = dot_product / (norm_vec1 * norm_vec2)
    return np.clip(similarity_val, -1.0, 1.0)

def get_single_embedding(client, text_to_embed, task_type="RETRIEVAL_DOCUMENT"):
    try:
        result = client.models.embed_content(
            model=MODEL_NAME_EMBEDDING,
            contents=text_to_embed,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )
        if result.embeddings and len(result.embeddings) > 0 and hasattr(result.embeddings[0], 'values'):
            return np.array(result.embeddings[0].values)
        else:
            print(f"Warning: No valid embeddings returned from API for text: '{text_to_embed[:50]}...'")
            return None
    except Exception as e:
        print(f"Error generating embedding for text: '{text_to_embed[:50]}...': {e}")
        return None

# --- Data Loading Functions ---
def load_terms_and_variations_for_step1(base_path):
    all_items = []
    processed_combinations = set() # To store (normalized_text, canonical_term, level) to avoid duplicates

    if not os.path.exists(base_path):
        print(f"Warning: Base data path {base_path} not found. Cannot load terms for Step 1.")
        return []

    for dir_name in os.listdir(base_path):
        if dir_name.startswith("lv") and os.path.isdir(os.path.join(base_path, dir_name)):
            level_name = dir_name
            
            resources_file_path = os.path.join(base_path, level_name, f"{level_name}_filtered_resources.json")
            if os.path.exists(resources_file_path):
                try:
                    with open(resources_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        # In filtered_resources, keys are canonical terms, values are lists of resource objects
                        # We are interested in the canonical term itself (the key)
                        for canonical_term_key in data.keys(): 
                            norm_text = normalize_string(canonical_term_key)
                            combo = (norm_text, canonical_term_key, level_name)
                            if combo not in processed_combinations:
                                all_items.append({
                                    "text_to_match": canonical_term_key, "text_normalized": norm_text,
                                    "canonical_term": canonical_term_key, "level": level_name,
                                    "source_file": resources_file_path, "is_canonical": True
                                })
                                processed_combinations.add(combo)
                except Exception as e:
                    print(f"Warning: Could not process {resources_file_path}: {e}")

            metadata_file_path = os.path.join(base_path, level_name, f"{level_name}_metadata.json")
            if os.path.exists(metadata_file_path):
                try:
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    if isinstance(metadata, dict):
                        for canonical_term_from_meta, meta_details in metadata.items():
                            variations = []
                            # graph_dedup.py uses "variations" key inside a dict, or the list directly
                            if isinstance(meta_details, dict) and "variations" in meta_details and isinstance(meta_details["variations"], list):
                                variations = meta_details["variations"]
                            elif isinstance(meta_details, list): 
                                variations = meta_details
                            
                            for variation in variations:
                                norm_text = normalize_string(variation)
                                combo = (norm_text, canonical_term_from_meta, level_name) # Variation maps to its canonical
                                if combo not in processed_combinations:
                                    all_items.append({
                                        "text_to_match": variation, "text_normalized": norm_text,
                                        "canonical_term": canonical_term_from_meta, "level": level_name,
                                        "source_file": metadata_file_path, "is_canonical": False
                                    })
                                    processed_combinations.add(combo)
                except Exception as e:
                    print(f"Warning: Could not process {metadata_file_path}: {e}")
    return all_items


def load_level_mapping_from_embeddings_db(db_path):
    """Loads level string to int mapping and vice-versa from the embeddings DB for consistent reporting."""
    level_str_to_int_map = {}
    level_int_to_str_map = {}
    if not os.path.exists(db_path):
        print(f"Warning: Embeddings database {db_path} not found. Cannot create level mapping.")
        return level_str_to_int_map, level_int_to_str_map
    try:
        with open(db_path, 'rb') as f:
            all_data = pickle.load(f)
        for item in all_data:
            level_str = item.get("level")
            if level_str and level_str not in level_str_to_int_map:
                try:
                    level_int_val = int(level_str.replace("lv", ""))
                    level_str_to_int_map[level_str] = level_int_val
                    level_int_to_str_map[level_int_val] = level_str
                except ValueError:
                    pass # Ignore if parsing fails for some reason
    except Exception as e:
        print(f"Error loading level mapping from {db_path}: {e}")
    return level_str_to_int_map, level_int_to_str_map


# --- Step Implementations ---

def find_direct_match_step1(input_string, all_terms_and_variations, similarity_threshold):
    normalized_input = normalize_string(input_string)
    best_match = None
    highest_similarity = -1.0

    for item in all_terms_and_variations:
        if normalized_input == item["text_normalized"]:
            return {
                "canonical_term": item["canonical_term"], "level": item["level"],
                "matched_text": item["text_to_match"], "similarity": 1.0, "match_type_detail": "exact"
            }

    for item in all_terms_and_variations:
        similarity = get_string_similarity(normalized_input, item["text_normalized"])
        if similarity >= similarity_threshold and similarity > highest_similarity:
            highest_similarity = similarity
            best_match = {
                "canonical_term": item["canonical_term"], "level": item["level"],
                "matched_text": item["text_to_match"], "similarity": similarity, "match_type_detail": "similar"
            }
    return best_match

def find_match_via_deduplication_step2(input_string, client, level_int_to_str_map_global):
    if not GRAPH_DEDUP_AVAILABLE:
        print("Skipping Step 2: graph_dedup components are not available.", file=sys.stderr)
        return None

    G_cached = None
    term_embeddings_cached = None

    if os.path.exists(GRAPH_CACHE_PATH_S2) and os.path.exists(EMBEDDINGS_CACHE_PATH_S2):
        try:
            with open(GRAPH_CACHE_PATH_S2, "rb") as f:
                G_cached = pickle.load(f)
            with open(EMBEDDINGS_CACHE_PATH_S2, "rb") as f:
                term_embeddings_cached = pickle.load(f)
            print(f"Successfully loaded cached graph ({G_cached.number_of_nodes()} nodes) and {len(term_embeddings_cached)} embeddings for Step 2.", file=sys.stderr)
        except Exception as e:
            print(f"Error loading cached graph/embeddings for Step 2: {e}. Skipping Step 2 deduplication.", file=sys.stderr)
            return None
    else:
        print(f"Cached graph ({GRAPH_CACHE_PATH_S2}) or embeddings ({EMBEDDINGS_CACHE_PATH_S2}) not found. Skipping Step 2 deduplication.", file=sys.stderr)
        return None

    print(f"Embedding input string '{input_string}' for Step 2 graph integration...", file=sys.stderr)
    input_embedding = get_single_embedding(client, input_string, task_type="RETRIEVAL_DOCUMENT")
    if input_embedding is None:
        print("Could not generate embedding for input string. Skipping Step 2 deduplication.", file=sys.stderr)
        return None

    G_temp = G_cached.copy()
    current_terms_by_level_int = {}
    max_existing_level_in_g = -1
    for node, data in G_cached.nodes(data=True):
        level = data.get('level')
        if isinstance(level, int):
            if level not in current_terms_by_level_int:
                current_terms_by_level_int[level] = []
            current_terms_by_level_int[level].append(node)
            if level > max_existing_level_in_g:
                max_existing_level_in_g = level
        else:
            print(f"Warning: Node '{node}' in cached graph has missing or non-integer level: {level}", file=sys.stderr)

    INPUT_TERM_LEVEL_INT = max_existing_level_in_g + 1 if max_existing_level_in_g != -1 else 0

    terms_by_level_for_processing = {lvl: list(terms) for lvl, terms in current_terms_by_level_int.items()}
    terms_by_level_for_processing[INPUT_TERM_LEVEL_INT] = [input_string]

    embeddings_for_processing = term_embeddings_cached.copy()
    embeddings_for_processing[input_string] = input_embedding

    G_temp.add_node(
        input_string,
        level=INPUT_TERM_LEVEL_INT,
        has_sciences_suffix=input_string.endswith(("sciences", "studies")),
        word_count=len(input_string.split()),
        embedding=input_embedding
    )
    if not G_temp.has_node(input_string):
        print(f"Error: Input string '{input_string}' failed to be added to the temporary graph. Step 2 cannot proceed.", file=sys.stderr)
        return None

    print("Adding rule-based edges involving the input string to the temporary graph...", file=sys.stderr)
    add_rule_based_edges(
        G_temp,
        all_terms_map=terms_by_level_for_processing, 
        current_processing_level_num=INPUT_TERM_LEVEL_INT, 
        current_processing_terms_list=[input_string]
    )

    print("Selecting canonical terms from the temporary graph...", file=sys.stderr)
    canonical_mapping = select_canonical_terms(G_temp, terms_by_level_for_processing)
    original_canonical_terms_from_cache = set(G_cached.nodes())

    for canonical, variations in canonical_mapping.items():
        if input_string in variations and canonical != input_string:
            if canonical in original_canonical_terms_from_cache:
                original_level_of_canonical_int = G_cached.nodes[canonical].get('level')
                if isinstance(original_level_of_canonical_int, int):
                    level_str = level_int_to_str_map_global.get(original_level_of_canonical_int, f"lv{original_level_of_canonical_int}")
                    return {
                        "canonical_term": canonical,
                        "level": level_str, # Return string level
                        "input_term_became_variation": True
                    }
                else:
                    print(f"Warning: Matched canonical '{canonical}' from cache has invalid level: {original_level_of_canonical_int}", file=sys.stderr)
    return None

def find_match_via_embedding_query_step3(input_string, client, db_path, similarity_threshold, top_n):
    print(f"Embedding query string '{input_string}' for Step 3...", file=sys.stderr)
    query_embedding_vector = get_single_embedding(client, input_string, task_type="RETRIEVAL_QUERY")
    if query_embedding_vector is None:
        print("Could not generate query embedding. Skipping Step 3.", file=sys.stderr)
        return []

    if not os.path.exists(db_path):
        print(f"Embeddings database {db_path} not found for Step 3. Skipping.", file=sys.stderr)
        return []
    
    all_db_items = []
    try:
        with open(db_path, 'rb') as f:
            all_db_items = pickle.load(f)
    except Exception as e:
        print(f"Error loading database {db_path} for Step 3: {e}", file=sys.stderr)
        return []

    if not all_db_items:
        print("Embedding database is empty for Step 3. Skipping.", file=sys.stderr)
        return []

    results_with_similarity = []
    for item in all_db_items:
        stored_vector = None
        pickled_embedding_data = item.get('embedding')
        
        if isinstance(pickled_embedding_data, list) and len(pickled_embedding_data) > 0:
            embedding_object = pickled_embedding_data[0]
            if hasattr(embedding_object, 'values') and isinstance(embedding_object.values, list):
                stored_vector = np.array(embedding_object.values)
        
        if stored_vector is not None:
            sim = cosine_similarity(query_embedding_vector, stored_vector)
            if sim >= similarity_threshold:
                results_with_similarity.append({
                    "term": item["term"], "level": item.get("level", "N/A"),
                    "resource_text": item.get("resource_text", ""), "similarity": float(sim) # Ensure JSON serializable
                })

    results_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)
    deduplicated_results = []
    seen_term_level_combinations = set()
    for res in results_with_similarity:
        term_level_combo = (res["term"], res["level"])
        if term_level_combo not in seen_term_level_combinations:
            deduplicated_results.append(res)
            seen_term_level_combinations.add(term_level_combo)
    
    return deduplicated_results[:top_n]

# --- Main Logic for a Single Term ---
def _map_single_term(input_to_map, client, terms_and_variations_s1, level_int_to_str_map_global, db_path_s3, similarity_threshold_s3, top_n_s3, direct_match_similarity_threshold_s1):
    print(f"Attempting to map input: '{input_to_map}'", file=sys.stderr)
    
    output_json = {
        "input_term": input_to_map,
        "status": "failure",
        "match_type": "no_match",
        "message": f"Could not map input '{input_to_map}' through any defined steps.",
        "mapped_term": None,
        "level": None,
        "similarity": None,
        "details": None
    }

    # Step 1: Direct Match
    print(f"--- Step 1: Direct Match for '{input_to_map}' ---", file=sys.stderr)
    if not terms_and_variations_s1:
        print("No terms or variations loaded for Step 1. Skipping.", file=sys.stderr)
    else:
        direct_match_result = find_direct_match_step1(input_to_map, terms_and_variations_s1, direct_match_similarity_threshold_s1)
        if direct_match_result:
            output_json.update({
                "status": "success",
                "match_type": "direct_match",
                "mapped_term": direct_match_result['canonical_term'],
                "level": direct_match_result['level'],
                "similarity": direct_match_result['similarity'],
                "details": {"matched_text": direct_match_result['matched_text'], "match_subtype": direct_match_result['match_type_detail']},
                "message": f"Direct match found for '{input_to_map}'."
            })
            return output_json # Exit after first successful match for this term

    # Step 2: Deduplication Pipeline
    # Note: Step 2 loads its own caches internally if GRAPH_DEDUP_AVAILABLE is True.
    # We pass the client and level_int_to_str_map_global.
    print(f"--- Step 2: Deduplication Pipeline Match for '{input_to_map}' ---", file=sys.stderr)
    dedup_match_result = find_match_via_deduplication_step2(input_to_map, client, level_int_to_str_map_global)
    if dedup_match_result:
        output_json.update({
            "status": "success",
            "match_type": "deduplication_pipeline",
            "mapped_term": dedup_match_result['canonical_term'],
            "level": dedup_match_result['level'],
            "details": {"input_term_became_variation": dedup_match_result.get('input_term_became_variation', False)},
            "message": f"Match found via deduplication pipeline for '{input_to_map}'."
        })
        return output_json

    # Step 3: Embedding Query
    print(f"--- Step 3: Embedding Query Match for '{input_to_map}' ---", file=sys.stderr)
    embedding_query_results = find_match_via_embedding_query_step3(
        input_to_map, client, db_path_s3,
        similarity_threshold_s3, top_n_s3
    )
    if embedding_query_results:
        top_embedding_match = embedding_query_results[0]
        output_json.update({
            "status": "success",
            "match_type": "embedding_query",
            "mapped_term": top_embedding_match['term'],
            "level": top_embedding_match['level'],
            "similarity": top_embedding_match['similarity'],
            "details": {"top_n_matches": embedding_query_results},
            "message": f"Found {len(embedding_query_results)} potential match(es) via embedding query for '{input_to_map}'."
        })
        return output_json

    print(f"--- Conclusion: No Match for '{input_to_map}' ---", file=sys.stderr)
    return output_json


# --- Main Function (Batch Processing) ---
def main(input_strings):
    results = []
    client = None
    try:
        print("Initializing AI client...", file=sys.stderr)
        client = genai.Client()
        print("AI client initialized.", file=sys.stderr)
    except Exception as e:
        print(f"Fatal: Failed to initialize AI client: {e}", file=sys.stderr)
        # For batch, if client fails, all will fail with this message.
        # We'll create error results for each input string.
        for input_str in input_strings:
            results.append({
                "input_term": input_str,
                "status": "failure",
                "match_type": "client_initialization_error",
                "message": f"Failed to initialize AI client: {e}",
                "mapped_term": None, "level": None, "similarity": None, "details": None
            })
        print(json.dumps(results))
        return

    # Load shared resources once
    print("Loading shared resources for batch processing...", file=sys.stderr)
    _, level_int_to_str_map = load_level_mapping_from_embeddings_db(EMBEDDINGS_DB_PATH)
    terms_and_variations_s1 = load_terms_and_variations_for_step1(BASE_DATA_PATH)
    if terms_and_variations_s1:
        print(f"Loaded {len(terms_and_variations_s1)} total terms/variations for direct matching (Step 1).", file=sys.stderr)
    else:
        print("No terms or variations loaded for Step 1. Direct matches will be skipped if data is missing.", file=sys.stderr)
    print("Shared resources loaded.", file=sys.stderr)

    for input_to_map in input_strings:
        if not isinstance(input_to_map, str) or not input_to_map.strip():
            print(f"Skipping invalid or empty input term: {input_to_map}", file=sys.stderr)
            results.append({
                "input_term": str(input_to_map), # Ensure it's a string for JSON
                "status": "failure",
                "match_type": "invalid_input",
                "message": "Input term was empty or invalid.",
                "mapped_term": None, "level": None, "similarity": None, "details": None
            })
            continue

        result = _map_single_term(
            input_to_map,
            client,
            terms_and_variations_s1,
            level_int_to_str_map,
            EMBEDDINGS_DB_PATH, # db_path_s3
            EMBEDDING_QUERY_SIMILARITY_THRESHOLD, # similarity_threshold_s3
            TOP_N_EMBEDDING_RESULTS, # top_n_s3
            DIRECT_MATCH_SIMILARITY_THRESHOLD # direct_match_similarity_threshold_s1
        )
        results.append(result)

    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map one or more input strings to terms in the glossary. Outputs a JSON list of results.")
    parser.add_argument("input_strings", type=str, nargs='+', help="The input string(s) to map.")
    args = parser.parse_args()
    main(args.input_strings) 