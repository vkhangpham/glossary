import os
import pickle
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
import argparse
from collections import defaultdict
import json
import sys

# --- Configuration ---
load_dotenv()
MODEL_NAME = "text-embedding-005"
DATABASE_FILE = "data/embeddings/resource_embeddings.pkl" # Matches process_resources.py

def get_embedding(client, text_to_embed, task_type_string):
    """
    Generates an embedding for the given text.
    Returns the flat embedding vector (list of floats).
    """
    try:
        # Always create a new client in each thread to avoid sharing issues
        print(f"Creating new client for embedding of '{text_to_embed[:20]}...'", file=sys.stderr)
        
        # Load environment variables in this process
        load_dotenv()
        
        # Force required environment variables to be set
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            # print(f"GOOGLE_APPLICATION_CREDENTIALS is set: {os.environ['GOOGLE_APPLICATION_CREDENTIALS'][:20]}...", file=sys.stderr)
        else:
            print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not found", file=sys.stderr)
            
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")
            # print(f"GOOGLE_CLOUD_PROJECT is set: {os.environ['GOOGLE_CLOUD_PROJECT']}", file=sys.stderr)
            
        if os.getenv("GOOGLE_CLOUD_LOCATION"):
            os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("GOOGLE_CLOUD_LOCATION")
        
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
        
        # Always create a new client
        thread_client = genai.Client()
        
        result = thread_client.models.embed_content(
            model=MODEL_NAME,
            contents=text_to_embed,
            config=types.EmbedContentConfig(
                task_type=task_type_string,
                output_dimensionality=768
            )
        )
        # result.embeddings is a list of ContentEmbedding objects.
        # For a single piece of content, it will have one element.
        # We need the .values attribute of that ContentEmbedding object.
        if result.embeddings and len(result.embeddings) > 0 and hasattr(result.embeddings[0], 'values'):
            return result.embeddings[0].values # Corrected to return the numerical vector
        else:
            print(f"Warning: No valid embeddings returned from API for text: '{text_to_embed[:50]}...'", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error generating embedding for text: '{text_to_embed[:50]}...': {e}", file=sys.stderr)
        return None

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    similarity_val = dot_product / (norm_vec1 * norm_vec2)
    return np.clip(similarity_val, -1.0, 1.0) # Clip to handle potential floating point inaccuracies

def load_database(file_path):
    """Loads the vector database from a pickle file."""
    if not os.path.exists(file_path):
        print(f"Database file {file_path} not found.", file=sys.stderr)
        return None
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # Suppressing verbose print for programmatic use: print(f"Loaded {len(data)} resources from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading database from {file_path}: {e}", file=sys.stderr)
        return None

def get_top_related_terms_by_level(
    client, 
    all_embeddings_data, 
    query_string, 
    top_n_overall_resources=10, 
    top_k_categories_per_level=3
):
    """
    Finds relevant resources and then returns a dict of top K related terms per level.
    Excludes the query_string itself from the related terms.
    """
    # print(f"Embedding query for related terms: '{query_string}'...", file=sys.stderr) # Less verbose
    query_embedding_vector = get_embedding(client, query_string, "RETRIEVAL_QUERY")

    if query_embedding_vector is None: # Check if list is empty or None
        print(f"Could not generate embedding for the query: '{query_string}'. Aborting search for this term.", file=sys.stderr)
        return {}

    if not all_embeddings_data:
        # This check should ideally be done before calling this func multiple times
        # print("Embeddings database is empty. Aborting.", file=sys.stderr) 
        return {}

    results_with_similarity = []
    for item in all_embeddings_data:
        stored_vector = None
        pickled_embedding_data = item.get('embedding')
        
        if isinstance(pickled_embedding_data, list) and len(pickled_embedding_data) > 0:
            embedding_object = pickled_embedding_data[0]
            if hasattr(embedding_object, 'values') and isinstance(embedding_object.values, list):
                stored_vector = np.array(embedding_object.values) # Ensure it's an array for cosine_similarity
        
        if stored_vector is not None and len(stored_vector) > 0 : # Ensure vector is not empty
            similarity = cosine_similarity(query_embedding_vector, stored_vector)
            # We are interested in item["term"] which is the canonical term from the DB
            # and item["level"]
            results_with_similarity.append({
                "term": item["term"], 
                "level": item.get("level", "N/A"),
                "similarity": similarity
            })

    results_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)
    top_overall_results = results_with_similarity[:top_n_overall_resources]

    if not top_overall_results:
        # print(f"No relevant resources found for '{query_string}' based on the initial query.", file=sys.stderr) # Less verbose
        return {}

    level_category_rank_weighted_scores = defaultdict(lambda: defaultdict(float))
    for i, res in enumerate(top_overall_results):
        level = res['level']
        category = res['term'] # This is the canonical term from the DB
        similarity_score = res['similarity']
        overall_rank = i + 1 

        if level != "N/A" and category != query_string: # Exclude self
            contribution = similarity_score / overall_rank 
            level_category_rank_weighted_scores[level][category] += contribution
    
    final_related_terms_by_level = {}
    if not level_category_rank_weighted_scores:
        # print(f"No related terms found for '{query_string}' after filtering.", file=sys.stderr) # Less verbose
        return {}
    
    for level, category_scores_for_level in sorted(level_category_rank_weighted_scores.items()):
        ranked_categories_for_level = sorted(
            category_scores_for_level.items(), 
            key=lambda item_val: item_val[1], 
            reverse=True
        )
        
        if ranked_categories_for_level:
            final_related_terms_by_level[level] = [
                cat_score[0] for cat_score in ranked_categories_for_level[:top_k_categories_per_level]
            ]
            # Filter out the query string again, just in case, though primary filter is above
            if query_string in final_related_terms_by_level[level]:
                final_related_terms_by_level[level].remove(query_string)
            
            if not final_related_terms_by_level[level]: # If list became empty after removal
                del final_related_terms_by_level[level]


    return final_related_terms_by_level

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for relevant resources, then rank categories by level and output as JSON.")
    parser.add_argument("query", type=str, help="The search query string.")
    parser.add_argument("--top_n_overall", dest="top_n_overall_resources", type=int, default=20, help="Number of top overall resources to initially fetch and consider for level ranking (default: 20). Increased for better chance of diverse related concepts.")
    parser.add_argument("--top_k_level", dest="top_k_categories_per_level", type=int, default=3, help="Number of top categories to show per level (default: 3).")
    
    args = parser.parse_args()

    client = None
    try:
        client = genai.Client()
        print(f"AI Client initialized for query: '{args.query}'", file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize AI client: {e}", file=sys.stderr)
        sys.exit(1)

    all_db_data = load_database(DATABASE_FILE)
    if not all_db_data:
        print(f"Failed to load database: {DATABASE_FILE}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Finding related terms for query: '{args.query}'", file=sys.stderr)
    related_terms = get_top_related_terms_by_level(
        client,
        all_db_data,
        args.query,
        top_n_overall_resources=args.top_n_overall_resources,
        top_k_categories_per_level=args.top_k_categories_per_level
    )
    
    if related_terms:
        print(f"--- Top Related Terms by Level for '{args.query}' ---", file=sys.stderr)
        # Output the main result as JSON to stdout
        print(json.dumps(related_terms, indent=4))
        # Also print to stderr for easy viewing during CLI run
        for level, terms in related_terms.items():
            print(f"  Level '{level}':", file=sys.stderr)
            for i, term in enumerate(terms):
                print(f"    {i+1}. {term}", file=sys.stderr)
    else:
        print(f"No related terms found for '{args.query}'.", file=sys.stderr)
        print(json.dumps({})) # Output empty JSON to stdout 