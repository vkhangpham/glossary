import argparse
import json
import os
import sys
import concurrent.futures
import pickle
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm

# Assuming query_embeddings.py is in the same directory or accessible in PYTHONPATH
try:
    from query_embeddings import load_database, DATABASE_FILE
except ImportError:
    print("ERROR: Could not import functions from query_embeddings.py. Make sure it's in the same directory or PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

# Load environment variables in the main process
load_dotenv()

BASE_METADATA_PATH = "data/final"
TERM_EMBEDDINGS_DB_PATH = "data/embeddings/term_embeddings.pkl"

DEFAULT_TOP_N_OVERALL = 1000
DEFAULT_TOP_K_PER_LEVEL = 10
DEFAULT_MAX_WORKERS = 16

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    similarity_val = dot_product / (norm_vec1 * norm_vec2)
    return np.clip(similarity_val, -1.0, 1.0)

def cosine_similarity_matrix(matrix, vector):
    """Computes cosine similarity between a matrix of vectors and a single vector."""
    # Ensure vector is a 2D column vector for broadcasting if necessary, though dot product handles it.
    # Ensure matrix and vector are float32
    matrix = np.array(matrix, dtype=np.float32) # Should already be, but good practice
    vector = np.array(vector, dtype=np.float32) # Should already be

    dot_products = np.dot(matrix, vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)

    # Handle potential division by zero if norms are zero
    # Create a mask for non-zero norms to avoid warnings/errors
    valid_norms_mask = (matrix_norms != 0) & (vector_norm != 0)
    
    similarities = np.zeros(matrix.shape[0], dtype=np.float32) # Initialize with zeros

    # Compute similarities only for valid norms
    if np.any(valid_norms_mask): # Check if there's at least one valid computation
        similarities[valid_norms_mask] = dot_products[valid_norms_mask] / (matrix_norms[valid_norms_mask] * vector_norm)
    
    return np.clip(similarities, -1.0, 1.0)

def load_term_embeddings(file_path):
    """Loads pre-computed term embeddings and maps them term_string -> embedding_vector."""
    if not os.path.exists(file_path):
        print(f"Term embeddings file {file_path} not found.", file=sys.stderr)
        return {}
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)  # Expected: list of {"term": str, "level": str, "embedding": list_of_float}
        
        term_to_embedding_map = {}
        for item in data:
            if 'term' in item and 'embedding' in item and isinstance(item['embedding'], (list, np.ndarray)):
                term_to_embedding_map[item['term']] = np.array(item['embedding'], dtype=np.float32)
            else:
                print(f"Warning: Skipping invalid term entry in term embeddings: {item.get('term', 'N/A_TERM')}", file=sys.stderr)

        print(f"Loaded {len(term_to_embedding_map)} term embeddings from {file_path}.", file=sys.stderr)
        return term_to_embedding_map
    except Exception as e:
        print(f"Error loading term embeddings from {file_path}: {e}", file=sys.stderr)
        return {}

def process_single_canonical_term_for_related_concepts(
    canonical_term_str,
    variations_list,
    term_to_embedding_map,
    preprocessed_resource_embeddings_info,
    resource_embeddings_matrix_np,
    top_n_overall_resources,
    top_k_categories_per_level
):
    """
    Finds related concepts for a canonical term by using its embedding and its variations' embeddings
    to query against resource_embeddings_data. Aggregates and re-ranks results.
    """
    overall_aggregated_scores = defaultdict(lambda: defaultdict(float))
    all_source_terms_to_query_with = [canonical_term_str] + (variations_list if variations_list else [])

    for current_query_term in all_source_terms_to_query_with:
        query_embedding_vector = term_to_embedding_map.get(current_query_term)
        
        if query_embedding_vector is None:
            continue

        num_resources = resource_embeddings_matrix_np.shape[0]
        if num_resources == 0: # No resource embeddings to compare against
            continue

        all_similarities = cosine_similarity_matrix(resource_embeddings_matrix_np, query_embedding_vector)
        
        k = top_n_overall_resources
        sorted_top_indices = []

        if k >= num_resources:
            # If k is larger or equal to num_resources, we need all items, sorted by similarity.
            # np.argsort returns indices that would sort the array in ascending order.
            # Slicing [::-1] reverses it for descending order.
            if num_resources > 0: # Avoid argsort on empty array if all_similarities could be empty
                sorted_top_indices = np.argsort(all_similarities)[::-1]
        else:
            # Find indices of the k largest similarities (unsorted among themselves).
            # np.argpartition places the k-th largest element in its sorted position
            # and partitions other elements accordingly. We want the -k largest elements.
            top_k_indices_unsorted = np.argpartition(all_similarities, -k)[-k:]
            
            # Get their actual similarity values.
            similarities_of_top_k = all_similarities[top_k_indices_unsorted]
            
            # Sort these k items by their similarities (descending) to get the correct rank order.
            # The indices from argsort will be relative to the `similarities_of_top_k` array.
            order_within_top_k = np.argsort(similarities_of_top_k)[::-1]
            
            # Use these sorted relative indices to get the final original indices, sorted by similarity.
            sorted_top_indices = top_k_indices_unsorted[order_within_top_k]

        top_resources_for_this_query = []
        # Iterate through the (at most) k top sorted indices
        for original_idx in sorted_top_indices:
            resource_info = preprocessed_resource_embeddings_info[original_idx]
            similarity_score = all_similarities[original_idx]
            
            top_resources_for_this_query.append({
                "term": resource_info["term"],
                "level": resource_info.get("level", "N/A"),
                "similarity": similarity_score
            })
        
        # top_resources_for_this_query is now sorted by similarity and has at most k items.
        # The previous explicit sort and slice are no longer needed here.

        for i, res_item in enumerate(top_resources_for_this_query):
            level = res_item['level']
            related_canonical_term_from_resource = res_item['term']
            similarity_score = res_item['similarity']
            rank_for_this_query = i + 1 

            if level != "N/A" and related_canonical_term_from_resource != canonical_term_str:
                contribution = similarity_score / rank_for_this_query
                overall_aggregated_scores[level][related_canonical_term_from_resource] += contribution
                
    final_related_terms_by_level = {}
    if not overall_aggregated_scores:
        return {} 
    
    for level, category_scores_for_level in sorted(overall_aggregated_scores.items()):
        ranked_categories_for_level = sorted(
            category_scores_for_level.items(), 
            key=lambda item_val_pair: item_val_pair[1],
            reverse=True
        )
        
        if ranked_categories_for_level:
            selected_terms = [
                term_score_pair[0] for term_score_pair in ranked_categories_for_level[:top_k_categories_per_level]
            ]
            if canonical_term_str in selected_terms:
                selected_terms.remove(canonical_term_str)
            
            if selected_terms:
                 final_related_terms_by_level[level] = selected_terms
    
    return final_related_terms_by_level

def collect_metadata_files(metadata_dir, level_filter=None):
    """Collect all metadata files that need to be processed"""
    metadata_files = []
    
    if level_filter:
        level_dir = os.path.join(metadata_dir, level_filter)
        metadata_file_path = os.path.join(level_dir, f"{level_filter}_metadata.json")
        if os.path.exists(metadata_file_path):
            metadata_files.append((level_filter, metadata_file_path))
    else:
        for item_name in os.listdir(metadata_dir):
            if item_name.startswith("lv") and os.path.isdir(os.path.join(metadata_dir, item_name)):
                level_name = item_name
                metadata_file_path = os.path.join(metadata_dir, level_name, f"{level_name}_metadata.json")
                if os.path.exists(metadata_file_path):
                    metadata_files.append((level_name, metadata_file_path))
    
    return metadata_files

def load_all_metadata_content(metadata_files_list):
    """Load all metadata files into a dictionary: {filepath: content}."""
    all_metadata_content_map = {}
    
    for _, file_path in metadata_files_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                all_metadata_content_map[file_path] = metadata
        except Exception as e:
            print(f"Error loading metadata file {file_path}: {e}", file=sys.stderr)
    
    return all_metadata_content_map

def save_metadata(metadata_dict_to_save):
    """Save all updated metadata files"""
    success_count = 0
    for file_path, metadata_content in metadata_dict_to_save.items():
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_content, f, indent=4)
            success_count += 1
        except Exception as e:
            print(f"Error saving metadata file {file_path}: {e}", file=sys.stderr)
    
    return success_count

def main():
    global DEFAULT_MAX_WORKERS
    
    parser = argparse.ArgumentParser(description="Enriches lvX_metadata.json files with related concepts using pre-computed term embeddings.")
    parser.add_argument(
        "--metadata_dir", type=str, default=BASE_METADATA_PATH,
        help=f"Base directory containing level subdirectories with metadata.json files (default: {BASE_METADATA_PATH})."
    )
    parser.add_argument(
        "--term_embeddings_path", type=str, default=TERM_EMBEDDINGS_DB_PATH,
        help=f"Path to the pre-computed term embeddings pickle file (default: {TERM_EMBEDDINGS_DB_PATH})."
    )
    parser.add_argument(
        "--resource_embeddings_path", type=str, default=DATABASE_FILE,
        help=f"Path to the resource embeddings pickle file (default: {DATABASE_FILE})."
    )
    parser.add_argument(
        "--top_n_overall", type=int, default=DEFAULT_TOP_N_OVERALL,
        help=f"Number of top overall resources to initially consider from resource DB for each query term (default: {DEFAULT_TOP_N_OVERALL})."
    )
    parser.add_argument(
        "--top_k_per_level", type=int, default=DEFAULT_TOP_K_PER_LEVEL,
        help=f"Number of top related concepts to store per level for each canonical term (default: {DEFAULT_TOP_K_PER_LEVEL})."
    )
    parser.add_argument(
        "--level", type=str, default=None,
        help="Specify a single level (e.g., 'lv0') to process. If not set, all levels in metadata_dir will be processed."
    )
    parser.add_argument(
        "--max_workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of worker threads for parallel processing (default: {DEFAULT_MAX_WORKERS})."
    )
    args = parser.parse_args()

    DEFAULT_MAX_WORKERS = args.max_workers

    print("Loading pre-computed term embeddings...", file=sys.stderr)
    term_to_embedding_map = load_term_embeddings(args.term_embeddings_path)
    if not term_to_embedding_map:
        print(f"FATAL: Failed to load term embeddings from {args.term_embeddings_path} or it's empty. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading resource embeddings database from: {args.resource_embeddings_path}...", file=sys.stderr)
    all_resource_db_data = load_database(args.resource_embeddings_path)
    if not all_resource_db_data:
        print(f"FATAL: Failed to load resource embeddings database from {args.resource_embeddings_path} or it's empty. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Resource embeddings database loaded with {len(all_resource_db_data)} items.", file=sys.stderr)

    print("Preprocessing resource embeddings data...", file=sys.stderr)
    preprocessed_resource_embeddings_info_list = []
    temp_embedding_vectors_for_matrix = []

    for item in tqdm(all_resource_db_data, desc="Preprocessing resources"):
        pickled_embedding_data = item.get('embedding')
        extracted_vector_raw = None

        if isinstance(pickled_embedding_data, list) and len(pickled_embedding_data) > 0:
            embedding_object = pickled_embedding_data[0]
            if hasattr(embedding_object, 'values') and isinstance(embedding_object.values, (list, np.ndarray)):
                extracted_vector_raw = embedding_object.values
        elif isinstance(pickled_embedding_data, np.ndarray):
            extracted_vector_raw = pickled_embedding_data
        elif isinstance(pickled_embedding_data, list) and all(isinstance(x, (int, float)) for x in pickled_embedding_data):
            extracted_vector_raw = pickled_embedding_data
        
        if extracted_vector_raw is not None:
            try:
                # Ensure it's a flat list/array of numbers before converting
                current_embedding_np = None
                if isinstance(extracted_vector_raw, np.ndarray) and extracted_vector_raw.ndim == 1:
                     current_embedding_np = np.array(extracted_vector_raw, dtype=np.float32)
                elif isinstance(extracted_vector_raw, list) and all(isinstance(x, (float, int)) for x in extracted_vector_raw):
                     current_embedding_np = np.array(extracted_vector_raw, dtype=np.float32)
                else:
                    print(f"Warning: Embedding for resource term '{item['term']}' has unexpected structure or type. Skipping embedding.", file=sys.stderr)

                if current_embedding_np is not None:
                    preprocessed_resource_embeddings_info_list.append({
                        "term": item["term"],
                        "level": item.get("level", "N/A")
                    })
                    temp_embedding_vectors_for_matrix.append(current_embedding_np)

            except Exception as e:
                print(f"Warning: Could not convert embedding to np.float32 for resource term '{item['term']}': {e}. Skipping embedding.", file=sys.stderr)
        
    resource_embeddings_matrix = np.array([], dtype=np.float32) # Initialize as empty
    if temp_embedding_vectors_for_matrix:
        try:
            resource_embeddings_matrix = np.vstack(temp_embedding_vectors_for_matrix)
        except ValueError as e:
            # This can happen if embeddings have inconsistent dimensions
            print(f"FATAL: Error creating resource embeddings matrix: {e}. Embeddings might have inconsistent dimensions. Exiting.", file=sys.stderr)
            # Attempt to find offending dimensions for debugging
            dim_counts = defaultdict(int)
            for vec in temp_embedding_vectors_for_matrix:
                if hasattr(vec, 'shape') and len(vec.shape) > 0:
                    dim_counts[vec.shape[0]] +=1
                else:
                    dim_counts[None] +=1
            print(f"Dimension counts: {dict(dim_counts)}", file=sys.stderr)
            sys.exit(1)

    print(f"Successfully preprocessed {len(preprocessed_resource_embeddings_info_list)} resource embeddings into a matrix of shape {resource_embeddings_matrix.shape}.", file=sys.stderr)

    if not os.path.isdir(args.metadata_dir):
        print(f"ERROR: Metadata directory '{args.metadata_dir}' not found. Exiting.", file=sys.stderr)
        sys.exit(1)

    metadata_files_to_process = collect_metadata_files(args.metadata_dir, args.level)
    if not metadata_files_to_process:
        print(f"No metadata files found to process in '{args.metadata_dir}' (filter: {args.level}). Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(metadata_files_to_process)} metadata files to process.", file=sys.stderr)
    
    print("Loading all metadata content...", file=sys.stderr)
    all_metadata_loaded_map = load_all_metadata_content(metadata_files_to_process)
    if not all_metadata_loaded_map:
        print(f"No metadata content successfully loaded. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded content for {len(all_metadata_loaded_map)} metadata files.", file=sys.stderr)

    tasks_for_enrichment = []
    missing_variation_count = 0
    for file_path, metadata_content in all_metadata_loaded_map.items():
        for canonical_term, term_data_dict in metadata_content.items():
            if not isinstance(term_data_dict, dict):
                print(f"Warning: Term data for '{canonical_term}' in {file_path} is not a dictionary. Skipping.", file=sys.stderr)
                continue
            
            variations = term_data_dict.get("variations", [])
            if not isinstance(variations, list):
                variations = []
            
            tasks_for_enrichment.append({
                "canonical_term": canonical_term,
                "variations": variations,
                "file_path": file_path,
            })
            
    if not tasks_for_enrichment:
        print("No canonical terms found in metadata to process. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Prepared {len(tasks_for_enrichment)} canonical terms for related concept enrichment.", file=sys.stderr)

    total_processed_canonical_terms = 0
    total_enriched_canonical_terms = 0
    
    print(f"Processing {len(tasks_for_enrichment)} canonical terms using up to {args.max_workers} workers...", file=sys.stderr)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task_info = {}
        for task_info in tasks_for_enrichment:
            future = executor.submit(
                process_single_canonical_term_for_related_concepts,
                canonical_term_str=task_info["canonical_term"],
                variations_list=task_info["variations"],
                term_to_embedding_map=term_to_embedding_map,
                preprocessed_resource_embeddings_info=preprocessed_resource_embeddings_info_list,
                resource_embeddings_matrix_np=resource_embeddings_matrix,
                top_n_overall_resources=args.top_n_overall,
                top_k_categories_per_level=args.top_k_per_level
            )
            future_to_task_info[future] = task_info

        for future in tqdm(concurrent.futures.as_completed(future_to_task_info), total=len(tasks_for_enrichment), desc="Enriching terms"):
            original_task_info = future_to_task_info[future]
            canonical_term = original_task_info["canonical_term"]
            file_path_to_update = original_task_info["file_path"]
            
            try:
                related_concepts_result = future.result()
                all_metadata_loaded_map[file_path_to_update][canonical_term]["related_concepts"] = related_concepts_result
                
                if related_concepts_result:
                    total_enriched_canonical_terms += 1
                total_processed_canonical_terms +=1
            except Exception as e:
                print(f"Error processing canonical term '{canonical_term}': {e}", file=sys.stderr)
                total_processed_canonical_terms +=1

    print(f"\nProcessed {total_processed_canonical_terms} canonical terms. Found related concepts for {total_enriched_canonical_terms} terms.", file=sys.stderr)
    print("Saving updated metadata files...", file=sys.stderr)
    
    saved_count = save_metadata(all_metadata_loaded_map)
    
    if saved_count == len(all_metadata_loaded_map):
        print(f"Successfully saved all {saved_count} updated metadata files.", file=sys.stderr)
    else:
        print(f"Warning: Saved {saved_count} out of {len(all_metadata_loaded_map)} metadata files.", file=sys.stderr)
        if saved_count < len(all_metadata_loaded_map):
             print("Some metadata files might not have been updated. Check logs for errors.", file=sys.stderr)

if __name__ == "__main__":
    main() 