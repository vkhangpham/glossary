import argparse
import json
import os
import pickle
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Explicitly set environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT", "")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("GOOGLE_CLOUD_LOCATION", "")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "")

MODEL_NAME = "text-embedding-005"
BASE_METADATA_PATH = "data/final"
EMBEDDINGS_DB_PATH = "data/embeddings/resource_embeddings.pkl"
TERM_EMBEDDINGS_DB_PATH = "data/embeddings/term_embeddings.pkl"

def get_embedding(client, text_to_embed, task_type_string):
    """Generates an embedding for the given text."""
    try:
        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=text_to_embed,
            config=types.EmbedContentConfig(
                task_type=task_type_string,
                output_dimensionality=768
            )
        )
        if result.embeddings and len(result.embeddings) > 0 and hasattr(result.embeddings[0], 'values'):
            return result.embeddings[0].values
        else:
            print(f"Warning: No valid embeddings returned from API for text: '{text_to_embed[:50]}...'", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error generating embedding for text: '{text_to_embed[:50]}...': {e}", file=sys.stderr)
        return None

def collect_metadata_files(metadata_dir, level=None):
    """Collect all metadata files that need to be processed"""
    metadata_files = []
    
    if level:
        level_dir = os.path.join(metadata_dir, level)
        metadata_file_path = os.path.join(level_dir, f"{level}_metadata.json")
        if os.path.exists(metadata_file_path):
            metadata_files.append((level, metadata_file_path))
    else:
        for item in os.listdir(metadata_dir):
            if item.startswith("lv") and os.path.isdir(os.path.join(metadata_dir, item)):
                metadata_file_path = os.path.join(metadata_dir, item, f"{item}_metadata.json")
                if os.path.exists(metadata_file_path):
                    metadata_files.append((item, metadata_file_path))
    
    return metadata_files

def load_all_metadata(metadata_files):
    """Load all metadata files and extract canonical terms and their variations."""
    all_metadata_loaded = {} # Stores the raw loaded metadata if needed elsewhere, though not directly used for terms list here
    unique_terms_to_embed = set() # Use a set to store (term_string, level) to ensure uniqueness
    
    for level, file_path in metadata_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata_content = json.load(f)
                all_metadata_loaded[file_path] = metadata_content
                
                for canonical_term, term_data in metadata_content.items():
                    # Add the canonical term itself
                    unique_terms_to_embed.add((canonical_term, level))
                    
                    # Add its variations
                    variations = []
                    if isinstance(term_data, dict) and "variations" in term_data:
                        if isinstance(term_data["variations"], list):
                            variations = term_data["variations"]
                        elif isinstance(term_data["variations"], str): # Handle if variations is accidentally a string
                            variations = [term_data["variations"]]
                    elif isinstance(term_data, list): # Older format where term_data itself was the list of variations
                        variations = term_data
                    
                    for variation_string in variations:
                        if isinstance(variation_string, str) and variation_string.strip():
                            unique_terms_to_embed.add((variation_string, level))
                            
        except Exception as e:
            print(f"Error loading or processing metadata file {file_path}: {e}", file=sys.stderr)
    
    # Convert set of tuples to list of tuples for ordered processing by tqdm
    all_terms_list = sorted(list(unique_terms_to_embed))
    return all_metadata_loaded, all_terms_list

def main():
    parser = argparse.ArgumentParser(description="Generate and save retrieval query embeddings for terms.")
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=BASE_METADATA_PATH,
        help=f"Base directory containing level subdirectories (e.g., lv0, lv1) with metadata.json files (default: {BASE_METADATA_PATH})."
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Specify a single level (e.g., 'lv0') to process. If not set, all levels in metadata_dir will be processed."
    )
    args = parser.parse_args()

    # Initialize AI client
    try:
        print("Initializing AI Client...", file=sys.stderr)
        client = genai.Client()
        print("AI Client initialized.", file=sys.stderr)
    except Exception as e:
        print(f"FATAL: Failed to initialize AI client: {e}", file=sys.stderr)
        sys.exit(1)

    # Collect metadata files to process
    metadata_files = collect_metadata_files(args.metadata_dir, args.level)
    if not metadata_files:
        print(f"No metadata files found to process. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(metadata_files)} metadata files to process: {[level for level, _ in metadata_files]}", file=sys.stderr)
    
    # Load all metadata files
    print("Loading all metadata files...", file=sys.stderr)
    all_metadata, all_terms = load_all_metadata(metadata_files)
    print(f"Loaded {len(all_terms)} terms from {len(all_metadata)} metadata files.", file=sys.stderr)

    # Create embeddings directory if it doesn't exist
    os.makedirs(os.path.dirname(TERM_EMBEDDINGS_DB_PATH), exist_ok=True)

    # Generate embeddings for each term
    term_embeddings = []
    print("Generating embeddings for terms...", file=sys.stderr)
    
    for term, level in tqdm(all_terms, desc="Processing terms"):
        embedding = get_embedding(client, term, "RETRIEVAL_QUERY")
        if embedding is not None:
            term_embeddings.append({
                "term": term,
                "level": level,
                "embedding": embedding
            })
        else:
            print(f"Warning: Failed to generate embedding for term '{term}'", file=sys.stderr)

    # Save embeddings
    print(f"Saving {len(term_embeddings)} term embeddings to {TERM_EMBEDDINGS_DB_PATH}...", file=sys.stderr)
    try:
        with open(TERM_EMBEDDINGS_DB_PATH, 'wb') as f:
            pickle.dump(term_embeddings, f)
        print("Successfully saved term embeddings.", file=sys.stderr)
    except Exception as e:
        print(f"Error saving term embeddings: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 