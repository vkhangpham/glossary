import os
import json
import pickle
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm
import time

# --- Configuration ---
load_dotenv()

MODEL_NAME = "text-embedding-005"
BASE_DATA_PATH = "data/final"
OUTPUT_DATABASE_FILE = "data/embeddings/resource_embeddings.pkl"
# Updated key for the text to be embedded based on the sample JSON
RESOURCE_TEXT_KEY = "processed_content"


def get_embedding(client, text_to_embed, task_type_string):
    """Generates an embedding for the given text using the specified client and task type string."""
    try:
        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=text_to_embed,
            config=types.EmbedContentConfig(
                task_type=task_type_string,
                output_dimensionality=768
            )
        )
        return result.embeddings
    except Exception as e:
        print(f"Error generating embedding for text: '{text_to_embed[:50]}...': {e}")
        return None

def build_vector_database():
    """
    Scans for resource files, generates embeddings, and saves them to a file.
    If an existing database file is found, it updates it with new resources only.
    """

    client = genai.Client()
    all_embeddings_data = []
    processed_resource_identifiers = set()

    # Load existing database if it exists
    if os.path.exists(OUTPUT_DATABASE_FILE):
        print(f"Loading existing database from {OUTPUT_DATABASE_FILE}...")
        try:
            with open(OUTPUT_DATABASE_FILE, 'rb') as f:
                all_embeddings_data = pickle.load(f)
            for item in all_embeddings_data:
                # Create an identifier for each loaded resource
                # Assumes 'level', 'term', and 'resource_text' uniquely identify a resource
                identifier = (item.get("level"), item.get("term"), item.get("resource_text"))
                processed_resource_identifiers.add(identifier)
            print(f"Loaded {len(all_embeddings_data)} existing embeddings. {len(processed_resource_identifiers)} unique identifiers created.")
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading existing database: {e}. Starting fresh.")
            all_embeddings_data = []
            processed_resource_identifiers = set()
        except Exception as e:
            print(f"An unexpected error occurred loading the database: {e}. Starting fresh.")
            all_embeddings_data = []
            processed_resource_identifiers = set()
    else:
        print(f"No existing database found at {OUTPUT_DATABASE_FILE}. Starting fresh.")

    new_embeddings_count = 0
    print(f"Scanning directories in {BASE_DATA_PATH}...")

    for term_dir_name in tqdm(os.listdir(BASE_DATA_PATH), desc="Processing Level Dirs"):
        if term_dir_name.startswith("lv") and os.path.isdir(os.path.join(BASE_DATA_PATH, term_dir_name)):
            level_name = term_dir_name # e.g., "lv0", "lv1" - this is the level
            file_name = f"{level_name}_filtered_resources.json"
            file_path = os.path.join(BASE_DATA_PATH, level_name, file_name)

            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # JSON structure is now a dictionary of categories,
                        # where each category is a list of resources.
                        data = json.load(f)

                    if not isinstance(data, dict):
                        print(f"Warning: Expected a dictionary of categories in {file_path}, found {type(data)}. Skipping.")
                        continue
                    
                    for category_name, resources_list in tqdm(data.items(), desc=f"Processing categories in {file_name}", leave=False):
                        if not isinstance(resources_list, list):
                            print(f"Warning: Expected a list of resources for category '{category_name}' in {file_path}, found {type(resources_list)}. Skipping category.")
                            continue
                        
                        for i, resource in enumerate(tqdm(resources_list, desc=f"Embedding {category_name[:20]}", leave=False)):
                            if not isinstance(resource, dict) or RESOURCE_TEXT_KEY not in resource:
                                print(f"Warning: Could not find key '{RESOURCE_TEXT_KEY}' in resource #{i} under category '{category_name}' in {file_path}. Skipping.")
                                continue

                            resource_text = resource[RESOURCE_TEXT_KEY]

                            if not resource_text or not isinstance(resource_text, str):
                                print(f"Warning: Empty or invalid text for resource #{i} under category '{category_name}' in {file_path}. Skipping.")
                                continue

                            # Check if this resource has already been processed
                            current_resource_identifier = (level_name, category_name, resource_text)
                            if current_resource_identifier in processed_resource_identifiers:
                                # print(f"Skipping already processed resource: {level_name}/{category_name}/{resource_text[:30]}...")
                                continue

                            embedding_result = get_embedding(client, resource_text, "RETRIEVAL_DOCUMENT")
                            if embedding_result:
                                all_embeddings_data.append({
                                    "term": category_name,  # THIS IS THE KEY CHANGE: term is now the JSON category
                                    "level": level_name,     # Storing the original 'lvX' as 'level'
                                    "resource_text": resource_text,
                                    "original_resource": resource,
                                    "embedding": embedding_result
                                })
                                processed_resource_identifiers.add(current_resource_identifier) # Add to set after successful embedding and append
                                new_embeddings_count += 1
                            time.sleep(0.1)
                    print(f"Processed resources from {file_path}")

                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}. Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred processing {file_path}: {e}. Skipping.")
            else:
                print(f"File not found: {file_path}. Skipping.")

    if not all_embeddings_data:
        print("No embeddings were generated or loaded. Exiting.")
        return

    print(f"Generated {new_embeddings_count} new embeddings.")
    print(f"Total embeddings in database: {len(all_embeddings_data)}.")
    print(f"Saving database to {OUTPUT_DATABASE_FILE}...")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_DATABASE_FILE)
    if output_dir: # Check if output_dir is not an empty string (e.g. if filename has no path)
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_DATABASE_FILE, 'wb') as f:
        pickle.dump(all_embeddings_data, f)
    print("Database saved successfully.")

if __name__ == "__main__":
    print("Attempting to build the vector database...")
    build_vector_database()
    print("Finished building vector database (if any resources were processed).")

    # To use this script:
    # 1. Create a .env file in the same directory as this script with your GEMINI_API_KEY:
    #    GEMINI_API_KEY=your_actual_api_key_here
    # 2. Install necessary libraries: pip install google-generativeai numpy python-dotenv tqdm
    # 3. Run the script: python process_resources.py
    #    It will create/overwrite `vector_database.pkl`.
 