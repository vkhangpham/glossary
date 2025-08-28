import json
import os
import sys

# --- Configuration ---
# Assuming the script is in 'generate_definition' and 'data' is in the parent 'glossary' directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
BASE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "final") # Path to the 'final' directory
METADATA_FILE_TEMPLATE = os.path.join(BASE_DATA_PATH, "lv{}", "lv{}_metadata.json") # data/final/lvX/lvX_metadata.json
LEVELS_TO_PROCESS = [0, 1, 2, 3]  # Specify which levels' metadata files to process

def capitalize_first_letter(text: str) -> str:
    """Capitalizes the first letter of a string, if the string is not empty."""
    if text and len(text) > 0:
        return text[0].upper() + text[1:]
    return text

def main():
    """Main function to load, process, and save definitions in metadata files."""
    print("Starting definition capitalization script for individual metadata files...")
    
    total_capitalized_overall = 0
    total_files_processed = 0
    total_files_with_errors = 0
    files_updated = 0

    for level_num in LEVELS_TO_PROCESS:
        level_key = f"lv{level_num}" # For consistent naming if needed elsewhere, though not directly used in path format here
        # Construct the metadata file path for the current level
        # The template METADATA_FILE_TEMPLATE.format(level_num, level_num) expects two level numbers
        metadata_file_path = METADATA_FILE_TEMPLATE.format(level_num, level_num)
        
        print(f"--- Processing Level {level_num}: {metadata_file_path} ---")

        try:
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"Successfully loaded metadata from {metadata_file_path}")
        except FileNotFoundError:
            print(f"ERROR: Metadata file not found at {metadata_file_path}. Skipping this level.")
            total_files_with_errors += 1
            continue
        except json.JSONDecodeError as e:
            print(f"ERROR: Could not decode JSON from {metadata_file_path}: {e}. Skipping this level.")
            total_files_with_errors += 1
            continue
        except Exception as e:
            print(f"An unexpected error occurred while loading {metadata_file_path}: {e}. Skipping this level.")
            total_files_with_errors += 1
            continue

        if not isinstance(metadata, dict):
            print(f"ERROR: Expected metadata in {metadata_file_path} to be a dictionary of terms. Skipping.")
            total_files_with_errors += 1
            continue

        capitalized_in_this_file = 0
        
        # Iterate through terms in the metadata
        # Metadata structure is expected to be: { "term_name": {"definition": "...", ...}, ... }
        for term, term_data in metadata.items():
            if isinstance(term_data, dict) and "definition" in term_data:
                original_definition = term_data.get("definition")
                if isinstance(original_definition, str):
                    capitalized_definition = capitalize_first_letter(original_definition)
                    if original_definition != capitalized_definition:
                        metadata[term]["definition"] = capitalized_definition
                        capitalized_in_this_file += 1
                else:
                    # This can be a bit noisy if many definitions are not strings, consider logging level or less verbose message
                    print(f"INFO: Definition for term '{term}' in {metadata_file_path} is not a string (type: {type(original_definition)}). Skipping capitalization for this definition.")
            # else:
                # Term data might not be a dict, or "definition" key might be missing.
                # This is not necessarily an error if some terms don't have definitions yet.
                # print(f"DEBUG: Term '{term}' in {metadata_file_path} does not have a dictionary structure or a 'definition' key. Skipping.")


        if capitalized_in_this_file > 0:
            try:
                with open(metadata_file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                print(f"Successfully saved updated metadata to {metadata_file_path}. Capitalized {capitalized_in_this_file} definitions in this file.")
                total_capitalized_overall += capitalized_in_this_file
                files_updated +=1
            except IOError as e:
                print(f"ERROR: Could not write updated metadata to {metadata_file_path}: {e}")
                total_files_with_errors += 1 # Count as an error if saving fails
            except Exception as e:
                print(f"An unexpected error occurred while saving {metadata_file_path}: {e}")
                total_files_with_errors += 1
        elif capitalized_in_this_file == 0 and os.path.exists(metadata_file_path): # Check if file existed before saying no changes
             print(f"No definitions required capitalization in {metadata_file_path}.")
        
        total_files_processed +=1

    print("--- Overall Summary ---")
    print(f"Attempted to process {len(LEVELS_TO_PROCESS)} levels.")
    print(f"Successfully processed {total_files_processed - total_files_with_errors} metadata files.")
    if files_updated > 0:
        print(f"Updated and saved {files_updated} metadata files.")
    else:
        print(f"No metadata files were updated with new capitalizations.")
    if total_files_with_errors > 0:
        print(f"Encountered errors with {total_files_with_errors} files (load/save issues or incorrect format).")
    print(f"Total definitions capitalized across all processed files: {total_capitalized_overall}.")
    print("Definition capitalization script finished.")

if __name__ == "__main__":
    main() 