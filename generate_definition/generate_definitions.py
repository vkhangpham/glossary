import json
import os
import logging
import sys
import time
import concurrent.futures
import threading
from typing import Optional, Any, Dict, List, Tuple

# Package structure now properly configured with pyproject.toml

try:
    from generate_glossary.llm import completion
    from generate_glossary.config import get_llm_config
    from generate_glossary.utils.logger import get_logger
    # Import exceptions if needed
except ImportError as e:
    print(f"Error importing 'generate_glossary' modules: {e}")
    print("Please ensure that 'generate_glossary' is in your PYTHONPATH or structured as a package accessible from the script's location.")
    sys.exit(1)

# --- Configuration ---
BASE_DATA_PATH = "data/final" # This path is relative to the project root
RESOURCES_DIR_TEMPLATE = os.path.join(BASE_DATA_PATH, "lv{}")
RESOURCE_FILE_TEMPLATE = "lv{}_filtered_resources.json"
OUTPUT_FILE = os.path.join(BASE_DATA_PATH, "generated_definitions_output_report.json") # Changed name for clarity
METADATA_FILE_TEMPLATE = os.path.join(BASE_DATA_PATH, "lv{}", "lv{}_metadata.json")  # Correct format for lvX_metadata.json
LEVELS_TO_PROCESS = [0, 1, 2, 3] # Process all levels
# LEVELS_TO_PROCESS = [1, 2, 3] # Process all levels

# Multithreading configuration
MAX_WORKERS = 8  # Number of concurrent API calls
BATCH_SIZE = 10   # Process terms in batches of this size
BATCH_TIMEOUT = 5  # Seconds to wait between batches to avoid rate limits
REQUEST_TIMEOUT = 30  # Seconds to wait for a single API call before timing out

# Token limits
MAX_CONTEXT_TOKENS = 800000  # Safe limit to avoid hitting the 1048575 token limit
MAX_CONTEXT_CHARS = 1600000  # Approximate character count (avg 2 chars per token)

# LLM Configuration - now handled automatically by llm_simple

# Resource context configuration
MAX_RESOURCES_TO_USE = 3  # Number of top resources to include in the context
PROCESSED_CONTENT_FIELD = "processed_content"  # Field containing the processed text

# --- Logger Setup ---
logger = get_logger(__name__)

# --- Global Counters for process_term (as it's run in threads) ---
# These are for the *entire run* of the script across all levels/batches.
# They are updated by process_term.
global_successful_definitions_in_run = 0
global_failed_definitions_in_run = 0
global_counter_lock = threading.Lock()

# --- Helper Functions ---
def load_json_file(file_path: str) -> Optional[Any]:
    """Loads a JSON file."""
    # Adjust file paths to be relative to project root if script is run from project root
    # or keep as is if script is run from its own directory and BASE_DATA_PATH is correct
    absolute_file_path = os.path.join(project_root, file_path) if not os.path.isabs(file_path) else file_path
    logger.info(f"Loading JSON file: {absolute_file_path}")
    try:
        with open(absolute_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {absolute_file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {absolute_file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {absolute_file_path}: {e}")
        return None

def save_json_file(data: Any, file_path: str) -> bool:
    """Saves data to a JSON file."""
    absolute_file_path = os.path.join(project_root, file_path) if not os.path.isabs(file_path) else file_path
    logger.info(f"Saving data to JSON file: {absolute_file_path}")
    try:
        os.makedirs(os.path.dirname(absolute_file_path), exist_ok=True)
        with open(absolute_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved data to {absolute_file_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving JSON to {absolute_file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving to {absolute_file_path}: {e}")
        return False

def extract_terms_from_resources(resource_data: Any) -> List[str]:
    """Extract terms from resource data.
    This function should be adapted based on the structure of your resource files.
    """
    terms = []
    
    # Example: If resource_data is a list of items with a 'term' field
    if isinstance(resource_data, list):
        for item in resource_data:
            if isinstance(item, dict) and 'term' in item:
                terms.append(item['term'])
    
    # Example: If resource_data has a 'terms' or 'concepts' key with a list of terms
    elif isinstance(resource_data, dict):
        # Try different possible keys that might contain terms
        for key in ['terms', 'concepts', 'keywords', 'topics']:
            if key in resource_data and isinstance(resource_data[key], list):
                terms.extend([t for t in resource_data[key] if isinstance(t, str)])
        
        # If there are no predefined term lists, extract keys as terms
        # This should be the primary way terms are identified from resource files
        if not terms and isinstance(resource_data, dict) and len(resource_data) > 0:
            terms = list(resource_data.keys())
    
    # Remove duplicates while preserving order
    unique_terms = []
    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)
    
    return unique_terms

def update_metadata_file(metadata_file_template: str, level_num: int, definitions_to_add: Dict[str, str]) -> bool:
    if not definitions_to_add:
        logger.info(f"No new definitions to add to metadata for lv{level_num}.")
        return True # Nothing to do, so considered successful in a way

    actual_metadata_path_str = metadata_file_template.format(level_num, level_num)
    logger.info(f"Attempting to update metadata file: {actual_metadata_path_str} with {len(definitions_to_add)} new definitions.")
    
    metadata = load_json_file(actual_metadata_path_str)
    if metadata is None: # File not found or failed to load
        logger.error(f"Original metadata file {actual_metadata_path_str} could not be loaded. Cannot update.")
        # Optionally, create a new one if that's desired, but safer to assume it should exist
        # metadata = {} 
        # logger.warning(f"Metadata file {actual_metadata_path_str} not found or empty. New definitions will form a new file content.")
        return False


    terms_updated_count = 0
    if isinstance(metadata, dict):
        for term, definition in definitions_to_add.items():
            if term in metadata:
                if not metadata[term].get("definition") or not metadata[term]["definition"].strip(): # Only update if missing/empty
                    metadata[term]["definition"] = definition
                    terms_updated_count += 1
                    logger.debug(f"Added definition for existing term (was missing): {term} in lv{level_num}")
                else:
                    logger.info(f"Term '{term}' in lv{level_num} already has a definition. Skipping update for this term.")
            else:
                # This case should ideally not happen if terms are sourced from metadata
                logger.warning(f"Term '{term}' (to be updated) not originally found in lv{level_num} metadata. Adding as new entry.")
                metadata[term] = {"definition": definition} # Create a new entry for the term
                terms_updated_count += 1
    
    if terms_updated_count > 0:
        logger.info(f"Saving {terms_updated_count} new/updated definitions to metadata file: {actual_metadata_path_str}")
        return save_json_file(metadata, actual_metadata_path_str)
    else:
        logger.info(f"No definitions were actually added or updated in metadata for lv{level_num} (e.g., all already existed).")
        return True # No changes made, but process was fine.

def truncate_context(context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate context to a safe size to avoid token limit errors.
    
    Args:
        context: The full context string
        max_chars: Maximum number of characters to keep
        
    Returns:
        Truncated context with a note about truncation
    """
    if len(context) <= max_chars:
        return context
    
    logger.warning(f"Context is too large ({len(context)} chars). Truncating to {max_chars} chars.")
    truncated = context[:max_chars]
    
    # Try to truncate at a reasonable spot like the end of a JSON object
    reasonable_break_points = [",", "}", "]"]
    for char in reasonable_break_points:
        last_pos = truncated.rfind(char)
        if last_pos > max_chars * 0.9:  # Only use if we found a break point in the last 10% of the truncated text
            truncated = truncated[:last_pos+1]
            break
    
    truncated += "\n\n... [Context was truncated due to size limitations] ..."
    return truncated

def generate_definition_for_term(term: str, context: str) -> Optional[str]:
    """Generates a definition for a given term using the LLM and context."""
    # Truncate context to avoid token limit errors
    safe_context = truncate_context(context)
    
    prompt = (
        f"Generate a formal, academic definition for the research topic or concept: '{term}'.\n\n"
        f"Important requirements:\n"
        f"1. Start your definition with '{term} is' followed by a concise explanation (1-2 sentences total).\n"
        f"2. Focus on defining it as a research field, academic concept, or methodological approach.\n"
        f"3. Include what this concept investigates, its key characteristics, and its significance.\n"
        f"4. Use formal academic language appropriate for a scholarly glossary.\n"
        f"5. DO NOT prefix your answer with phrases like 'Based on the context...' or 'According to...' - start directly with '{term} is'.\n\n"
        f"Context information (use this to inform your definition):\n{safe_context}\n\n"
        f"Definition (begin with '{term} is'):"
    )
    
    logger.debug(f"Generating definition for '{term}' with improved prompt...")
    
    try:
        config = get_llm_config()
        messages = [{"role": "user", "content": prompt}]
        result = completion(messages, tier="budget")  # Use budget tier for definition generation
        definition = result.strip() if result else None
        
        if definition:
            # Ensure definition starts with "term is" 
            if not definition.lower().startswith(term.lower() + " is"):
                definition = f"{term} is " + definition
                
            # Remove common prefixes that might be inserted by the LLM
            prefixes_to_remove = [
                "Based on the context",
                "Based on the provided context",
                "According to the context",
                "From the context",
                "In the context",
                "The context indicates",
                "As per the context",
            ]
            
            for prefix in prefixes_to_remove:
                if definition.startswith(prefix):
                    # Find where the actual definition starts
                    parts = definition.split(',', 1)
                    if len(parts) > 1:
                        definition = parts[1].strip()
                        # Add back the term is prefix
                        definition = f"{term} is " + definition
            
            logger.info(f"Successfully generated definition for '{term}'")
        else:
            logger.warning(f"LLM returned an empty definition for '{term}'.")
        return definition
    except LLMError as e:
        logger.error(f"LLMError generating definition for '{term}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error generating definition for '{term}': {e}")
        return None

def process_term(term: str, context: str, batch_num: int, level_key: str) -> Tuple[str, Optional[str]]:
    """Process a single term in a thread-safe manner.
    
    Args:
        term: The term to process
        context: The context to use for generation
        batch_num: The batch number this term belongs to (for logging)
        
    Returns:
        Tuple of (term, definition or None if failed)
    """
    global global_successful_definitions_in_run, global_failed_definitions_in_run, global_counter_lock
    
    try:
        logger.info(f"[{level_key} Batch {batch_num}] Processing term: '{term}'")
        start_time = time.time()
        
        # Generate definition with timeout
        definition = generate_definition_for_term(term, context)
        
        # Update counters in a thread-safe manner
        with global_counter_lock:
            if definition:
                global_successful_definitions_in_run += 1
                logger.info(f"[{level_key} Batch {batch_num}] Completed term '{term}' in {time.time() - start_time:.2f}s. (Overall success: {global_successful_definitions_in_run}, fail: {global_failed_definitions_in_run})")
            else:
                global_failed_definitions_in_run += 1
                logger.warning(f"[{level_key} Batch {batch_num}] Failed definition for '{term}' in {time.time() - start_time:.2f}s. (Overall success: {global_successful_definitions_in_run}, fail: {global_failed_definitions_in_run})")
                
        # Return the term and its definition (or None if generation failed)
        return (term, definition)
    except Exception as e:
        # Update failed counter in a thread-safe manner
        with global_counter_lock:
            global_failed_definitions_in_run += 1
            logger.error(f"[{level_key} Batch {batch_num}] Error processing term '{term}': {e} - Success: {global_successful_definitions_in_run}, Failed: {global_failed_definitions_in_run}")
        return (term, None)

def extract_processed_content(resource_data: Any, max_resources: int = MAX_RESOURCES_TO_USE) -> Optional[str]:
    """Extract processed_content from the top N resources ranked by score.
    
    Args:
        resource_data: The loaded resource data (dictionary or list)
        max_resources: Maximum number of resources to include
        
    Returns:
        Concatenated processed_content from top resources
    """
    resources_with_content = []
    
    # Collect resources with score and non-empty processed_content
    # Handle case where resource_data is a list of resources
    if isinstance(resource_data, list):
        for resource in resource_data:
            if isinstance(resource, dict) and PROCESSED_CONTENT_FIELD in resource:
                content = resource.get(PROCESSED_CONTENT_FIELD, "")
                # Only include resources with non-empty processed content
                if content and isinstance(content, str) and content.strip():
                    # Get the score, defaulting to 0 if not present
                    score = resource.get("score", 0)
                    resources_with_content.append({
                        "content": content,
                        "score": score
                    })
    
    # Handle case where resource_data is a dictionary with fields that might contain resources
    elif isinstance(resource_data, dict):
        # Try to find arrays of resources in the dictionary
        for key, value in resource_data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and PROCESSED_CONTENT_FIELD in item:
                        content = item.get(PROCESSED_CONTENT_FIELD, "")
                        # Only include resources with non-empty processed content
                        if content and isinstance(content, str) and content.strip():
                            # Get the score, defaulting to 0 if not present
                            score = item.get("score", 0)
                            resources_with_content.append({
                                "content": content,
                                "score": score
                            })
            # Check if the value itself has the processed_content field
            elif isinstance(value, dict) and PROCESSED_CONTENT_FIELD in value:
                content = value.get(PROCESSED_CONTENT_FIELD, "")
                # Only include resources with non-empty processed content
                if content and isinstance(content, str) and content.strip():
                    # Get the score, defaulting to 0 if not present
                    score = value.get("score", 0)
                    resources_with_content.append({
                        "content": content,
                        "score": score
                    })
    
    if not resources_with_content:
        logger.warning(f"No resources with non-empty '{PROCESSED_CONTENT_FIELD}' found in resource data.")
        return None
    
    # Sort resources by score (higher score first)
    resources_with_content.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take the top N resources
    top_resources = resources_with_content[:max_resources]
    
    # Extract just the content from the sorted resources
    processed_contents = [r["content"] for r in top_resources]
    
    # Join the processed contents with separators
    result = "\n\n---\n\n".join(processed_contents)
    logger.info(f"Extracted processed content from {len(processed_contents)} resources (ranked by score, total length: {len(result)} chars).")
    
    return result

def get_parent_based_context(term: str, term_attributes: Dict[str, Any], current_level_metadata: Dict[str, Any], level_key: str) -> Optional[str]:
    """
    Constructs context from parent definitions for a given term.
    Args:
        term: The term for which to find parent context.
        term_attributes: The metadata attributes of the term itself.
        current_level_metadata: The entire metadata dictionary for the current level.
        level_key: The current level being processed (e.g., "lv1") for logging.
    Returns:
        A string containing concatenated parent definitions, or None if no suitable parent context.
    """
    parent_terms = term_attributes.get("parents")
    if not isinstance(parent_terms, list) or not parent_terms:
        logger.debug(f"[{level_key}] No valid parent list for term '{term}'.")
        return None

    parent_definitions = []
    logger.debug(f"[{level_key}] Looking for parent definitions for '{term}': {parent_terms}")
    for parent_term_name in parent_terms:
        parent_meta = current_level_metadata.get(parent_term_name)
        if parent_meta:
            parent_definition = parent_meta.get("definition")
            if parent_definition and parent_definition.strip():
                parent_definitions.append(f"Definition of parent term '{parent_term_name}':\n{parent_definition.strip()}")
            else:
                logger.debug(f"[{level_key}] Parent '{parent_term_name}' for '{term}' has no definition.")
        else:
            logger.debug(f"[{level_key}] Parent '{parent_term_name}' for '{term}' not in metadata.")

    if not parent_definitions:
        logger.info(f"[{level_key}] No definitions found for any parents of '{term}'.")
        return None

    context_str = "\n\n---\n\n".join(parent_definitions)
    preamble = f"Context for '{term}' derived from its parent definitions in {level_key}:\n"
    full_context = preamble + context_str
    logger.info(f"[{level_key}] Built parent-based context for '{term}' (length: {len(full_context)} chars).")
    return full_context

# --- Main Processing Function ---
def main():
    """Main function to orchestrate the definition generation process."""
    global global_successful_definitions_in_run, global_failed_definitions_in_run, global_counter_lock
    global_successful_definitions_in_run = 0 # Reset for this run
    global_failed_definitions_in_run = 0     # Reset for this run
    
    run_start_time = time.time()
    logger.info("Starting glossary definition generation process...")
    logger.info(f"Concurrency: MAX_WORKERS={MAX_WORKERS}, BATCH_SIZE={BATCH_SIZE}")

    # LLM configuration is now handled automatically by llm_simple

    # This report stores definitions generated *in this specific run*
    run_generated_definitions_report = {} 

    for level_num in LEVELS_TO_PROCESS:
        level_key = f"lv{level_num}"
        level_start_time = time.time()
        logger.info(f"--- Processing Level: {level_key} ---")

        resource_file_path = os.path.join(RESOURCES_DIR_TEMPLATE.format(level_num), RESOURCE_FILE_TEMPLATE.format(level_num))
        resource_map_for_level = load_json_file(resource_file_path) or {} # Term -> List of resource objects
        
        metadata_file_path = METADATA_FILE_TEMPLATE.format(level_num, level_num)
        current_level_metadata = load_json_file(metadata_file_path) or {}
        if not current_level_metadata:
            logger.warning(f"Metadata for {level_key} not loaded or empty. Skipping this level.")
            continue

        terms_needing_defs_in_level = []
        for term, attributes in current_level_metadata.items():
            if not (attributes.get('definition') and attributes['definition'].strip()):
                terms_needing_defs_in_level.append(term)
        
        if not terms_needing_defs_in_level:
            logger.info(f"No terms in {level_key} metadata need definitions. Skipping generation for this level.")
            continue
        logger.info(f"Found {len(terms_needing_defs_in_level)} terms in {level_key} metadata needing definitions.")

        run_generated_definitions_report[level_key] = []
        definitions_to_update_in_metadata_for_level = {}
        
        terms_for_batch_processing = []
        term_to_context_cache = {} # Cache context for terms to pass to threads

        # --- Context Sourcing Loop ---
        for term in terms_needing_defs_in_level:
            final_context_for_term = None
            context_source_type = "none"

            # 1. Try Resource-based context
            term_specific_resources = resource_map_for_level.get(term) # This is a list for the specific term
            if term_specific_resources: # If the term exists as a key and has some resource data
                logger.debug(f"[{level_key}] Attempting resource context for '{term}'.")
                resource_context = extract_processed_content(term_specific_resources, MAX_RESOURCES_TO_USE)
                if resource_context:
                    final_context_for_term = resource_context
                    context_source_type = "resource"
                    logger.info(f"[{level_key}] Using RESOURCE context for '{term}'.")
                else:
                    logger.info(f"[{level_key}] No usable resource content for '{term}'. Trying parent context.")
            else:
                logger.info(f"[{level_key}] No entry/resources for '{term}' in resource map. Trying parent context.")

            # 2. Try Parent-based context if resource context failed or wasn't applicable
            if not final_context_for_term:
                term_attributes = current_level_metadata.get(term)
                if term_attributes: # Should always be true if term is from current_level_metadata keys
                    logger.debug(f"[{level_key}] Attempting parent context for '{term}'.")
                    parent_context = get_parent_based_context(term, term_attributes, current_level_metadata, level_key)
                    if parent_context:
                        final_context_for_term = parent_context
                        context_source_type = "parent"
                        logger.info(f"[{level_key}] Using PARENT context for '{term}'.")
                    else:
                        logger.warning(f"[{level_key}] Failed to get parent context for '{term}'.")
                else: # Should not happen
                     logger.error(f"[{level_key}] Term '{term}' (needing def) not found in its own metadata. Skipping.")


            if final_context_for_term:
                terms_for_batch_processing.append(term)
                term_to_context_cache[term] = final_context_for_term
            else:
                logger.warning(f"[{level_key}] No context (resource or parent) for term '{term}'. Cannot generate definition.")
                # This term failed to get context, effectively a failure for this run for this term
                with global_counter_lock: # Should this be a separate per-level counter for "no context found"?
                    global_failed_definitions_in_run += 1 


        if not terms_for_batch_processing:
            logger.info(f"No terms in {level_key} have sufficient context for definition generation. Skipping batch processing.")
            # Log level summary here if needed, but main summary is at the end of level loop
        else:
            logger.info(f"Will attempt to generate definitions for {len(terms_for_batch_processing)} terms in {level_key}.")
            # --- Batch Processing Loop ---
            batch_num_for_level = 0
            for i in range(0, len(terms_for_batch_processing), BATCH_SIZE):
                batch_num_for_level += 1
                current_batch_terms = terms_for_batch_processing[i:i+BATCH_SIZE]
                batch_start_time = time.time()
                logger.info(f"[{level_key}] Starting Batch {batch_num_for_level}/{ (len(terms_for_batch_processing) + BATCH_SIZE - 1)//BATCH_SIZE } with {len(current_batch_terms)} terms.")
                
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix=f"{level_key}_DefGen") as executor:
                    for term_in_batch in current_batch_terms:
                        context = term_to_context_cache.get(term_in_batch)
                        if not context: # Should not happen if logic is correct
                            logger.error(f"[{level_key}] CRITICAL: Context for term '{term_in_batch}' missing from cache. Skipping.")
                            with global_counter_lock: global_failed_definitions_in_run += 1
                            continue
                        futures.append(executor.submit(process_term, term_in_batch, context, batch_num_for_level, level_key))
                
                # Collect results from this batch
                for future in concurrent.futures.as_completed(futures):
                    try:
                        processed_term, new_definition = future.result()
                        if new_definition:
                            definitions_to_update_in_metadata_for_level[processed_term] = new_definition
                            run_generated_definitions_report[level_key].append({"term": processed_term, "definition": new_definition, "level": level_key})
                        # Failures are already counted by process_term's global counter
                    except Exception as exc:
                        logger.error(f"[{level_key}] Batch {batch_num_for_level} - Exception collecting future result: {exc}")
                        with global_counter_lock: global_failed_definitions_in_run += 1 # Count this as a failure too

                logger.info(f"[{level_key}] Batch {batch_num_for_level} finished in {time.time() - batch_start_time:.2f}s.")
                if i + BATCH_SIZE < len(terms_for_batch_processing):
                    logger.info(f"[{level_key}] Waiting {BATCH_TIMEOUT}s before next batch...")
                    time.sleep(BATCH_TIMEOUT)

        # --- After all batches for the level ---
        if definitions_to_update_in_metadata_for_level:
            logger.info(f"[{level_key}] Attempting to update metadata with {len(definitions_to_update_in_metadata_for_level)} newly generated definitions.")
            update_metadata_file(METADATA_FILE_TEMPLATE, level_num, definitions_to_update_in_metadata_for_level)
        else:
            logger.info(f"[{level_key}] No new definitions were successfully generated in this run to update metadata.")

        logger.info(f"--- Finished Level {level_key} in {time.time() - level_start_time:.2f}s ---")

    # --- After all levels ---
    logger.info(f"--- Overall Summary for This Run ---")
    logger.info(f"Total script duration: {time.time() - run_start_time:.2f}s")
    logger.info(f"Total definitions successfully generated (across all levels): {global_successful_definitions_in_run}")
    logger.info(f"Total terms failed or skipped (no context/LLM error): {global_failed_definitions_in_run}")
    
    if save_json_file(run_generated_definitions_report, OUTPUT_FILE):
        logger.info(f"Report of definitions generated in this run saved to: {OUTPUT_FILE}")
    else:
        logger.error(f"Failed to save report of definitions generated in this run to {OUTPUT_FILE}")

    logger.info("Glossary definition generation process finished for this run.")

# --- Script Entry Point ---
if __name__ == "__main__":
    # For type hinting Optional and Any
    main() 