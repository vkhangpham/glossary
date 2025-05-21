import json
import os
import logging
import sys
import time
import concurrent.futures
import threading
from typing import Optional, Any, Dict, List, Tuple

# Ensure the script can find the generate_glossary module
# This assumes the script is run from the root of the 'glossary' project
# and 'generate_glossary' is a package in that root.
# If your project structure is different, you might need to adjust sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Go up one level from generate_definition to glossary
sys.path.insert(0, project_root)

try:
    from generate_glossary.utils.llm import get_llm, LLMError, LLMConfigError, GEMINI_MODELS
    # Assuming logger and exceptions are in the same utils directory and handled by llm.py
except ImportError as e:
    print(f"Error importing 'generate_glossary' modules: {e}")
    print("Please ensure that 'generate_glossary' is in your PYTHONPATH or structured as a package accessible from the script's location.")
    print(f"Attempted to add to sys.path: {project_root}") # Added for debugging
    sys.exit(1)

# --- Configuration ---
BASE_DATA_PATH = "data/final" # This path is relative to the project root
RESOURCES_DIR_TEMPLATE = os.path.join(BASE_DATA_PATH, "lv{}")
RESOURCE_FILE_TEMPLATE = "lv{}_filtered_resources.json"
OUTPUT_FILE = os.path.join(BASE_DATA_PATH, "generated_definitions.json")
METADATA_FILE_TEMPLATE = os.path.join(BASE_DATA_PATH, "lv{}", "lv{}_metadata.json")  # Correct format for lvX_metadata.json
LEVELS_TO_PROCESS = [3] # Process all levels
# LEVELS_TO_PROCESS = [1, 2, 3] # Process all levels

# Multithreading configuration
MAX_WORKERS = 8  # Number of concurrent API calls
BATCH_SIZE = 10   # Process terms in batches of this size
BATCH_TIMEOUT = 5  # Seconds to wait between batches to avoid rate limits
REQUEST_TIMEOUT = 30  # Seconds to wait for a single API call before timing out

# Token limits
MAX_CONTEXT_TOKENS = 800000  # Safe limit to avoid hitting the 1048575 token limit
MAX_CONTEXT_CHARS = 1600000  # Approximate character count (avg 2 chars per token)

# LLM Configuration
LLM_PROVIDER = "gemini"
LLM_MODEL_TIER = "default"  # Corresponds to Gemini 2.5 Flash in llm.py

# Resource context configuration
MAX_RESOURCES_TO_USE = 3  # Number of top resources to include in the context
PROCESSED_CONTENT_FIELD = "processed_content"  # Field containing the processed text

# --- Logger Setup ---
# Using a basic logger. If generate_glossary.utils.logger.setup_logger is available
# and preferred, this can be changed.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Thread-safe counters for tracking progress
successful_definitions = 0
failed_definitions = 0
counter_lock = threading.Lock()

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
        if not terms and len(resource_data) > 0:
            terms = list(resource_data.keys())
    
    # Remove duplicates while preserving order
    unique_terms = []
    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)
    
    return unique_terms

def update_metadata_file(metadata_path: str, level_num: int, definitions: Dict[str, str]) -> bool:
    """Update the metadata file with generated definitions.
    
    Args:
        metadata_path: Template path for metadata file
        level_num: Level number (0, 1, 2, 3)
        definitions: Dictionary of term -> definition pairs
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Format the metadata path with the level number
        actual_metadata_path = metadata_path.format(level_num, level_num)
        logger.info(f"Updating metadata file: {actual_metadata_path}")
        
        # Load existing metadata file
        metadata = {}
        absolute_path = os.path.join(project_root, actual_metadata_path)
        if os.path.exists(absolute_path):
            metadata = load_json_file(actual_metadata_path) or {}
            logger.info(f"Loaded existing metadata file with {len(metadata)} terms")
        else:
            logger.warning(f"Metadata file {actual_metadata_path} not found. Will create new file.")
        
        # Update the metadata with new definitions
        terms_updated = 0
        if isinstance(metadata, dict):
            for term, definition in definitions.items():
                if term in metadata:
                    # Add the definition directly to the term's object
                    metadata[term]["definition"] = definition
                    terms_updated += 1
                    logger.debug(f"Updated definition for existing term: {term}")
                else:
                    # If the term doesn't exist, log a warning
                    logger.warning(f"Term '{term}' not found in metadata file. Skipping.")
        
        # Save the updated metadata back to the original file
        if terms_updated > 0:
            logger.info(f"Updated {terms_updated} term definitions in metadata")
            return save_json_file(metadata, actual_metadata_path)
        else:
            logger.warning(f"No terms were updated in metadata file")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata file {metadata_path}: {e}")
        return False

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

def generate_definition_for_term(llm_client: Any, term: str, context: str) -> Optional[str]:
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
        result = llm_client.infer(prompt=prompt)
        definition = result.text.strip() if result and result.text else None
        
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

def process_term(llm_client: Any, term: str, context: str, batch_num: int) -> Tuple[str, Optional[str]]:
    """Process a single term in a thread-safe manner.
    
    Args:
        llm_client: The LLM client to use
        term: The term to process
        context: The context to use for generation
        batch_num: The batch number this term belongs to (for logging)
        
    Returns:
        Tuple of (term, definition or None if failed)
    """
    global successful_definitions, failed_definitions
    
    try:
        logger.info(f"[Batch {batch_num}] Processing term: '{term}'")
        start_time = time.time()
        
        # Generate definition with timeout
        definition = generate_definition_for_term(llm_client, term, context)
        
        # Update counters in a thread-safe manner
        with counter_lock:
            if definition:
                successful_definitions += 1
                logger.info(f"[Batch {batch_num}] Completed term '{term}' in {time.time() - start_time:.2f}s - Success: {successful_definitions}, Failed: {failed_definitions}")
            else:
                failed_definitions += 1
                logger.warning(f"[Batch {batch_num}] Failed to generate definition for '{term}' - Success: {successful_definitions}, Failed: {failed_definitions}")
                
        # Return the term and its definition (or None if generation failed)
        return (term, definition)
    except Exception as e:
        # Update failed counter in a thread-safe manner
        with counter_lock:
            failed_definitions += 1
            logger.error(f"[Batch {batch_num}] Error processing term '{term}': {e} - Success: {successful_definitions}, Failed: {failed_definitions}")
        return (term, None)

def extract_processed_content(resource_data: Any, max_resources: int = MAX_RESOURCES_TO_USE) -> str:
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
        return "No processed content available in the resources."
    
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

# --- Main Processing Function ---
def main():
    """Main function to orchestrate the definition generation process."""
    global successful_definitions, failed_definitions
    
    start_time = time.time()
    logger.info("Starting glossary definition generation process with multithreading...")
    
    logger.info("Reminder: Ensure GOOGLE_API_KEY, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and GOOGLE_GENAI_USE_VERTEXAI environment variables are set.")
    logger.info(f"Using concurrency settings: MAX_WORKERS={MAX_WORKERS}, BATCH_SIZE={BATCH_SIZE}, BATCH_TIMEOUT={BATCH_TIMEOUT}s")

    try:
        logger.info(f"Initializing LLM ({LLM_PROVIDER} provider, model tier: {GEMINI_MODELS[LLM_MODEL_TIER]})...")
        llm = get_llm(provider=LLM_PROVIDER, model=GEMINI_MODELS[LLM_MODEL_TIER])
        logger.info("LLM initialized successfully.")
    except LLMConfigError as e:
        logger.error(f"LLM Configuration Error: {e}. Please check your API keys and environment setup.")
        return
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return

    all_definitions = {}
    total_terms_processed = 0
    total_batch_count = 0

    for level_num in LEVELS_TO_PROCESS:
        level_key = f"lv{level_num}"
        level_start_time = time.time()
        logger.info(f"--- Processing Level: {level_key} ---")

        # 1. Read resource file for context
        resource_dir = RESOURCES_DIR_TEMPLATE.format(level_num)
        resource_file_path = os.path.join(resource_dir, RESOURCE_FILE_TEMPLATE.format(level_num))
            
        # Load resource data for context
        resource_data = load_json_file(resource_file_path)
        if resource_data is None:
            logger.warning(f"Could not load resource data for {level_key}. Skipping this level.")
            continue
            
        # Extract processed content from the top resources instead of using the full JSON
        try:
            context_str = extract_processed_content(resource_data, MAX_RESOURCES_TO_USE)
            context_size = len(context_str)
            logger.info(f"Successfully prepared context from top {MAX_RESOURCES_TO_USE} resources for {level_key} (length: {context_size:,} chars).")
            
            if context_size > MAX_CONTEXT_CHARS:
                logger.warning(f"Context size ({context_size:,} chars) exceeds recommended limit ({MAX_CONTEXT_CHARS:,}). Will truncate during processing.")
        except Exception as e:
            logger.error(f"Error extracting processed content for {level_key}: {e}")
            continue

        # 2. Extract terms from resource data
        terms_to_process = extract_terms_from_resources(resource_data)
        if not terms_to_process:
            logger.warning(f"No terms found in resource data for {level_key}. Skipping this level.")
            continue
            
        logger.info(f"Found {len(terms_to_process)} terms for {level_key}.")
        all_definitions[level_key] = []
        level_definitions = {}
        
        # Reset counters for this level
        successful_definitions = 0
        failed_definitions = 0
        
        # 3. Process terms in batches using ThreadPoolExecutor
        batch_number = 0
        for i in range(0, len(terms_to_process), BATCH_SIZE):
            batch_number += 1
            total_batch_count += 1
            batch_terms = terms_to_process[i:i+BATCH_SIZE]
            batch_start_time = time.time()
            
            logger.info(f"Processing batch {batch_number} with {len(batch_terms)} terms...")
            futures = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit tasks for all terms in this batch
                for term in batch_terms:
                    future = executor.submit(process_term, llm, term, context_str, batch_number)
                    futures.append(future)
                
                # Collect results as they complete
                batch_results = {}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        term, definition = future.result()
                        if definition:
                            batch_results[term] = definition
                    except Exception as e:
                        logger.error(f"Error collecting result from future: {e}")
            
            # Add batch results to level definitions
            level_definitions.update(batch_results)
            
            # Convert batch results to the format for all_definitions
            for term, definition in batch_results.items():
                all_definitions[level_key].append({"term": term, "definition": definition})
            
            # Log batch completion
            batch_duration = time.time() - batch_start_time
            logger.info(f"Completed batch {batch_number} in {batch_duration:.2f}s - Added {len(batch_results)} definitions")
            total_terms_processed += len(batch_terms)
            
            # Wait between batches to avoid rate limiting, unless it's the last batch
            if i + BATCH_SIZE < len(terms_to_process):
                logger.info(f"Waiting {BATCH_TIMEOUT} seconds before next batch...")
                time.sleep(BATCH_TIMEOUT)

        # 4. Update metadata file with definitions
        if level_definitions:
            logger.info(f"Updating metadata file for {level_key} with {len(level_definitions)} definitions...")
            if update_metadata_file(METADATA_FILE_TEMPLATE, level_num, level_definitions):
                logger.info(f"Successfully updated metadata file for {level_key}.")
            else:
                logger.error(f"Failed to update metadata file for {level_key}.")

        level_duration = time.time() - level_start_time
        logger.info(f"Finished processing {level_key} in {level_duration:.2f}s. Success: {successful_definitions}, Failed: {failed_definitions}")

    total_duration = time.time() - start_time
    logger.info(f"--- Overall Summary ---")
    logger.info(f"Processed a total of {total_terms_processed} terms across {len(LEVELS_TO_PROCESS)} levels in {total_duration:.2f}s")
    logger.info(f"Total batches: {total_batch_count}, Total successful definitions: {successful_definitions}, Total failed: {failed_definitions}")
    
    if save_json_file(all_definitions, OUTPUT_FILE):
        logger.info(f"All definitions saved to {OUTPUT_FILE}")
    else:
        logger.error(f"Failed to save definitions to {OUTPUT_FILE}")

    logger.info("Glossary definition generation process finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    # For type hinting Optional and Any
    main() 