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
except ImportError as e:
    print(f"Error importing 'generate_glossary' modules: {e}")
    print("Please ensure that 'generate_glossary' is in your PYTHONPATH or structured as a package accessible from the script's location.")
    sys.exit(1)

# --- Configuration ---
BASE_DATA_PATH = "data/final"
RESOURCES_DIR_TEMPLATE = os.path.join(BASE_DATA_PATH, "lv{}")
RESOURCE_FILE_TEMPLATE = "lv{}_filtered_resources.json"
OUTPUT_FILE = os.path.join(BASE_DATA_PATH, "generated_definitions_split_terms.json")
METADATA_FILE_TEMPLATE = os.path.join(BASE_DATA_PATH, "lv{}", "lv{}_metadata.json")
LEVELS_TO_PROCESS = [3]  # Process levels that have split terms

# Multithreading configuration
MAX_WORKERS = 4  # Reduced to be more conservative with API calls
BATCH_SIZE = 5   # Smaller batches for better control
BATCH_TIMEOUT = 10  # Longer timeout between batches
REQUEST_TIMEOUT = 30

# Token limits
MAX_CONTEXT_TOKENS = 800000
MAX_CONTEXT_CHARS = 1600000

# LLM Configuration - now handled automatically by llm_simple

# Resource context configuration
MAX_RESOURCES_TO_USE = 3
PROCESSED_CONTENT_FIELD = "processed_content"

# Only process split terms (terms containing parentheses)
ONLY_SPLIT_TERMS = True

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- Debug settings for specific term check ---
DEBUG_TERMS_ONLY = False # Set to True to only process specific terms
DEBUG_TERM_LIST = [] # ["visualization (data visualization)", "product design (industrial design)"]
# --- End Debug settings ---

# Thread-safe counters
successful_definitions = 0
failed_definitions = 0
counter_lock = threading.Lock()

def load_json_file(file_path: str) -> Optional[Any]:
    """Loads a JSON file."""
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

def extract_split_terms_from_resources(resource_data: Any) -> List[str]:
    """Extract split terms (terms with parentheses) from resource data keys."""
    if not isinstance(resource_data, dict):
        logger.warning("Resource data is not a dictionary. Cannot extract terms.")
        return []
    
    split_terms = []
    for term in resource_data.keys():
        # Check if this is a split term (contains parentheses)
        if ONLY_SPLIT_TERMS and "(" in term and ")" in term:
            split_terms.append(term)
        elif not ONLY_SPLIT_TERMS:
            split_terms.append(term)
    
    logger.info(f"Found {len(split_terms)} split terms in resource data")
    return split_terms

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using simple word overlap.
    Returns a value between 0 and 1, where 1 means identical.
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity (could be enhanced with more sophisticated methods)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def should_exclude_resource_context(term: str, metadata: Dict, resource_data: Dict) -> bool:
    """
    Determine whether to exclude resource context for a split term.
    Only excludes context if the term is a split term AND has very similar 
    resource context to other split terms from the same original term.
    
    Args:
        term: The term being processed
        metadata: Metadata for the term including split information
        resource_data: Resource data dictionary
        
    Returns:
        True if resource context should be excluded, False otherwise
    """
    # Only consider split terms (terms with parentheses)
    if not ("(" in term and ")" in term):
        return False
    
    # Extract the original term (everything before the first parenthesis)
    original_term = term.split(" (")[0]
    
    # Find all other split terms that share the same original term
    related_split_terms = []
    for candidate_term in resource_data.keys():
        if (candidate_term != term and 
            "(" in candidate_term and ")" in candidate_term and 
            candidate_term.startswith(original_term + " (")):
            related_split_terms.append(candidate_term)
    
    # If no related split terms, resource context is unique, so keep it
    if not related_split_terms:
        return False
    
    # Get resource context for current term
    current_context = extract_context_for_specific_term(resource_data, term, MAX_RESOURCES_TO_USE)
    
    # Check similarity with related split terms
    high_similarity_count = 0
    similarity_threshold = 0.8  # 80% similarity threshold
    
    for related_term in related_split_terms:
        related_context = extract_context_for_specific_term(resource_data, related_term, MAX_RESOURCES_TO_USE)
        similarity = calculate_text_similarity(current_context, related_context)
        
        if similarity >= similarity_threshold:
            high_similarity_count += 1
            logger.info(f"High similarity ({similarity:.2f}) detected between '{term}' and '{related_term}' contexts")
    
    # Exclude resource context if we found high similarity with other split terms
    if high_similarity_count > 0:
        logger.info(f"Excluding resource context for split term '{term}' due to {high_similarity_count} highly similar contexts")
        return True
    else:
        logger.info(f"Keeping resource context for split term '{term}' - contexts are sufficiently distinct")
        return False

def extract_context_for_specific_term(resource_data: Dict, term: str, max_resources: int = MAX_RESOURCES_TO_USE) -> str:
    """Extract processed_content specifically for a given term from its resources."""
    if term not in resource_data:
        logger.warning(f"Term '{term}' not found in resource data")
        return f"No specific resources found for '{term}'"
    
    term_resources = resource_data[term]
    if not isinstance(term_resources, list):
        logger.warning(f"Resources for term '{term}' are not in list format")
        return f"Invalid resource format for '{term}'"
    
    resources_with_content = []
    
    for resource in term_resources:
        if isinstance(resource, dict) and PROCESSED_CONTENT_FIELD in resource:
            content = resource.get(PROCESSED_CONTENT_FIELD, "")
            if content and isinstance(content, str) and content.strip():
                score = resource.get("score", 0)
                resources_with_content.append({
                    "content": content,
                    "score": score
                })
    
    if not resources_with_content:
        logger.warning(f"No resources with processed content found for term '{term}'")
        return f"No processed content available for '{term}'"
    
    # Sort by score (higher score first)
    resources_with_content.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Take top N resources
    top_resources = resources_with_content[:max_resources]
    processed_contents = [r["content"] for r in top_resources]
    
    result = "\n\n---\n\n".join(processed_contents)
    logger.info(f"Extracted {len(processed_contents)} resources for term '{term}' (total length: {len(result)} chars)")
    
    return result

def get_term_metadata(metadata_data: Dict, term: str) -> Dict:
    """Get metadata for a specific term including sense tag and split info."""
    if term not in metadata_data:
        return {}
    
    term_metadata = metadata_data[term]
    
    # Extract relevant information for context
    metadata_info = {
        "original_term": term_metadata.get("original_term", ""),
        "sense_tag": term_metadata.get("sense_tag", ""),
        "split_reason": term_metadata.get("split_info", {}).get("split_reason", ""),
        "parents": term_metadata.get("parents", []),
        "sources": term_metadata.get("sources", [])
    }
    
    return metadata_info

def update_metadata_file(metadata_path: str, level_num: int, definitions: Dict[str, str]) -> bool:
    """Update the metadata file with generated definitions."""
    try:
        actual_metadata_path = metadata_path.format(level_num, level_num)
        logger.info(f"Updating metadata file: {actual_metadata_path}")
        
        metadata = {}
        absolute_path = os.path.join(project_root, actual_metadata_path)
        if os.path.exists(absolute_path):
            metadata = load_json_file(actual_metadata_path) or {}
            logger.info(f"Loaded existing metadata file with {len(metadata)} terms")
        else:
            logger.warning(f"Metadata file {actual_metadata_path} not found.")
            return False
        
        terms_updated = 0
        if isinstance(metadata, dict):
            for term, definition in definitions.items():
                if term in metadata:
                    metadata[term]["definition"] = definition
                    terms_updated += 1
                    logger.debug(f"Updated definition for: {term}")
                else:
                    logger.warning(f"Term '{term}' not found in metadata file. Skipping.")
        
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
    """Truncate context to avoid token limit errors."""
    if len(context) <= max_chars:
        return context
    
    logger.warning(f"Context is too large ({len(context)} chars). Truncating to {max_chars} chars.")
    truncated = context[:max_chars]
    
    reasonable_break_points = [",", "}", "]"]
    for char in reasonable_break_points:
        last_pos = truncated.rfind(char)
        if last_pos > max_chars * 0.9:
            truncated = truncated[:last_pos+1]
            break
    
    truncated += "\n\n... [Context was truncated due to size limitations] ..."
    return truncated

def generate_definition_for_split_term(term: str, context: str, metadata: Dict, use_resource_context: bool = True) -> Optional[str]:
    """Generate a definition specifically for a split term using context and metadata."""
    # Extract sense information
    original_term = metadata.get("original_term", term.split(" (")[0] if " (" in term else term)
    sense_tag = metadata.get("sense_tag", "")
    split_reason = metadata.get("split_reason", "")
    parents = metadata.get("parents", [])
    
    prompt = (
        f"Generate a formal, academic definition for the specific research concept: '{term}'.\n\n"
        f"This term is a disambiguated sense of the original term '{original_term}'.\n"
        f"Its specific sense is: '{sense_tag.replace('_', ' ') if sense_tag else 'N/A'}'.\n"
    )
    if split_reason and split_reason != "N/A":
        prompt += f"The key distinction for this sense, established during disambiguation, is: {split_reason}\n\n"
    else:
        prompt += "\n"
    
    prompt += (
        f"Hierarchical parent concepts (if any): {', '.join(parents) if parents else 'N/A'}.\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"1. The definition MUST begin exactly with the phrase '{term} is ...' (case-sensitive if '{term}' has capitals).\n"
        f"2. The definition MUST clearly define the concept '{term}' in its SPECIFIC disambiguated sense as identified by the SENSE TAG ('{sense_tag.replace('_', ' ') if sense_tag else 'general sense'}').\n"
        f"3. If a SPLIT REASON is provided above, the definition MUST incorporate and explain this distinction to differentiate '{term}' from other meanings of '{original_term}'.\n"
        f"4. Use formal, objective, academic language suitable for a scholarly glossary.\n"
        f"5. Concisely explain what this specific concept investigates or represents, and its significance within its disambiguated academic context.\n"
        f"6. DO NOT use phrases like 'Based on the context...', 'According to the provided information...', or similar meta-commentary about the source materials.\n"
    )
    
    if use_resource_context:
        safe_context = truncate_context(context)
        prompt += f"\nSupporting context from academic resources (NOTE: This context may be general or for the original term. You MUST PRIORITIZE the SENSE TAG and SPLIT REASON provided above for disambiguation to define '{term}' in its specific sense.):\n{safe_context}\n\n"
    else:
        prompt += f"\nNOTE: No resource context is provided. Generate the definition based solely on the sense tag, split reason, and hierarchical context provided above.\n\n"
    
    prompt += f"Definition (begin with '{term} is'):"
    
    logger.debug(f"Generating definition for split term '{term}' with sense '{sense_tag}' (use_resource_context={use_resource_context})...")
    
    try:
        config = get_llm_config()
        messages = [{"role": "user", "content": prompt}]
        result = completion(messages, tier="budget")  # Use budget tier for split term definition generation
        definition = result.strip() if result else None
        
        if definition:
            # Ensure proper formatting
            if not definition.lower().startswith(term.lower() + " is"):
                definition = f"{term} is " + definition
                
            # Clean up common LLM prefixes
            prefixes_to_remove = [
                "Based on the context", "According to the context", "From the context",
                "In the context", "The context indicates", "As per the context"
            ]
            
            for prefix in prefixes_to_remove:
                if definition.startswith(prefix):
                    parts = definition.split(',', 1)
                    if len(parts) > 1:
                        definition = f"{term} is " + parts[1].strip()
            
            logger.info(f"Generated definition for split term '{term}'")
        else:
            logger.warning(f"LLM returned empty definition for '{term}'")
        
        return definition
        
    except Exception as e:
        logger.error(f"Error generating definition for '{term}': {e}")
        return None

def process_split_term(term: str, resource_data: Dict, metadata_data: Dict, batch_num: int) -> Tuple[str, Optional[str]]:
    """Process a single split term with its specific context."""
    global successful_definitions, failed_definitions
    
    try:
        logger.info(f"[Batch {batch_num}] Processing split term: '{term}'")
        start_time = time.time()
        
        # Get metadata for this term first to determine if we should exclude resource context
        metadata = get_term_metadata(metadata_data, term)
        
        # Determine whether to use resource context
        use_resource_context = not should_exclude_resource_context(term, metadata, resource_data)
        
        # Extract specific context for this term (even if we won't use it, for potential future use)
        context = extract_context_for_specific_term(resource_data, term, MAX_RESOURCES_TO_USE)
        
        # Generate definition
        definition = generate_definition_for_split_term(term, context, metadata, use_resource_context)
        
        # Update counters
        with counter_lock:
            if definition:
                successful_definitions += 1
                logger.info(f"[Batch {batch_num}] Completed '{term}' in {time.time() - start_time:.2f}s (resource_context={'excluded' if not use_resource_context else 'included'}) - Success: {successful_definitions}, Failed: {failed_definitions}")
            else:
                failed_definitions += 1
                logger.warning(f"[Batch {batch_num}] Failed '{term}' - Success: {successful_definitions}, Failed: {failed_definitions}")
                
        return (term, definition)
        
    except Exception as e:
        with counter_lock:
            failed_definitions += 1
            logger.error(f"[Batch {batch_num}] Error processing '{term}': {e} - Success: {successful_definitions}, Failed: {failed_definitions}")
        return (term, None)

def main():
    """Main function for split term definition generation."""
    global successful_definitions, failed_definitions
    
    start_time = time.time()
    logger.info("Starting split term definition generation process...")
    
    logger.info("Ensure GOOGLE_API_KEY and related environment variables are set.")
    logger.info(f"Config: MAX_WORKERS={MAX_WORKERS}, BATCH_SIZE={BATCH_SIZE}, BATCH_TIMEOUT={BATCH_TIMEOUT}s")

    # LLM initialization is now handled automatically by llm_simple
    
    all_definitions = {}
    total_terms_processed = 0

    for level_num in LEVELS_TO_PROCESS:
        level_key = f"lv{level_num}"
        level_start_time = time.time()
        logger.info(f"--- Processing Level: {level_key} ---")

        # Load resource data
        resource_dir = RESOURCES_DIR_TEMPLATE.format(level_num)
        resource_file_path = os.path.join(resource_dir, RESOURCE_FILE_TEMPLATE.format(level_num))
        
        resource_data = load_json_file(resource_file_path)
        if resource_data is None:
            logger.warning(f"Could not load resource data for {level_key}. Skipping.")
            continue
        
        # Load metadata
        metadata_file_path = METADATA_FILE_TEMPLATE.format(level_num, level_num)
        metadata_data = load_json_file(metadata_file_path)
        if metadata_data is None:
            logger.warning(f"Could not load metadata for {level_key}. Skipping.")
            continue

        # Extract split terms
        split_terms = extract_split_terms_from_resources(resource_data)
        if not split_terms:
            logger.warning(f"No split terms found for {level_key}. Skipping.")
            continue
        
        # --- Debug filter for specific terms ---
        if DEBUG_TERMS_ONLY:
            original_term_count = len(split_terms)
            split_terms = [t for t in split_terms if t in DEBUG_TERM_LIST]
            logger.info(f"DEBUG: Filtering to DEBUG_TERM_LIST. Kept {len(split_terms)} out of {original_term_count} for level {level_key}.")
            if not split_terms:
                logger.info(f"DEBUG: No terms from DEBUG_TERM_LIST found in level {level_key}. Skipping level.")
                continue
        # --- End Debug filter ---
            
        logger.info(f"Found {len(split_terms)} split terms for {level_key}")
        all_definitions[level_key] = []
        level_definitions = {}
        
        # Reset counters
        successful_definitions = 0
        failed_definitions = 0
        
        # Process terms in batches
        batch_number = 0
        for i in range(0, len(split_terms), BATCH_SIZE):
            batch_number += 1
            batch_terms = split_terms[i:i+BATCH_SIZE]
            batch_start_time = time.time()
            
            logger.info(f"Processing batch {batch_number} with {len(batch_terms)} split terms...")
            futures = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for term in batch_terms:
                    future = executor.submit(process_split_term, term, resource_data, metadata_data, batch_number)
                    futures.append(future)
                
                batch_results = {}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        term, definition = future.result()
                        if definition:
                            batch_results[term] = definition
                    except Exception as e:
                        logger.error(f"Error collecting result: {e}")
            
            # Store results
            level_definitions.update(batch_results)
            for term, definition in batch_results.items():
                all_definitions[level_key].append({"term": term, "definition": definition})
            
            batch_duration = time.time() - batch_start_time
            logger.info(f"Batch {batch_number} completed in {batch_duration:.2f}s - {len(batch_results)} definitions")
            total_terms_processed += len(batch_terms)
            
            # Wait between batches
            if i + BATCH_SIZE < len(split_terms):
                logger.info(f"Waiting {BATCH_TIMEOUT} seconds before next batch...")
                time.sleep(BATCH_TIMEOUT)

        # Update metadata
        if level_definitions:
            logger.info(f"Updating metadata for {level_key} with {len(level_definitions)} definitions...")
            if update_metadata_file(METADATA_FILE_TEMPLATE, level_num, level_definitions):
                logger.info(f"Successfully updated metadata for {level_key}")
            else:
                logger.error(f"Failed to update metadata for {level_key}")

        level_duration = time.time() - level_start_time
        logger.info(f"Finished {level_key} in {level_duration:.2f}s. Success: {successful_definitions}, Failed: {failed_definitions}")

    # Save all definitions
    total_duration = time.time() - start_time
    logger.info(f"--- Overall Summary ---")
    logger.info(f"Processed {total_terms_processed} split terms across {len(LEVELS_TO_PROCESS)} levels in {total_duration:.2f}s")
    logger.info(f"Total successful: {successful_definitions}, Total failed: {failed_definitions}")
    
    if save_json_file(all_definitions, OUTPUT_FILE):
        logger.info(f"All definitions saved to {OUTPUT_FILE}")
    else:
        logger.error(f"Failed to save definitions to {OUTPUT_FILE}")

    logger.info("Split term definition generation completed.")

if __name__ == "__main__":
    main() 