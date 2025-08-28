#!/usr/bin/env python3
"""
Updates lvX_filtered_resources.json files.

This script takes new resource information from a source file
(data/analysis/updated_resources.json) and a mapping of terms
that were missing resources per level (data/analysis/terms_missing_resources.json).
It then updates the target lvX_filtered_resources.json files by replacing
the resource lists for the specified terms if new data is available.
"""
import json
from pathlib import Path
import os
import logging
from typing import Optional, Dict

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_PATH = PROJECT_ROOT / "data"
ANALYSIS_DIR = BASE_DATA_PATH / "analysis"
FINAL_DIR = BASE_DATA_PATH / "final"

UPDATED_RESOURCES_SOURCE_FILE = ANALYSIS_DIR / "updated_resources.json"
TERMS_MAPPING_FILE = ANALYSIS_DIR / "terms_missing_resources.json"
TARGET_RESOURCE_FILE_TEMPLATE = FINAL_DIR / "lv{}" / "lv{}_filtered_resources.json"

# Fields to keep in the target resource objects in lvX_filtered_resources.json
TARGET_RESOURCE_FIELDS = ["url", "title", "processed_content", "score", "educational_score"]

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def load_json_safely(file_path: Path) -> Optional[dict]:
    """Loads a JSON file safely, returning None on error."""
    logger.info(f"Loading JSON from: {file_path}")
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def save_json_safely(data: dict, file_path: Path) -> bool:
    """Saves data to a JSON file safely, creating parent directories."""
    logger.info(f"Saving JSON to: {file_path}")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved data to {file_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving JSON to {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving to {file_path}: {e}")
        return False

def transform_resource_item(source_item: dict) -> dict:
    """
    Transforms a single resource item from updated_resources.json format
    to the target lvX_filtered_resources.json format.
    """
    target_item = {}
    
    target_item["url"] = source_item.get("url", "")
    target_item["title"] = source_item.get("title", "")
    
    # processed_content logic: use if available, else use snippet
    processed_content_source = source_item.get("processed_content")
    if processed_content_source and isinstance(processed_content_source, str) and processed_content_source.strip():
        target_item["processed_content"] = processed_content_source
    else:
        target_item["processed_content"] = source_item.get("snippet", "")
        if not target_item["processed_content"]:
             logger.debug(f"No processed_content or snippet for URL: {target_item['url']}")
        
    # Handle scores, defaulting to 0.0 if missing or invalid
    try:
        score_val = source_item.get("score")
        target_item["score"] = float(score_val) if score_val is not None else 0.0
    except (ValueError, TypeError):
        logger.warning(f"Invalid score value '{source_item.get('score')}' for URL '{source_item.get('url')}'. Defaulting to 0.0.")
        target_item["score"] = 0.0
        
    try:
        edu_score_val = source_item.get("educational_score")
        target_item["educational_score"] = float(edu_score_val) if edu_score_val is not None else 0.0
    except (ValueError, TypeError):
        logger.warning(f"Invalid educational_score value '{source_item.get('educational_score')}' for URL '{source_item.get('url')}'. Defaulting to 0.0.")
        target_item["educational_score"] = 0.0
            
    # Ensure only target fields are present
    final_item = {k: target_item.get(k) for k in TARGET_RESOURCE_FIELDS}
    return final_item

# --- Main Processing Function ---
def main():
    logger.info("Starting resource update process...")

    source_updated_resources = load_json_safely(UPDATED_RESOURCES_SOURCE_FILE)
    if source_updated_resources is None: # Check for None, as empty dict is valid if file is empty JSON {}
        logger.error(f"Could not load source updated resources from '{UPDATED_RESOURCES_SOURCE_FILE}'. Aborting.")
        return

    terms_map_by_level = load_json_safely(TERMS_MAPPING_FILE)
    if terms_map_by_level is None:
        logger.error(f"Could not load terms mapping from '{TERMS_MAPPING_FILE}'. Aborting.")
        return

    overall_terms_updated_count = 0

    for i in range(4):  # Process levels lv0, lv1, lv2, lv3
        level_name_key = f"lv{i}" # Key for terms_map_by_level (e.g., "lv0")
        
        logger.info(f"--- Processing level: {level_name_key} ---")

        target_resource_file_path = Path(str(TARGET_RESOURCE_FILE_TEMPLATE).format(i, i))
        
        # Load existing target resources for the current level
        # If file doesn't exist or is invalid, start with an empty dict
        current_level_target_data = load_json_safely(target_resource_file_path) or {}

        terms_to_check_in_level = terms_map_by_level.get(level_name_key, [])
        if not terms_to_check_in_level:
            logger.info(f"No terms listed in '{TERMS_MAPPING_FILE}' for level '{level_name_key}'. Skipping resource updates for this level.")
            continue
        
        logger.info(f"Found {len(terms_to_check_in_level)} candidate term(s) for '{level_name_key}' from '{TERMS_MAPPING_FILE}'.")
        
        level_updated_term_count = 0
        for term_to_update in terms_to_check_in_level:
            if term_to_update in source_updated_resources:
                new_resource_list_raw = source_updated_resources[term_to_update]
                
                if isinstance(new_resource_list_raw, list):
                    transformed_resources_for_term = []
                    for raw_item in new_resource_list_raw:
                        if isinstance(raw_item, dict):
                            transformed_resources_for_term.append(transform_resource_item(raw_item))
                        else:
                            logger.warning(f"Item in resource list for term '{term_to_update}' (level '{level_name_key}') is not a dictionary: {raw_item}. Skipping this item.")
                    
                    # Replace the entire resource list for this term in the current level's data
                    current_level_target_data[term_to_update] = transformed_resources_for_term
                    logger.info(f"Updated resources for term '{term_to_update}' in '{level_name_key}' with {len(transformed_resources_for_term)} new resource(s).")
                    level_updated_term_count += 1
                else:
                    logger.warning(f"Resource data for term '{term_to_update}' in '{UPDATED_RESOURCES_SOURCE_FILE}' is not a list. Skipping update for this term in '{level_name_key}'.")
            # else:
                # This term was in terms_missing_resources.json for this level,
                # but no corresponding entry was found in updated_resources.json.
                # So, no update action for this term in this level.
                # logger.debug(f"Term '{term_to_update}' (for level '{level_name_key}') not found in the source '{UPDATED_RESOURCES_SOURCE_FILE}'. No update.")

        if level_updated_term_count > 0:
            logger.info(f"Attempting to save updated resources for '{level_name_key}' to '{target_resource_file_path}' ({level_updated_term_count} term(s) modified).")
            if save_json_safely(current_level_target_data, target_resource_file_path):
                overall_terms_updated_count += level_updated_term_count
            else:
                logger.error(f"FAILED to save updated resources for '{level_name_key}'.")
        else:
            logger.info(f"No terms were actually updated for '{level_name_key}'. Target file not re-saved.")
            
    logger.info("--- Resource update process finished. ---")
    logger.info(f"Total terms for which resource lists were updated across all levels: {overall_terms_updated_count}")

if __name__ == "__main__":
    main() 