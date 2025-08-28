#!/usr/bin/env python3
"""
Update Level-Specific Files with Split Terms

This script updates the individual level files (lvX_metadata.json, lvX_final.txt, etc.)
to include the split terms from the sense disambiguation process.
"""

import json
import os
import argparse
import shutil
from datetime import datetime
from typing import Dict, List, Set
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_split_summary(split_summary_file: str) -> Dict[str, List[Dict]]:
    """Load the split summary to get information about which terms were split."""
    with open(split_summary_file, 'r') as f:
        data = json.load(f)
    
    # Organize splits by level
    splits_by_level = {}
    for split in data['detailed_splits']:
        level = split['level']
        if level not in splits_by_level:
            splits_by_level[level] = []
        splits_by_level[level].append(split)
    
    return splits_by_level

def load_updated_hierarchy(hierarchy_file: str) -> Dict:
    """Load the updated hierarchy with splits."""
    with open(hierarchy_file, 'r') as f:
        return json.load(f)

def update_metadata_file(metadata_file: str, splits_for_level: List[Dict], updated_hierarchy: Dict) -> bool:
    """Update a metadata JSON file to replace original terms with split terms."""
    logger.info(f"Updating metadata file: {metadata_file}")
    
    # Load the metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create a mapping of original term to split terms
    original_to_splits = {}
    for split in splits_for_level:
        original_term = split['original_term']
        new_terms = [term_info['term'] for term_info in split['new_terms']]
        original_to_splits[original_term] = new_terms
    
    changes_made = False
    
    # Update the metadata
    new_metadata = {}
    for term, term_data in metadata.items():
        if term in original_to_splits:
            # This term was split - create entries for each split
            logger.info(f"Splitting term '{term}' into {len(original_to_splits[term])} new terms")
            changes_made = True
            
            for new_term in original_to_splits[term]:
                # Create new metadata entry based on the original
                new_term_data = term_data.copy()
                
                # Get additional data from the updated hierarchy
                if new_term in updated_hierarchy.get('terms', {}):
                    hierarchy_data = updated_hierarchy['terms'][new_term]
                    new_term_data['original_term'] = hierarchy_data.get('original_term', term)
                    new_term_data['sense_tag'] = hierarchy_data.get('sense_tag', '')
                    new_term_data['split_info'] = hierarchy_data.get('split_info', {})
                
                new_metadata[new_term] = new_term_data
        else:
            # Keep the original term as-is
            new_metadata[term] = term_data
    
    # Save the updated metadata
    if changes_made:
        # Create backup
        backup_file = metadata_file + '.backup'
        shutil.copy2(metadata_file, backup_file)
        logger.info(f"Created backup: {backup_file}")
        
        with open(metadata_file, 'w') as f:
            json.dump(new_metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated {metadata_file}")
    
    return changes_made

def update_final_txt_file(txt_file: str, splits_for_level: List[Dict]) -> bool:
    """Update a final.txt file to replace original terms with split terms."""
    logger.info(f"Updating final txt file: {txt_file}")
    
    # Load the text file
    with open(txt_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Create a mapping of original term to split terms
    original_to_splits = {}
    for split in splits_for_level:
        original_term = split['original_term']
        new_terms = [term_info['term'] for term_info in split['new_terms']]
        original_to_splits[original_term] = new_terms
    
    changes_made = False
    new_lines = []
    
    for line in lines:
        if line in original_to_splits:
            # Replace with split terms
            logger.info(f"Replacing '{line}' with {len(original_to_splits[line])} split terms")
            changes_made = True
            for new_term in original_to_splits[line]:
                new_lines.append(new_term)
        else:
            # Keep the original line
            new_lines.append(line)
    
    # Save the updated file
    if changes_made:
        # Create backup
        backup_file = txt_file + '.backup'
        shutil.copy2(txt_file, backup_file)
        logger.info(f"Created backup: {backup_file}")
        
        with open(txt_file, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')
        logger.info(f"Updated {txt_file}")
    
    return changes_made

def update_filtered_resources_file(resources_file: str, splits_for_level: List[Dict], updated_hierarchy: Dict) -> bool:
    """Update a filtered_resources.json file to handle split terms."""
    logger.info(f"Updating filtered resources file: {resources_file}")
    
    # Load the resources file
    with open(resources_file, 'r') as f:
        resources_data = json.load(f)
    
    # Create a mapping of original term to split terms
    original_to_splits = {}
    for split in splits_for_level:
        original_term = split['original_term']
        new_terms = [term_info['term'] for term_info in split['new_terms']]
        original_to_splits[original_term] = new_terms
    
    changes_made = False
    new_resources_data = {}
    
    for term, term_resources in resources_data.items():
        if term in original_to_splits:
            # This term was split - distribute resources to split terms
            logger.info(f"Distributing resources for split term '{term}'")
            changes_made = True
            
            for new_term in original_to_splits[term]:
                # For now, give each split term the same resources
                # In a more sophisticated version, you could distribute based on clustering
                new_resources_data[new_term] = term_resources.copy()
                
                # Add split metadata to resources if available
                if new_term in updated_hierarchy.get('terms', {}):
                    hierarchy_data = updated_hierarchy['terms'][new_term]
                    if 'split_info' in hierarchy_data:
                        # Add split info to each resource entry
                        for resource in new_resources_data[new_term]:
                            if isinstance(resource, dict):
                                resource['split_info'] = hierarchy_data['split_info']
        else:
            # Keep the original term as-is
            new_resources_data[term] = term_resources
    
    # Save the updated resources file
    if changes_made:
        # Create backup
        backup_file = resources_file + '.backup'
        shutil.copy2(resources_file, backup_file)
        logger.info(f"Created backup: {backup_file}")
        
        with open(resources_file, 'w') as f:
            json.dump(new_resources_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated {resources_file}")
    
    return changes_made

def update_level_directory(level_dir: str, level: int, splits_for_level: List[Dict], updated_hierarchy: Dict):
    """Update all files in a level directory."""
    logger.info(f"Updating level {level} directory: {level_dir}")
    
    # Update metadata file
    metadata_file = os.path.join(level_dir, f'lv{level}_metadata.json')
    if os.path.exists(metadata_file):
        update_metadata_file(metadata_file, splits_for_level, updated_hierarchy)
    
    # Update final txt file
    txt_file = os.path.join(level_dir, f'lv{level}_final.txt')
    if os.path.exists(txt_file):
        update_final_txt_file(txt_file, splits_for_level)
    
    # Update filtered resources file
    resources_file = os.path.join(level_dir, f'lv{level}_filtered_resources.json')
    if os.path.exists(resources_file):
        update_filtered_resources_file(resources_file, splits_for_level, updated_hierarchy)

def main():
    parser = argparse.ArgumentParser(description='Update level-specific files with split terms')
    parser.add_argument('--split-summary', required=True,
                        help='Path to the split summary JSON file')
    parser.add_argument('--updated-hierarchy', required=True,
                        help='Path to the updated hierarchy JSON file')
    parser.add_argument('--data-final-dir', default='data/final',
                        help='Path to the data/final directory containing level subdirectories')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.split_summary):
        logger.error(f"Split summary file not found: {args.split_summary}")
        return 1
    
    if not os.path.exists(args.updated_hierarchy):
        logger.error(f"Updated hierarchy file not found: {args.updated_hierarchy}")
        return 1
    
    if not os.path.exists(args.data_final_dir):
        logger.error(f"Data final directory not found: {args.data_final_dir}")
        return 1
    
    try:
        # Load split information
        logger.info("Loading split summary...")
        splits_by_level = load_split_summary(args.split_summary)
        
        # Load updated hierarchy
        logger.info("Loading updated hierarchy...")
        updated_hierarchy = load_updated_hierarchy(args.updated_hierarchy)
        
        # Process each level
        total_changes = 0
        for level in range(4):  # Levels 0-3
            level_dir = os.path.join(args.data_final_dir, f'lv{level}')
            
            if not os.path.exists(level_dir):
                logger.warning(f"Level directory not found: {level_dir}")
                continue
                
            splits_for_level = splits_by_level.get(level, [])
            if not splits_for_level:
                logger.info(f"No splits for level {level}")
                continue
                
            logger.info(f"Processing level {level} with {len(splits_for_level)} splits")
            update_level_directory(level_dir, level, splits_for_level, updated_hierarchy)
            total_changes += len(splits_for_level)
        
        logger.info(f"Level file updates completed successfully. Applied {total_changes} split changes.")
        return 0
        
    except Exception as e:
        logger.error(f"Error updating level files: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    exit(main()) 