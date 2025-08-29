"""
Main metadata collection logic for glossary terms.

This module orchestrates the collection of metadata from various sources,
including parent relationships, variations, and source information.
"""

import os
import json
import csv
from typing import Dict, List, Any, Set, Optional
from pathlib import Path
from collections import defaultdict

from .file_discovery import (
    find_step_file, 
    find_step_metadata, 
    find_final_file,
    find_dedup_files,
    ensure_final_dirs_exist
)
from .extractors import (
    extract_parent_from_college,
    clean_parent_term,
    is_department_or_college_source,
    extract_concept_from_source,
    extract_metadata_from_json,
    extract_variations_from_dedup
)
from .consolidator import consolidate_metadata_for_term


# Base data directory
DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    'generate_glossary', 'data'
))
FINAL_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    'data', 'final'
))


def collect_metadata(level: int, verbose: bool = False, include_variations: bool = True, 
                     merge_variation_metadata: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Collect metadata for a given level.
    
    Args:
        level: The hierarchy level (0, 1, 2, 3)
        verbose: Whether to print verbose output
        include_variations: Whether to include variations in the metadata
        merge_variation_metadata: Whether to merge all metadata from variations into canonical terms
        
    Returns:
        Dictionary of terms and their metadata
    """
    # First ensure the final directories exist
    ensure_final_dirs_exist()
    
    # Initialize metadata dictionary
    metadata: Dict[str, Dict[str, List[str]]] = {}
    
    # Dictionary to store metadata for variations
    variations_metadata: Dict[str, Dict[str, Any]] = {} if include_variations else None
    
    # Get the final terms file path
    final_terms_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_final.txt')
    if verbose:
        print(f'Using final terms file: {final_terms_file}')
    
    # Read final terms
    with open(final_terms_file, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f.readlines()]
    
    if verbose:
        print(f'Found {len(terms)} terms in final file')
    
    # Initialize metadata for each term
    for term in terms:
        metadata[term] = {
            'sources': [],
            'parents': [],
            'variations': []
        }
    
    # Load parent level information if applicable
    parent_terms, parent_variations_map, all_parent_terms = load_parent_level_info(
        level, verbose
    )
    
    # Process source metadata
    process_source_files(level, metadata, all_parent_terms, verbose)
    
    # Process deduplication files
    process_dedup_files(level, metadata, variations_metadata, include_variations, verbose)
    
    # Clean and normalize parents
    clean_parents(metadata)
    
    # Merge variation metadata if requested
    if merge_variation_metadata and variations_metadata:
        merge_variations(metadata, variations_metadata, verbose)
    
    # Convert to final format
    final_metadata = convert_to_final_format(
        metadata, variations_metadata, include_variations
    )
    
    # Save metadata
    save_metadata(level, final_metadata, verbose)
    
    return final_metadata


def load_parent_level_info(level: int, verbose: bool = False) -> tuple:
    """
    Load parent level terms and variations.
    
    Returns:
        Tuple of (parent_terms, parent_variations_map, all_parent_terms)
    """
    parent_terms = set()
    parent_variations_map = {}
    all_parent_terms = set()
    
    if level > 0:
        parent_level = level - 1
        parent_final_file = os.path.join(DATA_DIR, f'lv{parent_level}', f'lv{parent_level}_final.txt')
        parent_metadata_file = os.path.join(DATA_DIR, f'lv{parent_level}', f'lv{parent_level}_metadata.json')
        
        # Read parent terms from final.txt
        if os.path.exists(parent_final_file):
            if verbose:
                print(f'Reading parent terms from: {parent_final_file}')
            with open(parent_final_file, 'r', encoding='utf-8') as f:
                parent_terms = {line.strip().lower() for line in f.readlines() if line.strip()}
            if verbose:
                print(f'Found {len(parent_terms)} terms in parent level final file')
        
        # Read parent metadata to get variations
        if os.path.exists(parent_metadata_file):
            if verbose:
                print(f'Reading parent metadata from: {parent_metadata_file}')
            try:
                with open(parent_metadata_file, 'r', encoding='utf-8') as f:
                    parent_metadata = json.load(f)
                
                # Build variation to canonical term mapping
                for term, term_data in parent_metadata.items():
                    # If this is a variation, add to the map
                    if 'canonical_term' in term_data:
                        canonical = term_data['canonical_term']
                        parent_variations_map[term] = canonical
                    
                    # If this is a canonical term, add all its variations to the map
                    elif 'variations' in term_data and term in parent_terms:
                        for variation in term_data['variations']:
                            parent_variations_map[variation] = term
                
                if verbose:
                    print(f'Found {len(parent_variations_map)} variations in parent level metadata')
            except Exception as e:
                if verbose:
                    print(f"Error reading parent metadata file: {str(e)}")
    
    # Compile complete list of parent terms and variations
    if level > 0:
        all_parent_terms = set(parent_terms)
        for var, canon in parent_variations_map.items():
            if canon in parent_terms:
                all_parent_terms.add(var.lower())
    
    return parent_terms, parent_variations_map, all_parent_terms


def process_source_files(level: int, metadata: Dict, all_parent_terms: Set, 
                         verbose: bool = False) -> None:
    """Process source CSV and metadata files."""
    # Process source CSV file if it exists
    if level in [2, 3]:
        source_csv_file = os.path.join(DATA_DIR, f'lv{level}', 'raw', f'lv{level}_s1_hierarchical_concepts.csv')
    else:
        source_csv_file = os.path.join(DATA_DIR, f'lv{level}', 'raw', f'lv{level}_s1_department_concepts.csv')
    
    if os.path.exists(source_csv_file):
        if verbose:
            print(f'Processing source CSV: {source_csv_file}')
        process_source_csv(source_csv_file, metadata, level, all_parent_terms, verbose)
    
    # Process metadata files from each step
    raw_dir = Path(DATA_DIR) / f'lv{level}' / 'raw'
    for step in range(4):
        metadata_file = find_step_metadata(raw_dir, level, step)
        if metadata_file:
            if verbose:
                print(f'Processing metadata file: {metadata_file}')
            process_step_metadata(metadata_file, metadata, level, all_parent_terms)


def process_source_csv(csv_file: str, metadata: Dict, level: int, 
                       parent_terms: Set, verbose: bool = False) -> None:
    """Process source CSV file to extract metadata."""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                concept = row.get('concept', '').strip()
                source = row.get('source', '').strip()
                parent = row.get('parent', '').strip() if level in [2, 3] else None
                
                if concept and concept in metadata:
                    # Add source
                    if source:
                        metadata[concept]['sources'].append(source)
                    
                    # Add parent for hierarchical levels
                    if parent:
                        cleaned_parent = clean_parent_term(parent)
                        if cleaned_parent:
                            metadata[concept]['parents'].append(cleaned_parent)
    except Exception as e:
        if verbose:
            print(f"Error processing CSV file {csv_file}: {str(e)}")


def process_step_metadata(metadata_file: Path, metadata: Dict, level: int,
                          parent_terms: Set) -> None:
    """Process metadata JSON file from a specific step."""
    try:
        data = extract_metadata_from_json(metadata_file)
        
        # Extract relevant information based on step
        if 's0' in metadata_file.name:
            # Step 0 metadata - raw extraction
            process_s0_metadata(data, metadata, level, parent_terms)
        elif 's1' in metadata_file.name:
            # Step 1 metadata - concept extraction
            process_s1_metadata(data, metadata, level, parent_terms)
        elif 's2' in metadata_file.name:
            # Step 2 metadata - frequency filtering
            process_s2_metadata(data, metadata)
        elif 's3' in metadata_file.name:
            # Step 3 metadata - verification
            process_s3_metadata(data, metadata)
            
    except Exception as e:
        print(f"Error processing metadata file {metadata_file}: {str(e)}")


def process_s0_metadata(data: Dict, metadata: Dict, level: int, parent_terms: Set) -> None:
    """Process step 0 metadata."""
    # Implementation depends on specific format of s0 metadata
    pass


def process_s1_metadata(data: Dict, metadata: Dict, level: int, parent_terms: Set) -> None:
    """Process step 1 metadata."""
    # Implementation depends on specific format of s1 metadata
    pass


def process_s2_metadata(data: Dict, metadata: Dict) -> None:
    """Process step 2 metadata."""
    # Implementation depends on specific format of s2 metadata
    pass


def process_s3_metadata(data: Dict, metadata: Dict) -> None:
    """Process step 3 metadata."""
    # Implementation depends on specific format of s3 metadata
    pass


def process_dedup_files(level: int, metadata: Dict, variations_metadata: Optional[Dict],
                        include_variations: bool, verbose: bool = False) -> None:
    """Process deduplication files to extract variations."""
    dedup_files = find_dedup_files(level)
    
    for dedup_file in dedup_files:
        if verbose:
            print(f'Processing dedup file: {dedup_file}')
        
        if dedup_file.endswith('.json'):
            try:
                with open(dedup_file, 'r', encoding='utf-8') as f:
                    dedup_data = json.load(f)
                    variations = extract_variations_from_dedup(dedup_data)
                    
                    for primary, vars in variations.items():
                        if primary in metadata:
                            metadata[primary]['variations'].extend(vars)
                            
                        # Store variation metadata if requested
                        if include_variations and variations_metadata is not None:
                            for var in vars:
                                variations_metadata[var] = {
                                    'canonical_term': primary,
                                    'sources': [],
                                    'parents': []
                                }
            except Exception as e:
                if verbose:
                    print(f"Error processing dedup file {dedup_file}: {str(e)}")


def clean_parents(metadata: Dict) -> None:
    """Clean and normalize parent relationships."""
    for term, term_data in metadata.items():
        if 'parents' in term_data:
            # Remove duplicates and normalize
            cleaned_parents = []
            seen = set()
            for parent in term_data['parents']:
                cleaned = clean_parent_term(parent)
                if cleaned and cleaned not in seen:
                    cleaned_parents.append(cleaned)
                    seen.add(cleaned)
            term_data['parents'] = cleaned_parents


def merge_variations(metadata: Dict, variations_metadata: Dict, verbose: bool = False) -> None:
    """Merge metadata from variations into canonical terms."""
    for variation, var_data in variations_metadata.items():
        canonical = var_data.get('canonical_term')
        if canonical and canonical in metadata:
            # Merge sources
            if 'sources' in var_data:
                metadata[canonical]['sources'].extend(var_data['sources'])
            
            # Merge parents
            if 'parents' in var_data:
                metadata[canonical]['parents'].extend(var_data['parents'])
    
    # Remove duplicates after merging
    for term_data in metadata.values():
        term_data['sources'] = list(set(term_data['sources']))
        term_data['parents'] = list(set(term_data['parents']))


def convert_to_final_format(metadata: Dict, variations_metadata: Optional[Dict],
                            include_variations: bool) -> Dict[str, Dict[str, Any]]:
    """Convert metadata to final output format."""
    final_metadata = {}
    
    for term, term_data in metadata.items():
        final_metadata[term] = {
            'sources': term_data['sources'],
            'parents': term_data['parents'],
            'variations': term_data['variations']
        }
    
    # Add variation entries if requested
    if include_variations and variations_metadata:
        for variation, var_data in variations_metadata.items():
            final_metadata[variation] = var_data
    
    return final_metadata


def save_metadata(level: int, metadata: Dict, verbose: bool = False) -> None:
    """Save metadata to JSON file."""
    output_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_metadata.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f'Saved metadata to: {output_file}')


def collect_resources(level: int, verbose: bool = False, 
                     include_variations: bool = True) -> None:
    """
    Collect and consolidate web resources for terms at a given level.
    
    Args:
        level: The hierarchy level (0, 1, 2, 3)
        verbose: Whether to print verbose output
        include_variations: Whether to include variations in the resources
    """
    # Implementation for resource collection
    # This is a simplified version - the full implementation would be more complex
    pass