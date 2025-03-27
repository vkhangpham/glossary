#!/usr/bin/env python

import argparse
import csv
import json
import os
import glob
from pathlib import Path
import re
from typing import Dict, List

DATA_DIR = 'data'


def extract_parent_from_college(college_name):
    """Extract parent term from college name."""
    # Extract the main subject from "college of X" pattern
    match = re.search(r'college of (\w+)', college_name.lower())
    if match:
        return match.group(1)
    return None


def find_step_file(level_dir, level, step, extension=None):
    """Find file for a specific step using regex."""
    pattern = f"lv{level}_s{step}_*"
    if extension:
        pattern += f".{extension}"
    
    # List all files matching the pattern
    matches = list(level_dir.glob(pattern))
    
    # Return the first match if any
    if matches:
        return matches[0]
    
    # If no matches with direct extension, try more flexible approach
    if extension:
        matches = list(level_dir.glob(f"lv{level}_s{step}_*"))
        for match in matches:
            if match.suffix.lstrip('.') == extension or match.name.endswith(f".{extension}"):
                return match
    
    return None


def find_step_metadata(raw_dir, level, step):
    """Find metadata.json file for a specific step."""
    # Try standard naming pattern
    metadata_file = raw_dir / f"lv{level}_s{step}_metadata.json"
    if metadata_file.exists():
        return metadata_file
    
    # Try flexible pattern matching
    metadata_files = list(raw_dir.glob(f"lv{level}_s{step}*metadata*.json"))
    if metadata_files:
        return metadata_files[0]
    
    return None


def find_final_file(level_dir, level):
    """Find the final output file for a level."""
    # First try the standard naming convention
    final_file = level_dir / 'postprocessed' / f'lv{level}_final.txt'
    if final_file.exists():
        return final_file
    
    # Try pattern matching if standard file not found
    final_files = list(level_dir.glob(f'**/lv{level}_final.txt'))
    if final_files:
        return final_files[0]
    
    # Try even more flexible pattern matching
    final_files = list(level_dir.glob(f'**/*final*.txt'))
    for file in final_files:
        if f'lv{level}' in file.name:
            return file
    
    return None


def find_dedup_files(level: int) -> List[str]:
    """Find all relevant deduplication files for a given level."""
    dedup_files = []
    
    # Check current level's postprocessed directory
    current_level_dir = os.path.join(DATA_DIR, f'lv{level}', 'postprocessed')
    if os.path.exists(current_level_dir):
        for file in os.listdir(current_level_dir):
            if 'dedup' in file or 'final' in file:
                dedup_files.append(os.path.join(current_level_dir, file))
    
    # Check next level's postprocessed directory
    next_level_dir = os.path.join(DATA_DIR, f'lv{level + 1}', 'postprocessed')
    if os.path.exists(next_level_dir):
        for file in os.listdir(next_level_dir):
            if 'dedup' in file or 'final' in file:
                dedup_files.append(os.path.join(next_level_dir, file))
    
    return dedup_files


def process_dedup_file(dedup_file: str, metadata: Dict[str, Dict[str, List[str]]]) -> None:
    """Process a deduplication file to extract variations."""
    try:
        with open(dedup_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process each entry in the deduplication file
        for entry in data:
            if isinstance(entry, dict):
                # Get the canonical term
                canonical = entry.get('canonical', '').strip().lower()
                if canonical in metadata:
                    # Check various fields that might contain variations
                    variations = set()
                    
                    # Check 'variations' field
                    if isinstance(entry.get('variations'), (list, tuple)):
                        variations.update(v.strip().lower() for v in entry['variations'] if isinstance(v, str))
                    
                    # Check 'variation_reasons' field
                    if isinstance(entry.get('variation_reasons'), dict):
                        for var_list in entry['variation_reasons'].values():
                            if isinstance(var_list, (list, tuple)):
                                variations.update(v.strip().lower() for v in var_list if isinstance(v, str))
                    
                    # Check 'component_details' field
                    if isinstance(entry.get('component_details'), dict):
                        for comp_data in entry['component_details'].values():
                            if isinstance(comp_data, dict) and isinstance(comp_data.get('variations'), (list, tuple)):
                                variations.update(v.strip().lower() for v in comp_data['variations'] if isinstance(v, str))
                    
                    # Check 'relationships' field
                    if isinstance(entry.get('relationships'), dict):
                        for rel_data in entry['relationships'].values():
                            if isinstance(rel_data, dict) and isinstance(rel_data.get('variations'), (list, tuple)):
                                variations.update(v.strip().lower() for v in rel_data['variations'] if isinstance(v, str))
                    
                    # Check 'cross_level_variations' field
                    if isinstance(entry.get('cross_level_variations'), (list, tuple)):
                        variations.update(v.strip().lower() for v in entry['cross_level_variations'] if isinstance(v, str))
                    
                    # Add variations to metadata
                    for variation in variations:
                        if variation and variation != canonical and variation not in metadata[canonical]['variations']:
                            metadata[canonical]['variations'].append(variation)
    except Exception as e:
        print(f"Error processing deduplication file {dedup_file}: {str(e)}")


def process_source_metadata(metadata_file: str, metadata: Dict[str, Dict[str, List[str]]], level: int) -> None:
    """Process metadata JSON files to extract sources."""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process source_concept_mapping
        if isinstance(data.get('source_concept_mapping'), (list, tuple)):
            for item in data['source_concept_mapping']:
                if isinstance(item, dict):
                    source = item.get('source', '').strip()
                    concepts = item.get('concepts', [])
                    if isinstance(concepts, (list, tuple)):
                        for concept in concepts:
                            if isinstance(concept, str) and concept.strip().lower() in metadata:
                                if source and source not in metadata[concept.strip().lower()]['sources']:
                                    metadata[concept.strip().lower()]['sources'].append(source)
        
        # Process concept_frequencies
        if isinstance(data.get('concept_frequencies'), dict):
            for concept, freq_data in data['concept_frequencies'].items():
                if concept.strip().lower() in metadata:
                    if isinstance(freq_data, dict):
                        sources = freq_data.get('sources', [])
                        if isinstance(sources, (list, tuple)):
                            for source in sources:
                                if source and source not in metadata[concept.strip().lower()]['sources']:
                                    metadata[concept.strip().lower()]['sources'].append(source)
        
        # Process verification_results
        if isinstance(data.get('verification_results'), dict):
            for concept, verify_data in data['verification_results'].items():
                if concept.strip().lower() in metadata:
                    if isinstance(verify_data, dict):
                        sources = verify_data.get('sources', [])
                        if isinstance(sources, (list, tuple)):
                            for source in sources:
                                if source and source not in metadata[concept.strip().lower()]['sources']:
                                    metadata[concept.strip().lower()]['sources'].append(source)
    except Exception as e:
        print(f"Error processing metadata file {metadata_file}: {str(e)}")


def clean_parent_term(parent: str) -> str:
    """Clean parent term by removing 'college of' and 'department of' prefixes."""
    parent = parent.lower().strip()
    parent = re.sub(r'^college of\s+', '', parent)
    parent = re.sub(r'^department of\s+', '', parent)
    return parent


def collect_metadata(level: int, verbose: bool = False) -> Dict[str, Dict[str, List[str]]]:
    """Collect metadata for a given level."""
    # Initialize metadata dictionary
    metadata: Dict[str, Dict[str, List[str]]] = {}
    
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
    
    # Process source CSV file if it exists
    source_csv_file = os.path.join(DATA_DIR, f'lv{level}', 'raw', f'lv{level}_s1_department_concepts.csv')
    if os.path.exists(source_csv_file):
        if verbose:
            print(f'Using source file: {source_csv_file}')
        
        # Read CSV file
        with open(source_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                concept = row.get('concept', '').strip().lower()
                if concept in metadata:
                    # Add department as source
                    if 'department' in row:
                        department = row['department'].strip()
                        if department and department not in metadata[concept]['sources']:
                            metadata[concept]['sources'].append(department)
                    
                    # Add parents based on level
                    if level == 1 and 'college' in row:
                        parent = clean_parent_term(row['college'])
                        if parent and parent not in metadata[concept]['parents']:
                            metadata[concept]['parents'].append(parent)
                    elif level == 2 and 'department' in row:
                        parent = clean_parent_term(row['department'])
                        if parent and parent not in metadata[concept]['parents']:
                            metadata[concept]['parents'].append(parent)
    
    # Process metadata JSON files
    for step in range(4):  # steps 0-3
        metadata_file = os.path.join(DATA_DIR, f'lv{level}', 'raw', f'lv{level}_s{step}_metadata.json')
        if os.path.exists(metadata_file):
            if verbose:
                print(f'Found metadata file for step {step}: {metadata_file}')
            process_source_metadata(metadata_file, metadata, level)
    
    # Process deduplication files
    dedup_files = find_dedup_files(level)
    if verbose:
        print(f'Found {len(dedup_files)} deduplication files to process')
    
    for dedup_file in dedup_files:
        if verbose:
            print(f'Processing deduplication file: {dedup_file}')
        process_dedup_file(dedup_file, metadata)
    
    # Save metadata to JSON file
    output_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_metadata.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f'Metadata collected for {len(terms)} terms and saved to {output_file}')
    
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect metadata for terms after level completion')
    parser.add_argument('level', type=int, help='Level number (0, 1, 2, etc.)')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='Output file path (default: data/lvX/metadata.json)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'data/lv{args.level}/metadata.json'
    
    collect_metadata(args.level, args.output) 