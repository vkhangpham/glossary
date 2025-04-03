#!/usr/bin/env python

import argparse
import csv
import json
import os
import glob
from pathlib import Path
import re
from typing import Dict, List, Set, Any
from collections import defaultdict

DATA_DIR = 'data'
FINAL_DIR = os.path.join(DATA_DIR, 'final')


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
            if ('dedup' in file or 'final' in file) and file.endswith('.json'):
                dedup_files.append(os.path.join(current_level_dir, file))
    
    # Check next level's postprocessed directory
    next_level_dir = os.path.join(DATA_DIR, f'lv{level + 1}', 'postprocessed')
    if os.path.exists(next_level_dir):
        for file in os.listdir(next_level_dir):
            if ('dedup' in file or 'final' in file) and file.endswith('.json'):
                dedup_files.append(os.path.join(next_level_dir, file))
    
    return dedup_files


def process_dedup_file(dedup_file: str, metadata: Dict[str, Dict[str, List[str]]],
                       variations_metadata: Dict[str, Dict[str, Any]] = None) -> None:
    """Process a deduplication file to extract variations."""
    try:
        if not dedup_file.endswith('.json'):
            if dedup_file.endswith('.txt'):
                # Handle text file - skip processing as it's just a list of terms
                return
            return
            
        with open(dedup_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format 1: List of dictionaries with canonical and variations
        if isinstance(data, list):    
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
                                
                                # Create metadata for the variation if requested
                                if variations_metadata is not None:
                                    variations_metadata[variation] = {
                                        'canonical_term': canonical,
                                        'sources': [],
                                        'parents': [],
                                        'variations': []
                                    }
        
        # Format 2: Dictionary with variations and cross_level_variations
        elif isinstance(data, dict):
            # Process 'variations' field
            if isinstance(data.get('variations'), dict):
                for canonical, variations_list in data['variations'].items():
                    canonical_lower = canonical.lower().strip()
                    if canonical_lower in metadata and isinstance(variations_list, (list, tuple)):
                        for variation in variations_list:
                            variation_lower = variation.lower().strip()
                            if variation_lower and variation_lower != canonical_lower and variation_lower not in metadata[canonical_lower]['variations']:
                                metadata[canonical_lower]['variations'].append(variation_lower)
                                
                                # Create metadata for the variation if requested
                                if variations_metadata is not None:
                                    variations_metadata[variation_lower] = {
                                        'canonical_term': canonical_lower,
                                        'sources': [],
                                        'parents': [],
                                        'variations': []
                                    }
            
            # Process 'cross_level_variations' field
            if isinstance(data.get('cross_level_variations'), dict):
                for canonical, variations_list in data['cross_level_variations'].items():
                    canonical_lower = canonical.lower().strip()
                    if canonical_lower in metadata and isinstance(variations_list, (list, tuple)):
                        for variation in variations_list:
                            variation_lower = variation.lower().strip()
                            if variation_lower and variation_lower != canonical_lower and variation_lower not in metadata[canonical_lower]['variations']:
                                metadata[canonical_lower]['variations'].append(variation_lower)
                                
                                # Create metadata for the variation if requested
                                if variations_metadata is not None:
                                    variations_metadata[variation_lower] = {
                                        'canonical_term': canonical_lower,
                                        'sources': [],
                                        'parents': [],
                                        'variations': []
                                    }
            
            # Process 'variation_reasons' field
            if isinstance(data.get('variation_reasons'), dict):
                for variation, reason_data in data['variation_reasons'].items():
                    if isinstance(reason_data, dict) and 'canonical' in reason_data:
                        canonical = reason_data['canonical'].lower().strip()
                        variation_lower = variation.lower().strip()
                        if canonical in metadata and variation_lower and variation_lower != canonical and variation_lower not in metadata[canonical]['variations']:
                            metadata[canonical]['variations'].append(variation_lower)
                            
                            # Create metadata for the variation if requested
                            if variations_metadata is not None:
                                variations_metadata[variation_lower] = {
                                    'canonical_term': canonical,
                                    'sources': [],
                                    'parents': [],
                                    'variations': []
                                }
                            
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
                        
                        # For level 0, also check 'institutions' field
                        if level == 0 and isinstance(freq_data.get('institutions'), (list, tuple)):
                            for institution in freq_data['institutions']:
                                if institution and institution not in metadata[concept.strip().lower()]['sources']:
                                    metadata[concept.strip().lower()]['sources'].append(institution)
        
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
                        
                        # For level 0, also check 'colleges' field
                        if level == 0 and isinstance(verify_data.get('colleges'), (list, tuple)):
                            for college in verify_data['colleges']:
                                if college and college not in metadata[concept.strip().lower()]['sources']:
                                    metadata[concept.strip().lower()]['sources'].append(college)
    except Exception as e:
        print(f"Error processing metadata file {metadata_file}: {str(e)}")


def clean_parent_term(parent: str) -> str:
    """Clean parent term by removing 'college of' and 'department of' prefixes."""
    parent = parent.lower().strip()
    parent = re.sub(r'^college of\s+', '', parent)
    parent = re.sub(r'^department of\s+', '', parent)
    return parent


def ensure_final_dirs_exist():
    """Ensure the final data directory structure exists."""
    os.makedirs(FINAL_DIR, exist_ok=True)
    
    # Create subdirectories for each level
    for level in range(4):  # Levels 0, 1, 2, 3
        os.makedirs(os.path.join(FINAL_DIR, f'lv{level}'), exist_ok=True)


def collect_metadata(level: int, verbose: bool = False, include_variations: bool = True, merge_variation_metadata: bool = True) -> Dict[str, Dict[str, Any]]:
    """Collect metadata for a given level.
    
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
    
    # If this is level 1 or higher, read parent level's final terms and metadata
    parent_terms = set()
    parent_variations_map = {}  # Maps variation to canonical term
    
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
    
    # Process source CSV file if it exists
    # For level 2, the CSV file has a different name pattern
    if level == 2:
        source_csv_file = os.path.join(DATA_DIR, f'lv{level}', 'raw', f'lv{level}_s1_hierarchical_concepts.csv')
    else:
        source_csv_file = os.path.join(DATA_DIR, f'lv{level}', 'raw', f'lv{level}_s1_department_concepts.csv')
    
    if os.path.exists(source_csv_file):
        if verbose:
            print(f'Using source file: {source_csv_file}')
        
        # Read CSV file
        with open(source_csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Level 2 CSV has different column names
                if level == 2:
                    concept_key = 'concept'
                    current_source_key = 'topic'
                    parent_source_key = 'department'
                else:
                    concept_key = 'concept'
                    current_source_key = 'department'
                    parent_source_key = 'college'
                
                concept = row.get(concept_key, '').strip().lower()
                if concept in metadata:
                    # Add department as source
                    if current_source_key in row:
                        current_source = row[current_source_key].strip()
                        if current_source and current_source not in metadata[concept]['sources']:
                            metadata[concept]['sources'].append(current_source)
                    
                    # Add parents based on level
                    if parent_source_key in row:
                        parent = clean_parent_term(row[parent_source_key])
                        
                        # Check if parent exists in parent terms directly
                        if parent and parent in parent_terms:
                            if parent not in metadata[concept]['parents']:
                                metadata[concept]['parents'].append(parent)
                        
                        # If not, check if parent exists as a variation
                        elif parent and parent in parent_variations_map:
                            canonical_parent = parent_variations_map[parent]
                            if canonical_parent in parent_terms and canonical_parent not in metadata[concept]['parents']:
                                metadata[concept]['parents'].append(canonical_parent)
                                if verbose:
                                    print(f"Parent '{parent}' is a variation of '{canonical_parent}', using canonical parent for '{concept}'")
                        
                        elif verbose and parent:
                            print(f"Skipping parent '{parent}' for '{concept}' - not found in parent level terms or variations")
    
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
        process_dedup_file(dedup_file, metadata, variations_metadata)
    
    # Merge variations metadata into the main metadata, if variations are included
    if include_variations and variations_metadata:
        if verbose:
            print(f'Adding metadata for {len(variations_metadata)} variations')
        
        # First, process each variation's sources and parents
        for variation, var_data in variations_metadata.items():
            canonical = var_data['canonical_term']
            
            # Find data for this variation in source files and other metadata sources
            for step in range(4):  # steps 0-3
                metadata_file = os.path.join(DATA_DIR, f'lv{level}', 'raw', f'lv{level}_s{step}_metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Check for the variation in source_concept_mapping
                        if isinstance(data.get('source_concept_mapping'), (list, tuple)):
                            for item in data['source_concept_mapping']:
                                if isinstance(item, dict):
                                    source = item.get('source', '').strip()
                                    concepts = item.get('concepts', [])
                                    if isinstance(concepts, (list, tuple)):
                                        for concept in concepts:
                                            if isinstance(concept, str) and concept.strip().lower() == variation:
                                                if source and source not in var_data['sources']:
                                                    var_data['sources'].append(source)
                                                    # Also add to canonical term if not already there
                                                    if canonical in metadata and source not in metadata[canonical]['sources']:
                                                        metadata[canonical]['sources'].append(source)
                        
                        # Check for the variation in concept_frequencies
                        if isinstance(data.get('concept_frequencies'), dict):
                            freq_data = data['concept_frequencies'].get(variation)
                            if isinstance(freq_data, dict):
                                sources = freq_data.get('sources', [])
                                if isinstance(sources, (list, tuple)):
                                    for source in sources:
                                        if source and source not in var_data['sources']:
                                            var_data['sources'].append(source)
                                            # Also add to canonical term if not already there
                                            if canonical in metadata and source not in metadata[canonical]['sources']:
                                                metadata[canonical]['sources'].append(source)
                    except Exception as e:
                        if verbose:
                            print(f"Error processing metadata file for variation {variation}: {str(e)}")
            
            # Inherit parents from canonical term
            if canonical in metadata:
                var_data['parents'] = metadata[canonical]['parents'].copy()
                
                # Enhanced: If merge_variation_metadata is True, ensure all variation data is fully transferred to canonical term
                if merge_variation_metadata:
                    # Add sources from variation to canonical term
                    for source in var_data.get('sources', []):
                        if source not in metadata[canonical]['sources']:
                            metadata[canonical]['sources'].append(source)
                    
                    # Add parents from variation to canonical term
                    if 'parents' in var_data:
                        for parent in var_data['parents']:
                            if parent not in metadata[canonical]['parents']:
                                metadata[canonical]['parents'].append(parent)
        
        # Add variations metadata to the main metadata
        metadata.update(variations_metadata)
    
    # Save metadata to the original location
    output_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_metadata.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Also save to the final directory
    final_output_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_metadata.json')
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save final.txt with canonical terms only
    terms = [term for term in metadata if 'canonical_term' not in metadata[term]]
    final_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_final.txt')
    with open(final_file, 'w', encoding='utf-8') as f:
        for term in sorted(terms):
            f.write(f"{term}\n")
    
    if verbose:
        print(f'Metadata collected for {len(metadata)} terms (including {len(metadata) - len(terms)} variations)')
        print(f'Saved to original location: {output_file}')
        print(f'Saved to final location: {final_output_file}')
    
    return metadata


def collect_resources(level: int, verbose: bool = False, include_variations: bool = True) -> None:
    """Collect resources for a given level and save to a separate file.
    
    Only include resources that:
    1. Are for terms in the final.txt file or their variations (if include_variations=True)
    2. Have is_verified = True
    3. Have relevance_score > 0.75
    4. Include only these fields: url, title, processed_content, score, relevance_score
    """
    # First ensure the final directories exist
    ensure_final_dirs_exist()
    
    resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_resources.json')
    final_terms_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_final.txt')
    metadata_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_metadata.json')
    output_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
    
    if not os.path.exists(resources_file):
        if verbose:
            print(f"Resources file {resources_file} not found")
        return
        
    if not os.path.exists(final_terms_file):
        if verbose:
            print(f"Final terms file {final_terms_file} not found")
        return
    
    # Read final terms
    final_terms = set()
    with open(final_terms_file, 'r', encoding='utf-8') as f:
        for line in f:
            term = line.strip().lower()
            if term:
                final_terms.add(term)
    
    # Get variations from metadata if requested
    variations_map = {}  # Maps variation to canonical term
    if include_variations and os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Find all variations
            for term, term_data in metadata.items():
                if 'canonical_term' in term_data:
                    # This is a variation
                    canonical = term_data['canonical_term']
                    variations_map[term] = canonical
                elif 'variations' in term_data:
                    # This is a canonical term
                    for variation in term_data['variations']:
                        variations_map[variation] = term
        except Exception as e:
            if verbose:
                print(f"Error reading metadata file for variations: {str(e)}")
    
    if verbose:
        print(f"Found {len(final_terms)} terms in final file")
        if include_variations:
            print(f"Found {len(variations_map)} variations in metadata")
        print(f"Processing resources file: {resources_file}")
    
    # Initialize filtered resources dictionary
    filtered_resources = {}
    
    # Process resources file in chunks to handle large files
    try:
        with open(resources_file, 'r', encoding='utf-8') as f:
            resources_data = json.load(f)
            
            # Process each term's resources
            for term, resources in resources_data.items():
                term_lower = term.lower()
                
                # Check if this is a final term or a variation of a final term
                is_final_term = term_lower in final_terms
                canonical_term = None
                
                if not is_final_term and include_variations:
                    # Check if this is a variation
                    if term_lower in variations_map:
                        canonical_term = variations_map[term_lower]
                        if canonical_term in final_terms:
                            is_final_term = True
                
                # Skip terms not in final.txt or not variations of final terms
                if not is_final_term:
                    continue
                    
                if not isinstance(resources, list):
                    continue
                
                # Filter resources based on criteria
                filtered_term_resources = []
                for resource in resources:
                    # Check if resource has required fields and meets criteria
                    if (isinstance(resource, dict) and 
                        resource.get('is_verified') is True and 
                        resource.get('relevance_score', 0) > 0.75 and
                        'url' in resource and 
                        'title' in resource and 
                        'processed_content' in resource and 
                        'score' in resource):
                        
                        # Only include specified fields
                        filtered_resource = {
                            'url': resource['url'],
                            'title': resource['title'],
                            'processed_content': resource['processed_content'],
                            'score': resource['score'],
                            'relevance_score': resource['relevance_score']
                        }
                        filtered_term_resources.append(filtered_resource)
                
                if filtered_term_resources:
                    # If this is a variation, add resources to the canonical term
                    target_term = canonical_term if canonical_term else term_lower
                    
                    # Avoid duplicates by checking URLs
                    if target_term in filtered_resources:
                        existing_urls = {r["url"] for r in filtered_resources[target_term]}
                        for resource in filtered_term_resources:
                            if resource["url"] not in existing_urls:
                                filtered_resources[target_term].append(resource)
                                existing_urls.add(resource["url"])
                    else:
                        filtered_resources[target_term] = filtered_term_resources
        
        # Save filtered resources to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_resources, f, indent=2, ensure_ascii=False)
        
        # Also save to the final directory
        final_output_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_resources, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"Filtered resources saved to original location: {output_file}")
            print(f"Filtered resources saved to final location: {final_output_file}")
            print(f"Collected resources for {len(filtered_resources)} terms")
    
    except Exception as e:
        print(f"Error processing resources file {resources_file}: {str(e)}")


def promote_terms_based_on_parents(verbose: bool = False) -> None:
    """Promote terms between levels based on their parent-child relationships.
    
    This function identifies terms that need promotion due to level gaps with their parents:
    - Level 3 terms with only level 0/1 parents are promoted to level 2
    - Level 2 terms with only level 0 parents are promoted to level 1
    - Level 1 terms with no parents might be promoted to level 0 (needs review)
    
    The original metadata and resources remain untouched. All changes are written 
    to the data/final directory structure.
    """
    # First ensure the final directories exist
    ensure_final_dirs_exist()
    
    # Load metadata from all levels
    metadata = {}
    resources = {}
    
    # Load metadata and resources for all levels
    for level in range(4):  # Levels 0, 1, 2, 3
        metadata_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_metadata.json')
        
        if not os.path.exists(metadata_file):
            if verbose:
                print(f"Metadata file not found: {metadata_file}")
            metadata[level] = {}
            continue
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata[level] = json.load(f)
            
        if verbose:
            print(f"Loaded metadata for {len(metadata[level])} terms from level {level}")
        
        # Load resources
        filtered_resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
        resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_resources.json')
        
        if os.path.exists(filtered_resources_file):
            with open(filtered_resources_file, 'r', encoding='utf-8') as f:
                resources[level] = json.load(f)
        elif os.path.exists(resources_file):
            with open(resources_file, 'r', encoding='utf-8') as f:
                resources[level] = json.load(f)
        else:
            resources[level] = {}
        
        if verbose and resources[level]:
            print(f"Loaded resources for {len(resources[level])} terms from level {level}")
    
    # Create a working copy of the metadata to modify
    final_metadata = {}
    final_resources = {}
    
    # First, copy all existing metadata and resources to the final structure
    for level in range(4):
        final_metadata[level] = metadata[level].copy() if level in metadata else {}
        final_resources[level] = resources[level].copy() if level in resources else {}
    
    # Find terms that need promotion
    promotions = {}
    
    # Find level 3 terms with only level 0/1 parents
    if 3 in metadata:
        for term, term_data in metadata[3].items():
            # Skip variations
            if 'canonical_term' in term_data:
                continue
                
            parents = term_data.get('parents', [])
            if not parents:
                continue
                
            parent_levels = set()
            
            # Find levels of the parents
            for parent in parents:
                for level in range(3):  # Check levels 0, 1, 2
                    if parent in metadata[level] and 'canonical_term' not in metadata[level][parent]:
                        parent_levels.add(level)
            
            # If the term only has level 0/1 parents, promote to level 2
            if parent_levels and max(parent_levels) <= 1:
                promotions[term] = {
                    'from_level': 3,
                    'to_level': 2,
                    'data': term_data,
                    'reason': f"Term has only level 0/1 parents: {parents}"
                }
    
    # Find level 2 terms with only level 0 parents
    if 2 in metadata:
        for term, term_data in metadata[2].items():
            # Skip variations
            if 'canonical_term' in term_data:
                continue
                
            parents = term_data.get('parents', [])
            if not parents:
                continue
                
            parent_levels = set()
            
            # Find levels of the parents
            for parent in parents:
                for level in range(2):  # Check levels 0, 1
                    if parent in metadata[level] and 'canonical_term' not in metadata[level][parent]:
                        parent_levels.add(level)
            
            # If the term only has level 0 parents, promote to level 1
            if parent_levels and max(parent_levels) == 0:
                promotions[term] = {
                    'from_level': 2,
                    'to_level': 1,
                    'data': term_data,
                    'reason': f"Term has only level 0 parents: {parents}"
                }
    
    # Apply promotions to the final metadata and resources
    if promotions:
        if verbose:
            print(f"Found {len(promotions)} terms to promote")
            
        # Apply promotions
        for term, promotion_data in promotions.items():
            from_level = promotion_data['from_level']
            to_level = promotion_data['to_level']
            term_data = promotion_data['data']
            
            # Move the term to the new level in final metadata
            final_metadata[to_level][term] = term_data
            # Remove from the original level
            del final_metadata[from_level][term]
            
            # Move resources if available
            if term in final_resources[from_level]:
                if term not in final_resources[to_level]:
                    final_resources[to_level][term] = final_resources[from_level][term]
                del final_resources[from_level][term]
            
            if verbose:
                print(f"Promoted '{term}' from level {from_level} to level {to_level}: {promotion_data['reason']}")
    
    # Save the final metadata, terms, and resources to the final directory
    for level in range(4):
        if level not in final_metadata:
            continue
            
        # Save metadata
        metadata_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(final_metadata[level], f, indent=2, ensure_ascii=False)
        
        # Generate and save final.txt with canonical terms only
        terms = [term for term, data in final_metadata[level].items() if 'canonical_term' not in data]
        final_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_final.txt')
        with open(final_file, 'w', encoding='utf-8') as f:
            for term in sorted(terms):
                f.write(f"{term}\n")
        
        # Save resources if they exist
        if final_resources.get(level, {}):
            resources_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
            with open(resources_file, 'w', encoding='utf-8') as f:
                json.dump(final_resources[level], f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"Saved level {level} data to {FINAL_DIR}/lv{level}/")
            print(f"  - {len(terms)} terms in final.txt")
            print(f"  - {len(final_metadata[level])} entries in metadata")
            if final_resources.get(level, {}):
                print(f"  - Resources for {len(final_resources[level])} terms")
    
    # Create a promotion log
    log_file = os.path.join(FINAL_DIR, 'promotion_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            'promoted_terms': {term: data['reason'] for term, data in promotions.items()},
            'total_promotions': len(promotions),
            'level_stats': {
                level: len([t for t, d in final_metadata.get(level, {}).items() if 'canonical_term' not in d])
                for level in range(4)
            }
        }, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"Promotion log saved to {log_file}")
    
    return promotions


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Collect metadata for terms after level completion')
    parser.add_argument('level', type=int, help='Level number (0, 1, 2, 3, etc.)')
    parser.add_argument('-o', '--output', type=str, default=None, 
                        help='Output file path (default: data/lvX/metadata.json)')
    parser.add_argument('-r', '--resources', action='store_false', dest='skip_resources',
                        help='Skip collecting resources (default: collect resources)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-variations', action='store_true',
                        help='Do not include metadata for variations')
    parser.add_argument('-p', '--promote', action='store_false', dest='skip_promotion',
                        help='Skip promoting terms based on parent relationships (default: promote terms)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'data/lv{args.level}/metadata.json'
    
    # Ensure final directories exist
    ensure_final_dirs_exist()
    
    # Collect metadata
    collect_metadata(args.level, args.verbose, not args.no_variations, merge_variation_metadata=True)
    
    # Collect resources if not skipped
    if not args.skip_resources:
        collect_resources(args.level, args.verbose, not args.no_variations)
    
    # Promote terms if not skipped and we've processed the highest level
    if not args.skip_promotion and args.level == 3:  # Updated to level 3
        if args.verbose:
            print("\nPromoting terms based on parent relationships...")
        promote_terms_based_on_parents(args.verbose)


if __name__ == '__main__':
    main() 