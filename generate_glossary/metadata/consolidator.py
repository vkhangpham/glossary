"""
Metadata consolidation utilities.

This module handles consolidating metadata from multiple sources,
merging variations, and preparing final output.
"""

import json
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from pathlib import Path


def consolidate_metadata_for_term(term: str, metadata_sources: List[Dict]) -> Dict[str, Any]:
    """
    Consolidate metadata from multiple sources for a single term.
    
    Args:
        term: The term to consolidate metadata for
        metadata_sources: List of metadata dictionaries from different sources
        
    Returns:
        Consolidated metadata dictionary
    """
    consolidated = {
        'sources': [],
        'parents': [],
        'variations': [],
        'resources': []
    }
    
    for source_metadata in metadata_sources:
        if 'sources' in source_metadata:
            consolidated['sources'].extend(source_metadata['sources'])
        
        if 'parents' in source_metadata:
            consolidated['parents'].extend(source_metadata['parents'])
        
        if 'variations' in source_metadata:
            consolidated['variations'].extend(source_metadata['variations'])
        
        if 'resources' in source_metadata:
            consolidated['resources'].extend(source_metadata['resources'])
    
    # Remove duplicates while preserving order
    consolidated['sources'] = list(dict.fromkeys(consolidated['sources']))
    consolidated['parents'] = list(dict.fromkeys(consolidated['parents']))
    consolidated['variations'] = list(dict.fromkeys(consolidated['variations']))
    
    return consolidated


def merge_resource_data(resources: List[Dict]) -> Dict[str, Any]:
    """
    Merge resource data from multiple sources.
    
    Args:
        resources: List of resource dictionaries
        
    Returns:
        Merged resource dictionary
    """
    merged = {
        'urls': [],
        'titles': [],
        'descriptions': [],
        'content': []
    }
    
    for resource in resources:
        if 'url' in resource:
            merged['urls'].append(resource['url'])
        
        if 'title' in resource:
            merged['titles'].append(resource['title'])
        
        if 'description' in resource:
            merged['descriptions'].append(resource['description'])
        
        if 'content' in resource:
            merged['content'].append(resource['content'])
    
    # Remove duplicates
    merged['urls'] = list(dict.fromkeys(merged['urls']))
    merged['titles'] = list(dict.fromkeys(merged['titles']))
    
    return merged


def consolidate_variations(primary_term: str, variations: List[str], 
                          metadata_dict: Dict) -> Dict[str, Any]:
    """
    Consolidate metadata from variations into the primary term.
    
    Args:
        primary_term: The canonical/primary term
        variations: List of variation terms
        metadata_dict: Dictionary containing metadata for all terms
        
    Returns:
        Consolidated metadata for the primary term
    """
    # Start with primary term's metadata
    consolidated = metadata_dict.get(primary_term, {
        'sources': [],
        'parents': [],
        'variations': variations,
        'resources': []
    })
    
    # Merge metadata from each variation
    for variation in variations:
        if variation in metadata_dict:
            var_metadata = metadata_dict[variation]
            
            if 'sources' in var_metadata:
                consolidated['sources'].extend(var_metadata['sources'])
            
            if 'parents' in var_metadata:
                consolidated['parents'].extend(var_metadata['parents'])
            
            if 'resources' in var_metadata:
                consolidated['resources'].extend(var_metadata['resources'])
    
    # Remove duplicates
    consolidated['sources'] = list(dict.fromkeys(consolidated['sources']))
    consolidated['parents'] = list(dict.fromkeys(consolidated['parents']))
    consolidated['variations'] = list(dict.fromkeys(variations))
    
    return consolidated


def promote_terms_based_on_parents(metadata: Dict[str, Dict], parent_terms: Set[str],
                                   verbose: bool = False) -> Dict[str, Dict]:
    """
    Promote terms to final status based on parent relationships.
    
    Args:
        metadata: Dictionary of term metadata
        parent_terms: Set of valid parent terms
        verbose: Whether to print verbose output
        
    Returns:
        Updated metadata dictionary with promoted terms
    """
    promoted = {}
    
    for term, term_data in metadata.items():
        # Check if term has valid parent relationships
        valid_parents = [p for p in term_data.get('parents', []) if p in parent_terms]
        
        if valid_parents:
            # Promote term
            promoted[term] = term_data
            promoted[term]['promoted'] = True
            promoted[term]['valid_parents'] = valid_parents
            
            if verbose:
                print(f"Promoted '{term}' based on parents: {valid_parents}")
        else:
            # Keep term but mark as not promoted
            promoted[term] = term_data
            promoted[term]['promoted'] = False
    
    return promoted


def consolidate_hierarchy_relationships(metadata: Dict[str, Dict], 
                                       hierarchy_data: Dict) -> Dict[str, Dict]:
    """
    Consolidate hierarchy relationships into metadata.
    
    Args:
        metadata: Term metadata dictionary
        hierarchy_data: Hierarchy relationship data
        
    Returns:
        Updated metadata with hierarchy relationships
    """
    for term, term_data in metadata.items():
        if term in hierarchy_data:
            hierarchy_info = hierarchy_data[term]
            
            # Add children if present
            if 'children' in hierarchy_info:
                term_data['children'] = hierarchy_info['children']
            
            # Update parents if more accurate in hierarchy
            if 'parents' in hierarchy_info:
                # Merge with existing parents
                existing_parents = set(term_data.get('parents', []))
                hierarchy_parents = set(hierarchy_info['parents'])
                term_data['parents'] = list(existing_parents | hierarchy_parents)
            
            # Add level information
            if 'level' in hierarchy_info:
                term_data['level'] = hierarchy_info['level']
    
    return metadata


def create_summary_statistics(metadata: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Create summary statistics for the metadata.
    
    Args:
        metadata: Term metadata dictionary
        
    Returns:
        Dictionary containing summary statistics
    """
    stats = {
        'total_terms': len(metadata),
        'canonical_terms': 0,
        'variations': 0,
        'terms_with_sources': 0,
        'terms_with_parents': 0,
        'terms_with_resources': 0,
        'average_sources_per_term': 0,
        'average_parents_per_term': 0,
        'average_variations_per_term': 0
    }
    
    total_sources = 0
    total_parents = 0
    total_variations = 0
    
    for term, term_data in metadata.items():
        # Check if canonical or variation
        if 'canonical_term' in term_data:
            stats['variations'] += 1
        else:
            stats['canonical_terms'] += 1
        
        # Count terms with different metadata
        if term_data.get('sources'):
            stats['terms_with_sources'] += 1
            total_sources += len(term_data['sources'])
        
        if term_data.get('parents'):
            stats['terms_with_parents'] += 1
            total_parents += len(term_data['parents'])
        
        if term_data.get('resources'):
            stats['terms_with_resources'] += 1
        
        if term_data.get('variations'):
            total_variations += len(term_data['variations'])
    
    # Calculate averages
    if stats['canonical_terms'] > 0:
        stats['average_sources_per_term'] = round(total_sources / stats['canonical_terms'], 2)
        stats['average_parents_per_term'] = round(total_parents / stats['canonical_terms'], 2)
        stats['average_variations_per_term'] = round(total_variations / stats['canonical_terms'], 2)
    
    return stats


def save_consolidated_metadata(metadata: Dict[str, Dict], output_path: Path,
                               include_stats: bool = True, verbose: bool = False) -> None:
    """
    Save consolidated metadata to JSON file.
    
    Args:
        metadata: Consolidated metadata dictionary
        output_path: Path to save the output file
        include_stats: Whether to include summary statistics
        verbose: Whether to print verbose output
    """
    output_data = {
        'metadata': metadata
    }
    
    if include_stats:
        stats = create_summary_statistics(metadata)
        output_data['statistics'] = stats
        
        if verbose:
            print("\nMetadata Statistics:")
            print(f"  Total terms: {stats['total_terms']}")
            print(f"  Canonical terms: {stats['canonical_terms']}")
            print(f"  Variations: {stats['variations']}")
            print(f"  Terms with sources: {stats['terms_with_sources']}")
            print(f"  Terms with parents: {stats['terms_with_parents']}")
            print(f"  Average sources per term: {stats['average_sources_per_term']}")
            print(f"  Average parents per term: {stats['average_parents_per_term']}")
            print(f"  Average variations per term: {stats['average_variations_per_term']}")
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\nSaved consolidated metadata to: {output_path}")