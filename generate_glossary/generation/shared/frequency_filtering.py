"""
Shared frequency filtering functionality for levels 1-3.

This module provides the generic s2 (frequency-based filtering) logic that can be 
configured for different levels through the level_config module.
"""

import os
import json
import time
import csv
from typing import Dict, List, Any, Set, Counter as CounterType
from collections import Counter, defaultdict
from pathlib import Path

from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import ensure_directories
from generate_glossary.deduplicator.dedup_utils import normalize_text
from .level_config import get_level_config


# Configuration constants
MIN_CONCEPT_LENGTH = 2
MAX_CONCEPT_LENGTH = 100

# Non-academic terms to exclude
NON_ACADEMIC_TERMS = {
    "page", "home", "about", "contact", "staff", "faculty", "links",
    "click", "here", "website", "portal", "login", "apply", "apply now",
    "register", "registration", "more", "learn more", "read more",
    "back", "next", "previous", "link", "site", "menu", "navigation"
}


def is_valid_concept(concept: str) -> bool:
    """
    Check if a concept is valid based on basic criteria.
    
    Args:
        concept: The concept string to validate
        
    Returns:
        Boolean indicating if the concept is valid
    """
    # Check length criteria
    if not concept or len(concept) < MIN_CONCEPT_LENGTH or len(concept) > MAX_CONCEPT_LENGTH:
        return False
        
    # Exclude single characters and pure numbers
    if len(concept) <= 2 and (concept.isdigit() or concept.isalpha()):
        return False
        
    # Exclude non-academic terms
    if concept.lower() in NON_ACADEMIC_TERMS:
        return False
        
    # Check if concept has more than just punctuation and spaces
    if not any(c.isalnum() for c in concept):
        return False
        
    return True


def load_concept_source_mapping(input_file: str) -> tuple[List[str], Dict[str, List[str]]]:
    """
    Load concepts and create source mapping.
    
    Args:
        input_file: Path to input file
        
    Returns:
        Tuple of (all_concepts, source_concept_mapping)
    """
    all_concepts = []
    source_concept_mapping = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Handle format: "source - concept" or just "concept"
            if ' - ' in line:
                source, concept = line.split(' - ', 1)
                source = source.strip()
                concept = concept.strip()
            else:
                # No source info, treat as generic
                source = 'generic'
                concept = line.strip()
            
            if is_valid_concept(concept):
                all_concepts.append(concept)
                source_concept_mapping[source].append(concept)
    
    return all_concepts, dict(source_concept_mapping)


def count_source_frequencies(source_concept_mapping: Dict[str, List[str]]) -> tuple[
    Dict[str, int], Dict[str, str], Dict[str, Set[str]]
]:
    """
    Count how many sources each concept appears in.
    
    Args:
        source_concept_mapping: Dictionary mapping sources to their concepts
        
    Returns:
        Tuple containing:
        - Dictionary mapping concepts to their source frequency
        - Dictionary mapping normalized forms to original forms  
        - Dictionary mapping concepts to the set of sources they appear in
    """
    concept_source_count = Counter()
    normalized_to_original = {}
    concept_sources = defaultdict(set)
    
    for source, concepts in source_concept_mapping.items():
        unique_concepts_in_source = set()
        
        for concept in concepts:
            normalized = normalize_text(concept)
            
            # Track original form
            if normalized not in normalized_to_original:
                normalized_to_original[normalized] = concept
            
            # Count once per source (avoid double-counting within same source)
            unique_concepts_in_source.add(normalized)
            concept_sources[normalized].add(source)
        
        # Increment count for each unique concept in this source
        for normalized_concept in unique_concepts_in_source:
            concept_source_count[normalized_concept] += 1
    
    return dict(concept_source_count), normalized_to_original, dict(concept_sources)


def apply_institutional_frequency_filter(
    concepts: List[str],
    source_concept_mapping: Dict[str, List[str]],
    threshold_percent: float
) -> tuple[List[str], Dict[str, Any]]:
    """
    Apply institutional frequency filtering (for levels 1-2).
    
    Args:
        concepts: List of all concepts
        source_concept_mapping: Mapping from sources to concepts
        threshold_percent: Minimum percentage of sources required
        
    Returns:
        Tuple of (filtered_concepts, statistics)
    """
    logger = setup_logger("frequency_filter")
    
    # Count frequencies
    concept_frequencies, normalized_to_original, concept_sources = count_source_frequencies(
        source_concept_mapping
    )
    
    total_sources = len(source_concept_mapping)
    min_sources = int(total_sources * threshold_percent)
    
    logger.info(f"Total sources: {total_sources}, minimum required: {min_sources} ({threshold_percent*100}%)")
    
    # Filter concepts by frequency
    filtered_concepts = []
    for concept in concepts:
        normalized = normalize_text(concept)
        frequency = concept_frequencies.get(normalized, 0)
        
        if frequency >= min_sources:
            original_form = normalized_to_original.get(normalized, concept)
            filtered_concepts.append(original_form)
    
    # Generate statistics
    statistics = {
        'total_concepts_before': len(concepts),
        'total_concepts_after': len(filtered_concepts),
        'total_sources': total_sources,
        'minimum_sources_required': min_sources,
        'threshold_percent': threshold_percent,
        'concepts_removed': len(concepts) - len(filtered_concepts),
        'frequency_distribution': dict(Counter(concept_frequencies.values())),
        'concept_frequencies': {
            normalized_to_original.get(k, k): v 
            for k, v in concept_frequencies.items()
            if v >= min_sources
        }
    }
    
    logger.info(f"Filtered {len(concepts)} → {len(filtered_concepts)} concepts")
    
    return filtered_concepts, statistics


def apply_venue_frequency_filter(
    concepts: List[str],
    source_concept_mapping: Dict[str, List[str]]
) -> tuple[List[str], Dict[str, Any]]:
    """
    Apply venue-based frequency filtering (for level 3).
    
    Args:
        concepts: List of all concepts
        source_concept_mapping: Mapping from sources to concepts
        
    Returns:
        Tuple of (filtered_concepts, statistics)
    """
    logger = setup_logger("frequency_filter")
    
    # Count frequencies across venues
    concept_frequencies, normalized_to_original, concept_sources = count_source_frequencies(
        source_concept_mapping
    )
    
    total_venues = len(source_concept_mapping)
    
    # For conference topics, use a different strategy:
    # - Require appearance in multiple venues
    # - But be less strict than institutional filtering
    min_venues = max(2, int(total_venues * 0.1))  # At least 2 venues, or 10% of total
    
    logger.info(f"Total venues: {total_venues}, minimum required: {min_venues}")
    
    # Filter concepts by venue frequency
    filtered_concepts = []
    for concept in concepts:
        normalized = normalize_text(concept)
        frequency = concept_frequencies.get(normalized, 0)
        
        if frequency >= min_venues:
            original_form = normalized_to_original.get(normalized, concept)
            filtered_concepts.append(original_form)
    
    # Generate statistics
    statistics = {
        'total_concepts_before': len(concepts),
        'total_concepts_after': len(filtered_concepts),
        'total_venues': total_venues,
        'minimum_venues_required': min_venues,
        'concepts_removed': len(concepts) - len(filtered_concepts),
        'frequency_distribution': dict(Counter(concept_frequencies.values())),
        'concept_frequencies': {
            normalized_to_original.get(k, k): v 
            for k, v in concept_frequencies.items()
            if v >= min_venues
        }
    }
    
    logger.info(f"Venue-filtered {len(concepts)} → {len(filtered_concepts)} concepts")
    
    return filtered_concepts, statistics


def save_filtering_results(
    filtered_concepts: List[str],
    statistics: Dict[str, Any],
    output_file: str,
    metadata_file: str,
    level: int
):
    """Save filtering results to files."""
    logger = setup_logger(f"lv{level}.s2")
    
    # Save filtered concepts
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for concept in filtered_concepts:
            f.write(concept + '\n')
    
    # Save metadata with statistics
    metadata = {
        'level': level,
        'step': 's2',
        'processing_timestamp': time.time(),
        'config_used': {
            'threshold_percent': get_level_config(level).frequency_threshold,
        },
        'statistics': statistics
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(filtered_concepts)} filtered concepts to {output_file}")


def export_analysis_csv(
    concepts: List[str],
    source_concept_mapping: Dict[str, List[str]],
    statistics: Dict[str, Any],
    level: int
):
    """Export detailed analysis to CSV for manual review."""
    logger = setup_logger(f"lv{level}.s2")
    
    # Create analysis file
    analysis_file = f"data/lv{level}/raw/lv{level}_s2_analysis.csv"
    
    concept_frequencies, normalized_to_original, concept_sources = count_source_frequencies(
        source_concept_mapping
    )
    
    # Prepare CSV data
    csv_data = []
    for concept in concepts:
        normalized = normalize_text(concept)
        frequency = concept_frequencies.get(normalized, 0)
        sources = list(concept_sources.get(normalized, set()))
        
        csv_data.append({
            'concept': concept,
            'normalized': normalized,
            'frequency': frequency,
            'sources': ', '.join(sources[:5]),  # Limit for readability
            'total_sources': len(sources),
            'included': frequency >= statistics.get('minimum_sources_required', 0)
        })
    
    # Sort by frequency (descending)
    csv_data.sort(key=lambda x: x['frequency'], reverse=True)
    
    # Write CSV
    os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
    with open(analysis_file, 'w', newline='', encoding='utf-8') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    
    logger.info(f"Exported analysis to {analysis_file}")


def filter_by_frequency(
    input_file: str,
    level: int,
    output_file: str,
    metadata_file: str
) -> Dict[str, Any]:
    """
    Generic frequency filtering for any level.
    
    Args:
        input_file: Path to file containing concepts with source mapping
        level: Generation level (1, 2, or 3)
        output_file: Path to save filtered concepts
        metadata_file: Path to save processing metadata
        
    Returns:
        Dictionary containing processing results and metadata
    """
    logger = setup_logger(f"lv{level}.s2")
    config = get_level_config(level)
    
    # Ensure directories exist
    ensure_directories(level)
    
    logger.info(f"Starting Level {level} frequency filtering: {config.processing_description}")
    
    # Load concepts and source mapping
    concepts, source_concept_mapping = load_concept_source_mapping(input_file)
    logger.info(f"Loaded {len(concepts)} concepts from {len(source_concept_mapping)} sources")
    
    if not concepts:
        logger.warning("No concepts found to filter")
        return {'error': 'No concepts found'}
    
    # Apply appropriate filtering based on level
    if config.frequency_threshold == "venue_based":
        # Level 3: Venue-based filtering
        filtered_concepts, statistics = apply_venue_frequency_filter(
            concepts, source_concept_mapping
        )
    else:
        # Levels 1-2: Institutional frequency filtering
        filtered_concepts, statistics = apply_institutional_frequency_filter(
            concepts, source_concept_mapping, config.frequency_threshold
        )
    
    # Save results
    save_filtering_results(
        filtered_concepts, statistics, output_file, metadata_file, level
    )
    
    # Export analysis CSV for manual review
    export_analysis_csv(concepts, source_concept_mapping, statistics, level)
    
    # Return processing metadata
    return {
        'level': level,
        'step': 's2', 
        'success': True,
        'input_concepts_count': len(concepts),
        'filtered_concepts_count': len(filtered_concepts),
        'sources_count': len(source_concept_mapping),
        'filtering_method': 'venue_based' if config.frequency_threshold == 'venue_based' else 'institutional',
        'statistics': statistics,
        'processing_description': config.processing_description
    }