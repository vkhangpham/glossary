"""
Shared frequency filtering functionality for all levels.

This module provides the generic s2 (frequency-based filtering) logic that can be 
configured for different levels through the level_config module.
"""

import os
import json
import time
import csv
import math
from typing import Dict, List, Any, Set, Counter as CounterType, Optional
from collections import Counter, defaultdict
from pathlib import Path

from generate_glossary.utils.error_handler import (
    handle_error, processing_context
)
from generate_glossary.utils.logger import get_logger, log_processing_step
from generate_glossary.config import ensure_directories, get_step_config, get_processing_config
from generate_glossary.deduplication.utils import normalize_text
from generate_glossary.config import get_level_config


# Configuration constants are now centralized in config.py
# Functions below will read from get_processing_config(level) for consistency


def normalize_threshold_percent(threshold: float) -> float:
    """
    Normalize threshold percentage to 0-1 range consistently.
    
    Args:
        threshold: Threshold value that may be in 0-1 range or 1-100 range
        
    Returns:
        float: Normalized threshold in 0-1 range
    """
    if threshold > 1:
        return threshold / 100.0
    return threshold


def is_valid_concept(concept: str, processing_config) -> bool:
    """
    Check if a concept is valid based on basic criteria.
    
    Args:
        concept: The concept string to validate
        processing_config: ProcessingConfig instance with validation settings
        
    Returns:
        Boolean indicating if the concept is valid
    """
    # Check length criteria
    if not concept or len(concept) < processing_config.min_concept_length or len(concept) > processing_config.max_concept_length:
        return False
        
    # Exclude single characters and pure numbers
    if len(concept) <= 2 and (concept.isdigit() or concept.isalpha()):
        return False
        
    # Exclude non-academic terms
    if concept.lower() in processing_config.non_academic_terms:
        return False
        
    if not any(c.isalnum() for c in concept):
        return False
        
    return True


def load_concept_source_mapping(input_file: str, processing_config) -> tuple[List[str], Dict[str, List[str]]]:
    """
    Load concepts and create source mapping.
    
    Args:
        input_file: Path to input file
        processing_config: ProcessingConfig instance with validation settings
        
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
                
            if ' - ' in line:
                # Use rsplit to handle sources like "Institution - College" correctly
                source, concept = line.rsplit(' - ', 1)
                source = source.strip()
                concept = concept.strip()
            else:
                source = 'generic'
                concept = line.strip()
            
            if is_valid_concept(concept, processing_config):
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
            
            if normalized not in normalized_to_original:
                normalized_to_original[normalized] = concept
            
            unique_concepts_in_source.add(normalized)
            concept_sources[normalized].add(source)
        
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
        threshold_percent: Minimum percentage of sources required.
                          Accepts either fraction format (0.6 for 60%) or percentage format (60 for 60%).
                          Values > 1 are automatically normalized to fractions.
        
    Returns:
        Tuple of (filtered_concepts, statistics)
    """
    with processing_context("institutional_frequency_filter") as correlation_id:
        logger = get_logger("frequency_filter")
        
        log_processing_step(
            logger,
            "institutional_frequency_filter",
            "started",
            {
                "concepts_count": len(concepts),
                "sources_count": len(source_concept_mapping),
                "threshold_percent": threshold_percent
            }
        )
        
        try:
            # Count frequencies
            concept_frequencies, normalized_to_original, concept_sources = count_source_frequencies(
                source_concept_mapping
            )
            
            # Normalize threshold_percent to 0-1 range consistently
            threshold_percent = normalize_threshold_percent(threshold_percent)
            
            total_sources = len(source_concept_mapping)
            min_sources = max(1, math.ceil(total_sources * threshold_percent))
            
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
            
            log_processing_step(
                logger,
                "institutional_frequency_filter",
                "completed",
                {
                    "concepts_before": len(concepts),
                    "concepts_after": len(filtered_concepts),
                    "concepts_removed": len(concepts) - len(filtered_concepts)
                }
            )
            
            logger.info(f"Filtered {len(concepts)} → {len(filtered_concepts)} concepts")
            
            return filtered_concepts, statistics
            
        except Exception as e:
            handle_error(
                e,
                context={
                    "concepts_count": len(concepts),
                    "sources_count": len(source_concept_mapping),
                    "threshold_percent": threshold_percent,
                    "correlation_id": correlation_id
                },
                operation="institutional_frequency_filter",
                reraise=True
            )


def apply_level0_institution_frequency_filter(
    concepts: List[str],
    source_concept_mapping: Dict[str, List[str]],
    threshold_percent: float = 60
) -> tuple[List[str], Dict[str, Any]]:
    """
    Apply Level 0 institution-based frequency filtering.
    
    Level 0 has special filtering: concepts must appear in X% of institutions
    (where institution is extracted from source string format: "Institution - College").
    
    Args:
        concepts: List of all concepts
        source_concept_mapping: Mapping from sources to concepts (sources are "Institution - College" format)
        threshold_percent: Minimum percentage of institutions required (default: 60)
                          Accepts both percentage format (60 for 60%) and fraction format (0.6 for 60%)
        
    Returns:
        Tuple of (filtered_concepts, statistics)
    """
    with processing_context("level0_institution_filter") as correlation_id:
        logger = get_logger("lv0.s2")
        
        log_processing_step(
            logger,
            "level0_institution_filter",
            "started",
            {
                "concepts_count": len(concepts),
                "sources_count": len(source_concept_mapping),
                "threshold_percent": threshold_percent
            }
        )
        
        try:
            # Extract institutions from sources
            institution_concept_map = defaultdict(set)
            concept_institutions = defaultdict(set)
            
            for source, source_concepts in source_concept_mapping.items():
                # Extract institution from "Institution - College" format
                institution = source.split(' - ')[0].strip() if ' - ' in source else source
                
                for concept in source_concepts:
                    normalized = normalize_text(concept)
                    institution_concept_map[institution].add(normalized)
                    concept_institutions[normalized].add(institution)
            
            # Count unique institutions
            total_institutions = len(institution_concept_map)
            # Handle both percentage (60) and fraction (0.6) inputs
            threshold_percent = normalize_threshold_percent(threshold_percent)
            min_institutions = max(1, math.ceil(total_institutions * threshold_percent))
            
            logger.info(f"Total institutions: {total_institutions}, minimum required: {min_institutions} ({threshold_percent*100}%)")
            
            # Create normalized to original mapping
            normalized_to_original = {}
            for concept in concepts:
                normalized = normalize_text(concept)
                if normalized not in normalized_to_original:
                    normalized_to_original[normalized] = concept
            
            # Filter concepts by institution frequency
            filtered_concepts = []
            concept_frequencies = {}
            
            for concept in concepts:
                normalized = normalize_text(concept)
                institutions = concept_institutions.get(normalized, set())
                frequency = len(institutions)
                
                concept_frequencies[normalized] = {
                    'count': frequency,
                    'institutions': sorted(list(institutions))
                }
                
                if frequency >= min_institutions:
                    filtered_concepts.append(concept)
            
            # Generate frequency distribution
            freq_counts = [data['count'] for data in concept_frequencies.values()]
            freq_distribution = dict(Counter(freq_counts))
            
            # Generate statistics
            statistics = {
                'total_concepts_before': len(concepts),
                'total_concepts_after': len(filtered_concepts),
                'total_institutions': total_institutions,
                'institution_threshold_percent': threshold_percent,
                'min_institutions_required': min_institutions,
                'concepts_removed': len(concepts) - len(filtered_concepts),
                'frequency_distribution': freq_distribution,
                'selected_institutions': sorted(list(institution_concept_map.keys())),
                'concept_frequencies': {
                    normalized_to_original.get(k, k): v
                    for k, v in concept_frequencies.items()
                    if v['count'] >= min_institutions
                }
            }
            
            log_processing_step(
                logger,
                "level0_institution_filter",
                "completed",
                {
                    "concepts_before": len(concepts),
                    "concepts_after": len(filtered_concepts),
                    "institutions_count": total_institutions
                }
            )
            
            logger.info(f"Institution-filtered {len(concepts)} → {len(filtered_concepts)} concepts")
            logger.info("Frequency distribution:")
            for freq, count in sorted(freq_distribution.items()):
                logger.info(f"  {count} concepts appear in {freq} institutions")
            
            return sorted(filtered_concepts), statistics
            
        except Exception as e:
            handle_error(
                e,
                context={
                    "concepts_count": len(concepts),
                    "sources_count": len(source_concept_mapping),
                    "threshold_percent": threshold_percent,
                    "correlation_id": correlation_id
                },
                operation="level0_institution_filter",
                reraise=True
            )


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
    with processing_context("venue_frequency_filter") as correlation_id:
        logger = get_logger("frequency_filter")
        
        log_processing_step(
            logger,
            "venue_frequency_filter", 
            "started",
            {
                "concepts_count": len(concepts),
                "venues_count": len(source_concept_mapping)
            }
        )
        
        try:
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
            
            log_processing_step(
                logger,
                "venue_frequency_filter",
                "completed", 
                {
                    "concepts_before": len(concepts),
                    "concepts_after": len(filtered_concepts),
                    "concepts_removed": len(concepts) - len(filtered_concepts)
                }
            )
            
            logger.info(f"Venue-filtered {len(concepts)} → {len(filtered_concepts)} concepts")
            
            return filtered_concepts, statistics
            
        except Exception as e:
            handle_error(
                e,
                context={
                    "concepts_count": len(concepts),
                    "venues_count": len(source_concept_mapping),
                    "correlation_id": correlation_id
                },
                operation="venue_frequency_filter",
                reraise=True
            )


def save_filtering_results(
    filtered_concepts: List[str],
    statistics: Dict[str, Any],
    output_file: str,
    metadata_file: str,
    level: int
):
    """Save filtering results to files."""
    logger = get_logger(f"lv{level}.s2")
    
    # Save filtered concepts
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for concept in filtered_concepts:
            f.write(concept + '\n')
    
    # Save metadata with statistics
    # Get step config for threshold value
    step_config = get_step_config(level) or get_level_config(level)
    config_used = {
        'min_concept_length': get_processing_config(level).min_concept_length,
        'max_concept_length': get_processing_config(level).max_concept_length,
    }
    
    # For Level 0, only include institution_threshold_percent to avoid unit confusion
    if level == 0:
        config_used['institution_threshold_percent'] = statistics.get('institution_threshold_percent', step_config.frequency_threshold)
    else:
        config_used['threshold_percent'] = step_config.frequency_threshold
    
    metadata = {
        'level': level,
        'step': 's2',
        'processing_timestamp': time.time(),
        'config_used': config_used,
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
    logger = get_logger(f"lv{level}.s2")
    
    analysis_file = f"data/lv{level}/raw/lv{level}_s2_analysis.csv"
    
    concept_frequencies, normalized_to_original, concept_sources = count_source_frequencies(
        source_concept_mapping
    )
    
    # For Level 0, build institution mapping
    if level == 0:
        concept_institutions = defaultdict(set)
        for source, source_concepts in source_concept_mapping.items():
            # Extract institution from "Institution - College" format
            institution = source.split(' - ')[0].strip() if ' - ' in source else source
            for concept in source_concepts:
                normalized = normalize_text(concept)
                concept_institutions[normalized].add(institution)
    
    # Prepare CSV data
    csv_data = []
    for concept in concepts:
        normalized = normalize_text(concept)
        
        if level == 0:
            # For Level 0, count institutions not sources
            institutions = list(concept_institutions.get(normalized, set()))
            frequency = len(institutions)
            threshold = statistics.get('min_institutions_required', 0)
            csv_data.append({
                'concept': concept,
                'normalized': normalized,
                'frequency': frequency,
                'sources': ', '.join(list(concept_sources.get(normalized, set()))[:5]),
                'institutions': ', '.join(institutions[:5]),
                'total_sources': len(concept_sources.get(normalized, set())),
                'included': frequency >= threshold
            })
        else:
            frequency = concept_frequencies.get(normalized, 0)
            sources = list(concept_sources.get(normalized, set()))
            threshold = statistics.get('minimum_sources_required', 0)
            csv_data.append({
                'concept': concept,
                'normalized': normalized,
                'frequency': frequency,
                'sources': ', '.join(sources[:5]),
                'total_sources': len(sources),
                'included': frequency >= threshold
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
    metadata_file: str,
    min_frequency: Optional[int] = None,
    threshold_percent: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generic frequency filtering for any level.
    
    Args:
        input_file: Path to file containing concepts with source mapping
        level: Generation level (0, 1, 2, or 3)
        output_file: Path to save filtered concepts
        metadata_file: Path to save processing metadata
        min_frequency: Optional minimum absolute number of sources required
        threshold_percent: Optional override percentage for filtering
        
    Returns:
        Dictionary containing processing results and metadata
    """
    logger = get_logger(f"lv{level}.s2")
    config = get_level_config(level)
    
    # Ensure directories exist
    ensure_directories(level)
    
    logger.info(f"Starting Level {level} frequency filtering: {config.processing_description}")
    
    # Get processing config for this level
    processing_config = get_processing_config(level)
    
    # Load concepts and source mapping
    concepts, source_concept_mapping = load_concept_source_mapping(input_file, processing_config)
    logger.info(f"Loaded {len(concepts)} concepts from {len(source_concept_mapping)} sources")
    
    if not concepts:
        logger.warning("No concepts found to filter")
        return {'error': 'No concepts found'}
    
    # Determine threshold to use
    effective_threshold = threshold_percent if threshold_percent is not None else config.frequency_threshold
    
    # Apply appropriate filtering based on level
    if level == 0:
        # Level 0: Institution-based filtering (different from institutional frequency)
        # Level 0 uses a percentage of institutions threshold (default 60%)
        institution_threshold = threshold_percent if threshold_percent is not None else 60
        filtered_concepts, statistics = apply_level0_institution_frequency_filter(
            concepts, source_concept_mapping, institution_threshold
        )
    elif config.frequency_threshold == "venue_based":
        # Level 3: Venue-based filtering
        filtered_concepts, statistics = apply_venue_frequency_filter(
            concepts, source_concept_mapping
        )
    else:
        # Levels 1-2: Institutional frequency filtering
        filtered_concepts, statistics = apply_institutional_frequency_filter(
            concepts, source_concept_mapping, effective_threshold
        )
    
    # Apply min_frequency override if provided
    if min_frequency is not None:
        # Re-filter with absolute minimum
        concept_frequencies, normalized_to_original, concept_sources = count_source_frequencies(
            source_concept_mapping
        )
        
        refined_concepts = []
        for concept in filtered_concepts:
            normalized = normalize_text(concept)
            frequency = concept_frequencies.get(normalized, 0)
            if frequency >= min_frequency:
                refined_concepts.append(concept)
        
        logger.info(f"Applied min_frequency={min_frequency} override: {len(filtered_concepts)} → {len(refined_concepts)} concepts")
        statistics['min_frequency_override'] = min_frequency
        statistics['concepts_after_override'] = len(refined_concepts)
        filtered_concepts = refined_concepts
    
    # Update statistics with any overrides
    if threshold_percent is not None:
        statistics['threshold_percent_override'] = threshold_percent
    
    # Save results
    save_filtering_results(
        filtered_concepts, statistics, output_file, metadata_file, level
    )
    
    # Export analysis CSV for manual review
    export_analysis_csv(concepts, source_concept_mapping, statistics, level)
    
    return {
        'level': level,
        'step': 's2', 
        'success': True,
        'input_concepts_count': len(concepts),
        'filtered_concepts_count': len(filtered_concepts),
        'sources_count': len(source_concept_mapping),
        'filtering_method': 'institution_based' if level == 0 else ('venue_based' if config.frequency_threshold == 'venue_based' else 'institutional'),
        'statistics': statistics,
        'processing_description': config.processing_description
    }