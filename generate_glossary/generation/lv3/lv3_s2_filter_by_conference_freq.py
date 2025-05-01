import json
from typing import Dict, List, Any, Set, Tuple
import sys
import os
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Setup logging
logger = setup_logger("lv3.s2")

# Get the base directory (project root)
BASE_DIR = os.getcwd()

class Config:
    """Configuration for concept filtering"""

    INPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s1_metadata.json")
    CSV_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s1_hierarchical_concepts.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s2_filtered_concepts.txt")
    OUTPUT_CSV_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s2_filtered_concepts.csv")
    VALIDATION_META_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s2_metadata.json")
    CONFERENCE_FREQ_THRESHOLD = 1
    MIN_RESEARCH_AREA_APPEARANCE = 1  # Concept must appear in at least this many research areas
    MIN_RESEARCH_AREA_FREQ_PERCENT = 1  # Concept must appear in at least this % of conferences within a research area
    NUM_WORKERS = 4  # Number of parallel workers for processing
    BATCH_SIZE = 1000  # Size of batches for parallel processing

def load_concepts_and_metadata() -> Tuple[Set[str], Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    Load concepts and metadata from raw files
    
    Returns:
        Tuple containing:
        - Set of concepts from the input file
        - Dictionary mapping from conference topics to associated research areas
        - Dictionary mapping from research areas to concept lists
    """
    logger.info(f"Loading concepts from {Config.INPUT_FILE}")
    
    # Read raw concepts
    concepts = set()
    with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            concepts.add(line.strip())
    
    logger.info(f"Found {len(concepts)} unique concepts")
    
    # Load metadata
    logger.info(f"Loading metadata from {Config.META_FILE}")
    with open(Config.META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Extract conference topics to research areas mapping from metadata
    conference_to_research_areas = {}
    research_area_conference_topic_mapping = metadata.get("research_area_conference_topic_mapping", {})
    for area, topics in research_area_conference_topic_mapping.items():
        for topic in topics:
            if topic not in conference_to_research_areas:
                conference_to_research_areas[topic] = []
            conference_to_research_areas[topic].append(area)
    
    # Extract research area to concepts mapping from metadata
    research_area_to_concepts = metadata.get("research_area_concept_mapping", {})
    
    return concepts, conference_to_research_areas, research_area_to_concepts

def load_hierarchical_data() -> Dict[str, Dict[str, Set[str]]]:
    """
    Load hierarchical data from CSV file
    
    Returns:
        Dictionary mapping from conference topics to a dictionary of research areas and their concepts
    """
    logger.info(f"Loading hierarchical data from {Config.CSV_FILE}")
    
    hierarchy = defaultdict(lambda: defaultdict(set))
    
    with open(Config.CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 3:
                conference_topic = row[0]
                research_area = row[1]
                concept = row[2]
                
                hierarchy[conference_topic][research_area].add(concept)
    
    logger.info(f"Loaded data for {len(hierarchy)} conference topics")
    
    return hierarchy

def compute_conference_frequency(concepts: Set[str], hierarchy: Dict[str, Dict[str, Set[str]]]) -> Dict[str, int]:
    """
    Compute the frequency of each concept across different conference topics
    
    Args:
        concepts: Set of all concepts
        hierarchy: Hierarchical data mapping of conference topics to research areas and concepts
    
    Returns:
        Dictionary mapping from concept to its frequency across conference topics
    """
    logger.info("Computing concept frequency across conference topics")
    
    concept_freq: Dict[str, int] = defaultdict(int)
    
    # Count how many unique conference topics each concept appears in
    for conference_topic, research_areas in hierarchy.items():
        # Collect all concepts for this conference topic across all research areas
        conference_concepts = set()
        for area_concepts in research_areas.values():
            conference_concepts.update(area_concepts)
        
        # Increment frequency for each concept found in this conference topic
        for concept in conference_concepts:
            concept_freq[concept] += 1
    
    # Ensure all concepts from the input file are counted (even if zero)
    for concept in concepts:
        if concept not in concept_freq:
            concept_freq[concept] = 0
    
    logger.info(f"Computed frequency for {len(concept_freq)} concepts")
    
    return concept_freq

def filter_by_conference_frequency(
    concept_freq: Dict[str, int], 
    min_conferences: int
) -> List[str]:
    """
    Filter concepts by their frequency across conference topics
    
    Args:
        concept_freq: Dictionary mapping from concept to its frequency across conference topics
        min_conferences: Minimum number of conference topics a concept must appear in to be kept
    
    Returns:
        List of concepts that pass the filtering
    """
    logger.info(f"Filtering concepts by conference frequency (min: {min_conferences})")
    
    filtered_concepts = [
        concept
        for concept, freq in concept_freq.items()
        if freq >= min_conferences
    ]
    
    logger.info(f"Retained {len(filtered_concepts)} concepts after frequency filtering")
    
    return sorted(filtered_concepts)

def process_batch(batch, concept_to_keep):
    """Process a batch of rows from the CSV file for parallel filtering"""
    result = []
    
    for row in batch:
        if len(row) >= 3:
            concept = row[2]
            if concept in concept_to_keep:
                result.append(row)
    
    return result

def create_filtered_csv(filtered_concepts: List[str], csv_file: str, output_csv_file: str) -> None:
    """
    Create a filtered CSV file with only the rows containing kept concepts
    
    Args:
        filtered_concepts: List of concepts to keep
        csv_file: Path to the input CSV file
        output_csv_file: Path to the output CSV file
    """
    logger.info(f"Creating filtered CSV from {csv_file} to {output_csv_file}")
    
    # Convert list to a set for faster lookups
    concept_to_keep = set(filtered_concepts)
    
    # Read all rows from the CSV file
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header
        rows = list(reader)
    
    logger.info(f"Read {len(rows)} rows from the CSV file")
    
    # Split rows into batches for parallel processing
    batches = [rows[i:i + Config.BATCH_SIZE] for i in range(0, len(rows), Config.BATCH_SIZE)]
    
    # Process batches in parallel
    logger.info(f"Processing {len(batches)} batches with batch size {Config.BATCH_SIZE}")
    
    filtered_rows = []
    with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        # Process each batch in parallel
        process_fn = partial(process_batch, concept_to_keep=concept_to_keep)
        filtered_batches = list(executor.map(process_fn, batches))
        
        # Combine results
        for batch_result in filtered_batches:
            filtered_rows.extend(batch_result)
    
    logger.info(f"Retained {len(filtered_rows)} rows after filtering")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    
    # Write the filtered rows to the output CSV file
    with open(output_csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered_rows)
    
    logger.info(f"Wrote filtered CSV to {output_csv_file}")

def save_validation_metadata(
    filtered_concepts: List[str],
    concept_freq: Dict[str, int],
    input_concepts_count: int
) -> None:
    """
    Save validation metadata to JSON file
    
    Args:
        filtered_concepts: List of filtered concepts
        concept_freq: Dictionary mapping from concept to its frequency across conference topics
        input_concepts_count: Number of input concepts
    """
    logger.info(f"Saving validation metadata to {Config.VALIDATION_META_FILE}")
    
    # Filter frequency dictionary to only include kept concepts
    filtered_freq = {
        concept: freq
        for concept, freq in concept_freq.items()
        if concept in filtered_concepts
    }
    
    # Concept counts by frequency
    freq_counts = Counter(concept_freq.values())
    
    # Calculate what was filtered out at each frequency level
    freq_cutoffs = {}
    total_concepts = 0
    for i in range(max(concept_freq.values()) + 1):
        retained_count = sum(freq_counts[j] for j in range(i, max(concept_freq.values()) + 1))
        total_concepts += freq_counts.get(i, 0)
        freq_cutoffs[i] = {
            "retained_concepts": retained_count,
            "filtered_concepts": total_concepts - retained_count,
            "retention_percentage": (retained_count / total_concepts * 100) if total_concepts > 0 else 0
        }
    
    metadata = {
        "input_concepts_count": input_concepts_count,
        "output_concepts_count": len(filtered_concepts),
        "percentage_retained": (len(filtered_concepts) / input_concepts_count * 100) if input_concepts_count > 0 else 0,
        "conference_frequency_threshold": Config.CONFERENCE_FREQ_THRESHOLD,
        "min_research_area_appearance": Config.MIN_RESEARCH_AREA_APPEARANCE,
        "min_research_area_freq_percent": Config.MIN_RESEARCH_AREA_FREQ_PERCENT,
        "frequency_cutoff_analysis": freq_cutoffs,
        "concept_conference_frequencies": filtered_freq
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.VALIDATION_META_FILE), exist_ok=True)
    
    with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Saved validation metadata to {Config.VALIDATION_META_FILE}")

def is_concept_same_as_conference_topic(concept: str, conference_topic: str) -> bool:
    """
    Check if a concept is the same as its conference topic name
    
    Args:
        concept: The concept to check
        conference_topic: The conference topic name
        
    Returns:
        Boolean indicating if the concept matches the conference topic name
    """
    # Normalize both strings
    norm_concept = normalize_text(concept)
    norm_topic = normalize_text(conference_topic)
    
    # Clean common prefixes from conference topic 
    prefixes = [
        "special issue on ", "workshop on ", "symposium on ", 
        "conference on ", "track on ", "session on ",
        "topics in ", "advances in ", "recent developments in ",
    ]
    
    for prefix in prefixes:
        if norm_topic.startswith(prefix):
            norm_topic = norm_topic[len(prefix):]
            break
    
    # Check if concept matches the cleaned topic name
    if norm_concept == norm_topic:
        return True
    
    # Check if concept is contained in the topic name as a standalone phrase
    # For example "machine learning" in "advances in machine learning and applications"
    if norm_concept in norm_topic and (
        norm_topic.startswith(norm_concept + " ") or 
        norm_topic.endswith(" " + norm_concept) or 
        " " + norm_concept + " " in norm_topic
    ):
        return True
        
    # Check if concept is part of a compound conference topic name
    # e.g., "deep learning" in "deep learning and computer vision"
    if " and " in norm_topic:
        parts = [part.strip() for part in norm_topic.split(" and ")]
        if norm_concept in parts:
            return True
    
    return False

def analyze_research_area_distribution(
    hierarchy: Dict[str, Dict[str, Set[str]]]
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Set[str]], Dict[str, int]]:
    """
    Analyze concept distribution across research areas
    
    Args:
        hierarchy: Hierarchical data mapping of conference topics to research areas and concepts
    
    Returns:
        Tuple containing:
        - Research area to concept frequency mapping 
        - Concept to research areas mapping
        - Count of conferences per research area
    """
    logger.info("Analyzing concept distribution across research areas")
    
    # Research area -> concept -> count mapping
    area_concept_counts = defaultdict(Counter)
    
    # Concept -> set of research areas mapping
    concept_areas = defaultdict(set)
    
    # Count conferences per research area
    area_conference_counts = Counter()
    
    # Process hierarchy mapping
    for conference_topic, research_areas in hierarchy.items():
        for area, concepts in research_areas.items():
            # Count area as having this conference
            area_conference_counts[area] += 1
            
            # Update concept stats for this research area
            for concept in concepts:
                area_concept_counts[area][concept] += 1
                concept_areas[concept].add(area)
    
    logger.info(f"Found {len(area_concept_counts)} research areas with concepts")
    logger.info(f"Found {len(concept_areas)} unique concepts across research areas")
    
    return (
        {area: dict(counts) for area, counts in area_concept_counts.items()},
        {concept: areas for concept, areas in concept_areas.items()},
        dict(area_conference_counts)
    )

def filter_by_research_area_distribution(
    concepts: Set[str],
    area_concept_counts: Dict[str, Dict[str, int]],
    concept_areas: Dict[str, Set[str]],
    area_conference_counts: Dict[str, int],
    hierarchy: Dict[str, Dict[str, Set[str]]]
) -> List[str]:
    """
    Filter concepts based on their distribution across research areas
    
    Args:
        concepts: Set of all concepts
        area_concept_counts: Research area -> concept -> frequency mapping
        concept_areas: Concept -> research areas mapping
        area_conference_counts: Research area -> conference count mapping
        hierarchy: Hierarchical data mapping of conference topics to research areas and concepts
        
    Returns:
        List of filtered concepts
    """
    logger.info("Filtering concepts by research area distribution")
    
    filtered_concepts = []
    
    # Track skipped concepts that match conference topics
    skipped_topic_concepts = set()
    
    for concept in tqdm(concepts, desc="Filtering concepts"):
        # First check: is this concept the same as any conference topic?
        is_conference_name = False
        for conference_topic in hierarchy.keys():
            if is_concept_same_as_conference_topic(concept, conference_topic):
                skipped_topic_concepts.add((concept, conference_topic))
                is_conference_name = True
                break
                
        if is_conference_name:
            continue
            
        # Second check: research area distribution
        # Get research areas where this concept appears
        areas = concept_areas.get(concept, set())
        
        # Check if concept appears in enough research areas
        if len(areas) >= Config.MIN_RESEARCH_AREA_APPEARANCE:
            # Check research area-level frequencies
            area_percentages = []
            
            for area in areas:
                if area in area_conference_counts and area_conference_counts[area] > 0:
                    # Get concept frequency in this research area
                    concept_freq = area_concept_counts.get(area, {}).get(concept, 0)
                    
                    # Calculate percentage of conferences in this research area having this concept
                    percentage = (concept_freq / area_conference_counts[area]) * 100
                    area_percentages.append(percentage)
            
            # Filter based on minimum percentage threshold
            if any(pct >= Config.MIN_RESEARCH_AREA_FREQ_PERCENT for pct in area_percentages):
                filtered_concepts.append(concept)
    
    # Log skipped concepts
    if skipped_topic_concepts:
        logger.info("\nSkipped concepts that match conference topic names:")
        for concept, topic in sorted(skipped_topic_concepts):
            logger.info(f"  Skipped '{concept}' from '{topic}'")
    
    logger.info(f"Filtered to {len(filtered_concepts)} concepts after research area distribution analysis")
    
    return sorted(filtered_concepts)

def write_research_area_stats(
    validation_meta_file: str,
    area_concept_counts: Dict[str, Dict[str, int]],
    concept_areas: Dict[str, Set[str]],
    area_conference_counts: Dict[str, int],
    filtered_concepts: List[str]
) -> None:
    """
    Add research area distribution statistics to the metadata file
    
    Args:
        validation_meta_file: Path to the metadata file
        area_concept_counts: Research area -> concept -> frequency mapping
        concept_areas: Concept -> research areas mapping
        area_conference_counts: Research area -> conference count mapping
        filtered_concepts: List of filtered concepts
    """
    try:
        # Read existing metadata
        with open(validation_meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Add research area statistics
        metadata['research_area_statistics'] = {
            area: {
                'total_conferences': area_conference_counts.get(area, 0),
                'total_concepts': len(concepts),
                'filtered_concepts': sum(1 for c in concepts if c in filtered_concepts)
            }
            for area, concepts in area_concept_counts.items()
        }
        
        # Add concept distribution across research areas
        metadata['concept_research_area_distribution'] = {
            concept: {
                'total_research_areas': len(concept_areas.get(concept, [])),
                'research_areas': sorted(list(concept_areas.get(concept, [])))
            }
            for concept in filtered_concepts
        }
        
        # Write updated metadata
        with open(validation_meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        logger.info("Research area statistics added to metadata file")
        
    except Exception as e:
        logger.error(f"Error writing research area statistics: {str(e)}")

def main():
    """Main execution function"""
    try:
        # Get custom threshold from command line arguments if provided
        if len(sys.argv) > 1:
            try:
                threshold = int(sys.argv[1])
                Config.CONFERENCE_FREQ_THRESHOLD = threshold
                logger.info(f"Using conference frequency threshold: {threshold}")
            except ValueError:
                logger.warning(f"Invalid threshold value: {sys.argv[1]}. Using default: {Config.CONFERENCE_FREQ_THRESHOLD}")
        
        # Load concepts and metadata
        concepts, conference_to_research_areas, research_area_to_concepts = load_concepts_and_metadata()
        input_concepts_count = len(concepts)
        
        # Load hierarchical data
        hierarchy = load_hierarchical_data()
        
        # Compute conference frequency
        concept_freq = compute_conference_frequency(concepts, hierarchy)
        
        # Analyze research area distribution
        area_concept_counts, concept_areas, area_conference_counts = analyze_research_area_distribution(hierarchy)
        
        # Filter by research area distribution (this also filters out concepts matching conference topics)
        filtered_concepts = filter_by_research_area_distribution(
            concepts, area_concept_counts, concept_areas, area_conference_counts, hierarchy
        )
        logger.info(f"Filtered to {len(filtered_concepts)} concepts after distribution filtering")
        
        # Create filtered CSV
        create_filtered_csv(filtered_concepts, Config.CSV_FILE, Config.OUTPUT_CSV_FILE)
        
        # Save filtered concepts to file
        logger.info(f"Saving filtered concepts to {Config.OUTPUT_FILE}")
        os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
        
        # Save validation metadata
        save_validation_metadata(filtered_concepts, concept_freq, input_concepts_count)
        
        # Write research area statistics
        write_research_area_stats(
            Config.VALIDATION_META_FILE, 
            area_concept_counts, 
            concept_areas, 
            area_conference_counts, 
            filtered_concepts
        )
        
        logger.info("Filtering completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()