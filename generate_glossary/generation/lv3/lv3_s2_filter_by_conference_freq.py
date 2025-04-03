import json
from typing import Dict, List, Any, Set, Tuple
import sys
import os
import csv
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger

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
        "frequency_cutoff_analysis": freq_cutoffs,
        "concept_conference_frequencies": filtered_freq
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.VALIDATION_META_FILE), exist_ok=True)
    
    with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Saved validation metadata to {Config.VALIDATION_META_FILE}")

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
        
        # Filter by conference frequency
        filtered_concepts = filter_by_conference_frequency(
            concept_freq, Config.CONFERENCE_FREQ_THRESHOLD
        )
        
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
        
        logger.info("Filtering completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()