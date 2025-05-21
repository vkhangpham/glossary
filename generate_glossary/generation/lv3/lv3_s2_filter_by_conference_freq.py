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
    CONFERENCE_FREQ_THRESHOLD = 1  # Minimum number of conference topics a concept must appear in
    MIN_JOURNAL_APPEARANCE = 1  # Extracted concept must appear in at least this many journals
    MIN_JOURNAL_FREQ_PERCENT = 1  # Extracted concept must appear in at least this % of topics within a journal
    NUM_WORKERS = 4  # Number of parallel workers for processing
    BATCH_SIZE = 1000  # Size of batches for parallel processing


def read_hierarchy_csv(csv_file: str) -> List[Dict[str, str]]:
    """
    Read the hierarchy CSV file and return a list of dictionaries
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        List of dictionaries containing conference_topic, journal, and extracted_concept
    """
    entries = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Use csv module to handle quotes and escaping properly
            reader = csv.DictReader(f)
            for row in reader:
                # Check if row has all required fields using the actual header names
                if 'conference_topic' in row and 'conference_journal' in row and 'extracted_concept' in row:
                    entries.append({
                        'conference_topic': row['conference_topic'],
                        'journal': row['conference_journal'],  # Map conference_journal to journal
                        'extracted_concept': row['extracted_concept']  # Keep extracted_concept as is
                    })
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        
    return entries


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
        "international conference on ", "journal of ", "studies in ",
        "research in ", "methods in ", "approaches to ", "trends in ",
        "perspectives on ", "applications of ", "theory of "
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
            
    # Check for pipe-separated conference topics
    if "|" in norm_topic:
        parts = [part.strip() for part in norm_topic.split("|")]
        if norm_concept in parts:
            return True
            
    # Check for slash-separated topics
    if "/" in norm_topic:
        parts = [part.strip() for part in norm_topic.split("/")]
        if norm_concept in parts:
            return True
    
    return False


def create_mappings_from_csv(entries: List[Dict[str, str]]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    Create conference_topic-concept and hierarchical mappings from CSV entries
    
    Args:
        entries: List of dictionaries with hierarchy information
        
    Returns:
        Tuple of (conference_topic_mapping, hierarchy_mapping)
    """
    # Create conference topic to concept mapping
    conference_topic_mapping = defaultdict(list)
    
    # Create hierarchical mapping (conference_topic -> journal -> concepts)
    hierarchy_mapping = defaultdict(lambda: defaultdict(list))
    
    # Track skipped journal-concept matches
    skipped_journal_concepts = set()
    
    for entry in entries:
        # Get fields
        conference_topic = entry.get('conference_topic', '')
        journal = entry.get('journal', '')
        extracted_concept = entry.get('extracted_concept', '')
        
        # Skip entries with missing data
        if not all([conference_topic, journal, extracted_concept]):
            continue
        
        # Skip if extracted concept matches journal name
        # Using the existing comparison function but comparing extracted_concept with journal
        if is_concept_same_as_conference_topic(extracted_concept, journal):
            skipped_journal_concepts.add((extracted_concept, journal))
            continue
            
        # Update topic mapping
        conference_topic_mapping[conference_topic].append(extracted_concept)
        
        # Update hierarchy mapping
        hierarchy_mapping[conference_topic][journal].append(extracted_concept)
        
    # Convert defaultdicts to regular dicts for serialization
    hierarchy_dict = {}
    for topic, journals in hierarchy_mapping.items():
        hierarchy_dict[topic] = {}
        for journal, concepts in journals.items():
            # Remove duplicates and sort
            hierarchy_dict[topic][journal] = sorted(list(set(concepts)))
    
    # Remove duplicates from topic mapping
    topic_dict = {
        topic: sorted(list(set(concepts)))
        for topic, concepts in conference_topic_mapping.items()
    }
    
    # Log skipped concepts
    if skipped_journal_concepts:
        logger.info("\nSkipped extracted concepts that match their journal names:")
        for concept, journal in sorted(skipped_journal_concepts):
            logger.info(f"  Skipped '{concept}' because it matched journal '{journal}'")
    
    return topic_dict, hierarchy_dict


def count_topic_frequencies_worker(
    conference_concepts: List[List[str]]
) -> Counter:
    """Count concept frequencies for a batch of conference topics"""
    concept_counts = Counter()
    
    for concepts in conference_concepts:
        # Convert list to set to avoid counting duplicates within same conference topic
        unique_concepts = set(concepts)
        concept_counts.update(unique_concepts)
        
    return concept_counts


def count_topic_frequencies(
    conference_mapping: Dict[str, List[str]],
    num_workers: int = Config.NUM_WORKERS,
    batch_size: int = Config.BATCH_SIZE
) -> Dict[str, int]:
    """Count how many conference topics each concept appears in using parallel processing"""
    # Convert mapping to list of concept lists for parallel processing
    concept_lists = list(conference_mapping.values())
    
    # Process in batches
    batches = [
        concept_lists[i:i + batch_size]
        for i in range(0, len(concept_lists), batch_size)
    ]
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        batch_counters = list(
            tqdm(
                executor.map(count_topic_frequencies_worker, batches),
                total=len(batches),
                desc="Counting frequencies"
            )
        )
    
    # Combine counters
    combined_counter = Counter()
    for counter in batch_counters:
        combined_counter.update(counter)
    
    return dict(combined_counter)


def filter_concepts_worker(args: tuple) -> List[str]:
    """Filter a batch of concepts based on frequency threshold"""
    concepts, concept_frequencies, threshold = args
    return [
        concept for concept in concepts
        if concept_frequencies.get(concept, 0) >= threshold
    ]


def filter_concepts(
    concepts: List[str],
    concept_frequencies: Dict[str, int],
    threshold: int,
    num_workers: int = Config.NUM_WORKERS,
    batch_size: int = Config.BATCH_SIZE
) -> List[str]:
    """Filter concepts based on their frequency across conference topics using parallel processing"""
    # Split concepts into batches
    batches = [
        concepts[i:i + batch_size]
        for i in range(0, len(concepts), batch_size)
    ]
    
    # Process batches in parallel
    filtered = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create argument tuples for each batch
        args_list = [(batch, concept_frequencies, threshold) for batch in batches]
        
        batch_results = list(
            tqdm(
                executor.map(filter_concepts_worker, args_list),
                total=len(batches),
                desc="Filtering concepts"
            )
        )
        
    # Combine results
    for batch_result in batch_results:
        filtered.extend(batch_result)
    
    return sorted(filtered)


def process_batch(batch, concept_to_keep):
    """Process a batch of rows from the CSV file for parallel filtering"""
    result = []
    
    for row in batch:
        if len(row) >= 3:
            concept = row[2]
            if concept in concept_to_keep:
                result.append(row)
    
    return result


def analyze_research_area_distribution(
    hierarchy: Dict[str, Dict[str, List[str]]]
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
    for conference_topic, journals in hierarchy.items():
        for journal, concepts in journals.items():
            # Count area as having this conference
            area_conference_counts[journal] += 1
            
            # Update concept stats for this research area
            for concept in concepts:
                area_concept_counts[journal][concept] += 1
                concept_areas[concept].add(journal)
    
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
    area_conference_counts: Dict[str, int]
) -> List[str]:
    """
    Filter concepts based on their distribution across research areas
    
    Args:
        concepts: Set of all concepts
        area_concept_counts: Research area -> concept -> frequency mapping
        concept_areas: Concept -> research areas mapping
        area_conference_counts: Research area -> conference count mapping
        
    Returns:
        List of filtered concepts
    """
    logger.info("Filtering concepts by research area distribution")
    
    filtered_concepts = []
    
    for concept in tqdm(concepts, desc="Filtering concepts"):
        # Get research areas where this concept appears
        areas = concept_areas.get(concept, set())
        
        # Check if concept appears in enough research areas
        if len(areas) >= Config.MIN_JOURNAL_APPEARANCE:
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
            if any(pct >= Config.MIN_JOURNAL_FREQ_PERCENT for pct in area_percentages):
                filtered_concepts.append(concept)
    
    logger.info(f"Filtered to {len(filtered_concepts)} concepts after research area distribution analysis")
    
    return sorted(filtered_concepts)


def write_filtered_csv(
    csv_file: str,
    entries: List[Dict[str, str]],
    filtered_concepts: Set[str]
) -> None:
    """
    Write a filtered CSV file containing only the filtered concepts
    
    Args:
        csv_file: Path to the output CSV file
        entries: List of dictionaries with hierarchy information
        filtered_concepts: Set of concepts that passed the filtering
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['conference_topic', 'journal', 'extracted_concept'])
            writer.writeheader()
            
            for entry in entries:
                concept = entry.get('extracted_concept', '')
                if concept in filtered_concepts:
                    writer.writerow(entry)
                    
        logger.info(f"Filtered hierarchy saved to {csv_file}")
    except Exception as e:
        logger.error(f"Error writing filtered CSV: {str(e)}")


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
        logger.info("Starting concept filtering by conference topic and research area frequency")

        # Create output directories if needed
        for path in [Config.OUTPUT_FILE, Config.VALIDATION_META_FILE, Config.OUTPUT_CSV_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get custom threshold from command line arguments if provided
        if len(sys.argv) > 1:
            try:
                threshold = int(sys.argv[1])
                Config.CONFERENCE_FREQ_THRESHOLD = threshold
                logger.info(f"Using conference frequency threshold: {threshold}")
            except ValueError:
                logger.warning(f"Invalid threshold value: {sys.argv[1]}. Using default: {Config.CONFERENCE_FREQ_THRESHOLD}")
        
        # Read input concepts
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            concepts = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(concepts)} concepts from input file")
        input_concepts_count = len(concepts)
        
        # Read the hierarchy CSV file
        csv_entries = read_hierarchy_csv(Config.CSV_FILE)
        logger.info(f"Read {len(csv_entries)} entries from CSV file")
        
        # Create mappings from CSV (this filters out concepts that match conference topics)
        conference_topic_mapping, hierarchy_mapping = create_mappings_from_csv(csv_entries)
        logger.info(f"Created mappings for {len(conference_topic_mapping)} conference topics")
        
        # Count concept frequencies across conference topics
        concept_freq = count_topic_frequencies(
            conference_topic_mapping,
            num_workers=Config.NUM_WORKERS,
            batch_size=Config.BATCH_SIZE
        )
        logger.info(f"Counted frequencies for {len(concept_freq)} unique concepts")
        
        # Analyze frequency distribution
        freq_dist = Counter(concept_freq.values())
        logger.info("Frequency distribution:")
        for freq, count in sorted(freq_dist.items()):
            logger.info(f"  {count} concepts appear in {freq} conference topics")
        
        # Filter by conference frequency threshold
        filtered_by_freq = filter_concepts(
            concepts,
            concept_freq,
            Config.CONFERENCE_FREQ_THRESHOLD,
            Config.NUM_WORKERS,
            Config.BATCH_SIZE
        )
        logger.info(f"Filtered to {len(filtered_by_freq)} concepts by frequency threshold")
        
        # Analyze research area distribution
        area_concept_counts, concept_areas, area_conference_counts = analyze_research_area_distribution(hierarchy_mapping)
        logger.info(f"Analyzed concept distribution across {len(area_concept_counts)} research areas")
        
        # Filter by research area distribution
        filtered_concepts = filter_by_research_area_distribution(
            set(filtered_by_freq), area_concept_counts, concept_areas, area_conference_counts
        )
        logger.info(f"Filtered to {len(filtered_concepts)} concepts after distribution filtering")
        
        # Write filtered CSV
        write_filtered_csv(Config.OUTPUT_CSV_FILE, csv_entries, set(filtered_concepts))
        
        # Save filtered concepts to file
        logger.info(f"Saving filtered concepts to {Config.OUTPUT_FILE}")
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
        
        # Save validation metadata
        validation_metadata = {
            "metadata": {
                "input_count": len(concepts),
                "output_count": len(filtered_concepts),
                "conference_freq_threshold": Config.CONFERENCE_FREQ_THRESHOLD,
                "min_research_area_appearance": Config.MIN_JOURNAL_APPEARANCE,
                "min_research_area_freq_percent": Config.MIN_JOURNAL_FREQ_PERCENT,
                "num_workers": Config.NUM_WORKERS,
                "batch_size": Config.BATCH_SIZE,
                "frequency_distribution": {str(k): v for k, v in freq_dist.items()},
                "conference_topic_count": len(conference_topic_mapping),
                "research_area_count": len(area_concept_counts),
            }
        }
        
        with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=4, ensure_ascii=False)
        
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