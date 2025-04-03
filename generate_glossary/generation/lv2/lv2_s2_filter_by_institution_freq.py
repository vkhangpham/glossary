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
logger = setup_logger("lv2.s2")

# Get the base directory (project root)
BASE_DIR = os.getcwd()

class Config:
    """Configuration for concept filtering"""

    INPUT_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s1_metadata.json")
    CSV_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s1_hierarchical_concepts.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s2_filtered_concepts.txt")
    OUTPUT_CSV_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s2_filtered_concepts.csv")
    VALIDATION_META_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s2_metadata.json")
    TOPIC_FREQ_THRESHOLD = 1
    NUM_WORKERS = 4  # Number of parallel workers for processing
    BATCH_SIZE = 1000  # Size of batches for parallel processing


def count_topic_frequencies_worker(
    topic_concepts: List[List[str]]
) -> Counter:
    """Count concept frequencies for a batch of topics"""
    concept_counts = Counter()
    
    for concepts in topic_concepts:
        # Convert list to set to avoid counting duplicates within same topic
        unique_concepts = set(concepts)
        concept_counts.update(unique_concepts)
        
    return concept_counts


def count_topic_frequencies(
    topic_mapping: Dict[str, List[str]],
    num_workers: int = Config.NUM_WORKERS,
    batch_size: int = Config.BATCH_SIZE
) -> Dict[str, int]:
    """Count how many topics each concept appears in using parallel processing"""
    # Convert mapping to list of concept lists for parallel processing
    concept_lists = list(topic_mapping.values())
    
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
    """Filter concepts based on their frequency across topics using parallel processing"""
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


def analyze_frequency_distribution(
    concept_frequencies: Dict[str, int]
) -> Dict[str, int]:
    """Analyze and log the frequency distribution of concepts"""
    freq_dist = Counter(concept_frequencies.values())
    
    logger.info("Frequency distribution:")
    for freq, count in sorted(freq_dist.items()):
        logger.info(f"  {count} concepts appear in {freq} topics")
        
    return freq_dist


def read_hierarchy_csv(csv_file: str) -> List[Dict[str, str]]:
    """
    Read the hierarchy CSV file and return a list of dictionaries
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        List of dictionaries containing topic, department, and concept
    """
    entries = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Use csv module to handle quotes and escaping properly
            reader = csv.DictReader(f)
            for row in reader:
                # Check if row has all required fields
                if all(field in row for field in ['topic', 'department', 'concept']):
                    entries.append({
                        'topic': row['topic'],
                        'department': row['department'],
                        'concept': row['concept']
                    })
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        
    return entries


def create_mappings_from_csv(entries: List[Dict[str, str]]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    Create topic-concept and hierarchical mappings from CSV entries
    
    Args:
        entries: List of dictionaries with hierarchy information
        
    Returns:
        Tuple of (topic_mapping, hierarchy_mapping)
    """
    # Create topic to concept mapping
    topic_mapping = defaultdict(list)
    
    # Create hierarchical mapping (topic -> department -> concepts)
    hierarchy_mapping = defaultdict(lambda: defaultdict(list))
    
    for entry in entries:
        # Get fields
        topic = entry.get('topic', '')
        department = entry.get('department', '')
        concept = entry.get('concept', '')
        
        # Skip entries with missing data
        if not all([topic, department, concept]):
            continue
            
        # Update topic mapping
        topic_mapping[topic].append(concept)
        
        # Update hierarchy mapping
        hierarchy_mapping[topic][department].append(concept)
        
    # Convert defaultdicts to regular dicts for serialization
    hierarchy_dict = {}
    for topic, departments in hierarchy_mapping.items():
        hierarchy_dict[topic] = {}
        for department, concepts in departments.items():
            # Remove duplicates and sort
            hierarchy_dict[topic][department] = sorted(list(set(concepts)))
    
    # Remove duplicates from topic mapping
    topic_dict = {
        topic: sorted(list(set(concepts)))
        for topic, concepts in topic_mapping.items()
    }
    
    return topic_dict, hierarchy_dict


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
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['topic', 'department', 'concept'])
            writer.writeheader()
            
            for entry in entries:
                concept = entry.get('concept', '')
                if concept in filtered_concepts:
                    writer.writerow(entry)
                    
        logger.info(f"Filtered hierarchy saved to {csv_file}")
    except Exception as e:
        logger.error(f"Error writing filtered CSV: {str(e)}")


def main():
    """Main execution function"""
    try:
        logger.info("Starting concept filtering by topic frequency")

        # Create output directories if needed
        for path in [Config.OUTPUT_FILE, Config.VALIDATION_META_FILE, Config.OUTPUT_CSV_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read input concepts
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            concepts = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(concepts)} concepts from input file")

        # Read the hierarchy CSV file
        csv_entries = read_hierarchy_csv(Config.CSV_FILE)
        logger.info(f"Read {len(csv_entries)} entries from CSV file")
        
        # Create mappings from CSV
        topic_mapping, hierarchy_mapping = create_mappings_from_csv(csv_entries)
        logger.info(f"Created mappings for {len(topic_mapping)} topics")

        # Count concept frequencies across topics
        concept_frequencies = count_topic_frequencies(
            topic_mapping,
            num_workers=Config.NUM_WORKERS,
            batch_size=Config.BATCH_SIZE
        )
        logger.info(f"Counted frequencies for {len(concept_frequencies)} unique concepts")

        # Analyze frequency distribution
        freq_dist = analyze_frequency_distribution(concept_frequencies)

        # Filter concepts
        filtered_concepts = filter_concepts(
            concepts,
            concept_frequencies,
            Config.TOPIC_FREQ_THRESHOLD,
            num_workers=Config.NUM_WORKERS,
            batch_size=Config.BATCH_SIZE
        )
        logger.info(f"Filtered to {len(filtered_concepts)} concepts")

        # Save filtered concepts
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
                
        # Write filtered CSV
        write_filtered_csv(Config.OUTPUT_CSV_FILE, csv_entries, set(filtered_concepts))

        # Save validation metadata
        validation_metadata = {
            "metadata": {
                "input_count": len(concepts),
                "output_count": len(filtered_concepts),
                "topic_threshold": Config.TOPIC_FREQ_THRESHOLD,
                "num_workers": Config.NUM_WORKERS,
                "batch_size": Config.BATCH_SIZE,
                # "frequency_distribution": {str(k): v for k, v in freq_dist.items()},
                "topic_count": len(topic_mapping),
                "department_count": sum(len(depts) for depts in hierarchy_mapping.values()),
            },
            # "concept_frequencies": concept_frequencies,
            "hierarchy_mapping": hierarchy_mapping,
        }

        with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=4, ensure_ascii=False)

        logger.info("Concept filtering completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
