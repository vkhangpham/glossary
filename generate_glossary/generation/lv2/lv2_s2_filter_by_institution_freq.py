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
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Setup logging
logger = setup_logger("lv2.s2")

# Get the base directory (project root)
BASE_DIR = os.getcwd()

# Use centralized configuration
LEVEL = 2
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

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
    batch_size: int = processing_config.batch_size
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
    batch_size: int = processing_config.batch_size
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


def is_concept_same_as_department(concept: str, department: str) -> bool:
    """
    Check if a concept is the same as its department name
    
    Args:
        concept: The concept to check
        department: The department name
        
    Returns:
        Boolean indicating if the concept matches the department name
    """
    # Normalize both strings
    norm_concept = normalize_text(concept)
    norm_department = normalize_text(department)
    
    # Extract department name without prefix/suffix qualifiers (if any)
    # First try: "department of X" pattern
    dept_match = re.search(r'department\s+of\s+(.*?)(?:\s+and\s+|\s*$)', norm_department)
    if dept_match:
        core_dept = dept_match.group(1).strip()
        # Check if concept matches core department name
        if norm_concept == core_dept:
            return True
            
        # Also check individual components of compound department names
        # e.g. "electrical" in "department of electrical and computer engineering"
        if ' and ' in core_dept:
            parts = [p.strip() for p in core_dept.split(' and ')]
            if norm_concept in parts:
                return True
    
    # Second try: Just match the whole name or significant part
    if norm_concept == norm_department:
        return True
        
    # Check if concept is a standalone word in department name
    words = re.findall(r'\b\w+\b', norm_department)
    if len(words) > 1 and norm_concept in words:
        return True
        
    return False


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
    
    # Track skipped department-concept matches
    skipped_dept_concepts = set()
    
    for entry in entries:
        # Get fields
        topic = entry.get('topic', '')
        department = entry.get('department', '')
        concept = entry.get('concept', '')
        
        # Skip entries with missing data
        if not all([topic, department, concept]):
            continue
        
        # Skip if concept matches department name
        if is_concept_same_as_department(concept, department):
            skipped_dept_concepts.add((concept, department))
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
    
    # Log skipped concepts
    if skipped_dept_concepts:
        logger.info("\nSkipped concepts that match their department names:")
        for concept, department in sorted(skipped_dept_concepts):
            logger.info(f"  Skipped '{concept}' from {department}")
    
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


def analyze_department_distribution(
    hierarchy_mapping: Dict[str, Dict[str, List[str]]]
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Set[str]], Dict[str, int]]:
    """
    Analyze concept distribution across departments
    
    Args:
        hierarchy_mapping: Hierarchical mapping of topic -> department -> concepts
        
    Returns:
        Tuple containing:
        - Department to concept frequency mapping 
        - Concept to departments mapping
        - Count of topics per department
    """
    # Department -> concept -> count mapping
    dept_concept_counts = defaultdict(Counter)
    
    # Concept -> set of departments mapping
    concept_departments = defaultdict(set)
    
    # Count topics per department
    dept_topic_counts = Counter()
    
    # Process hierarchy mapping
    for topic, departments in hierarchy_mapping.items():
        for department, concepts in departments.items():
            # Count department as having this topic
            dept_topic_counts[department] += 1
            
            # Update concept stats for this department
            for concept in concepts:
                dept_concept_counts[department][concept] += 1
                concept_departments[concept].add(department)
    
    return (
        {dept: dict(counts) for dept, counts in dept_concept_counts.items()},
        {concept: depts for concept, depts in concept_departments.items()},
        dict(dept_topic_counts)
    )


def filter_by_department_distribution(
    concepts: List[str],
    dept_concept_counts: Dict[str, Dict[str, int]],
    concept_departments: Dict[str, Set[str]],
    dept_topic_counts: Dict[str, int]
) -> List[str]:
    """
    Filter concepts based on their distribution across departments
    
    Args:
        concepts: List of concepts to filter
        dept_concept_counts: Department -> concept -> frequency mapping
        concept_departments: Concept -> departments mapping
        dept_topic_counts: Department -> topic count mapping
        
    Returns:
        List of filtered concepts
    """
    filtered_concepts = []
    
    for concept in tqdm(concepts, desc="Filtering by department distribution"):
        # Get departments where this concept appears
        departments = concept_departments.get(concept, set())
        
        # Check if concept appears in enough departments
        if len(departments) >= Config.MIN_DEPT_APPEARANCE:
            # Check department-level frequencies
            dept_percentages = []
            
            for dept in departments:
                if dept in dept_topic_counts and dept_topic_counts[dept] > 0:
                    # Get concept frequency in this department
                    concept_freq = dept_concept_counts.get(dept, {}).get(concept, 0)
                    
                    # Calculate percentage of topics in this department having this concept
                    percentage = (concept_freq / dept_topic_counts[dept]) * 100
                    dept_percentages.append(percentage)
            
            # Filter based on minimum percentage threshold
            if any(pct >= Config.MIN_DEPT_FREQ_PERCENT for pct in dept_percentages):
                filtered_concepts.append(concept)
    
    return sorted(filtered_concepts)


def write_department_stats(
    validation_meta_file: str,
    dept_concept_counts: Dict[str, Dict[str, int]],
    concept_departments: Dict[str, Set[str]],
    dept_topic_counts: Dict[str, int],
    filtered_concepts: List[str]
) -> None:
    """
    Add department distribution statistics to the metadata file
    
    Args:
        validation_meta_file: Path to the metadata file
        dept_concept_counts: Department -> concept -> frequency mapping
        concept_departments: Concept -> departments mapping
        dept_topic_counts: Department -> topic count mapping
        filtered_concepts: List of filtered concepts
    """
    try:
        # Read existing metadata
        with open(validation_meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Add department statistics
        metadata['department_statistics'] = {
            dept: {
                'total_topics': dept_topic_counts.get(dept, 0),
                'total_concepts': len(concepts),
                'filtered_concepts': sum(1 for c in concepts if c in filtered_concepts)
            }
            for dept, concepts in dept_concept_counts.items()
        }
        
        # Add concept distribution across departments
        metadata['concept_department_distribution'] = {
            concept: {
                'total_departments': len(concept_departments.get(concept, [])),
                'departments': sorted(list(concept_departments.get(concept, [])))
            }
            for concept in filtered_concepts
        }
        
        # Write updated metadata
        with open(validation_meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        logger.info("Department statistics added to metadata file")
        
    except Exception as e:
        logger.error(f"Error writing department statistics: {str(e)}")


def main():
    """Main execution function"""
    try:
        logger.info("Starting concept filtering by topic and department frequency")

        # Create output directories if needed
        for path in [level_config.get_step_output_file(2), level_config.get_validation_metadata_file(3), Config.OUTPUT_CSV_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read input concepts
        with open(level_config.get_step_input_file(2), "r", encoding="utf-8") as f:
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
            batch_size=processing_config.batch_size
        )
        logger.info(f"Counted frequencies for {len(concept_frequencies)} unique concepts")

        # Analyze frequency distribution
        freq_dist = analyze_frequency_distribution(concept_frequencies)
        
        # Analyze department distribution
        dept_concept_counts, concept_departments, dept_topic_counts = analyze_department_distribution(hierarchy_mapping)
        logger.info(f"Analyzed concept distribution across {len(dept_concept_counts)} departments")

        # Filter concepts by department distribution
        filtered_concepts = filter_by_department_distribution(
            concepts,
            dept_concept_counts,
            concept_departments,
            dept_topic_counts
        )
        logger.info(f"Filtered to {len(filtered_concepts)} concepts by department distribution")

        # Save filtered concepts
        with open(level_config.get_step_output_file(2), "w", encoding="utf-8") as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
                
        # Write filtered CSV
        write_filtered_csv(Config.OUTPUT_CSV_FILE, csv_entries, set(filtered_concepts))

        # Save validation metadata
        validation_metadata = {
            "metadata": {
                "input_count": len(concepts),
                "output_count": len(filtered_concepts),
                "min_dept_appearance": Config.MIN_DEPT_APPEARANCE,
                "min_dept_freq_percent": Config.MIN_DEPT_FREQ_PERCENT,
                "num_workers": Config.NUM_WORKERS,
                "batch_size": processing_config.batch_size,
                "frequency_distribution": {str(k): v for k, v in freq_dist.items()},
                "topic_count": len(topic_mapping),
                "department_count": sum(len(depts) for depts in hierarchy_mapping.values()),
            },
            "hierarchy_mapping": hierarchy_mapping,
        }

        with open(level_config.get_validation_metadata_file(3), "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=4, ensure_ascii=False)

        # Write department statistics
        write_department_stats(
            level_config.get_validation_metadata_file(3), 
            dept_concept_counts, 
            concept_departments, 
            dept_topic_counts, 
            filtered_concepts
        )

        logger.info("Concept filtering completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
