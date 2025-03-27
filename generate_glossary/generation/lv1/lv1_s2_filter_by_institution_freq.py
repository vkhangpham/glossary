import json
from typing import Dict, List, Any, Set
import sys
import os
import csv
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Setup logging
logger = setup_logger("lv1.s2")

class Config:
    """Configuration for concept filtering"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s1_department_concepts.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s2_filtered_concepts.txt")
    VALIDATION_META_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s2_metadata.json")
    DEPARTMENT_FREQ_THRESHOLD = 2  # Minimum number of departments a concept must appear in
    MIN_CONCEPT_LENGTH = 3  # Minimum length of a valid concept
    MAX_CONCEPT_LENGTH = 50  # Maximum length of a valid concept

def is_valid_concept(concept: str) -> bool:
    """
    Check if a concept is valid based on basic criteria
    
    Args:
        concept: The concept string to validate
        
    Returns:
        Boolean indicating if the concept is valid
    """
    # Check length criteria
    if not concept or len(concept) < Config.MIN_CONCEPT_LENGTH or len(concept) > Config.MAX_CONCEPT_LENGTH:
        return False
        
    # Exclude single characters and numbers
    if len(concept) <= 2 and (concept.isdigit() or concept.isalpha()):
        return False
        
    # Exclude concepts that are likely not academic terms
    non_academic_terms = [
        "page", "home", "about", "contact", "staff", "faculty", "links",
        "click", "here", "website", "portal", "login", "apply", "apply now",
        "register", "registration", "more", "learn more", "read more",
        "back", "next", "previous", "link", "site"
    ]
    
    if concept.lower() in non_academic_terms:
        return False
        
    # Check if concept has more than just punctuation and spaces
    if not any(c.isalnum() for c in concept):
        return False
        
    return True

def count_department_frequencies(input_file: str) -> Dict[str, int]:
    """
    Count how many departments each concept appears in
    
    Args:
        input_file: Path to CSV file containing department-concept mappings
        
    Returns:
        Dictionary mapping concepts to their department frequency and normalized mapping
    """
    # Use Counter for efficient counting
    concept_counts = Counter()
    
    # Normalize concepts and track original forms
    normalized_to_original = {}
    
    # Read CSV file using csv module to handle quoted fields properly
    with open(input_file, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        
        # Skip header
        next(csv_reader)
        
        # Process each line
        for row in csv_reader:
            if len(row) >= 3:  # Ensure we have at least department, college, concept
                department, college, concept = row[0], row[1], row[2]
                
                # Skip invalid concepts
                if not is_valid_concept(concept):
                    continue
                    
                # Normalize the concept
                norm_concept = normalize_text(concept)
                
                # Track the original form (prefer shorter versions)
                if norm_concept in normalized_to_original:
                    current = normalized_to_original[norm_concept]
                    if len(concept) < len(current):
                        normalized_to_original[norm_concept] = concept
                else:
                    normalized_to_original[norm_concept] = concept
                    
                # Add to counter
                concept_counts[norm_concept] += 1
    
    return dict(concept_counts), normalized_to_original

def filter_concepts(
    concept_frequencies: Dict[str, int],
    normalized_to_original: Dict[str, str],
    threshold: int
) -> List[str]:
    """
    Filter concepts based on their frequency across departments
    
    Args:
        concept_frequencies: Dictionary mapping normalized concepts to their department frequency
        normalized_to_original: Dictionary mapping normalized forms to original forms
        threshold: Minimum number of departments a concept must appear in
        
    Returns:
        List of concepts that meet the threshold
    """
    filtered = []
    
    for norm_concept, freq in tqdm(concept_frequencies.items(), desc="Filtering concepts"):
        # Apply frequency threshold
        if freq >= threshold:
            # Use original form if available, otherwise use normalized form
            orig_concept = normalized_to_original.get(norm_concept, norm_concept)
            filtered.append(orig_concept)
    
    return sorted(filtered)

def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        os.path.dirname(Config.OUTPUT_FILE),
        os.path.dirname(Config.VALIDATION_META_FILE)
    ]
    
    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        logger.info("Starting concept filtering by department frequency")
        
        # Ensure directories exist
        ensure_dirs_exist()
        
        # Count concept frequencies across departments
        concept_frequencies, normalized_to_original = count_department_frequencies(Config.INPUT_FILE)
        logger.info(f"Counted frequencies for {len(concept_frequencies)} unique normalized concepts")
        
        # Filter concepts
        filtered_concepts = filter_concepts(
            concept_frequencies,
            normalized_to_original,
            Config.DEPARTMENT_FREQ_THRESHOLD
        )
        
        # Log frequency distribution
        freq_dist = Counter(concept_frequencies.values())
        logger.info("Frequency distribution:")
        for freq, count in sorted(freq_dist.items()):
            logger.info(f"  {count} concepts appear in {freq} departments")
        
        logger.info(f"Filtered to {len(filtered_concepts)} concepts")
        
        # Create output directories if needed
        for path in [Config.OUTPUT_FILE, Config.VALIDATION_META_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filtered concepts
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
        
        # Save validation metadata
        validation_metadata = {
            "metadata": {
                "output_count": len(filtered_concepts),
                "normalized_concepts_count": len(concept_frequencies),
                "department_threshold": Config.DEPARTMENT_FREQ_THRESHOLD,
                "min_concept_length": Config.MIN_CONCEPT_LENGTH,
                "max_concept_length": Config.MAX_CONCEPT_LENGTH,
                "frequency_distribution": {
                    str(k): v for k, v in freq_dist.items()
                }
            },
            "concept_frequencies": {
                normalized_to_original.get(concept, concept): freq 
                for concept, freq in concept_frequencies.items()
                if freq >= Config.DEPARTMENT_FREQ_THRESHOLD
            }
        }
        
        with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=4, ensure_ascii=False)
        
        logger.info("Concept filtering completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 