import json
from typing import Dict, List, Any, Set
import sys
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)

from generate_glossary.utils.logger import setup_logger

# Setup logging
logger = setup_logger("lv0.s2")

class Config:
    """Configuration for concept filtering"""
    INPUT_FILE = "data/lv0/lv0_s1_extracted_concepts.txt"
    META_FILE = "data/lv0/lv0_s1_metadata.json"
    METADATA_S0_FILE = "data/lv0/lv0_s0_metadata.json"
    OUTPUT_FILE = "data/lv0/lv0_s2_filtered_concepts.txt"
    VALIDATION_META_FILE = "data/lv0/lv0_s2_metadata.json"
    INSTITUTION_FREQ_THRESHOLD_PERCENT = 60  # Minimum percentage of institutions a concept must appear in

def get_institution_from_source(source: str) -> str:
    """
    Extract institution name from source string
    
    Args:
        source: Source string in format "institution - college"
        
    Returns:
        Institution name
    """
    if " - " not in source:
        return source
    
    return source.split(" - ")[0].strip()

def count_institution_frequencies(
    source_concept_mapping: Dict[str, List[str]]
) -> Dict[str, Dict[str, int]]:
    """
    Count how many institutions each concept appears in
    
    Args:
        source_concept_mapping: Dictionary mapping sources to their concepts
        
    Returns:
        Dictionary with concept frequencies by institution
    """
    # Map concepts to the institutions they appear in
    concept_institutions = {}
    
    # First, create a mapping from institution to all its concepts
    institution_concepts = {}
    
    for source, concepts in source_concept_mapping.items():
        institution = get_institution_from_source(source)
        
        if institution not in institution_concepts:
            institution_concepts[institution] = set()
            
        institution_concepts[institution].update(concepts)
    
    # Now count unique institutions for each concept
    for institution, concepts in institution_concepts.items():
        for concept in concepts:
            if concept not in concept_institutions:
                concept_institutions[concept] = set()
            concept_institutions[concept].add(institution)
    
    # Convert sets to counts
    concept_frequencies = {
        concept: {
            "count": len(institutions),
            "institutions": sorted(list(institutions))
        }
        for concept, institutions in concept_institutions.items()
    }
    
    return concept_frequencies

def filter_concepts(
    concepts: List[str],
    concept_frequencies: Dict[str, Dict[str, Any]],
    total_institutions: int,
    threshold_percent: int
) -> List[str]:
    """
    Filter concepts based on their frequency across institutions
    
    Args:
        concepts: List of concepts to filter
        concept_frequencies: Dictionary mapping concepts to their institution frequency
        total_institutions: Total number of institutions
        threshold_percent: Minimum percentage of institutions a concept must appear in
        
    Returns:
        List of concepts that meet the threshold
    """
    # Calculate the minimum number of institutions required
    min_institutions = int(total_institutions * threshold_percent / 100)
    logger.info(f"Requiring concepts to appear in at least {min_institutions}/{total_institutions} institutions ({threshold_percent}%)")
    
    filtered = []
    
    for concept in tqdm(concepts, desc="Filtering concepts"):
        if concept in concept_frequencies:
            freq_data = concept_frequencies[concept]
            if freq_data["count"] >= min_institutions:
                filtered.append(concept)
    
    return sorted(filtered)

def main():
    """Main execution function"""
    try:
        logger.info("Starting concept filtering by institution frequency")
        
        # Read input concepts
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            concepts = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(concepts)} concepts from input file")
        
        # Read metadata with source-concept mapping
        with open(Config.META_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        # Get source-concept mapping from metadata
        source_concept_mapping = metadata.get("source_concept_mapping", {})
        logger.info(f"Read mappings for {len(source_concept_mapping)} sources")
        
        # Read s0 metadata to get the list of selected institutions
        with open(Config.METADATA_S0_FILE, "r", encoding="utf-8") as f:
            s0_metadata = json.load(f)
            
        selected_institutions = s0_metadata.get("selected_institutions", [])
        total_institutions = len(selected_institutions)
        logger.info(f"Found {total_institutions} selected institutions")
        
        # Count concept frequencies across institutions
        concept_frequencies = count_institution_frequencies(source_concept_mapping)
        logger.info(f"Counted frequencies for {len(concept_frequencies)} unique concepts")
        
        # Filter concepts
        filtered_concepts = filter_concepts(
            concepts,
            concept_frequencies,
            total_institutions,
            Config.INSTITUTION_FREQ_THRESHOLD_PERCENT
        )
        
        # Log frequency distribution
        freq_counts = [freq_data["count"] for freq_data in concept_frequencies.values()]
        freq_dist = Counter(freq_counts)
        logger.info("Frequency distribution:")
        for freq, count in sorted(freq_dist.items()):
            logger.info(f"  {count} concepts appear in {freq} institutions")
        
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
                "input_count": len(concepts),
                "output_count": len(filtered_concepts),
                "total_institutions": total_institutions,
                "institution_threshold_percent": Config.INSTITUTION_FREQ_THRESHOLD_PERCENT,
                "min_institutions_required": int(total_institutions * Config.INSTITUTION_FREQ_THRESHOLD_PERCENT / 100),
                "frequency_distribution": {
                    str(k): v for k, v in freq_dist.items()
                }
            },
            "concept_frequencies": {
                concept: freq_data
                for concept, freq_data in concept_frequencies.items()
                if freq_data["count"] >= int(total_institutions * Config.INSTITUTION_FREQ_THRESHOLD_PERCENT / 100)
            },
            "selected_institutions": selected_institutions
        }
        
        with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=4, ensure_ascii=False)
        
        logger.info("Concept filtering completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 