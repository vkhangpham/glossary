import json
from typing import Dict, List, Any
import sys
import os
from pathlib import Path
from collections import Counter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("lv4.s2")


class Config:
    """Configuration for concept filtering"""

    INPUT_FILE = "../data/lv4/lv4_s1_extracted_concepts.txt"
    META_FILE = "../data/lv4/lv4_s1_metadata.json"
    OUTPUT_FILE = "../data/lv4/lv4_s2_filtered_concepts.txt"
    VALIDATION_META_FILE = "../data/lv4/lv4_s2_metadata.json"
    VENUE_FREQ_THRESHOLD = 2  # Minimum number of venues a concept must appear in


def count_venue_frequencies(venue_mapping: Dict[str, List[str]], input_concepts: set[str]) -> Dict[str, int]:
    """
    Count how many venues each concept appears in

    Args:
        venue_mapping: Dictionary mapping venues to their concepts
        input_concepts: Set of concepts we want to track frequencies for

    Returns:
        Dictionary mapping concepts to their venue frequency
    """
    concept_counts = Counter()

    for concepts in venue_mapping.values():
        # Only count concepts that are in our input list
        relevant_concepts = set(concepts) & input_concepts
        concept_counts.update(relevant_concepts)

    return dict(concept_counts)


def filter_concepts(
    concepts: List[str], concept_frequencies: Dict[str, int], threshold: int
) -> List[str]:
    """
    Filter concepts based on their frequency across venues
    """
    filtered = []

    for concept in tqdm(concepts, desc="Filtering concepts"):
        if concept_frequencies.get(concept, 0) >= threshold:
            filtered.append(concept)

    return sorted(filtered)


def main():
    """Main execution function"""
    try:
        logger.info("Starting concept filtering by venue frequency")

        # Read input concepts
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            concepts = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(concepts)} concepts from input file")

        # Read metadata with venue mapping
        with open(Config.META_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Get venue mapping from metadata
        venue_mapping = metadata.get("source_concept_mapping", {})
        logger.info(f"Read mappings for {len(venue_mapping)} venues")

        # Count concept frequencies across venues
        concept_frequencies = count_venue_frequencies(venue_mapping, set(concepts))
        logger.info(
            f"Counted frequencies for {len(concept_frequencies)} unique concepts"
        )

        # Filter concepts
        filtered_concepts = filter_concepts(
            concepts, concept_frequencies, Config.VENUE_FREQ_THRESHOLD
        )

        # Log frequency distribution
        freq_dist = Counter(concept_frequencies.values())
        logger.info("Frequency distribution:")
        for freq, count in sorted(freq_dist.items()):
            logger.info(f"  {count} concepts appear in {freq} venues")

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
                "venue_threshold": Config.VENUE_FREQ_THRESHOLD,
                "frequency_distribution": {str(k): v for k, v in freq_dist.items()},
            },
            "concept_frequencies": {
                concept: freq
                for concept, freq in concept_frequencies.items()
                if freq >= Config.VENUE_FREQ_THRESHOLD
            },
        }

        with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=4, ensure_ascii=False)

        logger.info("Concept filtering completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
