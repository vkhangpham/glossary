import json
from typing import Dict, List, Any, Set
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from generate_glossary.utils.logger import setup_logger

logger = setup_logger("lv0.s2")

LEVEL = 0
INSTITUTION_FREQ_THRESHOLD_PERCENT = (
    60  # Percentage of institutions a concept must appear in
)

DATA_DIR = Path("data/generation/lv0")
INPUT_FILE = DATA_DIR / "lv0_s1_output.txt"
OUTPUT_FILE = DATA_DIR / "lv0_s2_output.txt"
META_FILE = DATA_DIR / "lv0_s2_metadata.json"
S1_META_FILE = DATA_DIR / "lv0_s1_metadata.json"
S0_META_FILE = DATA_DIR / "lv0_s0_metadata.json"

TEST_DATA_DIR = Path("data/generation/tests")
TEST_INPUT_FILE = TEST_DATA_DIR / "lv0_s1_output.txt"
TEST_OUTPUT_FILE = TEST_DATA_DIR / "lv0_s2_output.txt"
TEST_META_FILE = TEST_DATA_DIR / "lv0_s2_metadata.json"
TEST_S1_META_FILE = TEST_DATA_DIR / "lv0_s1_metadata.json"
TEST_S0_META_FILE = TEST_DATA_DIR / "lv0_s0_metadata.json"


def get_institution_from_source(source: str) -> str:
    """Extract institution name from source string"""
    if " - " not in source:
        return source
    return source.split(" - ")[0].strip()


def calculate_min_institutions(total_institutions: int, threshold_percent: int) -> int:
    """Calculate minimum institutions required based on threshold"""
    return int(total_institutions * threshold_percent / 100)


def count_institution_frequencies(
    source_concept_mapping: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Count how many institutions each concept appears in

    Args:
        source_concept_mapping: Dictionary mapping sources to their concepts

    Returns:
        Dictionary with concept frequencies by institution
    """
    concept_institutions: Dict[str, Set[str]] = {}

    for source, concepts in source_concept_mapping.items():
        institution = get_institution_from_source(source)

        for concept in concepts:
            if concept not in concept_institutions:
                concept_institutions[concept] = set()
            concept_institutions[concept].add(institution)

    concept_frequencies = {
        concept: {
            "count": len(institutions),
            "institutions": sorted(list(institutions)),
        }
        for concept, institutions in concept_institutions.items()
    }

    return concept_frequencies


def filter_concepts(
    concepts: List[str],
    concept_frequencies: Dict[str, Dict[str, Any]],
    total_institutions: int,
    threshold_percent: int,
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
    min_institutions = calculate_min_institutions(total_institutions, threshold_percent)
    logger.info(
        f"Requiring concepts to appear in at least {min_institutions}/{total_institutions} institutions ({threshold_percent}%)"
    )

    filtered = []
    for concept in tqdm(concepts, desc="Filtering concepts"):
        if concept in concept_frequencies:
            freq_data = concept_frequencies[concept]
            if freq_data["count"] >= min_institutions:
                filtered.append(concept)

    return sorted(filtered)


def test():
    """Test mode: Read from test directory and save to test directory"""
    global INPUT_FILE, OUTPUT_FILE, META_FILE, S1_META_FILE, S0_META_FILE

    original_input = INPUT_FILE
    original_output = OUTPUT_FILE
    original_meta = META_FILE
    original_s1_meta = S1_META_FILE
    original_s0_meta = S0_META_FILE

    INPUT_FILE = TEST_INPUT_FILE
    OUTPUT_FILE = TEST_OUTPUT_FILE
    META_FILE = TEST_META_FILE
    S1_META_FILE = TEST_S1_META_FILE
    S0_META_FILE = TEST_S0_META_FILE

    logger.info("Running in TEST MODE")

    try:
        main()
    finally:
        INPUT_FILE = original_input
        OUTPUT_FILE = original_output
        META_FILE = original_meta
        S1_META_FILE = original_s1_meta
        S0_META_FILE = original_s0_meta


def main():
    """Main execution function"""
    logger.info("Starting concept filtering by institution frequency")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f.readlines()]
    logger.info(f"Read {len(concepts)} concepts from input file")

    with open(S1_META_FILE, "r", encoding="utf-8") as f:
        s1_metadata = json.load(f)

    source_concept_mapping = s1_metadata.get("source_concept_mapping", {})
    logger.info(f"Read mappings for {len(source_concept_mapping)} sources")

    with open(S0_META_FILE, "r", encoding="utf-8") as f:
        s0_metadata = json.load(f)

    selected_institutions = s0_metadata.get("selected_institutions", [])
    total_institutions = len(selected_institutions)
    logger.info(f"Found {total_institutions} selected institutions")

    concept_frequencies = count_institution_frequencies(source_concept_mapping)
    logger.info(f"Counted frequencies for {len(concept_frequencies)} unique concepts")

    filtered_concepts = filter_concepts(
        concepts,
        concept_frequencies,
        total_institutions,
        INSTITUTION_FREQ_THRESHOLD_PERCENT,
    )

    freq_counts = [freq_data["count"] for freq_data in concept_frequencies.values()]
    freq_dist = Counter(freq_counts)
    logger.info("Frequency distribution:")
    for freq, count in sorted(freq_dist.items()):
        logger.info(f"  {count} concepts appear in {freq} institutions")

    logger.info(f"Filtered to {len(filtered_concepts)} concepts")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for concept in filtered_concepts:
            f.write(f"{concept}\n")

    min_institutions = calculate_min_institutions(
        total_institutions, INSTITUTION_FREQ_THRESHOLD_PERCENT
    )
    metadata = {
        "input_count": len(concepts),
        "output_count": len(filtered_concepts),
        "total_institutions": total_institutions,
        "institution_threshold_percent": INSTITUTION_FREQ_THRESHOLD_PERCENT,
        "min_institutions_required": min_institutions,
        "frequency_distribution": {str(k): v for k, v in freq_dist.items()},
        "concept_frequencies": {
            concept: freq_data
            for concept, freq_data in concept_frequencies.items()
            if freq_data["count"] >= min_institutions
        },
        "selected_institutions": selected_institutions,
        "source_concept_mapping": source_concept_mapping,
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info("Concept filtering completed successfully")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test()
    else:
        main()
