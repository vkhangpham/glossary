import json
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk


from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import structured_completion_consensus
from generate_glossary.generation.shared import process_with_checkpoint

load_dotenv()
logger = setup_logger("lv0.s1")

# Configuration constants - simple and direct
LEVEL = 0
BATCH_SIZE = 20
CHUNK_SIZE = 100
MAX_WORKERS = 4
LLM_ATTEMPTS = 3
AGREEMENT_THRESHOLD = 2
FREQUENCY_THRESHOLD = 2

# File paths - explicit and clear
DATA_DIR = Path("data/generation/lv0")
INPUT_FILE = DATA_DIR / "lv0_s0_output.txt"  # Output from step 0
OUTPUT_FILE = DATA_DIR / "lv0_s1_output.txt"  # Main output
META_FILE = DATA_DIR / "lv0_s1_metadata.json"  # Metadata output
CHECKPOINT_DIR = DATA_DIR / ".checkpoints"


class ConceptExtraction(BaseModel):
    source: str = Field(description="Source text being processed")
    concepts: List[str] = Field(description="List of extracted concepts")


class ConceptExtractionList(BaseModel):
    extractions: List[ConceptExtraction] = Field(
        description="List of concept extractions"
    )


SYSTEM_PROMPT = """Extract academic disciplines from college/school names.

Rules:
1. Extract ONLY terms explicitly mentioned in the text
2. Remove administrative qualifiers (college of, school of, department of)
3. Keep compound terms intact (e.g., "computer science", "applied mathematics")
4. Exclude: generic terms, acronyms, proper nouns, locations

Examples:
- "College of Arts and Sciences" → ["arts", "sciences"]
- "School of Computer Science" → ["computer science"]
- "Department of Electrical Engineering" → ["electrical engineering"]"""


def create_extraction_prompt(sources: List[str]) -> str:
    """Create LLM prompt for concept extraction from college names"""
    sources_str = "\n".join(f"- {source}" for source in sources)

    return f"""Extract broad academic disciplines and research areas from these college/school/division names.
For each name, identify the core academic fields and disciplines represented.

Return the concepts in this exact JSON format:
{{
    "extractions": [
        {{"source": "source1", "concepts": ["concept1", "concept2"]}},
        {{"source": "source2", "concepts": ["concept3", "concept4"]}}
    ]
}}

College/school/division names to process:
{sources_str}"""


def extract_concepts_with_consensus(
    batch: List[str], num_attempts: int = LLM_ATTEMPTS
) -> List[List[ConceptExtraction]]:
    """Extract concepts from a batch using LLM consensus mechanism"""
    prompt = create_extraction_prompt(batch)

    # Build messages
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Use the new consensus method to get all responses for analysis
        consensus, all_responses = structured_completion_consensus(
            messages=messages,
            response_model=ConceptExtractionList,
            model="openai/gpt-4o-mini",
            num_responses=num_attempts,
            return_all=True
        )
        
        # Convert to the expected format (list of extractions for each response)
        llm_responses = [response.extractions for response in all_responses]
        
        logger.info(f"Generated {len(llm_responses)} responses using consensus method")
        return llm_responses
        
    except Exception as e:
        logger.error(f"Consensus extraction failed: {e}")
        return []


def aggregate_by_agreement_threshold(
    extractions_list: List[List[ConceptExtraction]], agreement_threshold: int
) -> Dict[str, set]:
    """Aggregate concepts and filter by agreement threshold across LLM attempts"""
    concept_counts = {}

    for extractions in extractions_list:
        for extraction in extractions:
            source = extraction.source
            if source not in concept_counts:
                concept_counts[source] = {}

            for concept in extraction.concepts:
                concept_lower = concept.lower()
                concept_counts[source][concept_lower] = (
                    concept_counts[source].get(concept_lower, 0) + 1
                )

    # Filter by agreement threshold
    return {
        source: {c for c, count in concepts.items() if count >= agreement_threshold}
        for source, concepts in concept_counts.items()
        if any(count >= agreement_threshold for count in concepts.values())
    }


def process_source_chunk(chunk_data: Tuple[List[str], int]) -> Dict[str, set]:
    """Process a chunk of college name sources for concept extraction"""
    sources, num_attempts = chunk_data
    source_concepts = {}

    for batch in chunk(sources, BATCH_SIZE):
        multiple_extractions = extract_concepts_with_consensus(batch, num_attempts)

        if multiple_extractions:
            filtered_results = aggregate_by_agreement_threshold(
                multiple_extractions, AGREEMENT_THRESHOLD
            )
            source_concepts.update(filtered_results)

    return source_concepts


def load_unique_college_names(input_file: Path) -> List[str]:
    """Load college names from file and remove duplicates while preserving order"""
    with open(input_file, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f.readlines()]

    # Remove duplicates while preserving order
    seen = set()
    return [s for s in sources if not (s in seen or seen.add(s))]


def extract_all_concepts_with_checkpoints(sources: List[str]) -> Tuple[Dict[str, set], Counter]:
    """Extract concepts from all sources with checkpoint recovery support"""
    checkpoint_file = CHECKPOINT_DIR / "lv0_s1_checkpoint.json"
    
    # Process with simple checkpointing
    source_concept_mapping = process_with_checkpoint(
        items=sources,
        batch_size=CHUNK_SIZE,  # Process in larger chunks for efficiency
        checkpoint_file=checkpoint_file,
        process_batch_func=lambda chunk: process_source_chunk((chunk, LLM_ATTEMPTS))
    )
    
    # Count concept frequencies
    all_concepts = [
        concept.lower()
        for concepts in source_concept_mapping.values()
        for concept in concepts
    ]
    
    return source_concept_mapping, Counter(all_concepts)


def save_results(
    sources: List[str],
    source_concepts: Dict[str, set],
    concept_counts: Counter,
    verified_concepts: List[str],
) -> None:
    """Save all results to files"""
    # Ensure directories exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save concept list (main output)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for concept in verified_concepts:
            f.write(f"{concept}\n")
    logger.info(f"Saved {len(verified_concepts)} concepts to {OUTPUT_FILE}")

    # Save metadata
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "step": "lv0_s1_extract_concepts",
                    "input_file": str(INPUT_FILE),
                    "input_count": len(sources),
                    "output_file": str(OUTPUT_FILE),
                    "output_count": len(verified_concepts),
                    "processing_params": {
                        "batch_size": BATCH_SIZE,
                        "chunk_size": CHUNK_SIZE,
                        "max_workers": MAX_WORKERS,
                        "llm_attempts": LLM_ATTEMPTS,
                        "concept_agreement_threshold": AGREEMENT_THRESHOLD,
                        "concept_frequency_threshold": FREQUENCY_THRESHOLD,
                    },
                },
                "source_concept_mapping": {
                    source: sorted(list(concepts))
                    for source, concepts in source_concepts.items()
                },
                "concept_frequencies": {
                    concept: count
                    for concept, count in concept_counts.items()
                    if count >= FREQUENCY_THRESHOLD
                },
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    logger.info(f"Saved metadata to {META_FILE}")


# Removed CSV generation - we only need the text file and metadata


def main():
    """Main execution function"""
    try:
        logger.info("Starting concept extraction from college names")
        logger.info(f"Consensus attempts: {LLM_ATTEMPTS}, Agreement threshold: {AGREEMENT_THRESHOLD}")

        # Read input
        sources = load_unique_college_names(INPUT_FILE)
        logger.info(f"Read {len(sources)} unique sources")

        # Process concepts
        source_concepts, concept_counts = extract_all_concepts_with_checkpoints(sources)

        # Filter by frequency threshold
        verified_concepts = sorted(
            [
                concept
                for concept, count in concept_counts.items()
                if count >= FREQUENCY_THRESHOLD
            ]
        )
        logger.info(f"Extracted {len(verified_concepts)} verified concepts")

        # Save all results
        save_results(sources, source_concepts, concept_counts, verified_concepts)

        logger.info(f"Concept extraction completed: {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"Error in concept extraction: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
