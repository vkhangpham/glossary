import json
import asyncio
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

# Test mode paths
TEST_DATA_DIR = Path("data/generation/tests")
TEST_INPUT_FILE = TEST_DATA_DIR / "lv0_s0_output.txt"  # Read from test s0 output
TEST_OUTPUT_FILE = TEST_DATA_DIR / "lv0_s1_output.txt"  # Write test output
TEST_META_FILE = TEST_DATA_DIR / "lv0_s1_metadata.json"  # Test metadata
TEST_CHECKPOINT_DIR = TEST_DATA_DIR / ".checkpoints"


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
        # Run the async consensus method in a synchronous context
        consensus, all_responses = asyncio.run(
            structured_completion_consensus(
                messages=messages,
                response_model=ConceptExtractionList,
                tier="budget",  # Use tier instead of specific model
                num_responses=num_attempts,
                return_all=True,
            )
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


# Removed unused functions - logic is now integrated in main()


def test():
    """Test mode: Read from test directory and save to test directory"""
    global INPUT_FILE, OUTPUT_FILE, META_FILE, CHECKPOINT_DIR
    
    # Save original values
    original_input = INPUT_FILE
    original_output = OUTPUT_FILE
    original_meta = META_FILE
    original_checkpoint = CHECKPOINT_DIR
    
    # Set test values
    INPUT_FILE = TEST_INPUT_FILE
    OUTPUT_FILE = TEST_OUTPUT_FILE
    META_FILE = TEST_META_FILE
    CHECKPOINT_DIR = TEST_CHECKPOINT_DIR
    
    # Ensure test directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Running in TEST MODE")
    
    try:
        # Run main with test settings
        main()
    finally:
        # Restore original values
        INPUT_FILE = original_input
        OUTPUT_FILE = original_output
        META_FILE = original_meta
        CHECKPOINT_DIR = original_checkpoint


def main():
    """Main execution function"""
    try:
        input_file = INPUT_FILE
        output_file = OUTPUT_FILE
        meta_file = META_FILE
        checkpoint_dir = CHECKPOINT_DIR
            
        logger.info("Starting concept extraction from college names")
        logger.info(
            f"Consensus attempts: {LLM_ATTEMPTS}, Agreement threshold: {AGREEMENT_THRESHOLD}"
        )

        # Read input
        sources = load_unique_college_names(input_file)
        logger.info(f"Read {len(sources)} unique sources")
        

        # Process concepts with appropriate checkpoint
        checkpoint_file = checkpoint_dir / "lv0_s1_checkpoint.json"
        source_concepts = process_with_checkpoint(
            items=sources,
            batch_size=CHUNK_SIZE,
            checkpoint_file=checkpoint_file,
            process_batch_func=lambda chunk: process_source_chunk((chunk, LLM_ATTEMPTS)),
        )
        
        # Count concept frequencies
        all_concepts = [
            concept.lower()
            for concepts in source_concepts.values()
            for concept in concepts
        ]
        concept_counts = Counter(all_concepts)

        # Filter by frequency threshold
        verified_concepts = sorted(
            [
                concept
                for concept, count in concept_counts.items()
                if count >= FREQUENCY_THRESHOLD
            ]
        )
        logger.info(f"Extracted {len(verified_concepts)} verified concepts")

        # Save all results with appropriate paths
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save concept list (main output)
        with open(output_file, "w", encoding="utf-8") as f:
            for concept in verified_concepts:
                f.write(f"{concept}\n")
        logger.info(f"Saved {len(verified_concepts)} concepts to {output_file}")

        # Save metadata
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "step": "lv0_s1_extract_concepts",
                        "input_file": str(input_file),
                        "input_count": len(sources),
                        "output_file": str(output_file),
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
        logger.info(f"Saved metadata to {meta_file}")

        logger.info(f"Concept extraction completed: {output_file}")

    except Exception as e:
        logger.error(f"Error in concept extraction: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
