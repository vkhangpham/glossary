import json
import asyncio
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk


from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import structured_completion_consensus, load_prompt_from_file
from generate_glossary.generation.shared import process_with_checkpoint

load_dotenv()
logger = setup_logger("lv0.s1")

LEVEL = 0
BATCH_SIZE = 20
CHUNK_SIZE = 100
MAX_WORKERS = 4
LLM_ATTEMPTS = 3  # Number of LLM responses for consensus (majority wins)
FREQUENCY_THRESHOLD = 2  # Minimum frequency across sources to keep concept
CACHE_TTL = 3600  # Cache consensus results for 1 hour
TEMPERATURE = 1.0  # GPT-5 models only support temperature=1

DATA_DIR = Path("data/generation/lv0")
INPUT_FILE = DATA_DIR / "lv0_s0_output.txt"  # Output from step 0
OUTPUT_FILE = DATA_DIR / "lv0_s1_output.txt"  # Main output
META_FILE = DATA_DIR / "lv0_s1_metadata.json"  # Metadata output
CHECKPOINT_DIR = DATA_DIR / ".checkpoints"

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


# Default prompts - can be overridden by loading from file
SYSTEM_PROMPT = """You are an expert at extracting academic concepts from institutional names.
Extract the core academic disciplines and fields of study from the given college/school names.
Focus on the academic subjects, not the institution type or location."""

USER_TEMPLATE = """Extract academic concepts from these institutions:

{sources}

For each institution, identify the core academic fields and disciplines."""

# Try to load optimized prompts if they exist
optimized_system = load_prompt_from_file("data/prompts/lv0_s1_system_latest.json")
if optimized_system:
    SYSTEM_PROMPT = optimized_system
    logger.info("Loaded optimized system prompt from file")

optimized_user = load_prompt_from_file("data/prompts/lv0_s1_user_latest.json")
if optimized_user:
    USER_TEMPLATE = optimized_user
    logger.info("Loaded optimized user prompt template from file")

# Semantic validation prompt for quality assurance
SEMANTIC_VALIDATION = """Verify that each extracted concept is:
1. An academic discipline or field of study
2. Not a location, person name, or institution name
3. A meaningful educational or research area
4. Properly normalized (lowercase, no abbreviations)"""


def create_extraction_prompt(sources: List[str]) -> str:
    """Create LLM prompt for concept extraction from college names"""
    sources_str = "\n".join(f"- {source}" for source in sources)
    return USER_TEMPLATE.format(sources=sources_str)


def extract_concepts_with_consensus(
    batch: List[str], num_attempts: int = LLM_ATTEMPTS
) -> List[ConceptExtraction]:
    """Extract concepts from a batch using LLM consensus mechanism

    Returns the consensus extraction directly - no need for additional aggregation
    since the API already finds the most agreed-upon response.
    """
    prompt = create_extraction_prompt(batch)

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    try:
        # Run the async consensus method in a synchronous context
        # The API returns the consensus - the most common response among all attempts
        consensus = asyncio.run(
            structured_completion_consensus(
                messages=messages,
                response_model=ConceptExtractionList,
                tier="budget",  # Use tier instead of specific model
                num_responses=num_attempts,
                return_all=False,  # We only need consensus, not all responses
                temperature=TEMPERATURE,  # Control response diversity
                cache_ttl=CACHE_TTL,  # Cache for 1 hour to speed up re-runs
                # semantic_validation=SEMANTIC_VALIDATION,  # TODO: Fix semantic validation
            )
        )

        logger.info(f"Consensus extraction completed for batch of {len(batch)}")
        return consensus.extractions

    except (ConnectionError, TimeoutError) as e:
        # Critical network errors - re-raise after logging
        logger.error(
            f"Network error in consensus extraction for batch of {len(batch)}: {e}"
        )
        raise
    except KeyError as e:
        # API key or configuration errors - re-raise
        logger.error(f"Configuration error in consensus extraction: {e}")
        raise
    except ValueError as e:
        # Data validation errors - log but continue with empty result
        logger.warning(
            f"Data validation error in consensus extraction for batch of {len(batch)}: {e}"
        )
        return []
    except Exception as e:
        # Unexpected errors - log with full context but don't crash
        logger.error(
            f"Unexpected error in consensus extraction for batch of {len(batch)} items: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return []


def process_source_chunk(chunk_data: Tuple[List[str], int]) -> Dict[str, List[str]]:
    """Process a chunk of college name sources for concept extraction"""
    sources, num_attempts = chunk_data
    source_concepts = {}

    for batch in chunk(sources, BATCH_SIZE):
        consensus_extractions = extract_concepts_with_consensus(batch, num_attempts)

        if consensus_extractions:
            for extraction in consensus_extractions:
                source = extraction.source
                concepts = [concept.lower() for concept in extraction.concepts]
                concepts = list(dict.fromkeys(concepts))
                source_concepts[source] = concepts

    return source_concepts


def load_unique_college_names(input_file: Path) -> List[str]:
    """Load college names from file and remove duplicates while preserving order"""
    with open(input_file, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f.readlines()]

    seen = set()
    return [s for s in sources if not (s in seen or seen.add(s))]



def test():
    """Test mode: Read from test directory and save to test directory"""
    global INPUT_FILE, OUTPUT_FILE, META_FILE, CHECKPOINT_DIR

    original_input = INPUT_FILE
    original_output = OUTPUT_FILE
    original_meta = META_FILE
    original_checkpoint = CHECKPOINT_DIR

    INPUT_FILE = TEST_INPUT_FILE
    OUTPUT_FILE = TEST_OUTPUT_FILE
    META_FILE = TEST_META_FILE
    CHECKPOINT_DIR = TEST_CHECKPOINT_DIR

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running in TEST MODE")

    try:
        main()
    finally:
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
            f"Consensus attempts: {LLM_ATTEMPTS} (majority vote wins), Frequency threshold: {FREQUENCY_THRESHOLD}"
        )

        sources = load_unique_college_names(input_file)
        logger.info(f"Read {len(sources)} unique sources")
        checkpoint_file = checkpoint_dir / "lv0_s1_checkpoint.json"
        source_concepts = process_with_checkpoint(
            items=sources,
            batch_size=CHUNK_SIZE,
            checkpoint_file=checkpoint_file,
            process_batch_func=lambda chunk: process_source_chunk(
                (chunk, LLM_ATTEMPTS)
            ),
        )

        all_concepts = [
            concept.lower()
            for concepts in source_concepts.values()
            for concept in concepts
        ]
        concept_counts = Counter(all_concepts)
        verified_concepts = sorted(
            [
                concept
                for concept, count in concept_counts.items()
                if count >= FREQUENCY_THRESHOLD
            ]
        )
        logger.info(f"Extracted {len(verified_concepts)} verified concepts")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for concept in verified_concepts:
                f.write(f"{concept}\n")
        logger.info(f"Saved {len(verified_concepts)} concepts to {output_file}")
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
                            "consensus_mode": "majority_vote",
                            "concept_frequency_threshold": FREQUENCY_THRESHOLD,
                            "temperature": TEMPERATURE,
                            "cache_ttl": CACHE_TTL,
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
