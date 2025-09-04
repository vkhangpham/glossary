import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import structured_completion_consensus
from generate_glossary.deduplication.utils import normalize_text
from generate_glossary.generation.shared import process_with_checkpoint

load_dotenv()
logger = setup_logger("lv0.s3")

# Configuration constants - simple and direct
LEVEL = 0
BATCH_SIZE = 20
CHUNK_SIZE = 100
MAX_WORKERS = 4
LLM_ATTEMPTS = 3  # Number of LLM responses for consensus
MAX_EXAMPLES = 10  # Maximum number of college examples to show
CACHE_TTL = 3600  # Cache consensus results for 1 hour
TEMPERATURE = 0.3  # Lower temperature for verification task

# File paths - explicit and clear
DATA_DIR = Path("data/generation/lv0")
INPUT_FILE = DATA_DIR / "lv0_s2_output.txt"  # Output from step 2
INPUT_META_FILE = DATA_DIR / "lv0_s2_metadata.json"  # Metadata from step 2
OUTPUT_FILE = DATA_DIR / "lv0_s3_output.txt"  # Main output
META_FILE = DATA_DIR / "lv0_s3_metadata.json"  # Metadata output
CHECKPOINT_DIR = DATA_DIR / ".checkpoints"

# Test mode paths
TEST_DATA_DIR = Path("data/generation/tests")
TEST_INPUT_FILE = TEST_DATA_DIR / "lv0_s2_output.txt"
TEST_INPUT_META_FILE = TEST_DATA_DIR / "lv0_s2_metadata.json"
TEST_OUTPUT_FILE = TEST_DATA_DIR / "lv0_s3_output.txt"
TEST_META_FILE = TEST_DATA_DIR / "lv0_s3_metadata.json"
TEST_CHECKPOINT_DIR = TEST_DATA_DIR / ".checkpoints"

class VerificationResult(BaseModel):
    is_valid: bool = Field(description="Whether the term is a valid broad academic discipline")


SYSTEM_PROMPT = """You are an expert in academic research classification with a deep understanding of research domains, 
academic departments, scientific disciplines, and specialized fields of study.

Your task is to verify whether terms represent legitimate broad academic disciplines by considering:
1. Academic relevance - Is it a recognized field of study or broad academic discipline?
2. Disciplinary context - Does it represent a major division of knowledge in academia?
3. Scope - Is it broad enough to encompass multiple research areas or subdisciplines?

Accept:
- Broad academic disciplines (e.g., humanities, sciences, engineering)
- Major fields of study (e.g., arts, medicine, law)
- Traditional knowledge domains (e.g., social sciences, natural sciences)

DO NOT accept:
- Narrow specializations or subdisciplines (e.g., organic chemistry, medieval history)
- Technical methodologies (e.g., spectroscopy, chromatography)
- Specific research topics (e.g., climate change, artificial intelligence)
- Acronyms (e.g., STEM, AI) unless they are universally recognized as standalone concepts
- Proper nouns or names (e.g., Harvard, MIT)
- Informal or colloquial terms (e.g., stuff, thing)
- General English words without specific academic meaning"""

def create_verification_prompt(keyword: str, colleges: List[str]) -> str:
    """Create prompt for single keyword verification"""
    # Take up to MAX_EXAMPLES example colleges
    example_colleges = colleges[:MAX_EXAMPLES]
    colleges_str = "\n".join(f"- {college}" for college in example_colleges)
    
    return f"""Analyze whether "{keyword}" is a valid broad academic discipline.

Evidence - Colleges/schools/divisions that mention this concept:
{colleges_str}

Consider:
1. Is it a recognized major field of study or broad academic discipline?
2. Does it represent a major division of knowledge in academia?
3. Is it broad enough to encompass multiple research areas or subdisciplines?

Answer with true if it meets these criteria, false otherwise."""


def verify_keyword_with_consensus(
    keyword: str, colleges: List[str], num_attempts: int = LLM_ATTEMPTS
) -> bool:
    """Verify if a keyword is a valid academic discipline using consensus"""
    prompt = create_verification_prompt(keyword, colleges)
    
    # Build messages
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Run the async consensus method
        consensus = asyncio.run(
            structured_completion_consensus(
                messages=messages,
                response_model=VerificationResult,
                tier="budget",  # Use budget tier for verification
                num_responses=num_attempts,
                return_all=False,  # Only need consensus
                temperature=TEMPERATURE,
                cache_ttl=CACHE_TTL,
            )
        )
        
        return consensus.is_valid
        
    except Exception as e:
        logger.error(f"Error verifying keyword '{keyword}': {e}")
        return False


def process_keyword_chunk(chunk_data: Tuple[List[Tuple[str, List[str]]], int]) -> Dict[str, bool]:
    """Process a chunk of keywords for verification"""
    keyword_colleges_list, num_attempts = chunk_data
    results = {}
    
    for keyword, colleges in keyword_colleges_list:
        is_valid = verify_keyword_with_consensus(keyword, colleges, num_attempts)
        results[keyword] = is_valid
        logger.debug(f"Verified '{keyword}': {is_valid}")
    
    return results

def get_concept_colleges(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract colleges for each concept from metadata
    
    Args:
        metadata: Metadata from s1
        
    Returns:
        Dictionary mapping concepts to their colleges
    """
    concept_colleges = {}
    
    # Get source->concepts mapping from metadata
    source_mapping = metadata.get("source_concept_mapping", {})
    
    # Invert mapping from source->concepts to concept->sources
    for source, concepts in source_mapping.items():
        # Skip sources with empty concept lists
        if not concepts:
            continue
            
        for concept in concepts:
            if concept not in concept_colleges:
                concept_colleges[concept] = []
            if source not in concept_colleges[concept]:
                concept_colleges[concept].append(source)
    
    return concept_colleges

def is_single_word(keyword: str) -> bool:
    """Check if keyword is a single word (no spaces)"""
    return len(keyword.split()) == 1


def test():
    """Test mode: Read from test directory and save to test directory"""
    global INPUT_FILE, INPUT_META_FILE, OUTPUT_FILE, META_FILE, CHECKPOINT_DIR
    
    # Save original values
    original_input = INPUT_FILE
    original_input_meta = INPUT_META_FILE
    original_output = OUTPUT_FILE
    original_meta = META_FILE
    original_checkpoint = CHECKPOINT_DIR
    
    # Set test values
    INPUT_FILE = TEST_INPUT_FILE
    INPUT_META_FILE = TEST_INPUT_META_FILE
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
        INPUT_META_FILE = original_input_meta
        OUTPUT_FILE = original_output
        META_FILE = original_meta
        CHECKPOINT_DIR = original_checkpoint


def main():
    """Main execution function"""
    try:
        input_file = INPUT_FILE
        input_meta_file = INPUT_META_FILE
        output_file = OUTPUT_FILE
        meta_file = META_FILE
        checkpoint_dir = CHECKPOINT_DIR
        
        logger.info("Starting single-word academic discipline verification")
        logger.info(f"Consensus attempts: {LLM_ATTEMPTS} (majority vote wins)")
        
        # Read input keywords
        with open(input_file, "r", encoding='utf-8') as f:
            all_keywords = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(all_keywords)} total academic disciplines")
        
        # Read metadata from s2 (was s1, now s2 provides the metadata)
        with open(input_meta_file, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        concept_colleges = get_concept_colleges(metadata)
        logger.info(f"Loaded college data for {len(concept_colleges)} concepts")
        
        # Filter for single-word keywords
        single_word_keywords = [kw for kw in all_keywords if is_single_word(kw)]
        multi_word_keywords = [kw for kw in all_keywords if not is_single_word(kw)]
        logger.info(f"Found {len(single_word_keywords)} single-word disciplines to verify")
        logger.info(f"Found {len(multi_word_keywords)} multi-word disciplines to bypass")
        
        # Prepare data for checkpoint processing
        keyword_colleges_list = [
            (kw, concept_colleges.get(kw, [])) 
            for kw in single_word_keywords
        ]
        
        # Process with checkpoint support
        checkpoint_file = checkpoint_dir / "lv0_s3_checkpoint.json"
        verification_results = process_with_checkpoint(
            items=keyword_colleges_list,
            batch_size=CHUNK_SIZE,
            checkpoint_file=checkpoint_file,
            process_batch_func=lambda chunk: process_keyword_chunk(
                (chunk, LLM_ATTEMPTS)
            ),
        )
        
        # Split into verified and unverified
        verified_single_words = [
            kw for kw, is_valid in verification_results.items() 
            if is_valid
        ]
        unverified_keywords = [
            kw for kw, is_valid in verification_results.items() 
            if not is_valid
        ]
        
        logger.info(f"Verified {len(verified_single_words)} single-word disciplines")
        logger.info(f"Rejected {len(unverified_keywords)} single-word disciplines")
        
        # Combine verified single words with multi-word keywords
        all_verified_keywords = verified_single_words + multi_word_keywords
        logger.info(f"Total verified disciplines: {len(all_verified_keywords)}")
        
        # Normalize and deduplicate
        normalized_keywords = [normalize_text(kw) for kw in all_verified_keywords]
        seen = set()
        final_keywords = []
        for kw, norm_kw in zip(all_verified_keywords, normalized_keywords):
            if norm_kw not in seen:
                seen.add(norm_kw)
                final_keywords.append(kw)  # Keep original form
        
        logger.info(f"Final unique disciplines after normalization: {len(final_keywords)}")
        
        # Save all results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save verified keywords
        with open(output_file, "w", encoding='utf-8') as f:
            for kw in sorted(final_keywords):
                f.write(f"{kw}\n")
        logger.info(f"Saved {len(final_keywords)} disciplines to {output_file}")
        
        # Save metadata
        with open(meta_file, "w", encoding='utf-8') as f:
            json.dump(
                {
                    "metadata": {
                        "step": "lv0_s3_verify_single_token",
                        "input_file": str(input_file),
                        "input_count": len(all_keywords),
                        "single_word_count": len(single_word_keywords),
                        "multi_word_count": len(multi_word_keywords),
                        "verified_single_word_count": len(verified_single_words),
                        "unverified_single_word_count": len(unverified_keywords),
                        "output_file": str(output_file),
                        "output_count": len(final_keywords),
                        "processing_params": {
                            "batch_size": BATCH_SIZE,
                            "chunk_size": CHUNK_SIZE,
                            "max_workers": MAX_WORKERS,
                            "llm_attempts": LLM_ATTEMPTS,
                            "consensus_mode": "majority_vote",
                            "temperature": TEMPERATURE,
                            "cache_ttl": CACHE_TTL,
                        },
                    },
                    "verification_results": {
                        kw: {"is_verified": is_valid}
                        for kw, is_valid in verification_results.items()
                    },
                    "verified_single_words": verified_single_words,
                    "unverified_keywords": unverified_keywords,
                    "multi_word_keywords": multi_word_keywords,
                    "final_keywords": final_keywords,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Saved metadata to {meta_file}")
        
        logger.info("Academic discipline verification completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        main() 