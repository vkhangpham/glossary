import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
from generate_glossary.utils.llm_simple import infer_structured, infer_text, get_random_llm_config
from generate_glossary.deduplicator.dedup_utils import normalize_text
from generate_glossary.utils.resilient_processing import (
    KeywordVerificationProcessor, create_processing_config, get_checkpoint_dir
)

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv3.s3")

# Get the base directory
BASE_DIR = os.getcwd()

# Use centralized configuration
LEVEL = 3
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

# No longer need init_llm - using direct calls

SYSTEM_PROMPT = """You are an expert in academic research classification with a deep understanding of research domains, 
academic conferences, scientific journals, and specialized fields of study.

Your task is to verify whether terms represent legitimate research concepts frequently discussed in academic conferences 
and journals. You should consider the context of the conferences or journal special issues where these terms appear.

DO NOT accept any of the following:
- Acronyms (unless they are widely established in the field like "AI" or "NLP")
- Proper nouns or names
- Informal or colloquial terms
- Generic terms without specific academic meaning (e.g., "applications", "methods", "systems" on their own)"""

def build_verification_prompt(
    keyword: str,
    journals: List[str]
) -> str:
    """
    Build prompt for keyword verification using metadata
    
    Args:
        keyword: Keyword to verify
        journals: List of journals where this concept appears
        
    Returns:
        Formatted prompt for LLM
    """
    # Take up to MAX_EXAMPLES example journals
    example_journals = journals[:processing_config.max_examples]
    journals_str = "\n".join(f"- {journal}" for journal in example_journals)
    
    return f"""Analyze whether "{keyword}" is a valid research concept based on the following evidence:

Example Journal Topics where this concept appears:
{journals_str}

Return only a JSON with an is_valid boolean field:
{{
    "is_valid": true/false
}}"""

def verify_keyword(
    keyword: str,
    journals: List[str],
    provider: Optional[str] = None
) -> bool:
    """
    Verify a single keyword using LLM and metadata
    
    Args:
        keyword: Keyword to verify
        journals: List of journals where this concept appears
        provider: Optional LLM provider (openai or gemini)
        
    Returns:
        True if keyword is verified, False otherwise
    """
    # Skip if no journals found
    if not journals:
        logger.debug(f"No journals found for keyword '{keyword}'")
        return False
        
    prompt = build_verification_prompt(
        keyword,
        journals
    )
    
    try:
        response = infer_text(
            provider=provider or "openai",
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3
        )
        try:
            result = json.loads(response.text)
            return result.get("is_valid", False)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for '{keyword}': {response.text}")
            return False
            
    except Exception as e:
        if "insufficient_quota" in str(e):
            raise QuotaExceededError(f"API quota exceeded for provider {provider}")
        logger.error(f"Error verifying keyword '{keyword}': {str(e)}")
        return False

def verify_keywords_batch(
    keywords: List[str],
    concept_journals: Dict[str, List[str]],
    provider: Optional[str] = None,
    batch_size: int = processing_config.batch_size,
    cooldown: int = processing_config.cooldown_period,
    cooldown_freq: int = processing_config.cooldown_frequency
) -> Dict[str, Dict[str, Any]]:
    """
    Verify a batch of keywords with rate limiting
    
    Args:
        keywords: List of keywords to verify
        concept_journals: Dictionary mapping keywords to their journals
        provider: Optional LLM provider (openai or gemini)
        batch_size: Number of keywords to process before cooldown
        cooldown: Cooldown period in seconds
        cooldown_freq: Number of batches to process before cooling down
        
    Returns:
        Dictionary mapping keywords to their verification results
    """
    results = {}
    
    for i in tqdm(range(0, len(keywords), batch_size), desc="Verifying keywords"):
        batch = keywords[i:i + batch_size]
        batch_results = {}
        
        for keyword in batch:
            try:
                journals = concept_journals.get(keyword, [])
                is_verified = verify_keyword(keyword, journals, provider)
                batch_results[keyword] = {
                    "is_verified": is_verified,
                    "journals": journals,
                    "provider": provider or "openai",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.error(f"Failed to verify '{keyword}': {str(e)}")
                batch_results[keyword] = {
                    "is_verified": False,
                    "journals": [],
                    "provider": provider or "openai",
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
        
        results.update(batch_results)
        
        # Apply cooldown every cooldown_freq batches
        batch_num = i // batch_size
        if batch_num > 0 and batch_num % cooldown_freq == 0:
            logger.debug(f"Processed {i + len(batch)}/{len(keywords)} keywords. Cooling down...")
            time.sleep(cooldown)
    
    return results

def get_concept_journals(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract journals for each concept from metadata
    
    Args:
        metadata: Metadata from s1
        
    Returns:
        Dictionary mapping concepts to their source journals
    """
    # Map concepts to their journals
    concept_journals = {}
    
    # Try different possible keys (in case of schema changes)
    concept_journal_map = metadata.get("concept_to_conference_journal_mapping", {})
    if not concept_journal_map:
        concept_journal_map = metadata.get("concept_to_journal_mapping", {})
        
    if not concept_journal_map:
        if "metadata" in metadata:
            nested_meta = metadata.get("metadata", {})
            concept_journal_map = nested_meta.get("concept_to_conference_journal_mapping", {})
            if not concept_journal_map:
                concept_journal_map = nested_meta.get("concept_to_journal_mapping", {})
    
    # If still no mapping found
    if not concept_journal_map:
        logger.warning("Could not find concept-journal mapping in metadata.")
        return {}
        
    # Copy journal names for each concept
    for concept, journals in concept_journal_map.items():
        if isinstance(journals, list):
            concept_journals[concept] = journals
        else:
            # Handle string or other formats
            concept_journals[concept] = [str(journals)]
            
    return concept_journals

def is_single_word(keyword: str) -> bool:
    """Check if keyword is a single word (no spaces)"""
    return len(keyword.split()) == 1

def main():
    """Main entry point for concept verification"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(level_config.get_step_output_file(3)), exist_ok=True)
        
        # Read input concepts
        with open(level_config.get_step_input_file(3), "r", encoding="utf-8") as f:
            concepts = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(concepts)} concepts from input file")
        
        # Load metadata to get journal sources for each concept
        with open(level_config.get_step_metadata_file(3), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {level_config.get_step_metadata_file(3)}")
        
        # Get journals for each concept
        concept_journals = get_concept_journals(metadata)
        journals_found = len([c for c, j in concept_journals.items() if j])
        logger.info(f"Found journal sources for {journals_found}/{len(concepts)} concepts")
        
        # Focus on single-word concepts that need verification
        single_word_concepts = [c for c in concepts if is_single_word(c)]
        logger.info(f"Identified {len(single_word_concepts)} single-word concepts for verification")
        
        # Initialize resilient processor with checkpoint support
        checkpoint_dir = get_checkpoint_dir(level_config)
        processor = KeywordVerificationProcessor(checkpoint_dir, LEVEL)
        
        # Create configuration for checkpointing
        config = create_processing_config(processing_config)
        
        # Verify concepts with automatic checkpointing
        results = processor.verify_keywords(
            keywords=single_word_concepts,
            concept_journals=concept_journals,
            verify_func=verify_keyword,
            batch_size=processing_config.batch_size,
            cooldown_period=processing_config.cooldown_period,
            cooldown_frequency=processing_config.cooldown_frequency,
            config=config,
            provider=None,  # Use default provider
            force_restart=False  # Set to True to ignore existing checkpoints
        )
        
        # Filter out unverified concepts
        verified_concepts = set()
        unverified_concepts = set()
        
        # First add all multi-word concepts (automatically verified)
        multi_word_concepts = set(concepts) - set(single_word_concepts)
        verified_concepts.update(multi_word_concepts)
        logger.info(f"Added {len(multi_word_concepts)} multi-word concepts (automatically verified)")
        
        # Then add verified single-word concepts
        verified_single_words = set()
        for concept, result in results.items():
            if result.get("is_verified", False):
                verified_concepts.add(concept)
                verified_single_words.add(concept)
            else:
                unverified_concepts.add(concept)
                
        logger.info(f"Verified {len(verified_single_words)}/{len(single_word_concepts)} single-word concepts")
        logger.info(f"Total verified concepts: {len(verified_concepts)}")
        logger.info(f"Unverified concepts: {len(unverified_concepts)}")
        
        # Save verified concepts
        with open(level_config.get_step_output_file(3), "w", encoding="utf-8") as f:
            for concept in sorted(verified_concepts):
                f.write(f"{concept}\n")
        logger.info(f"Saved {len(verified_concepts)} verified concepts to {level_config.get_step_output_file(3)}")
        
        # Save metadata about the verification process
        validation_metadata = {
            "metadata": {
                "input_concepts": len(concepts),
                "single_word_concepts": len(single_word_concepts),
                "multi_word_concepts": len(multi_word_concepts),
                "verified_single_word_concepts": len(verified_single_words),
                "unverified_single_word_concepts": len(single_word_concepts) - len(verified_single_words),
                "total_verified_concepts": len(verified_concepts),
                "total_unverified_concepts": len(unverified_concepts)
            },
            "concept_verification_details": results
        }
        
        with open(level_config.get_validation_metadata_file(3), "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=2)
        logger.info(f"Saved validation metadata to {level_config.get_validation_metadata_file(3)}")
        
    except Exception as e:
        logger.error(f"Error during concept verification: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 