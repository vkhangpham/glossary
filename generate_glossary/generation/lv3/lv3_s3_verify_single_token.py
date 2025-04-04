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
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv3.s3")

# Get the base directory
BASE_DIR = os.getcwd()

class Config:
    """Configuration for concept filtering"""
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s2_filtered_concepts.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s3_verified_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s1_metadata.json")
    VALIDATION_META_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s3_metadata.json")
    BATCH_SIZE = 10
    COOLDOWN_PERIOD = 1
    COOLDOWN_FREQUENCY = 10
    MAX_RETRIES = 3
    MAX_EXAMPLES = 5  # Maximum number of example conference topics to show in prompt

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

def init_llm(provider: Optional[str] = None) -> BaseLLM:
    """Initialize LLM with specified provider"""
    if not provider:
        provider = Provider.OPENAI  # Default to OpenAI
        
    return LLMFactory.create_llm(
        provider=provider,
        model=OPENAI_MODELS["mini"] if provider == Provider.OPENAI else GEMINI_MODELS["pro"],
        temperature=0.3
    )

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
    conferences: List[str]
) -> str:
    """
    Build prompt for keyword verification using metadata
    
    Args:
        keyword: Keyword to verify
        conferences: List of conferences/journals where this concept appears
        
    Returns:
        Formatted prompt for LLM
    """
    # Take up to MAX_EXAMPLES example conferences
    example_conferences = conferences[:Config.MAX_EXAMPLES]
    conferences_str = "\n".join(f"- {conf}" for conf in example_conferences)
    
    return f"""Analyze whether "{keyword}" is a valid research concept based on the following evidence:

Example Conference Topics / Journal Special Issues where this concept appears:
{conferences_str}

Return only a JSON with an is_valid boolean field:
{{
    "is_valid": true/false
}}"""

def verify_keyword(
    keyword: str,
    conferences: List[str],
    provider: Optional[str] = None
) -> bool:
    """
    Verify a single keyword using LLM and metadata
    
    Args:
        keyword: Keyword to verify
        conferences: List of conferences/journals where this concept appears
        provider: Optional LLM provider (openai or gemini)
        
    Returns:
        True if keyword is verified, False otherwise
    """
    # Skip if no conferences found
    if not conferences:
        logger.debug(f"No conferences found for keyword '{keyword}'")
        return False
        
    prompt = build_verification_prompt(
        keyword,
        conferences
    )
    
    try:
        llm = init_llm(provider)
        response = llm.infer(
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
    concept_conferences: Dict[str, List[str]],
    provider: Optional[str] = None,
    batch_size: int = Config.BATCH_SIZE,
    cooldown: int = Config.COOLDOWN_PERIOD,
    cooldown_freq: int = Config.COOLDOWN_FREQUENCY
) -> Dict[str, Dict[str, Any]]:
    """
    Verify a batch of keywords with rate limiting
    
    Args:
        keywords: List of keywords to verify
        concept_conferences: Dictionary mapping keywords to their conferences/journals
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
                conferences = concept_conferences.get(keyword, [])
                is_verified = verify_keyword(keyword, conferences, provider)
                batch_results[keyword] = {
                    "is_verified": is_verified,
                    "conferences": conferences,
                    "provider": provider or Provider.OPENAI,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.error(f"Failed to verify '{keyword}': {str(e)}")
                batch_results[keyword] = {
                    "is_verified": False,
                    "conferences": [],
                    "provider": provider or Provider.OPENAI,
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

def get_concept_conferences(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract conferences for each concept from metadata
    
    Args:
        metadata: Metadata from s1
        
    Returns:
        Dictionary mapping concepts to their conferences/journals
    """
    concept_conferences = {}
    
    # Get source->concepts mapping from metadata
    source_mapping = metadata.get("conference_topic_concept_mapping", {})
    
    # Invert mapping from source->concepts to concept->sources
    for source, concepts in source_mapping.items():
        # Skip sources with empty concept lists
        if not concepts:
            continue
            
        for concept in concepts:
            if concept not in concept_conferences:
                concept_conferences[concept] = []
            if source not in concept_conferences[concept]:
                concept_conferences[concept].append(source)
    
    return concept_conferences

def is_single_word(keyword: str) -> bool:
    """Check if keyword is a single word (no spaces)"""
    return len(keyword.split()) == 1

def main():
    """Main execution function"""
    try:
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")
            
        logger.info("Starting single-word concept verification for conference/journal topics by LLM")
        
        # Read input keywords
        with open(Config.INPUT_FILE, "r", encoding='utf-8') as f:
            all_keywords = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(all_keywords)} total concepts")
        
        # Read metadata from s1
        with open(Config.META_FILE, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        concept_conferences = get_concept_conferences(metadata)
        logger.info(f"Loaded conference/journal data for {len(concept_conferences)} concepts")
        
        # Filter for single-word keywords
        single_word_keywords = [kw for kw in all_keywords if is_single_word(kw)]
        multi_word_keywords = [kw for kw in all_keywords if not is_single_word(kw)]
        logger.info(f"Found {len(single_word_keywords)} single-word concepts to verify")
        logger.info(f"Found {len(multi_word_keywords)} multi-word concepts to bypass")
        
        # Verify single-word keywords
        verification_results = verify_keywords_batch(
            single_word_keywords,
            concept_conferences,
            provider=provider
        )
        logger.info(f"Completed verification of {len(verification_results)} single-word concepts")
        
        # Split into verified and unverified
        verified_single_words = [
            k for k, v in verification_results.items() 
            if v.get("is_verified", False)
        ]
        unverified_keywords = [
            k for k, v in verification_results.items() 
            if not v.get("is_verified", False)
        ]
        logger.info(f"Verified {len(verified_single_words)} single-word concepts")
        logger.info(f"Rejected {len(unverified_keywords)} single-word concepts")
        
        # Combine verified single words with multi-word keywords
        all_verified_keywords = verified_single_words + multi_word_keywords
        logger.info(f"Total verified concepts before normalization: {len(all_verified_keywords)}")
        
        # Normalize all verified keywords
        normalized_keywords = [normalize_text(kw) for kw in all_verified_keywords]
        # Remove duplicates that normalize to the same text while maintaining order
        seen = set()
        final_keywords = []
        for kw, norm_kw in zip(all_verified_keywords, normalized_keywords):
            if norm_kw not in seen:
                seen.add(norm_kw)
                final_keywords.append(kw)  # Keep original form
        
        logger.info(f"Final unique concepts after normalization: {len(final_keywords)}")
        
        # Create output directory if needed
        for path in [Config.OUTPUT_FILE, Config.VALIDATION_META_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save verified keywords to text file
        logger.info(f"Saving verified concepts to {Config.OUTPUT_FILE}")
        with open(Config.OUTPUT_FILE, "w", encoding='utf-8') as f:
            for kw in sorted(final_keywords):
                f.write(f"{kw}\n")
        
        # Save detailed metadata to JSON file
        logger.info(f"Saving metadata to {Config.VALIDATION_META_FILE}")
        metadata = {
            "metadata": {
                "total_input_count": len(all_keywords),
                "single_word_count": len(single_word_keywords),
                "multi_word_count": len(multi_word_keywords),
                "verified_single_word_count": len(verified_single_words),
                "unverified_single_word_count": len(unverified_keywords),
                "total_verified_count": len(all_verified_keywords),
                "normalized_output_count": len(final_keywords),
                "model": init_llm(provider).model,
                "temperature": init_llm(provider).temperature,
                "batch_size": Config.BATCH_SIZE,
                "cooldown_period": Config.COOLDOWN_PERIOD,
                "cooldown_frequency": Config.COOLDOWN_FREQUENCY,
                "max_retries": Config.MAX_RETRIES
            },
            "verification_results": verification_results,
            "verified_single_words": verified_single_words,
            "unverified_keywords": unverified_keywords,
            "multi_word_keywords": multi_word_keywords,
            "final_keywords": final_keywords
        }
        
        with open(Config.VALIDATION_META_FILE, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        logger.info("Conference/journal concept verification completed successfully")
        logger.info(f"Saved {len(final_keywords)} verified concepts to {Config.OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 