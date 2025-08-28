print("Starting lv0_s3_verify_single_token.py script...")

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Package structure now properly configured with pyproject.toml

from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
from generate_glossary.utils.llm_simple import infer_text
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv0.s3")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use centralized configuration
LEVEL = 0
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

# No longer need init_llm - using direct calls

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

def build_verification_prompt(
    keyword: str,
    colleges: List[str]
) -> str:
    """
    Build prompt for keyword verification using metadata
    
    Args:
        keyword: Keyword to verify
        colleges: List of colleges where this concept appears
        
    Returns:
        Formatted prompt for LLM
    """
    # Take up to MAX_EXAMPLES example colleges
    example_colleges = colleges[:processing_config.max_examples]
    colleges_str = "\n".join(f"- {college}" for college in example_colleges)
    
    return f"""Analyze whether "{keyword}" is a valid broad academic discipline based on the following criteria:

1. Is it a recognized major field of study or broad academic discipline?
2. Does it represent a major division of knowledge in academia?
3. Is it broad enough to encompass multiple research areas or subdisciplines?

Evidence - Colleges/schools/divisions that mention this concept:
{colleges_str}

Consider:
- The academic context where the term appears
- Its usage in higher education organizational structures
- Whether it represents a well-defined broad discipline in academia

Return only a JSON with an is_valid boolean field:
{{
    "is_valid": true/false
}}"""

def clean_json_response(response: str) -> str:
    """Clean JSON response by removing code block markers"""
    # Remove code block markers if present
    response = response.replace('```json\n', '').replace('\n```', '')
    # Remove any leading/trailing whitespace
    response = response.strip()
    return response

def verify_keyword(
    keyword: str,
    colleges: List[str],
    provider: Optional[str] = None
) -> bool:
    """Verify if a keyword is a valid academic discipline."""
    try:
        prompt = build_verification_prompt(keyword, colleges)
        logger.debug(f"Verification prompt for '{keyword}':\n{prompt}")
        
        try:
            response = infer_text(
                provider=provider or "openai",
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3
            )
            logger.debug(f"Raw response for '{keyword}':\n{response.text}")
            response_json = clean_json_response(response.text)
            logger.debug(f"Cleaned response for '{keyword}':\n{response_json}")
            result = json.loads(response_json)
            return result.get("is_valid", False)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing response for keyword '{keyword}': {e}")
            logger.error(f"Raw response: {response}")
            return False
        except Exception as e:
            if "insufficient_quota" in str(e):
                raise QuotaExceededError(f"API quota exceeded for provider {provider}")
            logger.error(f"Error verifying keyword '{keyword}': {e}")
            return False
    except QuotaExceededError as e:
        logger.error(str(e))
        raise  # Re-raise to be handled by caller
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return False

def verify_keywords_batch(
    keywords: List[str],
    concept_colleges: Dict[str, List[str]],
    provider: Optional[str] = None,
    batch_size: int = processing_config.batch_size,
    cooldown: int = processing_config.cooldown_period,
    cooldown_freq: int = processing_config.cooldown_frequency
) -> Dict[str, Dict[str, Any]]:
    """
    Verify a batch of keywords with rate limiting
    
    Args:
        keywords: List of keywords to verify
        concept_colleges: Dictionary mapping keywords to their colleges
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
                colleges = concept_colleges.get(keyword, [])
                is_verified = verify_keyword(keyword, colleges, provider)
                batch_results[keyword] = {
                    "is_verified": is_verified,
                    "colleges": colleges,
                    "provider": provider or Provider.OPENAI,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.error(f"Failed to verify '{keyword}': {str(e)}")
                batch_results[keyword] = {
                    "is_verified": False,
                    "colleges": [],
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

def main():
    """Main execution function"""
    try:
        print("Starting main function of lv0_s3_verify_single_token.py...")
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")
        
        logger.info("Starting single-word academic discipline verification by LLM")
        
        # Read input keywords
        with open(level_config.get_step_input_file(3), "r", encoding='utf-8') as f:
            all_keywords = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(all_keywords)} total academic disciplines")
        
        # Read metadata from s1
        with open(level_config.get_step_metadata_file(3), "r", encoding='utf-8') as f:
            metadata = json.load(f)
        concept_colleges = get_concept_colleges(metadata)
        logger.info(f"Loaded college data for {len(concept_colleges)} concepts")
        
        # Filter for single-word keywords
        single_word_keywords = [kw for kw in all_keywords if is_single_word(kw)]
        multi_word_keywords = [kw for kw in all_keywords if not is_single_word(kw)]
        logger.info(f"Found {len(single_word_keywords)} single-word disciplines to verify")
        logger.info(f"Found {len(multi_word_keywords)} multi-word disciplines to bypass")
        
        # Verify single-word keywords
        verification_results = verify_keywords_batch(
            single_word_keywords,
            concept_colleges,
            provider=provider
        )
        logger.info(f"Completed verification of {len(verification_results)} single-word disciplines")
        
        # Split into verified and unverified
        verified_single_words = [
            k for k, v in verification_results.items() 
            if v.get("is_verified", False)
        ]
        unverified_keywords = [
            k for k, v in verification_results.items() 
            if not v.get("is_verified", False)
        ]
        logger.info(f"Verified {len(verified_single_words)} single-word disciplines")
        logger.info(f"Rejected {len(unverified_keywords)} single-word disciplines")
        
        # Combine verified single words with multi-word keywords
        all_verified_keywords = verified_single_words + multi_word_keywords
        logger.info(f"Total verified disciplines before normalization: {len(all_verified_keywords)}")
        
        # Normalize all verified keywords
        normalized_keywords = [normalize_text(kw) for kw in all_verified_keywords]
        # Remove duplicates that normalize to the same text while maintaining order
        seen = set()
        final_keywords = []
        for kw, norm_kw in zip(all_verified_keywords, normalized_keywords):
            if norm_kw not in seen:
                seen.add(norm_kw)
                final_keywords.append(kw)  # Keep original form
        
        logger.info(f"Final unique disciplines after normalization: {len(final_keywords)}")
        
        # Create output directory if needed
        for path in [level_config.get_step_output_file(3), level_config.get_validation_metadata_file(3)]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save verified keywords to text file
        logger.info(f"Saving verified disciplines to {level_config.get_step_output_file(3)}")
        with open(level_config.get_step_output_file(3), "w", encoding='utf-8') as f:
            for kw in sorted(final_keywords):
                f.write(f"{kw}\n")
        
        # Save detailed metadata to JSON file
        logger.info(f"Saving metadata to {level_config.get_validation_metadata_file(3)}")
        metadata = {
            "metadata": {
                "total_input_count": len(all_keywords),
                "single_word_count": len(single_word_keywords),
                "multi_word_count": len(multi_word_keywords),
                "verified_single_word_count": len(verified_single_words),
                "unverified_single_word_count": len(unverified_keywords),
                "total_verified_count": len(all_verified_keywords),
                "normalized_output_count": len(final_keywords),
                "provider": provider or Provider.OPENAI,
                "batch_size": processing_config.batch_size,
                "cooldown_period": processing_config.cooldown_period,
                "cooldown_frequency": processing_config.cooldown_frequency,
                "max_retries": processing_config.max_retries
            },
            "verification_results": verification_results,
            "verified_single_words": verified_single_words,
            "unverified_keywords": unverified_keywords,
            "multi_word_keywords": multi_word_keywords,
            "final_keywords": final_keywords
        }
        
        with open(level_config.get_validation_metadata_file(3), "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        logger.info("Academic discipline verification completed successfully")
        logger.info(f"Saved {len(final_keywords)} verified disciplines to {level_config.get_step_output_file(3)}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 