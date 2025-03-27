import os
import sys
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv1.s3")

class Config:
    """Configuration for concept filtering"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s2_filtered_concepts.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s3_verified_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s1_metadata.json")
    VALIDATION_META_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s3_metadata.json")
    BATCH_SIZE = 10
    COOLDOWN_PERIOD = 1
    COOLDOWN_FREQUENCY = 10
    MAX_RETRIES = 3
    MAX_EXAMPLES = 5  # Maximum number of example departments to show in prompt
    # List of common words that are not academic concepts, even if they appear in department names
    NON_ACADEMIC_TERMS = [
        "about", "all", "also", "and", "any", "are", "back", "can", "come", 
        "could", "day", "even", "first", "for", "from", "get", "give", "have", 
        "here", "home", "how", "info", "into", "just", "know", "like", "look", 
        "main", "make", "many", "more", "most", "new", "not", "now", "one", 
        "only", "our", "out", "over", "page", "part", "site", "some", "such", 
        "than", "that", "the", "their", "them", "then", "there", "these", 
        "they", "this", "time", "two", "use", "view", "was", "way", "web", 
        "well", "what", "when", "which", "will", "with", "would", "year", "you",
        "next", "click", "link", "links", "menu", "contact", "visit", "see",
        "faculty", "staff", "student", "students", "office", "list", "directory",
        "department", "university", "college", "school", "program", "programs"
    ]

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
academic departments, scientific disciplines, and specialized fields of study.

Your task is to verify whether terms represent legitimate research concepts by considering:
1. Academic relevance - Is it a recognized field of study, research methodology, or scientific concept?
2. Disciplinary context - Does it appear in academic departments and scholarly literature?
3. Technical validity - Is it a well-defined term used in academic or scientific contexts?

Accept:
- Academic disciplines (e.g., biology, physics, mathematics)
- Research methodologies (e.g., spectroscopy, chromatography)
- Scientific concepts (e.g., entropy, evolution)
- Technical terms (e.g., algorithm, protein)
- Specialized fields (e.g., bioinformatics, econometrics)

DO NOT accept:
- Common English words without specific academic meaning (e.g., "about", "main", "here")
- Acronyms (e.g., DNA, RNA) unless they are universally recognized as standalone concepts
- Proper nouns or names (e.g., Harvard, MIT) 
- General administrative terms (e.g., office, department, program, student)
- Informal or colloquial terms (e.g., stuff, thing)
- Website navigation terms (e.g., home, back, links)

Your decision must be especially strict for single-word terms, as these are more likely to be ambiguous.
"""

def is_obviously_invalid(keyword: str) -> bool:
    """
    Check if a keyword is obviously not a valid research concept
    
    Args:
        keyword: The keyword to check
        
    Returns:
        True if the keyword is obviously invalid, False otherwise
    """
    # Check if empty or too short
    if not keyword or len(keyword) < 3:
        return True
    
    # Check if it's just a common English word with no academic meaning
    if keyword.lower() in Config.NON_ACADEMIC_TERMS:
        return True
    
    # Check if it's just numbers or a single character
    if keyword.isdigit() or (len(keyword) == 1 and keyword.isalpha()):
        return True
    
    # Check if it contains HTML-like content or URLs
    if re.search(r'<[^>]+>|https?://|www\.', keyword):
        return True
    
    # Check if it's a generic/navigation term
    navigation_terms = ["back", "home", "next", "previous", "top", "menu", "site"]
    if keyword.lower() in navigation_terms:
        return True
    
    return False

def build_verification_prompt(
    keyword: str,
    departments: List[str]
) -> str:
    """
    Build prompt for keyword verification using metadata
    
    Args:
        keyword: Keyword to verify
        departments: List of departments where this concept appears
        
    Returns:
        Formatted prompt for LLM
    """
    # Take up to MAX_EXAMPLES example departments
    example_departments = departments[:Config.MAX_EXAMPLES]
    departments_str = "\n".join(f"- {dept}" for dept in example_departments)
    
    return f"""Analyze whether "{keyword}" is a valid research concept based on the following criteria:

1. Is it a recognized academic discipline, research methodology, or scientific concept?
2. Is it used in scholarly contexts and academic literature?
3. Does it have a clear technical or scientific meaning?

The term "{keyword}" appears in these academic departments:
{departments_str}

Consider:
- Whether this term has specific academic/scientific meaning, not just general English usage
- Whether it represents a field of study, methodology, or specific academic concept
- Whether it's found consistently in scholarly literature with a technical meaning
- Whether experts in a field would recognize it as a distinct concept

Note: Be especially critical of single-word terms, which require strong evidence of specific academic usage.

Return only a JSON with an is_valid boolean field:
{{
    "is_valid": true/false
}}"""

def clean_json_response(response: str) -> str:
    """Clean JSON response by removing code block markers"""
    # Remove code block markers if present
    response = response.replace('```json\n', '').replace('\n```', '')
    response = response.replace('```\n', '').replace('\n```', '')
    # Remove any leading/trailing whitespace
    response = response.strip()
    return response

def verify_keyword(
    keyword: str,
    departments: List[str],
    provider: Optional[str] = None
) -> bool:
    """Verify if a keyword is a valid academic/research concept."""
    # First, apply rule-based filtering
    if is_obviously_invalid(keyword):
        logger.debug(f"Skipping obviously invalid keyword: '{keyword}'")
        return False
    
    try:
        llm = init_llm(provider)
        prompt = build_verification_prompt(keyword, departments)
        logger.debug(f"Verification prompt for '{keyword}':\n{prompt}")
        
        try:
            response = llm.infer(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.3
            )
            logger.debug(f"Raw response for '{keyword}':\n{response.text}")
            response_json = clean_json_response(response.text)
            logger.debug(f"Cleaned response for '{keyword}':\n{response_json}")
            
            try:
                result = json.loads(response_json)
                return result.get("is_valid", False)
            except json.JSONDecodeError as e:
                # Try to extract boolean value from text if JSON parsing fails
                text = response_json.lower()
                if "true" in text and "false" not in text:
                    return True
                elif "false" in text and "true" not in text:
                    return False
                else:
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
    concept_departments: Dict[str, List[str]],
    provider: Optional[str] = None,
    batch_size: int = Config.BATCH_SIZE,
    cooldown: int = Config.COOLDOWN_PERIOD,
    cooldown_freq: int = Config.COOLDOWN_FREQUENCY
) -> Dict[str, Dict[str, Any]]:
    """
    Verify a batch of keywords with rate limiting
    
    Args:
        keywords: List of keywords to verify
        concept_departments: Dictionary mapping keywords to their departments
        provider: Optional LLM provider (openai or gemini)
        batch_size: Number of keywords to process before cooldown
        cooldown: Cooldown period in seconds
        cooldown_freq: Number of batches to process before cooling down
        
    Returns:
        Dictionary mapping keywords to their verification results
    """
    results = {}
    
    # First, filter out obviously invalid keywords
    filtered_keywords = [kw for kw in keywords if not is_obviously_invalid(kw)]
    logger.info(f"Pre-filtered from {len(keywords)} to {len(filtered_keywords)} keywords")
    
    for i in tqdm(range(0, len(filtered_keywords), batch_size), desc="Verifying keywords"):
        batch = filtered_keywords[i:i + batch_size]
        batch_results = {}
        
        for keyword in batch:
            try:
                departments = concept_departments.get(keyword, [])
                is_verified = verify_keyword(keyword, departments, provider)
                batch_results[keyword] = {
                    "is_verified": is_verified,
                    "departments": departments,
                    "provider": provider or Provider.OPENAI,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.error(f"Failed to verify '{keyword}': {str(e)}")
                batch_results[keyword] = {
                    "is_verified": False,
                    "departments": [],
                    "provider": provider or Provider.OPENAI,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
        
        results.update(batch_results)
        
        # Apply cooldown every cooldown_freq batches
        batch_num = i // batch_size
        if batch_num > 0 and batch_num % cooldown_freq == 0:
            logger.debug(f"Processed {i + len(batch)}/{len(filtered_keywords)} keywords. Cooling down...")
            time.sleep(cooldown)
    
    return results

def get_concept_departments(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract departments for each concept from metadata
    
    Args:
        metadata: Metadata from s1
        
    Returns:
        Dictionary mapping concepts to their departments
    """
    concept_departments = {}
    
    # Get source->concepts mapping from metadata
    source_mapping = metadata.get("source_concept_mapping", {})
    
    # Invert mapping from source->concepts to concept->sources
    for source, concepts in source_mapping.items():
        # Skip sources with empty concept lists
        if not concepts:
            continue
            
        for concept in concepts:
            if concept not in concept_departments:
                concept_departments[concept] = []
            if source not in concept_departments[concept]:
                concept_departments[concept].append(source)
    
    return concept_departments

def is_single_word(keyword: str) -> bool:
    """Check if keyword is a single word (no spaces)"""
    return len(keyword.split()) == 1

def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        os.path.dirname(Config.OUTPUT_FILE),
        os.path.dirname(Config.VALIDATION_META_FILE)
    ]
    
    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")
        
        logger.info("Starting single-word research concept verification by LLM")
        
        # Ensure directories exist
        ensure_dirs_exist()
        
        # Read input keywords
        with open(Config.INPUT_FILE, "r", encoding='utf-8') as f:
            all_keywords = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(all_keywords)} total research concepts")
        
        # Read metadata from s1
        with open(Config.META_FILE, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        concept_departments = get_concept_departments(metadata)
        logger.info(f"Loaded department data for {len(concept_departments)} concepts")
        
        # Pre-filter obviously invalid keywords
        pre_filtered = [kw for kw in all_keywords if not is_obviously_invalid(kw)]
        logger.info(f"Pre-filtered to {len(pre_filtered)} concepts (removed {len(all_keywords) - len(pre_filtered)} obviously invalid)")
        
        # Filter for single-word keywords
        single_word_keywords = [kw for kw in pre_filtered if is_single_word(kw)]
        multi_word_keywords = [kw for kw in pre_filtered if not is_single_word(kw)]
        logger.info(f"Found {len(single_word_keywords)} single-word concepts to verify")
        logger.info(f"Found {len(multi_word_keywords)} multi-word concepts to bypass")
        
        # Verify single-word keywords
        verification_results = verify_keywords_batch(
            single_word_keywords,
            concept_departments,
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
                "pre_filtered_count": len(pre_filtered),
                "single_word_count": len(single_word_keywords),
                "multi_word_count": len(multi_word_keywords),
                "verified_single_word_count": len(verified_single_words),
                "unverified_single_word_count": len(unverified_keywords),
                "total_verified_count": len(all_verified_keywords),
                "normalized_output_count": len(final_keywords),
                "provider": provider or Provider.OPENAI,
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
        
        logger.info("Research concept verification completed successfully")
        logger.info(f"Saved {len(final_keywords)} verified concepts to {Config.OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 