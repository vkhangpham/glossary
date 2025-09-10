"""
Shared token verification functionality for all levels.

This module provides the generic s3 (single token verification) logic that can be
configured for different levels through the level_config module.
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from generate_glossary.utils.error_handler import (
    ProcessingError, ValidationError, handle_error, processing_context
)
from generate_glossary.utils.logger import get_logger, log_processing_step
from generate_glossary.config import ensure_directories
from generate_glossary.utils.llm import completion, get_model_by_tier
from generate_glossary.deduplication.utils import normalize_text

# Resilient processing not yet implemented
# from generate_glossary.utils.resilient_processing import (
#     ConceptExtractionProcessor, create_processing_config
# )
from generate_glossary.config import get_level_config

# Provider to model mapping for token verification
MODEL_MAPPING = {
    "openai": "openai/gpt-4o-mini",
    "anthropic": "anthropic/claude-3-haiku-20240307", 
    "gemini": "gemini/gemini-1.5-flash"
}


class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""

    pass


def create_verification_system_prompt(level: int) -> str:
    """Create level-specific verification system prompt."""
    base_prompt = """You are an expert in academic research classification with a deep understanding of research domains, 
academic departments, scientific disciplines, and specialized fields of study."""

    if level == 0:
        return f"""{base_prompt}

Your task is to verify whether terms represent legitimate academic disciplines or broad fields of study that would be associated with colleges or schools by considering:
1. Academic relevance - Is it a recognized academic discipline at the college/school level?
2. Institutional context - Does it represent a field typically having its own college or school?
3. Educational scope - Is it broad enough to encompass an entire college or school?

Accept:
- Broad academic disciplines (e.g., engineering, medicine, business, law)
- Major academic fields (e.g., sciences, liberal arts, education)
- Professional schools (e.g., nursing, architecture, pharmacy)
- Established college-level disciplines (e.g., agriculture, communications)
- Core academic areas (e.g., humanities, social sciences)

DO NOT accept:
- Department-level specializations (too narrow for college level)
- Common English words without specific academic meaning
- Administrative terms (e.g., college, school, university, campus)
- Website navigation terms
- Geographic locations or institution names
- Terms too specific for college-level organization

Your decision must be especially strict for single-word terms, focusing on broad academic disciplines only."""

    elif level == 1:
        return f"""{base_prompt}

Your task is to verify whether terms represent legitimate academic departments or fields of study by considering:
1. Academic relevance - Is it a recognized field of study or academic discipline?
2. Departmental context - Does it appear in university departments and academic programs?
3. Educational validity - Is it a well-defined academic field taught in universities?

Accept:
- Academic disciplines (e.g., biology, physics, mathematics, psychology)
- Professional fields (e.g., engineering, medicine, law, business)
- Interdisciplinary fields (e.g., bioinformatics, environmental science)
- Established academic areas (e.g., literature, history, philosophy)
- Applied sciences (e.g., agriculture, nursing, education)

DO NOT accept:
- Common English words without specific academic meaning (e.g., "about", "main", "general")
- Administrative terms (e.g., office, department, program, student, staff)
- Website navigation terms (e.g., home, back, links, menu)
- Generic descriptors (e.g., advanced, basic, special, international)
- Proper nouns or institution names (e.g., Harvard, MIT)
- Informal or colloquial terms

Your decision must be especially strict for single-word terms, as these are more likely to be ambiguous."""

    elif level == 2:
        return f"""{base_prompt}

Your task is to verify whether terms represent legitimate research areas or specializations by considering:
1. Research relevance - Is it a recognized research field or specialization?
2. Academic context - Does it appear in research literature and academic settings?
3. Scientific validity - Is it a well-defined research area with established methodologies?

Accept:
- Research specializations (e.g., machine learning, genetics, oceanography)
- Scientific methodologies (e.g., spectroscopy, proteomics, bioinformatics)
- Technical fields (e.g., robotics, nanotechnology, cryptography)
- Research areas (e.g., sustainability, neuroscience, materials science)
- Applied research fields (e.g., biomedical engineering, data science)

DO NOT accept:
- Common English words without specific research meaning
- General academic terms (e.g., research, study, analysis)
- Administrative or organizational terms
- Website or navigation terms
- Generic descriptors or qualifiers
- Proper nouns or specific institution names

Your decision must be especially strict for single-word terms in research contexts."""

    elif level == 3:
        return f"""{base_prompt}

Your task is to verify whether terms represent legitimate conference topics or themes by considering:
1. Conference relevance - Is it appropriate for academic conference presentations?
2. Research specificity - Is it specific enough to be a conference track or topic?
3. Academic validity - Is it a recognized theme in scholarly discourse?

Accept:
- Specific research topics (e.g., deep learning, quantum computing, climate modeling)
- Conference themes (e.g., sustainability, security, optimization)
- Technical areas (e.g., algorithms, databases, networks)
- Methodological approaches (e.g., simulation, visualization, modeling)
- Applied research topics (e.g., healthcare informatics, smart cities)

DO NOT accept:
- Overly broad terms (e.g., science, technology, research)
- Common English words without conference context
- Administrative conference terms (e.g., registration, venue, schedule)
- Generic qualifiers (e.g., advanced, modern, future)
- Website or navigation terms
- Proper nouns or specific event names

Your decision must be especially strict for single-word terms in conference contexts."""

    else:
        raise ValueError(f"Unknown level: {level}")


def load_terms(input_file: str) -> List[str]:
    """Load terms from input file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def separate_terms_by_length(terms: List[str]) -> tuple[List[str], List[str]]:
    """Separate single-word from multi-word terms."""
    single_word_terms = []
    multi_word_terms = []

    for term in terms:
        # Clean and check word count
        cleaned_term = re.sub(r"[^\w\s-]", "", term).strip()
        word_count = len(cleaned_term.split())

        if word_count == 1:
            single_word_terms.append(term)
        else:
            multi_word_terms.append(term)

    return single_word_terms, multi_word_terms


def verify_single_term(
    term: str, level: int, system_prompt: str, provider: Optional[str] = None
) -> bool:
    """
    Verify a single term using LLM.

    Args:
        term: Term to verify
        level: Generation level
        system_prompt: System prompt for verification
        provider: Optional LLM provider

    Returns:
        Boolean indicating if term is valid
    """
    logger = get_logger(f"lv{level}.s3")
    config = get_level_config(level)

    # Create verification prompt
    prompt = f"""Is "{term}" a valid academic concept for {config.context_description}?

Consider the term in the context of {config.processing_description}.

Answer with only "YES" or "NO" followed by a brief justification."""

    # Create messages for completion API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Determine model/tier - align with concept_extraction mapping
    if provider in MODEL_MAPPING:
        model_str = MODEL_MAPPING[provider]
    else:
        # Use tier-based selection when provider is None
        tier = "budget" if level == 0 else "balanced"
        model_str = get_model_by_tier(tier)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def _verify_with_retry():
        try:
            response = completion(
                messages=messages, model=model_str, temperature=0.3, max_tokens=100
            )

            if not response:
                raise ProcessingError(f"Empty response for term '{term}'")

            # Parse response
            response_lower = response.lower().strip()

            # Look for clear YES/NO indicators
            if (
                response_lower.startswith("yes")
                or "yes," in response_lower
                or "yes." in response_lower
            ):
                return True
            elif (
                response_lower.startswith("no")
                or "no," in response_lower
                or "no." in response_lower
            ):
                return False
            else:
                # Try to extract decision from response
                if any(
                    positive in response_lower
                    for positive in ["valid", "legitimate", "appropriate", "accepted"]
                ):
                    return True
                elif any(
                    negative in response_lower
                    for negative in [
                        "invalid",
                        "not valid",
                        "inappropriate",
                        "rejected",
                    ]
                ):
                    return False
                else:
                    raise ValidationError(
                        f"Unclear response for term '{term}': {response[:100]}...",
                        invalid_data={"term": term, "response": response}
                    )
                    
        except QuotaExceededError:
            # Re-raise quota errors immediately
            raise
        except Exception as e:
            error_msg = str(e)
            
            # Check for quota exceeded errors
            if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                raise QuotaExceededError(f"API quota exceeded: {error_msg}")
            
            # Convert other errors to ProcessingError for proper handling
            if "rate" in error_msg.lower() and "limit" in error_msg.lower():
                raise ProcessingError(f"Rate limit exceeded for term '{term}': {error_msg}")
            else:
                raise ProcessingError(f"Error verifying term '{term}': {error_msg}")
    
    try:
        return _verify_with_retry()
    except Exception as e:
        handle_error(
            e,
            context={
                "term": term,
                "level": level,
                "provider": provider,
                "model": model_str
            },
            operation="term_verification"
        )
        
        # If quota exceeded, re-raise to stop processing
        if isinstance(e, QuotaExceededError):
            raise
            
        # For other errors, return False as fallback
        logger.error(f"Failed to verify term '{term}': {e}")
        return False


def verify_terms_batch(
    terms: List[str], level: int, system_prompt: str, provider: Optional[str] = None
) -> List[str]:
    """
    Verify a batch of terms using LLM with checkpointing.

    Args:
        terms: List of terms to verify
        level: Generation level
        system_prompt: System prompt for verification
        provider: Optional LLM provider

    Returns:
        List of verified (valid) terms
    """
    with processing_context(f"terms_verification_lv{level}") as correlation_id:
        logger = get_logger(f"lv{level}.s3")
        
        log_processing_step(
            logger,
            f"terms_verification_lv{level}",
            "started",
            {
                "terms_count": len(terms),
                "level": level,
                "provider": provider
            }
        )
        
        if not terms:
            return []

        try:
            # Process without checkpoint system for now
            # TODO: Re-enable checkpoint system when resilient_processing is available

            verified_terms = []

            for idx, term in enumerate(tqdm(terms, desc=f"Verifying terms")):
                try:
                    # Verify term without checkpoint system
                    is_valid = verify_single_term(term, level, system_prompt, provider)

                    if is_valid:
                        verified_terms.append(term)
                        
                except QuotaExceededError:
                    logger.error("API quota exceeded, stopping verification")
                    break
                except Exception as e:
                    # Use handle_error with reraise=False and skip the term
                    handle_error(
                        e,
                        context={
                            "term": term,
                            "term_index": idx,
                            "level": level,
                            "provider": provider,
                            "correlation_id": correlation_id
                        },
                        operation=f"term_verification_lv{level}",
                        reraise=False
                    )
                    logger.error(f"Error verifying term '{term}': {str(e)}")
                    # Skip this term and continue with next
                    continue

            log_processing_step(
                logger,
                f"terms_verification_lv{level}",
                "completed",
                {
                    "terms_verified": len(verified_terms),
                    "terms_total": len(terms),
                    "verification_rate": len(verified_terms) / len(terms) if terms else 0
                }
            )
            
            logger.info(f"Verified {len(verified_terms)} out of {len(terms)} terms")
            return verified_terms
            
        except Exception as e:
            handle_error(
                e,
                context={
                    "terms_count": len(terms),
                    "level": level,
                    "provider": provider,
                    "correlation_id": correlation_id
                },
                operation=f"terms_verification_batch_lv{level}",
                reraise=True
            )


def save_verification_results(
    verified_terms: List[str],
    multi_word_terms: List[str],
    output_file: str,
    metadata_file: str,
    level: int,
    verification_stats: Dict[str, Any],
):
    """Save verification results to files."""
    logger = get_logger(f"lv{level}.s3")

    # Combine verified single-word terms with multi-word terms (auto-pass)
    all_final_terms = verified_terms + multi_word_terms

    # Remove duplicates while preserving order
    seen = set()
    final_terms = []
    for term in all_final_terms:
        normalized = normalize_text(term)
        if normalized not in seen:
            seen.add(normalized)
            final_terms.append(term)

    # Save final terms
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for term in final_terms:
            f.write(term + "\n")

    # Save metadata
    metadata = {
        "level": level,
        "step": "s3",
        "total_verified_terms": len(final_terms),
        "single_word_verified": len(verified_terms),
        "multi_word_auto_passed": len(multi_word_terms),
        "processing_timestamp": time.time(),
        "verification_stats": verification_stats,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {len(final_terms)} verified terms to {output_file}")
    logger.info(f"  - Single-word verified: {len(verified_terms)}")
    logger.info(f"  - Multi-word auto-passed: {len(multi_word_terms)}")


def verify_single_tokens(
    input_file: str,
    level: int,
    output_file: str,
    metadata_file: str,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic single token verification for any level.

    Args:
        input_file: Path to file containing terms to verify
        level: Generation level (0, 1, 2, or 3)
        output_file: Path to save verified terms
        metadata_file: Path to save processing metadata
        provider: Optional LLM provider override

    Returns:
        Dictionary containing processing results and metadata
    """
    logger = get_logger(f"lv{level}.s3")
    config = get_level_config(level)

    # Ensure directories exist
    ensure_directories(level)

    logger.info(
        f"Starting Level {level} token verification: {config.processing_description}"
    )

    # Load terms
    all_terms = load_terms(input_file)
    logger.info(f"Loaded {len(all_terms)} terms")

    if not all_terms:
        logger.warning("No terms found to verify")
        return {"error": "No terms found"}

    # Separate single-word from multi-word terms
    single_word_terms, multi_word_terms = separate_terms_by_length(all_terms)
    logger.info(
        f"Single-word terms: {len(single_word_terms)}, Multi-word terms: {len(multi_word_terms)}"
    )

    # Create level-specific system prompt
    system_prompt = create_verification_system_prompt(level)
    
    # Determine actual provider to use
    if provider:
        actual_provider = provider
    else:
        # Use tier-based selection
        tier = "budget" if level == 0 else "balanced"
        model_str = get_model_by_tier(tier)
        actual_provider = model_str.split("/")[0] if "/" in model_str else "tier_based"

    # Verify single-word terms only (multi-word terms auto-pass)
    start_time = time.time()
    if single_word_terms:
        verified_single_word_terms = verify_terms_batch(
            single_word_terms, level, system_prompt, provider
        )
    else:
        verified_single_word_terms = []

    verification_time = time.time() - start_time

    # Verification statistics
    verification_stats = {
        "total_input_terms": len(all_terms),
        "single_word_terms_count": len(single_word_terms),
        "multi_word_terms_count": len(multi_word_terms),
        "single_word_verified_count": len(verified_single_word_terms),
        "single_word_rejected_count": len(single_word_terms)
        - len(verified_single_word_terms),
        "verification_time_seconds": verification_time,
        "provider_used": actual_provider,
        "verification_rate": (
            len(verified_single_word_terms) / len(single_word_terms)
            if single_word_terms
            else 0
        ),
    }

    # Save results
    save_verification_results(
        verified_single_word_terms,
        multi_word_terms,
        output_file,
        metadata_file,
        level,
        verification_stats,
    )

    # Return processing metadata
    return {
        "level": level,
        "step": "s3",
        "success": True,
        **verification_stats,
        "processing_description": config.processing_description,
    }
