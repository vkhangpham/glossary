import os
import sys
import time
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
from generate_glossary.utils.llm_simple import infer_structured, get_random_llm_config
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv1.s4")

# Get the base directory
BASE_DIR = os.getcwd()

# Use centralized configuration
LEVEL = 1
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

class SplitResult(BaseModel):
    """Model for split term results"""
    original_term: str = Field(description="Original compound term")
    should_split: bool = Field(description="Whether the term should be split")
    split_terms: List[str] = Field(description="List of split terms if should_split is true")

class SplitResultList(BaseModel):
    """Model for batch processing results"""
    results: List[SplitResult] = Field(description="List of split results")

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

# Now using centralized get_random_llm_config from llm_simple

# init_llm function removed - using direct LLM calls
        provider=provider,
        model=selected_model,
        temperature=0.2  # Low temperature for consistency
    )

SYSTEM_PROMPT = """You are an expert academic terminology analyst specializing in research fields and academic concepts.

Your task is to analyze compound or complex academic terms and determine if they should be split into separate terms that represent distinct academic fields or concepts.

**CRITICAL GUIDELINES FOR SPLITTING:**

1. ONLY split terms when they clearly represent distinct academic fields or research areas that stand alone.
2. Split terms that contain "and" that connects separate research areas (e.g., "biology and chemistry" → ["biology", "chemistry"]).
3. Split terms with commas separating distinct fields (e.g., "physics, chemistry, biology" → ["physics", "chemistry", "biology"]).
4. Split compound terms with apostrophes indicating conjunction (e.g., "children's and women's health" → ["children's health", "women's health"]).
5. DO NOT split terms where:
   - The "and" is part of an established field name (e.g., "supply and demand" is a unified economics concept)
   - The compound term represents a specialized interdisciplinary field (e.g., "biomedical engineering" is a single field)
   - The parts before or after "and" cannot stand as independent research areas
6. Be careful with shared prefixes/suffixes:
   - "Cognitive and behavioral neuroscience" → ["cognitive neuroscience", "behavioral neuroscience"]
   - "Machine learning and artificial intelligence" → ["machine learning", "artificial intelligence"]

Return ONLY valid academic fields. Do not return parts that are not valid academic fields on their own.
"""

def is_compound_term(term: str) -> bool:
    """
    Check if a term is potentially a compound term that could be split
    
    Args:
        term: The term to check
        
    Returns:
        True if the term contains "and", commas, or apostrophes that suggest it might be compound
    """
    # Check for "and" that isn't part of common established terms
    common_established_terms = [
        "supply and demand", "research and development", "teaching and learning",
        "structures and functions", "peer review and feedback"
    ]
    
    if any(established in term.lower() for established in common_established_terms):
        return False
    
    if " and " in term or "," in term or ("'" in term and " and " in term):
        return True
    
    return False

def build_split_prompt(terms: List[str]) -> str:
    """
    Build prompt for compound term splitting
    
    Args:
        terms: List of terms to analyze for splitting
        
    Returns:
        Formatted prompt for LLM
    """
    terms_str = "\n".join([f"- {term}" for term in terms])
    
    return f"""Analyze the following academic terms and determine if they should be split into multiple distinct academic fields:

{terms_str}

For each term, determine:
1. Should it be split into multiple terms?
2. If yes, what are the individual academic fields it should be split into?

Return your analysis in this exact JSON format:
{{
    "results": [
        {{
            "original_term": "term1",
            "should_split": true/false,
            "split_terms": ["split1", "split2"] // Only include if should_split is true
        }},
        // Repeat for each term
    ]
}}

IMPORTANT: Ensure your response is valid JSON. All strings must be properly quoted.
"""

def split_term_batch(
    terms: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> List[SplitResult]:
    """
    Process a batch of terms to determine if they should be split
    
    Args:
        terms: List of terms to analyze
        provider: Optional LLM provider
        model: Optional LLM model
        
    Returns:
        List of SplitResult objects with splitting information
    """
    if not terms:
        return []
        
    prompt = build_split_prompt(terms)
    
    try:
        response = infer_structured(
            provider=provider or "openai",
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=SplitResultList,
            model=model
        )
        
        return response.text.results
        
    except Exception as e:
        if "insufficient_quota" in str(e):
            raise QuotaExceededError(f"API quota exceeded for provider {provider}")
        logger.error(f"Error processing batch: {str(e)}")
        return []

def get_consensus_split(results: List[List[SplitResult]], min_consensus: int = Config.MIN_CONSENSUS) -> Dict[str, SplitResult]:
    """
    Get consensus from multiple LLM runs using majority voting
    
    Args:
        results: List of lists of SplitResult from multiple LLM runs
        min_consensus: Minimum number of identical responses to accept
        
    Returns:
        Dictionary mapping original terms to their consensus split
    """
    consensus = {}
    
    # Group results by original term
    grouped_results = {}
    for result_list in results:
        for result in result_list:
            if result.original_term not in grouped_results:
                grouped_results[result.original_term] = []
            grouped_results[result.original_term].append(result)
    
    # For each original term, find consensus
    for original_term, term_results in grouped_results.items():
        # Count should_split values
        split_counts = Counter([r.should_split for r in term_results])
        
        # Determine consensus on whether to split
        if split_counts.most_common(1)[0][1] >= min_consensus:
            should_split = split_counts.most_common(1)[0][0]
            
            if should_split:
                # For terms that should be split, find consensus on split terms
                # Create a string representation of each list of split terms for counting
                split_terms_counts = Counter()
                for result in term_results:
                    if result.should_split:
                        split_terms_counts[tuple(sorted(result.split_terms))] += 1
                
                # If there's consensus on the split terms
                if split_terms_counts and split_terms_counts.most_common(1)[0][1] >= min_consensus:
                    consensus_split = list(split_terms_counts.most_common(1)[0][0])
                    consensus[original_term] = SplitResult(
                        original_term=original_term,
                        should_split=True,
                        split_terms=consensus_split
                    )
                else:
                    # No consensus on split terms, treat as not split
                    consensus[original_term] = SplitResult(
                        original_term=original_term,
                        should_split=False,
                        split_terms=[]
                    )
            else:
                # Consensus not to split
                consensus[original_term] = SplitResult(
                    original_term=original_term,
                    should_split=False,
                    split_terms=[]
                )
        else:
            # No consensus, default to not splitting
            consensus[original_term] = SplitResult(
                original_term=original_term,
                should_split=False,
                split_terms=[]
            )
    
    return consensus

def process_compound_terms(
    terms: List[str],
    batch_size: int = processing_config.batch_size,
    num_attempts: int = processing_config.llm_attempts,
    cooldown: int = processing_config.cooldown_period,
    cooldown_freq: int = processing_config.cooldown_frequency
) -> Dict[str, SplitResult]:
    """
    Process all compound terms using multiple LLM attempts and consensus
    
    Args:
        terms: List of terms to process
        batch_size: Number of terms to process in each batch
        num_attempts: Number of LLM attempts for consensus
        cooldown: Cooldown period in seconds
        cooldown_freq: Number of batches before cooldown
        
    Returns:
        Dictionary mapping original terms to their consensus split results
    """
    results = {}
    
    compound_terms = [term for term in terms if is_compound_term(term)]
    logger.info(f"Found {len(compound_terms)} potential compound terms to analyze")
    
    # Process in batches
    for i in tqdm(range(0, len(compound_terms), batch_size), desc="Processing compound terms"):
        batch = compound_terms[i:i + batch_size]
        
        # Run multiple LLM attempts for this batch
        batch_results = []
        for attempt in range(num_attempts):
            provider, model = get_random_llm_config()
            logger.debug(f"Attempt {attempt+1}/{num_attempts} with provider={provider}, model={model}")
            
            try:
                result = split_term_batch(batch, provider, model)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                # Add empty list for the failed attempt
                batch_results.append([])
        
        # Get consensus from multiple attempts
        batch_consensus = get_consensus_split(batch_results)
        results.update(batch_consensus)
        
        # Apply cooldown periodically
        batch_num = i // batch_size
        if batch_num > 0 and batch_num % cooldown_freq == 0:
            logger.debug(f"Processed {i + len(batch)}/{len(compound_terms)} terms. Cooling down...")
            time.sleep(cooldown)
    
    return results

def expand_concepts(concepts: List[str], split_results: Dict[str, SplitResult]) -> List[str]:
    """
    Expand the concept list by replacing compound terms with their split versions
    
    Args:
        concepts: Original list of concepts
        split_results: Dictionary mapping original terms to their split results
        
    Returns:
        Expanded list of concepts with compounds replaced by their splits
    """
    expanded = []
    
    for concept in concepts:
        if concept in split_results and split_results[concept].should_split and split_results[concept].split_terms:
            # Add the split terms
            expanded.extend(split_results[concept].split_terms)
        else:
            # Keep the original concept
            expanded.append(concept)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expanded = []
    for concept in expanded:
        normalized = normalize_text(concept)
        if normalized not in seen:
            seen.add(normalized)
            unique_expanded.append(concept)
    
    return sorted(unique_expanded)

def main():
    """Main execution function"""
    try:
        logger.info("Starting compound term splitting process for level 1 concepts")
        
        # Read input concepts
        with open(level_config.get_step_input_file(4), "r", encoding="utf-8") as f:
            concepts = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(concepts)} concepts from input")
        
        # Process compound terms
        split_results = process_compound_terms(concepts)
        
        # Count split terms
        split_count = sum(1 for result in split_results.values() if result.should_split)
        logger.info(f"Split {split_count} compound terms out of {len(split_results)} analyzed")
        
        # Expand concepts by replacing compounds with their splits
        expanded_concepts = expand_concepts(concepts, split_results)
        logger.info(f"Expanded to {len(expanded_concepts)} concepts after splitting")
        
        # Create output directories if needed
        for path in [level_config.get_step_output_file(4), Config.SPLIT_META_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save expanded concepts to output file
        with open(level_config.get_step_output_file(4), "w", encoding="utf-8") as f:
            for concept in expanded_concepts:
                f.write(f"{concept}\n")
        
        # Save metadata
        metadata = {
            "metadata": {
                "input_count": len(concepts),
                "compound_terms_count": len(split_results),
                "split_terms_count": split_count,
                "output_count": len(expanded_concepts),
                "batch_size": processing_config.batch_size,
                "cooldown_period": processing_config.cooldown_period,
                "cooldown_frequency": processing_config.cooldown_frequency,
                "llm_attempts": processing_config.llm_attempts,
                "min_consensus": Config.MIN_CONSENSUS
            },
            "split_results": {
                term: {
                    "should_split": result.should_split,
                    "split_terms": result.split_terms if result.should_split else []
                }
                for term, result in split_results.items()
            }
        }
        
        with open(Config.SPLIT_META_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved {len(expanded_concepts)} processed concepts to {level_config.get_step_output_file(4)}")
        logger.info("Compound term splitting for level 1 completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 