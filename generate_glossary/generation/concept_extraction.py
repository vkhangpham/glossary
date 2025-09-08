"""
Shared concept extraction functionality for levels 1-3.

This module provides the generic s1 (LLM concept extraction) logic that can be
configured for different levels through the level_config module.
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Optional
from collections import Counter
from pydantic import BaseModel, Field


def chunk(items: list, size: int):
    """Simple chunking function to replace pydash.chunk"""
    for i in range(0, len(items), size):
        yield items[i : i + size]


from tqdm import tqdm

from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import ensure_directories
from generate_glossary.utils.llm import completion, get_model_by_tier
from generate_glossary.deduplication.utils import normalize_text

# Resilient processing not yet implemented
# from generate_glossary.utils.resilient_processing import (
#     ConceptExtractionProcessor, create_processing_config, get_checkpoint_dir
# )
from generate_glossary.generation.level_config import get_level_config


class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""

    source: str = Field(description="Source text being processed")
    concepts: List[str] = Field(description="List of extracted concepts")


class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""

    extractions: List[ConceptExtraction] = Field(
        description="List of concept extractions"
    )


def create_system_prompt(level: int) -> str:
    """Create level-specific system prompt."""
    config = get_level_config(level)

    base_prompt = """You are an expert in academic research classification with deep knowledge of research domains, scientific disciplines, and academic institutions."""

    if level == 1:
        return f"""{base_prompt}

Your task is to extract academic concepts and fields of study from university department names.

**CORE TASK:** Extract ONLY well-established, recognized academic disciplines, sub-disciplines, or broad research fields explicitly mentioned in the provided department names.

**CRITICAL GUIDELINES:**
1. Extract 1-3 core academic concepts per department name
2. Focus on established academic fields and disciplines  
3. Handle compound departments by extracting individual fields
4. Normalize terminology to standard academic language
5. Exclude administrative terms, locations, or generic words

**EXAMPLES:**
- "Department of Electrical and Computer Engineering" → ["electrical engineering", "computer engineering"]  
- "School of Business Administration" → ["business administration"]
- "College of Arts and Sciences" → ["arts", "sciences"]

Return structured JSON with extracted concepts for each input."""

    elif level == 2:
        return f"""{base_prompt}

Your task is to extract research areas and specializations from academic department or research descriptions.

**CORE TASK:** Extract ONLY established research areas, specializations, or research fields that represent genuine academic research domains.

**CRITICAL GUIDELINES:**  
1. Extract 1-3 core research areas per input
2. Focus on specific research specializations within departments
3. Prioritize established research fields over generic terms
4. Normalize to standard research area terminology
5. Exclude administrative content or overly broad terms

**EXAMPLES:**
- "Machine Learning Research Group" → ["machine learning"]
- "Biomedical Engineering Lab" → ["biomedical engineering"]
- "Computational Biology Center" → ["computational biology"]

Return structured JSON with extracted research areas for each input."""

    elif level == 3:
        return f"""{base_prompt}

Your task is to extract conference topics and themes from research area or conference descriptions.

**CORE TASK:** Extract ONLY specific topics, themes, or research areas that would be appropriate for academic conferences.

**CRITICAL GUIDELINES:**
1. Extract 1-3 specific conference topics per input  
2. Focus on research themes suitable for conference presentations
3. Prioritize specific topics over broad research areas
4. Normalize to standard academic terminology
5. Exclude general conference information or administrative content

**EXAMPLES:**
- "Deep Learning Workshop Call for Papers" → ["deep learning"]
- "Natural Language Processing Conference" → ["natural language processing"] 
- "Computer Vision and Pattern Recognition" → ["computer vision", "pattern recognition"]

Return structured JSON with extracted conference topics for each input."""

    else:
        raise ValueError(f"Unknown level: {level}")


def load_input_data(input_file: str) -> List[str]:
    """Load input data from file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def process_concept_batch(
    batch: List[str], level: int, system_prompt: str, provider: Optional[str] = None
) -> List[ConceptExtraction]:
    """Process a single batch of concepts with LLM."""
    logger = setup_logger(f"lv{level}.s1")
    config = get_level_config(level)

    # Build prompt for batch
    prompt_parts = ["Extract academic concepts from the following:"]
    for i, item in enumerate(batch, 1):
        prompt_parts.append(f"{i}. {item}")

    prompt = "\n".join(prompt_parts)

    # Build messages for structured completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        # Determine model based on provider or tier
        if provider:
            # Map provider to a specific model
            if provider == "openai":
                model_str = "openai/gpt-4o-mini"
            elif provider == "anthropic":
                model_str = "anthropic/claude-3-haiku-20240307"
            elif provider == "gemini":
                model_str = "gemini/gemini-1.5-flash"
            else:
                # Fallback to tier-based selection
                tier = "budget" if level == 0 else "balanced"
                model_str = get_model_by_tier(tier)
        else:
            # Use tier-based selection
            tier = "budget" if level == 0 else "balanced"
            model_str = get_model_by_tier(tier)

        # Use structured completion for reliable parsing
        response = completion(
            messages=messages, response_model=ConceptExtractionList, model=model_str
        )

        if response and hasattr(response, "extractions"):
            return response.extractions
        else:
            logger.warning("No extractions returned from LLM")
            return []

    except Exception as e:
        logger.error(f"Error in concept extraction: {str(e)}")
        return []


def process_concept_batches(
    input_data: List[str],
    level: int,
    system_prompt: str,
    provider: Optional[str] = None,
) -> List[ConceptExtraction]:
    """Process all batches with agreement-based filtering."""
    logger = setup_logger(f"lv{level}.s1")
    config = get_level_config(level)

    # Split into batches
    batches = list(chunk(input_data, config.batch_size))
    logger.info(f"Processing {len(batches)} batches of size {config.batch_size}")

    # Process without checkpoint system for now
    # TODO: Re-enable checkpoint system when resilient_processing is available

    # Process batches with checkpointing
    all_extractions = []

    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        try:
            # Process batch multiple times for agreement
            batch_results = []
            for attempt in range(config.agreement_threshold + 1):
                attempt_extractions = process_concept_batch(
                    batch, level, system_prompt, provider
                )
                if attempt_extractions:
                    batch_results.append(attempt_extractions)

            # Combine results from multiple attempts
            batch_extractions = []
            if batch_results:
                # Create mapping from source to all concepts
                concept_votes = {}
                for result in batch_results:
                    for extraction in result:
                        source = extraction.source
                        if source not in concept_votes:
                            concept_votes[source] = []
                        concept_votes[source].extend(extraction.concepts)

                # Apply agreement threshold
                for source, all_concepts in concept_votes.items():
                    concept_counts = Counter(
                        normalize_text(concept) for concept in all_concepts
                    )
                    agreed_concepts = [
                        concept
                        for concept, count in concept_counts.items()
                        if count >= config.agreement_threshold
                    ]

                    if agreed_concepts:
                        batch_extractions.append(
                            ConceptExtraction(source=source, concepts=agreed_concepts)
                        )

                # Save checkpoint
                # Checkpoint saving disabled until resilient_processing is available
                # processor.save_checkpoint(batch_idx, batch_extractions)

            all_extractions.extend(batch_extractions)

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    logger.info(f"Extracted concepts from {len(input_data)} inputs")
    return all_extractions


def save_extraction_results(
    extractions: List[ConceptExtraction],
    output_file: str,
    metadata_file: str,
    level: int,
    processing_stats: Dict[str, Any],
):
    """Save extraction results to files with source-concept mapping."""
    logger = setup_logger(f"lv{level}.s1")

    # Save source-concept pairs for s2 frequency filtering
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    total_pairs = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for extraction in extractions:
            for concept in extraction.concepts:
                f.write(f"{extraction.source} - {concept}\n")
                total_pairs += 1

    # Count unique concepts
    unique_concepts = set()
    for extraction in extractions:
        for concept in extraction.concepts:
            unique_concepts.add(normalize_text(concept))

    # Save metadata
    metadata = {
        "level": level,
        "step": "s1",
        "total_source_concept_pairs": total_pairs,
        "total_unique_concepts": len(unique_concepts),
        "total_sources_processed": len(extractions),
        "processing_timestamp": time.time(),
        "config_used": {
            "batch_size": get_level_config(level).batch_size,
            "agreement_threshold": get_level_config(level).agreement_threshold,
        },
        **processing_stats,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Saved {total_pairs} source-concept pairs ({len(unique_concepts)} unique concepts) to {output_file}"
    )


def extract_concepts_llm(
    input_file: str,
    level: int,
    output_file: str,
    metadata_file: str,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic LLM concept extraction for any level.

    Args:
        input_file: Path to file containing input data
        level: Generation level (1, 2, or 3)
        output_file: Path to save extracted concepts
        metadata_file: Path to save processing metadata
        provider: Optional LLM provider override

    Returns:
        Dictionary containing processing results and metadata
    """
    logger = setup_logger(f"lv{level}.s1")
    config = get_level_config(level)

    # Ensure directories exist
    ensure_directories(level)

    logger.info(
        f"Starting Level {level} concept extraction: {config.processing_description}"
    )

    # Load input data
    input_data = load_input_data(input_file)
    logger.info(f"Loaded {len(input_data)} input items")

    if not input_data:
        logger.warning("No input data found")
        return {"error": "No input data"}

    # Create level-specific system prompt
    system_prompt = create_system_prompt(level)

    # Process concepts with LLM
    start_time = time.time()
    extracted_concepts = process_concept_batches(
        input_data=input_data,
        level=level,
        system_prompt=system_prompt,
        provider=provider,
    )
    processing_time = time.time() - start_time

    # Calculate statistics from ConceptExtraction objects
    total_concepts = sum(len(extraction.concepts) for extraction in extracted_concepts)
    unique_concepts = set()
    for extraction in extracted_concepts:
        for concept in extraction.concepts:
            unique_concepts.add(normalize_text(concept))

    # Determine which provider was actually used
    if provider:
        actual_provider = provider
    else:
        # Tier-based selection was used
        tier = "budget" if level == 0 else "balanced"
        model_str = get_model_by_tier(tier)
        actual_provider = model_str.split("/")[0] if "/" in model_str else "tier_based"

    # Processing statistics
    processing_stats = {
        "input_items_count": len(input_data),
        "extracted_concepts_count": len(unique_concepts),
        "processing_time_seconds": processing_time,
        "concepts_per_input": total_concepts / len(input_data) if input_data else 0,
        "provider_used": actual_provider,
    }

    # Save results
    save_extraction_results(
        extracted_concepts, output_file, metadata_file, level, processing_stats
    )

    # Return processing metadata
    return {
        "level": level,
        "step": "s1",
        "success": True,
        **processing_stats,
        "processing_description": config.processing_description,
    }


# Alias for backward compatibility with runner imports
extract_concepts_llm_simple = extract_concepts_llm
