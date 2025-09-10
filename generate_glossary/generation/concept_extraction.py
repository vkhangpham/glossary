"""
Shared concept extraction functionality for all levels.

This module provides the generic s1 (LLM concept extraction) logic that can be
configured for different levels through the centralized config module.
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Optional
from collections import Counter
from pydantic import BaseModel, Field

from generate_glossary.utils.error_handler import (
    ValidationError, handle_error, processing_context
)
from generate_glossary.utils.logger import get_logger, log_processing_step


def chunk(items: list, size: int):
    """Simple chunking function to replace pydash.chunk"""
    for i in range(0, len(items), size):
        yield items[i : i + size]


from tqdm import tqdm
from generate_glossary.config import (
    ensure_directories,
    get_processing_config,
    get_step_config,
    get_llm_config
)
from generate_glossary.utils.llm import (
    completion, get_model_by_tier, smart_structured_completion_consensus, 
    run_async_safely
)
from generate_glossary.deduplication.utils import normalize_text

# Use centralized configuration
from generate_glossary.config import get_level_config


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
    # Use step config from centralized system
    config = get_step_config(level) or get_level_config(level)  # Fallback for compatibility

    base_prompt = """You are an expert in academic research classification with deep knowledge of research domains, scientific disciplines, and academic institutions."""

    if level == 0:
        return f"""{base_prompt}

Your task is to extract academic disciplines and fields of study from college and school names.

**CORE TASK:** Extract ONLY well-established, recognized academic disciplines or broad fields explicitly mentioned in or directly associated with the provided college/school names.

**CRITICAL GUIDELINES:**
1. Extract 1-3 core academic disciplines per college/school name
2. Focus on established academic fields and disciplines
3. Handle multi-disciplinary colleges by extracting individual fields
4. Normalize terminology to standard academic language
5. Exclude administrative terms, locations, or institutional identifiers

**EXAMPLES:**
- "College of Engineering" → ["engineering"]
- "School of Medicine" → ["medicine"]
- "College of Liberal Arts and Sciences" → ["liberal arts", "sciences"]
- "School of Business" → ["business"]

Return structured JSON with extracted academic disciplines for each input.
IMPORTANT: Your response must be in this exact format:
{
  "extractions": [
    {
      "source": "<exact input string>",
      "concepts": ["concept1", "concept2"]
    }
  ]
}"""

    elif level == 1:
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

Return structured JSON with extracted concepts for each input.
IMPORTANT: Your response must be in this exact format:
{
  "extractions": [
    {
      "source": "<exact input string>",
      "concepts": ["concept1", "concept2"]
    }
  ]
}"""

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

Return structured JSON with extracted research areas for each input.
IMPORTANT: Your response must be in this exact format:
{
  "extractions": [
    {
      "source": "<exact input string>",
      "concepts": ["concept1", "concept2"]
    }
  ]
}"""

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

Return structured JSON with extracted conference topics for each input.
IMPORTANT: Your response must be in this exact format:
{
  "extractions": [
    {
      "source": "<exact input string>",
      "concepts": ["concept1", "concept2"]
    }
  ]
}"""

    else:
        raise ValueError(f"Unknown level: {level}")


def load_input_data(input_file: str) -> List[str]:
    """Load input data from file."""
    with open(input_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def process_concept_batch(
    batch: List[str], level: int, system_prompt: str, provider: Optional[str] = None
) -> List[ConceptExtraction]:
    """Process a single batch of concepts with optimized smart consensus."""
    logger = get_logger(f"lv{level}.s1")
    # Get configuration from centralized system
    step_config = get_step_config(level) or get_level_config(level)
    processing_config = get_processing_config(level)
    llm_config = get_llm_config()
    
    # Semantic validation for concept extraction quality
    semantic_validation = f"""Verify that each extracted concept is:
1. An appropriate {step_config.context_description} term
2. Not a location, person name, or institution name
3. A meaningful academic or research term
4. Properly normalized (lowercase, no abbreviations)"""

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
        # Determine tier based on level
        tier = "budget" if level == 0 else "balanced"
        
        # Use smart consensus for improved accuracy and reduced API calls
        response = run_async_safely(
            smart_structured_completion_consensus,
            messages=messages,
            response_model=ConceptExtractionList,
            tier=tier,
            num_responses=step_config.consensus_attempts,  # Use consensus_attempts instead of agreement_threshold
            semantic_validation=semantic_validation,  # Add semantic validation
            use_case="concept_extraction",
            enable_enhanced_cache=True,
            enable_smart_consensus=True
        )

        if response and hasattr(response, "extractions"):
            return response.extractions
        else:
            # Treat empty LLM response as validation error
            raise ValidationError(
                "LLM returned empty extractions",
                invalid_data={
                    "batch_size": len(batch),
                    "batch_sample": str(batch[:2]) if batch else "empty",
                    "provider": provider,
                    "tier": tier
                }
            )

    except Exception as e:
        # Use handle_error with reraise=True for internal helpers
        handle_error(
            e,
            context={
                "batch_size": len(batch),
                "batch_sample": str(batch[:2]) if batch else "empty",
                "provider": provider
            },
            operation="concept_extraction_batch",
            reraise=True
        )


def process_concept_batches(
    input_data: List[str],
    level: int,
    system_prompt: str,
    provider: Optional[str] = None,
) -> List[ConceptExtraction]:
    """Process all batches with agreement-based filtering."""
    with processing_context(f"concept_batches_lv{level}") as correlation_id:
        logger = get_logger(f"lv{level}.s1")
        # Get configuration from centralized system
        config = get_step_config(level) or get_level_config(level)
        
        log_processing_step(
            logger,
            f"concept_extraction_lv{level}",
            "started",
            {"input_count": len(input_data), "batch_size": config.batch_size}
        )

        batches = list(chunk(input_data, config.batch_size))
        logger.info(f"Processing {len(batches)} batches of size {config.batch_size}")

        # TODO: Re-enable checkpoint system when resilient_processing is available
        all_extractions = []

        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            try:
                # Smart consensus handles agreement internally, so we only need one call per batch
                batch_extractions = process_concept_batch(
                    batch, level, system_prompt, provider
                )
                
                if batch_extractions:
                    all_extractions.extend(batch_extractions)
                    logger.debug(f"Batch {batch_idx}: extracted {len(batch_extractions)} concept sets")
                
                    # Save checkpoint
                    # Checkpoint saving disabled until resilient_processing is available
                    # processor.save_checkpoint(batch_idx, batch_extractions)

            except Exception as e:
                handle_error(
                    e,
                    context={
                        "batch_idx": batch_idx,
                        "batch_size": len(batch),
                        "level": level,
                        "correlation_id": correlation_id
                    },
                    operation=f"concept_batch_processing_lv{level}"
                )
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

        log_processing_step(
            logger,
            f"concept_extraction_lv{level}",
            "completed",
            {
                "total_inputs": len(input_data),
                "total_extractions": len(all_extractions),
                "batches_processed": len(batches)
            }
        )
        
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
    logger = get_logger(f"lv{level}.s1")

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

    # Save metadata - use centralized config source
    step_cfg = get_step_config(level) or get_level_config(level)  # Centralized config with fallback
    metadata = {
        "level": level,
        "step": "s1",
        "total_source_concept_pairs": total_pairs,
        "total_unique_concepts": len(unique_concepts),
        "total_sources_processed": len(extractions),
        "processing_timestamp": time.time(),
        "config_used": {
            "batch_size": step_cfg.batch_size,
            "agreement_threshold": step_cfg.agreement_threshold,
            "consensus_attempts": step_cfg.consensus_attempts,
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
        level: Generation level (0, 1, 2, or 3)
        output_file: Path to save extracted concepts
        metadata_file: Path to save processing metadata
        provider: Optional LLM provider override

    Returns:
        Dictionary containing processing results and metadata
    """
    logger = get_logger(f"lv{level}.s1")
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

    return {
        "level": level,
        "step": "s1",
        "success": True,
        **processing_stats,
        "processing_description": config.processing_description,
    }


# Alias for backward compatibility with runner imports
extract_concepts_llm_simple = extract_concepts_llm
