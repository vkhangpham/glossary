import os
import sys
import time
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk
from tqdm import tqdm

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)

from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv0.s1")

class Config:
    """Configuration for concept extraction"""
    INPUT_FILE = "data/lv0/lv0_s0_college_names.txt"
    OUTPUT_FILE = "data/lv0/lv0_s1_extracted_concepts.txt"
    META_FILE = "data/lv0/lv0_s1_metadata.json"
    BATCH_SIZE = 20
    NUM_LLM_ATTEMPTS = 2
    KW_APPEARANCE_THRESH = 2
    MAX_WORKERS = 4  # Number of parallel workers
    CHUNK_SIZE = 100  # Size of chunks for parallel processing

class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""
    source: str = Field(description="Source text being processed")
    concepts: List[str] = Field(description="List of extracted concepts")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")

def init_llm(provider: Optional[str] = None) -> BaseLLM:
    """Initialize LLM with specified provider"""
    if not provider:
        provider = Provider.OPENAI  # Default to OpenAI
        
    # Convert string provider to Provider constant
    provider_name = provider.lower()
    if provider_name not in [Provider.OPENAI, Provider.GEMINI]:
        raise ValueError(f"Unsupported provider: {provider}")
        
    return LLMFactory.create_llm(
        provider=provider_name,
        model=OPENAI_MODELS["mini"] if provider_name == Provider.OPENAI else GEMINI_MODELS["pro"],
        temperature=0.3
    )

SYSTEM_PROMPT = """You are an expert in academic research classification with deep knowledge 
of research domains, scientific disciplines, and academic organizational structures.

Your task is to extract broad academic disciplines and research areas from college/school/division names.

Guidelines for extraction:
1. Extract broad academic disciplines and research domains
2. Focus on the main fields of study represented by the college/school/division
3. Include both traditional disciplines and interdisciplinary areas
4. Ensure each concept is a well-defined academic or research field

DO NOT include:
- Generic terms (e.g., studies, research)
- Administrative terms (e.g., department, school, college, division)
- Organizational descriptors (e.g., center, institute)
- Acronyms or abbreviations
- Proper nouns or names
- Location-specific terms

Example decomposition:
"College of Arts and Sciences"
Valid concepts:
- arts
- sciences
- humanities
- natural sciences
- social sciences

"School of Engineering and Applied Sciences"
Valid concepts:
- engineering
- applied sciences
- technology"""

def build_prompt(sources: List[str]) -> str:
    """Build prompt for concept extraction"""
    sources_str = "\n".join(f"- {source}" for source in sources)

    return f"""Extract broad academic disciplines and research areas from these college/school/division names.
For each name, identify the core academic fields and disciplines represented.

Return the concepts in this exact JSON format:
{{
    "extractions": [
        {{"source": "source1", "concepts": ["concept1", "concept2"]}},
        {{"source": "source2", "concepts": ["concept3", "concept4"]}}
    ]
}}

College/school/division names to process:
{sources_str}"""

async def process_batch_async(
    batch: List[str],
    provider: Optional[str] = None,
    num_attempts: int = Config.NUM_LLM_ATTEMPTS
) -> List[List[ConceptExtraction]]:
    """Process a batch of sources using multiple LLM attempts asynchronously"""
    llm_responses = []
    llm = init_llm(provider)

    try:
        # Get multiple responses for each batch
        tasks = []
        for _ in range(num_attempts):
            prompt = build_prompt(batch)
            # Use synchronous call for providers without async support
            if not hasattr(llm, 'infer_async'):
                response = llm.infer(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    response_model=ConceptExtractionList,
                )
                tasks.append(response)
            else:
                tasks.append(
                    asyncio.create_task(
                        llm.infer_async(
                            prompt=prompt,
                            system_prompt=SYSTEM_PROMPT,
                            response_model=ConceptExtractionList,
                        )
                    )
                )

        # Wait for all attempts to complete
        if not hasattr(llm, 'infer_async'):
            responses = tasks  # Already have responses for synchronous calls
        else:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Attempt failed: {str(response)}")
                continue
                
            extractions = response.text.extractions
            llm_responses.append(extractions)
            logger.debug(f"Processed {len(extractions)} sources")

        if not llm_responses:
            logger.error("All attempts failed")
            return []

        return llm_responses

    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        return []

def process_chunk(chunk_data: tuple[List[str], str, int]) -> Dict[str, set]:
    """
    Process a chunk of sources in parallel
    
    Args:
        chunk_data: Tuple of (sources, provider, num_attempts)
        
    Returns:
        Dictionary mapping sources to their concepts
    """
    sources, provider, num_attempts = chunk_data
    source_concepts = {}
    
    # Process batches within the chunk
    batches = chunk(sources, Config.BATCH_SIZE)
    for batch in batches:
        # Run async batch processing in this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            responses = loop.run_until_complete(
                process_batch_async(batch, provider, num_attempts)
            )
            
            # Update source-concept mapping
            for response in responses:
                for extraction in response:
                    if extraction.source not in source_concepts:
                        source_concepts[extraction.source] = set()
                    source_concepts[extraction.source].update(extraction.concepts)
                    
        finally:
            loop.close()
            
    return source_concepts

async def main_async():
    """Async main execution function"""
    try:
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")

        logger.info("Starting concept extraction from college names")

        # Read input sources
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            sources = [line.strip() for line in f.readlines()]
        
        # Remove duplicates while preserving order
        seen = set()
        sources = [s for s in sources if not (s in seen or seen.add(s))]
        logger.info(f"Read {len(sources)} unique sources")

        # Split sources into chunks for parallel processing
        source_chunks = list(chunk(sources, Config.CHUNK_SIZE))
        chunk_data = [(chunk, provider, Config.NUM_LLM_ATTEMPTS) for chunk in source_chunks]
        
        # Process chunks in parallel
        source_concept_mapping = {}
        with ProcessPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_chunk, data)
                for data in chunk_data
            ]
            
            # Collect results
            for future in tqdm(futures, desc="Processing chunks"):
                chunk_results = future.result()
                source_concept_mapping.update(chunk_results)

        # Extract all concepts and apply frequency threshold
        all_concepts = [
            concept.lower()
            for concepts in source_concept_mapping.values()
            for concept in concepts
        ]
        concept_counts = Counter(all_concepts)
        verified_concepts = sorted([
            concept
            for concept, count in concept_counts.items()
            if count >= Config.KW_APPEARANCE_THRESH
        ])
        
        logger.info(f"Extracted {len(verified_concepts)} verified concepts")

        # Create output directories if needed
        for path in [Config.OUTPUT_FILE, Config.META_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for concept in verified_concepts:
                f.write(f"{concept}\n")

        # Get LLM info for metadata
        llm = init_llm(provider)
        
        # Get model name based on provider
        if provider == Provider.GEMINI:
            model_name = GEMINI_MODELS["pro"]  # Use the same model name from initialization
        else:
            model_name = OPENAI_MODELS["mini"]  # Use the same model name from initialization
        
        with open(Config.META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "input_count": len(sources),
                        "output_count": len(verified_concepts),
                        "batch_size": Config.BATCH_SIZE,
                        "chunk_size": Config.CHUNK_SIZE,
                        "num_workers": Config.MAX_WORKERS,
                        "llm_attempts": Config.NUM_LLM_ATTEMPTS,
                        "concept_threshold": Config.KW_APPEARANCE_THRESH,
                        "provider": provider or Provider.OPENAI,
                        "model": model_name,
                        "temperature": llm.temperature,
                    },
                    "source_concept_mapping": {
                        source: sorted(list(concepts))
                        for source, concepts in source_concept_mapping.items()
                    },
                    "concept_frequencies": {
                        concept: count 
                        for concept, count in concept_counts.items()
                        if count >= Config.KW_APPEARANCE_THRESH
                    }
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.info("Concept extraction from college names completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def main():
    """Main execution function"""
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_async())
    finally:
        loop.close()

if __name__ == "__main__":
    main() 