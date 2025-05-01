import os
import sys
import time
import json
import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
    CSV_FILE = "data/lv0/lv0_s1_college_concepts.csv"  # New CSV file path
    BATCH_SIZE = 20
    NUM_LLM_ATTEMPTS = 3  # Updated to 3 attempts
    CONCEPT_AGREEMENT_THRESH = 2  # Minimum number of models that must agree on a concept
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

def init_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Initialize LLM with specified provider and model"""
    if not provider:
        provider = Provider.OPENAI  # Default to OpenAI
        
    # Convert string provider to Provider constant
    provider_name = provider.lower()
    if provider_name not in [Provider.OPENAI, Provider.GEMINI]:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Choose model based on parameters
    if model == "default":
        selected_model = OPENAI_MODELS["default"] if provider_name == Provider.OPENAI else GEMINI_MODELS["pro"]
    else:  # mini
        selected_model = OPENAI_MODELS["mini"] if provider_name == Provider.OPENAI else GEMINI_MODELS["default"]
        
    return LLMFactory.create_llm(
        provider=provider_name,
        model=selected_model,
        temperature=0.3
    )

def get_random_llm_config() -> Tuple[str, str]:
    """Get a random LLM provider and model configuration"""
    provider = random.choice([Provider.OPENAI, Provider.GEMINI])
    model = random.choice(["default", "mini"])
    return provider, model

SYSTEM_PROMPT = """You are an expert in academic research classification with deep knowledge 
of research domains, scientific disciplines, and academic organizational structures.

Your task is to extract broad academic disciplines and research areas from college/school/division names.

CRITICAL GUIDELINE: ONLY extract concepts that are EXPLICITLY mentioned in the text. Do NOT add related or inferred concepts that don't appear directly.

Guidelines for extraction:
1. Extract ONLY academic disciplines that are DIRECTLY MENTIONED in the name
2. Remove administrative qualifiers (e.g., "department of", "school of", "college of") 
3. Extract the EXACT TERMS as they appear, do not decompose unless clearly separate entities
4. Do not infer or add specialized subfields that aren't directly mentioned

DO NOT include:
- Generic terms (e.g., studies, research)
- Administrative terms (e.g., department, school, college, division)
- Organizational descriptors (e.g., center, institute)
- Acronyms or abbreviations
- Proper nouns or names
- Location-specific terms
- ANY concepts that don't appear verbatim in the input text

Example decomposition:
"College of Arts and Sciences"
Valid concepts:
- arts
- sciences

"School of Engineering and Applied Sciences"
Valid concepts:
- engineering
- applied sciences

"Department of Electrical Engineering"
Valid concepts:
- electrical engineering
NOT valid:
- electronics
- power systems
- signal processing
(These are not valid because they don't appear directly in the text)

"College of Computer Science and Mathematics"
Valid concepts:
- computer science
- mathematics
NOT valid:
- algorithms
- data structures
- algebra
(These are not valid because they don't appear directly in the text)"""

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
    num_attempts: int = Config.NUM_LLM_ATTEMPTS
) -> List[List[ConceptExtraction]]:
    """Process a batch of sources using multiple LLM attempts asynchronously with different providers/models"""
    llm_responses = []

    try:
        # Get multiple responses for each batch
        tasks = []
        llms = []
        
        # Create a task for each attempt with a random provider/model combination
        for _ in range(num_attempts):
            provider, model = get_random_llm_config()
            llm = init_llm(provider, model)
            llms.append(llm)
            
            logger.info(f"Running attempt with provider: {provider}, model: {model}")
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
        if not all(hasattr(llm, 'infer_async') for llm in llms):
            # At least one llm uses synchronous calls
            responses = []
            for i, task in enumerate(tasks):
                if hasattr(llms[i], 'infer_async'):
                    # This was an async task
                    responses.append(await task)
                else:
                    # This was already a response
                    responses.append(task)
        else:
            # All llms use async calls
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

def combine_extractions(extractions_list: List[List[ConceptExtraction]]) -> Dict[str, Dict[str, int]]:
    """
    Combine extractions from multiple LLM runs and count concept frequencies
    
    Args:
        extractions_list: List of lists of extractions from multiple LLM runs
        
    Returns:
        Dictionary mapping sources to concepts and their occurrence counts
    """
    combined_results = {}
    
    # Process each list of extractions
    for extractions in extractions_list:
        # Process each extraction in the list
        for extraction in extractions:
            source = extraction.source
            
            # Initialize source in combined results if not present
            if source not in combined_results:
                combined_results[source] = {}
                
            # Update concept counts for this source
            for concept in extraction.concepts:
                concept_lower = concept.lower()
                if concept_lower not in combined_results[source]:
                    combined_results[source][concept_lower] = 0
                combined_results[source][concept_lower] += 1
    
    return combined_results

def filter_concepts_by_agreement(
    combined_results: Dict[str, Dict[str, int]], 
    agreement_threshold: int
) -> Dict[str, set]:
    """
    Filter concepts based on agreement threshold across multiple LLM runs
    
    Args:
        combined_results: Dictionary mapping sources to concepts and their occurrence counts
        agreement_threshold: Minimum number of times a concept must appear
        
    Returns:
        Dictionary mapping sources to filtered concepts
    """
    filtered_results = {}
    
    for source, concepts in combined_results.items():
        # Keep only concepts that meet the agreement threshold
        filtered_concepts = {
            concept for concept, count in concepts.items() 
            if count >= agreement_threshold
        }
        
        # Add filtered concepts to results
        if filtered_concepts:
            filtered_results[source] = filtered_concepts
    
    return filtered_results

def process_chunk(chunk_data: tuple[List[str], int]) -> Dict[str, set]:
    """
    Process a chunk of sources in parallel
    
    Args:
        chunk_data: Tuple of (sources, num_attempts)
        
    Returns:
        Dictionary mapping sources to their concepts
    """
    sources, num_attempts = chunk_data
    source_concepts = {}
    
    # Process batches within the chunk
    batches = chunk(sources, Config.BATCH_SIZE)
    for batch in batches:
        # Run async batch processing in this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Get responses from multiple LLM runs
            multiple_extractions = loop.run_until_complete(
                process_batch_async(batch, num_attempts)
            )
            
            if multiple_extractions:
                # Combine and filter concepts based on agreement threshold
                combined_results = combine_extractions(multiple_extractions)
                filtered_results = filter_concepts_by_agreement(
                    combined_results, Config.CONCEPT_AGREEMENT_THRESH
                )
                
                # Update source-concept mapping
                source_concepts.update(filtered_results)
                    
        finally:
            loop.close()
            
    return source_concepts

async def main_async():
    """Async main execution function"""
    try:
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
        chunk_data = [(chunk, Config.NUM_LLM_ATTEMPTS) for chunk in source_chunks]
        
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
        for path in [Config.OUTPUT_FILE, Config.META_FILE, Config.CSV_FILE]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for concept in verified_concepts:
                f.write(f"{concept}\n")

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
                        "concept_agreement_threshold": Config.CONCEPT_AGREEMENT_THRESH,
                        "concept_frequency_threshold": Config.KW_APPEARANCE_THRESH,
                        "temperature": 0.3,
                        "providers_and_models": "random selection of OpenAI default/mini and Gemini default/mini"
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

        # Create a CSV file with college-concept relationships
        try:
            with open(Config.CSV_FILE, "w", encoding="utf-8") as f:
                # Write header
                f.write("college,concept\n")
                
                # Write data rows
                for college, concepts in source_concept_mapping.items():
                    # Only include concepts that meet the threshold
                    filtered_concepts = [
                        concept for concept in concepts 
                        if concept_counts[concept.lower()] >= Config.KW_APPEARANCE_THRESH
                    ]
                    
                    for concept in sorted(filtered_concepts):
                        # Escape commas and quotes in fields
                        college_str = college.replace('"', '""')
                        concept_str = concept.replace('"', '""')
                        
                        # Add quotes if the field contains commas
                        college_csv = f'"{college_str}"' if ',' in college else college_str
                        concept_csv = f'"{concept_str}"' if ',' in concept else concept_str
                        
                        f.write(f"{college_csv},{concept_csv}\n")
            
            logger.info(f"College-concept relationships saved to {Config.CSV_FILE}")
        except Exception as e:
            logger.error(f"Failed to write CSV file: {str(e)}")

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