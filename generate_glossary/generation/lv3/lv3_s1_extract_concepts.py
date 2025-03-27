import os
import sys
import time
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger
from utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv3.s1")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    """Configuration for concept extraction"""
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv3/lv3_s0_venue_names.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv3/lv3_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv3/lv3_s1_metadata.json")
    BATCH_SIZE = 30        # Increased batch size for better efficiency
    NUM_WORKERS = 4        # Number of parallel workers
    KW_APPEARANCE_THRESH = 3
    COOLDOWN_PERIOD = 1    # Seconds between batches
    COOLDOWN_FREQUENCY = 5 # Number of batches before cooldown

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

# List of unwanted keywords specific to venues
UNWANTED_KEYWORDS = [
    "journal", "journals", "conference", "conferences", "symposium", "symposia",
    "proceedings", "transactions", "review", "workshop", "association", "society",
    "magazine", "quarterly", "annual", "biannual", "institute", "bulletin"
]

class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""
    source: str = Field(description="Source being processed (venue name)")
    concepts: List[str] = Field(description="List of extracted concepts")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")

SYSTEM_PROMPT = """You are an expert in academic research classification with deep knowledge 
of research domains, scientific disciplines, and academic fields.

Your task is to extract academic research topics and concepts from academic venue names.

For each venue name:
- Extract specific research domains that represent the academic focus areas
- Avoid generic terms, administrative terms, or organizational descriptors
- Avoid acronyms, proper nouns, and names of people
- Only include keywords that are directly mentioned in the name
- Decompose compound keywords into individual keywords
For example: "communication sciences and disorders" should be decomposed into "communication sciences" and "communication disorders"

Example:
- "IEEE Transactions on Neural Networks and Learning Systems" -> ["neural networks", "learning systems"]
- "PLoS Biology" -> ["biology"]
- "ACM Computing Surveys" -> ["computing"]
- "NeurIPS Conference" -> []
"""

def preprocess_venue_name(venue: str) -> str:
    """Preprocess venue name to clean and normalize it"""
    # Remove unwanted suffixes
    for suffix in ["journal", "conference", "symposium", "proceedings", "transactions"]:
        if venue.lower().endswith(f" {suffix}"):
            venue = venue[:-len(suffix)-1]
    
    # Replace special characters with spaces
    for char in ["&", "/", "-", ":", ";"]:
        venue = venue.replace(char, " ")
    
    # Remove extra whitespace
    venue = " ".join(venue.split())
    
    return venue.strip()

def build_prompt(venues: List[str]) -> str:
    """Build prompt for concept extraction"""
    venues_str = "\n".join([f"- {preprocess_venue_name(venue)}" for venue in venues])
    
    return f"""Extract research topics and concepts from these academic venue names.
Return the concepts in this exact JSON format:
{{
    "extractions": [
        {{
            "source": "venue name",
            "concepts": ["concept1", "concept2"]
        }}
    ]
}}

Venue names to process:
{venues_str}
"""

def process_batch_worker(args: tuple) -> List[ConceptExtraction]:
    """Worker function for parallel processing"""
    batch, provider = args
    try:
        prompt = build_prompt(batch)
        llm = init_llm(provider)
        response = llm.infer(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=ConceptExtractionList,
        )
        return response.text.extractions
    except Exception as e:
        if "insufficient_quota" in str(e):
            raise QuotaExceededError(f"API quota exceeded for provider {provider}")
        logger.error(f"Failed to process batch: {str(e)}")
        return []

def process_batches_parallel(
    batches: List[List[str]],
    provider: Optional[str] = None
) -> List[List[ConceptExtraction]]:
    """Process batches in parallel using ProcessPoolExecutor"""
    all_results = []
    
    with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_batch_worker, (batch, provider))
            for batch in batches
        ]
        
        for i, future in enumerate(tqdm(futures, desc="Processing batches")):
            try:
                result = future.result()
                all_results.extend([result])
                
                # Apply cooldown periodically
                if i > 0 and i % Config.COOLDOWN_FREQUENCY == 0:
                    time.sleep(Config.COOLDOWN_PERIOD)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
                
    return all_results

@lru_cache(maxsize=1000)
def filter_concept(concept: str) -> bool:
    """Filter out unwanted concepts"""
    concept = concept.lower()
    
    # Filter out unwanted keywords
    if concept in UNWANTED_KEYWORDS:
        return False
    
    # Filter out short concepts (likely not meaningful)
    if len(concept) <= 3:
        return False
    
    # Filter out concepts that are just numbers
    if concept.isdigit():
        return False
    
    return True

def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")
            
        logger.info("Starting concept extraction")

        # Read input venues
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            venues = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(venues)} venues from input file")

        # Process in batches
        all_extracted_concepts = []
        source_concept_mapping = {}  # Track concepts per source
        
        # Split into batches
        batches = chunk(venues, Config.BATCH_SIZE)
        logger.info(f"Split into {len(batches)} batches of size {Config.BATCH_SIZE}")
        
        # Process batches in parallel
        batch_results = process_batches_parallel(batches, provider)
        
        # Collect results
        for responses in batch_results:
            if responses:
                all_extracted_concepts.extend(responses)
                
                # Update mappings
                for extraction in responses:
                    # Update source mapping
                    if extraction.source not in source_concept_mapping:
                        source_concept_mapping[extraction.source] = set()
                    
                    # Filter concepts
                    filtered_concepts = [
                        concept for concept in extraction.concepts
                        if filter_concept(concept)
                    ]
                    
                    source_concept_mapping[extraction.source].update(filtered_concepts)

        # Apply frequency threshold to all concepts
        all_concepts = [
            concept.lower()
            for response in all_extracted_concepts
            for concept in response.concepts
            if filter_concept(concept)
        ]
        
        concept_counts = Counter(all_concepts)
        verified_concepts = sorted(
            [
                concept
                for concept, count in concept_counts.items()
                if count >= Config.KW_APPEARANCE_THRESH
            ]
        )

        logger.info(f"Extracted {len(verified_concepts)} verified concepts from {len(venues)} venues")
        
        # Analyze concept frequency distribution
        frequency_dist = Counter(concept_counts.values())
        logger.info("Frequency distribution:")
        for freq, count in sorted(frequency_dist.items()):
            logger.info(f"  {count} concepts appear {freq} times")

        # Create output directories if needed
        for path in [Config.OUTPUT_FILE, Config.META_FILE]:
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
                        "input_count": len(venues),
                        "output_count": len(verified_concepts),
                        "batch_size": Config.BATCH_SIZE,
                        "num_workers": Config.NUM_WORKERS,
                        "concept_threshold": Config.KW_APPEARANCE_THRESH,
                        "model": init_llm(provider).model,
                        "temperature": init_llm(provider).temperature,
                        "processing_time_seconds": round(time.time() - start_time, 2),
                        "frequency_distribution": {str(k): v for k, v in frequency_dist.items()},
                    },
                    "source_concept_mapping": {
                        source: sorted(list(concepts))
                        for source, concepts in source_concept_mapping.items()
                    }
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Concept extraction completed successfully in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
