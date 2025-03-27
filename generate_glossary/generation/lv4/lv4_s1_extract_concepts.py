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
logger = setup_logger("lv4.s1")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    """Configuration for concept extraction"""
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv4/lv4_s0_cfp_content.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv4/lv4_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv4/lv4_s1_metadata.json")
    BATCH_SIZE = 10  # Batch size for venue processing
    NUM_WORKERS = 4  # Number of parallel workers
    KW_APPEARANCE_THRESH = 3  # Minimum keyword appearance threshold
    MAX_CONTENT_WORDS = 10000  # Maximum words per CFP content
    COOLDOWN_PERIOD = 1  # Seconds between batches
    COOLDOWN_FREQUENCY = 5  # Number of batches before cooldown

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

class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""
    source: str = Field(description="Source being processed (venue name)")
    concepts: List[str] = Field(description="List of extracted concepts")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")

# List of unwanted keywords specific to venues and CFPs
UNWANTED_KEYWORDS = [
    "journal", "journals", "conference", "conferences", "symposium", "symposia",
    "proceedings", "transactions", "review", "workshop", "association", "society",
    "magazine", "quarterly", "annual", "biannual", "institute", "bulletin",
    "submission", "paper", "author", "deadline", "committee", "date", "program",
    "registration", "schedule", "venue", "publication", "presenter", "abstract"
]

SYSTEM_PROMPT = """You are an expert in academic research classification with deep knowledge 
of research domains, scientific disciplines, and academic fields.

Your task is to extract academic research topics and concepts from Call for Papers (CFP) and 
Aims & Scope documents for academic venues.

For each document:
- Extract specific research domains that represent the academic focus areas
- Avoid generic terms, administrative terms, or organizational descriptors
- Avoid acronyms, proper nouns, and names of people
- Try to identify the specific academic fields and subfields
- Decompose compound keywords into individual keywords
For example: "communication sciences and disorders" should be decomposed into "communication sciences" and "communication disorders"
- Focus on extracting broad academic disciplines and specific research areas
- Ignore terms that describe the event or publication process

Extract ONLY technical subject-matter concepts from the documents.
"""

@lru_cache(maxsize=1000)
def preprocess_content(content: str) -> str:
    """Preprocess content to clean and normalize it"""
    if not content:
        return ""
        
    # Remove extra whitespace
    content = " ".join(content.split())
    
    # Truncate if too long
    words = content.split()
    if len(words) > Config.MAX_CONTENT_WORDS:
        content = " ".join(words[:Config.MAX_CONTENT_WORDS])
    
    return content

def build_prompt(venue: str, content_type: str, content: str) -> str:
    """Build prompt for concept extraction from venue content"""
    # Preprocess content
    processed_content = preprocess_content(content)
    
    return f"""Extract academic research concepts from this {content_type} document.
Return the concepts in this exact JSON format:
{{
    "extractions": [
        {{
            "source": "{venue}",
            "concepts": ["concept1", "concept2", "concept3"]
        }}
    ]
}}

Document content:
{processed_content}
"""

def process_venue_worker(args: tuple) -> List[ConceptExtraction]:
    """Process a venue's content and extract concepts using LLM"""
    venue, content_type, content, provider = args
    try:
        # Build prompt
        prompt = build_prompt(venue, content_type, content)
        
        # Initialize LLM
        llm = init_llm(provider)
        
        # Get response
        response = llm.infer(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=ConceptExtractionList,
        )
        
        return response.text.extractions
    except Exception as e:
        if "insufficient_quota" in str(e):
            raise QuotaExceededError(f"API quota exceeded for provider {provider}")
        logger.error(f"Failed to process venue {venue}: {str(e)}")
        return []

def process_venues_parallel(
    venue_data: Dict[str, List[Dict[str, Any]]],
    provider: Optional[str] = None
) -> List[ConceptExtraction]:
    """Process venues in parallel using ProcessPoolExecutor"""
    all_results = []
    all_tasks = []
    
    # Create tasks for all venues
    for venue, documents in venue_data.items():
        for doc in documents:
            content = doc.get("content", "")
            content_type = doc.get("search_type", "document")
            
            if not content:
                continue
                
            all_tasks.append((venue, content_type, content, provider))
    
    logger.info(f"Created {len(all_tasks)} tasks for processing")
    
    # Process in batches
    batch_size = Config.BATCH_SIZE
    task_batches = [all_tasks[i:i + batch_size] for i in range(0, len(all_tasks), batch_size)]
    
    with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        for batch_idx, batch in enumerate(tqdm(task_batches, desc="Processing venues")):
            futures = [
                executor.submit(process_venue_worker, task)
                for task in batch
            ]
            
            # Wait for results
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        all_results.extend(result)
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx+1}, task {i}: {str(e)}")
                    continue
            
            # Apply cooldown periodically
            if batch_idx > 0 and batch_idx % Config.COOLDOWN_FREQUENCY == 0:
                logger.info(f"Cooldown after batch {batch_idx+1}/{len(task_batches)}")
                time.sleep(Config.COOLDOWN_PERIOD)
                
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

        # Read input content
        venue_data = {}
        try:
            with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
                venue_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read input file: {str(e)}")
            sys.exit(1)
            
        logger.info(f"Read content for {len(venue_data)} venues")
        
        # Process venues in parallel
        all_extracted_concepts = process_venues_parallel(venue_data, provider)
        
        # Track mapping and count occurrences
        source_concept_mapping = {}  # Track concepts per source
        all_concepts = []
        
        for extraction in all_extracted_concepts:
            venue = extraction.source
            
            # Filter concepts
            filtered_concepts = [
                concept.lower() for concept in extraction.concepts
                if filter_concept(concept)
            ]
            
            # Add to mappings
            if venue not in source_concept_mapping:
                source_concept_mapping[venue] = set()
            source_concept_mapping[venue].update(filtered_concepts)
            
            # Add to all concepts
            all_concepts.extend(filtered_concepts)
        
        # Apply frequency threshold
        concept_counts = Counter(all_concepts)
        verified_concepts = sorted(
            [
                concept
                for concept, count in concept_counts.items()
                if count >= Config.KW_APPEARANCE_THRESH
            ]
        )
        
        logger.info(f"Extracted {len(verified_concepts)} verified concepts")
        
        # Analyze frequency distribution
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
                        "input_count": len(venue_data),
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
        elapsed_time = time.time() - start_time
        logger.error(f"An error occurred after {elapsed_time:.2f} seconds: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 