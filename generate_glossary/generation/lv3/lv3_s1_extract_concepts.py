import os
import sys
import time
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import threading

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv3.s1")

# Get the base directory
BASE_DIR = os.getcwd()  # Get the current working directory, which should be the project root

# Process-local storage for LLM instances
_process_local = threading.local()

def normalize_research_area_name(research_area: str) -> str:
    """
    Normalize research area name by stripping common prefixes if present
    
    Args:
        research_area: Research area name to normalize
        
    Returns:
        Normalized research area name
    """
    prefixes = [
        "tracks on ", "topics in ", "track on ", "topic in ", 
        "workshops on ", "workshop on ", "special issues on ", "special issue on "
    ]
    
    research_area_lower = research_area.lower()
    for prefix in prefixes:
        if research_area_lower.startswith(prefix):
            return research_area[len(prefix):]
    
    return research_area

class Config:
    """Configuration for concept extraction"""
    # Use the output from step 0
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s0_conference_topics.txt")
    META_FILE_STEP0 = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s0_metadata.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s1_metadata.json")
    BATCH_SIZE = 10  # Size of batches for processing
    NUM_WORKERS = 64  # Number of parallel workers
    KW_APPEARANCE_THRESH = 1  # Minimum frequency threshold for concepts
    MAX_CONTENT_WORDS = 10000  # Maximum words per content chunk
    COOLDOWN_PERIOD = 1  # Seconds between batches
    COOLDOWN_FREQUENCY = 10  # Number of batches before cooldown

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """Get or initialize LLM with specified provider (process-local)"""
    if not hasattr(_process_local, 'llm'):
        # logger.info(f"Initializing LLM with provider: {provider} for process {os.getpid()}")
        _process_local.llm = init_llm(provider)
    return _process_local.llm

def init_llm(provider: Optional[str] = None) -> BaseLLM:
    """Initialize LLM with specified provider"""
    if not provider:
        provider = Provider.GEMINI  # Default to Gemini
        
    return LLMFactory.create_llm(
        provider=provider,
        model=OPENAI_MODELS["mini"] if provider == Provider.OPENAI else GEMINI_MODELS["pro"],
        temperature=0.3
    )

class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""
    source: str = Field(description="Conference/journal topic from which concepts are extracted")
    concepts: List[str] = Field(description="List of extracted concepts")
    research_area: str = Field(description="Research area associated with the source topic")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")

SYSTEM_PROMPT = """
You are an expert academic concept extractor specializing in conference/journal topics and research areas. Your goal is to identify and extract fundamental, standardized academic concepts from text describing conference topics, journal special issues, or workshop themes.

**CRITICAL EXTRACTION GUIDELINES:**

1.  **Atomic Concepts Only:** Extract the core academic subject or field. Decompose compound concepts into their individual components ONLY if they represent distinct fields.
    *   Example: "Deep Learning for Natural Language Processing" -> "deep learning", "natural language processing"
    *   Example: "Advances in Cloud Computing and Security" -> "cloud computing", "security"
    *   Example: "Blockchain Technology" -> "blockchain technology" (Do not decompose established fields)

2.  **Explicit Mention:** Extract ONLY concepts EXPLICITLY mentioned in the text. Do NOT infer or add related concepts.

3.  **Strict Exclusions - DO NOT EXTRACT:**
    *   **Conference/Journal Identifiers:** "special issue on", "workshop on", "symposium for", "track on", "session about"
    *   **Organizational Terms:** "committee", "presentation", "submission", "review", "deadline"
    *   **Generic Terms:** "advances in", "recent developments", "emerging trends", "novel approaches", "state-of-the-art"
    *   **Proper Nouns:** Names of people, specific places, journals, or conferences
    *   **Acronyms:** Unless the acronym is the standard name of the field (e.g., "NLP", "AI")
    *   **Action/Process Words:** "understanding", "application", "development", "analysis", "methods"
    *   **Navigation Terms:** "home", "about", "contact", "back", "agenda", "schedule"

4.  **Standardization:**
    *   Output concepts in lowercase.
    *   Prefer singular forms where appropriate (e.g., "algorithm" instead of "algorithms"), but maintain established plural forms (e.g., "data structures").

**EXAMPLES:**

1.  Input Topic: "Special Issue on Machine Learning for Healthcare Applications"
    Concepts: `["machine learning", "healthcare"]`
    *   *Reasoning:* "Applications" is excluded as a process word. "Special Issue on" is excluded as journal identifier.

2.  Input Topic: "Workshop on Explainable AI and Ethics in Decision Systems"
    Concepts: `["explainable ai", "ethics", "decision systems"]`
    *   *Reasoning:* "Workshop on" is excluded as conference identifier.

3.  Input Topic: "Track on Deep Reinforcement Learning: Theory and Applications"
    Concepts: `["deep reinforcement learning"]`
    *   *Reasoning:* "Track on" and "Theory and Applications" are excluded as non-core concepts.

4.  Input Topic: "Advances in Blockchain for Supply Chain Management"
    Concepts: `["blockchain", "supply chain management"]`
    *   *Reasoning:* "Advances in" is excluded as a generic term.

5.  Input Topic: "Smart Cities and Internet of Things"
    Concepts: `["smart cities", "internet of things"]`
    *   *Reasoning:* Both are core concepts, correctly decomposed.

**Output Format:**
Return ONLY the list of extracted concepts for each source topic in the specified JSON format. If a topic yields no valid concepts based on the strict rules, return an empty list `[]` for its `concepts` field. Ensure the overall output is valid JSON.
"""

def preprocess_content(content: str) -> str:
    """Preprocess content to reduce size and improve quality"""
    # Remove extra whitespace
    content = " ".join(content.split())
    
    # Truncate if too long
    words = content.split()
    if len(words) > Config.MAX_CONTENT_WORDS:
        content = " ".join(words[:Config.MAX_CONTENT_WORDS])
    
    return content

def build_prompt(entries: List[Dict[str, Any]]) -> str:
    """Build prompt for concept extraction"""
    entries_str = ""
    for entry in entries:
        research_area = entry.get("research_area", "")
        content = entry.get("content", "")
        conference_topic = entry.get("conference_topic", content)  # Use conference_topic if available, otherwise content

        if content:
            entries_str += f"\nResearch Area: {research_area}\nConference Topic: {conference_topic}\n---\n"

    return f"""Extract academic concepts from these conference topics and journal special issues.
Return the concepts in this exact JSON format:
{{
    "extractions": [
        {{
            "source": "conference_topic_name",
            "concepts": ["concept1", "concept2"],
            "research_area": "research_area_name"
        }}
    ]
}}

IMPORTANT: Ensure your response is valid JSON. All strings must be properly quoted and all objects must be properly closed.
Double-check your JSON before returning it.

Conference topics to process:
{entries_str}"""

def process_batch_worker(args: tuple) -> List[ConceptExtraction]:
    """Worker function for parallel processing"""
    batch, provider = args
    try:
        prompt = build_prompt(batch)
        
        # Get process-local LLM instance
        llm = get_llm(provider)
        
        logger.info(f"Processing batch with {len(batch)} items in process {os.getpid()}")
        
        try:
            logger.info("Attempting structured extraction with Pydantic model")
            response = llm.infer(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                response_model=ConceptExtractionList,
            )
            logger.info(f"Successfully extracted {len(response.text.extractions)} concepts with structured validation")
            return response.text.extractions
        except Exception as e:
            if "insufficient_quota" in str(e):
                logger.error(f"API quota exceeded for provider {provider}")
                raise QuotaExceededError(f"API quota exceeded for provider {provider}")
            
            # Handle JSON validation errors specifically
            if "Invalid JSON" in str(e) or "JSONDecodeError" in str(e) or "model_validate_json" in str(e):
                logger.error(f"JSON validation error: {str(e)}")
                
                # Try a simpler approach without the Pydantic model
                try:
                    logger.info("Attempting to extract concepts without structured validation")
                    simple_response = llm.infer(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                    )
                    
                    logger.debug(f"Raw response: {simple_response.text[:500]}...")
                    
                    # Try to manually extract the JSON
                    import json
                    
                    # Function to find balanced JSON objects
                    def find_balanced_json(text):
                        """Find balanced JSON objects in text"""
                        results = []
                        stack = []
                        start_indices = []
                        
                        for i, char in enumerate(text):
                            if char == '{':
                                if not stack:  # Start of a new object
                                    start_indices.append(i)
                                stack.append('{')
                            elif char == '}':
                                if stack and stack[-1] == '{':
                                    stack.pop()
                                    if not stack:  # End of an object
                                        start = start_indices.pop()
                                        results.append(text[start:i+1])
                        
                        return results
                    
                    # Look for JSON-like structures in the response
                    matches = find_balanced_json(simple_response.text)
                    logger.info(f"Found {len(matches)} potential JSON objects in response")
                    
                    for i, match in enumerate(matches):
                        try:
                            logger.debug(f"Attempting to parse JSON object {i+1}: {match[:100]}...")
                            data = json.loads(match)
                            if 'extractions' in data:
                                # Convert to ConceptExtraction objects
                                extractions = []
                                for item in data['extractions']:
                                    try:
                                        extraction = ConceptExtraction(
                                            source=item.get('source', ''),
                                            concepts=item.get('concepts', []),
                                            research_area=item.get('research_area', '')
                                        )
                                        extractions.append(extraction)
                                    except Exception as item_error:
                                        logger.warning(f"Failed to parse extraction item: {str(item_error)}")
                                
                                if extractions:
                                    logger.info(f"Successfully extracted {len(extractions)} concepts manually")
                                    return extractions
                        except Exception as json_error:
                            logger.debug(f"Failed to parse JSON object {i+1}: {str(json_error)}")
                            continue
                    
                    # If we still couldn't parse the JSON, try a simplified approach
                    if not matches or all('"extractions"' not in match for match in matches):
                        logger.info("Attempting extraction with simplified prompt")
                        
                        # Create a simplified prompt for each entry individually
                        all_extractions = []
                        
                        for entry in batch:
                            research_area = entry.get("research_area", "")
                            content = entry.get("content", "")
                            
                            if not content:
                                continue
                                
                            simple_prompt = f"""Extract academic concepts from this conference topic / journal special issue.
Return ONLY a comma-separated list of concepts, nothing else.

Research Area: {research_area}
Conference Topic: {content}
"""
                            
                            try:
                                concepts_response = llm.infer(
                                    prompt=simple_prompt,
                                    system_prompt=SYSTEM_PROMPT,
                                    temperature=0.1  # Lower temperature for more consistent results
                                )
                                
                                # Parse comma-separated list
                                concepts_text = concepts_response.text.strip()
                                # Remove any markdown formatting
                                concepts_text = concepts_text.replace('```', '').strip()
                                
                                # Split by commas and clean up
                                concepts = [c.strip() for c in concepts_text.split(',') if c.strip()]
                                
                                if concepts:
                                    source = entry.get("conference_topic", entry.get("content", ""))
                                    extraction = ConceptExtraction(
                                        source=source,
                                        concepts=concepts,
                                        research_area=research_area
                                    )
                                    all_extractions.append(extraction)
                                    logger.info(f"Extracted {len(concepts)} concepts from {source} using simplified approach")
                            except Exception as simple_error:
                                logger.error(f"Failed simplified extraction for {research_area}: {str(simple_error)}")
                        
                        if all_extractions:
                            logger.info(f"Successfully extracted concepts for {len(all_extractions)} sources using simplified approach")
                            return all_extractions
                
                except Exception as manual_error:
                    logger.error(f"Failed manual extraction attempt: {str(manual_error)}")
            
            # If all attempts fail, log the error and return empty list
            logger.error(f"Failed to process batch: {str(e)}")
            return []
            
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return []

def process_batches_parallel(
    batches: List[List[Dict[str, Any]]],
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

def prepare_entries_from_conference_topics(topics_file: str, metadata_file: str) -> List[Dict[str, Any]]:
    """
    Prepare entries for concept extraction from conference topics file and metadata
    
    Args:
        topics_file: Path to conference topics file
        metadata_file: Path to metadata file
        
    Returns:
        List of entries ready for concept extraction
    """
    # Read conference topics
    with open(topics_file, 'r', encoding='utf-8') as f:
        conference_topics = [line.strip() for line in f if line.strip()]
    
    # Read metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Extract level2_to_conference_topics mapping from metadata
    level2_to_topics = metadata.get('level2_to_conference_topics_mapping', {})
    
    # Create entries that match the expected format
    entries = []
    
    for level2_term, topics in level2_to_topics.items():
        if not topics:
            continue
            
        # Create an entry for EACH conference topic individually to preserve the mapping
        for topic in topics:
            entry = {
                "research_area": level2_term,
                "content": topic,
                "conference_topic": topic  # Track the specific conference topic
            }
            entries.append(entry)
    
    # If there are conference topics without level2 mapping, add them as a general entry
    unmapped_topics = []
    for topic in conference_topics:
        mapped = False
        for topics_list in level2_to_topics.values():
            if topic in topics_list:
                mapped = True
                break
        if not mapped:
            unmapped_topics.append(topic)
    
    for topic in unmapped_topics:
        entry = {
            "research_area": "Conference Topics",
            "content": topic,
            "conference_topic": topic  # Track the specific conference topic
        }
        entries.append(entry)
    
    return entries

def main():
    """Main execution function"""
    try:
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")
            
        logger.info("Starting concept extraction from conference topics")

        # Prepare input data from the step 0 output
        entries = prepare_entries_from_conference_topics(Config.INPUT_FILE, Config.META_FILE_STEP0)
        logger.info(f"Prepared {len(entries)} entries from conference topics")

        # Process in batches
        all_extracted_concepts = []
        source_concept_mapping = {}  # Map sources to concepts
        conference_topic_concept_mapping = {}  # Map conference topics to concepts
        research_area_conference_topic_mapping = {}  # Map research areas to conference topics
        research_area_concept_mapping = {}  # Map research areas to concepts
        
        # Split into batches
        batches = chunk(entries, Config.BATCH_SIZE)
        logger.info(f"Split into {len(batches)} batches of size {Config.BATCH_SIZE}")
        
        # Process batches in parallel
        batch_results = process_batches_parallel(batches, provider)
        
        # Collect results and build mappings
        for responses in batch_results:
            if responses:
                all_extracted_concepts.extend(responses)
                
                # Update mappings
                for extraction in responses:
                    # The source is now the conference topic
                    source = extraction.source
                    
                    # Find the corresponding entry to get conference topic and research area
                    matching_entries = [
                        entry for entry in entries 
                        if entry.get("conference_topic", entry.get("content", "")) == source
                    ]
                    
                    # If no matching entries found, continue to next extraction
                    if not matching_entries:
                        # Update source mapping
                        if source not in source_concept_mapping:
                            source_concept_mapping[source] = set()
                        source_concept_mapping[source].update(extraction.concepts)
                        continue
                    
                    # For each matching entry, update the mappings
                    for entry in matching_entries:
                        conference_topic = entry.get("conference_topic", entry.get("content", ""))
                        research_area = entry.get("research_area", "")
                        
                        # Map conference topic to concept
                        if conference_topic not in conference_topic_concept_mapping:
                            conference_topic_concept_mapping[conference_topic] = set()
                        conference_topic_concept_mapping[conference_topic].update(extraction.concepts)
                        
                        # Map research area to conference topic
                        if research_area not in research_area_conference_topic_mapping:
                            research_area_conference_topic_mapping[research_area] = set()
                        research_area_conference_topic_mapping[research_area].add(conference_topic)
                        
                        # Map research area to concept
                        if research_area not in research_area_concept_mapping:
                            research_area_concept_mapping[research_area] = set()
                        research_area_concept_mapping[research_area].update(extraction.concepts)

        # Apply frequency threshold to all concepts
        all_concepts = [
            concept.lower()
            for response in all_extracted_concepts
            for concept in response.concepts
        ]
        concept_counts = Counter(all_concepts)
        verified_concepts = sorted(
            [
                concept
                for concept, count in concept_counts.items()
                if count >= Config.KW_APPEARANCE_THRESH
            ]
        )

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
            
        # Create clean, consolidated mappings for metadata
        source_concept_mapping_clean = {
            source: sorted(list(concepts))
            for source, concepts in source_concept_mapping.items()
        }
        
        conference_topic_concept_mapping_clean = {
            topic: sorted(list(concepts))
            for topic, concepts in conference_topic_concept_mapping.items()
        }
        
        research_area_conference_topic_mapping_clean = {
            area: sorted(list(topics))
            for area, topics in research_area_conference_topic_mapping.items()
        }
        
        research_area_concept_mapping_clean = {
            area: sorted(list(concepts))
            for area, concepts in research_area_concept_mapping.items()
        }
        
        with open(Config.META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "input_count": len(entries),
                        "output_count": len(verified_concepts),
                        "batch_size": Config.BATCH_SIZE,
                        "num_workers": Config.NUM_WORKERS,
                        "concept_threshold": Config.KW_APPEARANCE_THRESH,
                        "provider": provider or Provider.GEMINI,
                        "model": model_name,
                        "temperature": llm.temperature,
                    },
                    "source_concept_mapping": source_concept_mapping_clean,
                    "conference_topic_concept_mapping": conference_topic_concept_mapping_clean,
                    "research_area_conference_topic_mapping": research_area_conference_topic_mapping_clean,
                    "research_area_concept_mapping": research_area_concept_mapping_clean,
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

        # Create a CSV file with the complete hierarchy
        csv_file = os.path.join(Path(Config.META_FILE).parent, "lv3_s1_hierarchical_concepts.csv")
        try:
            with open(csv_file, "w", encoding="utf-8") as f:
                # Write header with conference topic, research area, and concept
                f.write("conference_topic,research_area,concept\n")
                
                # Create a set to track unique entries and avoid duplicates
                unique_entries = set()
                
                # Validate mappings before writing
                validated_mappings = {}  # conference_topic -> (concepts, research_areas)
                
                # Process each extraction to validate the relationships
                for extraction in all_extracted_concepts:
                    source = extraction.source  # This is the conference topic
                    concepts = extraction.concepts
                    research_area = extraction.research_area
                    
                    # Skip if source is empty or too generic
                    if not source or source in ["Conference Topics", "General"]:
                        continue
                    
                    # Format research area name if needed
                    research_area = research_area or "General"
                    
                    # Add to validated mappings
                    if source not in validated_mappings:
                        validated_mappings[source] = (set(concepts), {research_area})
                    else:
                        existing_concepts, existing_research_areas = validated_mappings[source]
                        existing_concepts.update(concepts)
                        existing_research_areas.add(research_area)
                
                # Write the validated mappings to CSV
                for conference_topic, (concepts, research_areas) in validated_mappings.items():
                    for research_area in research_areas:
                        # Check if this is a valid relationship
                        # Skip excessively broad research_area-concept associations
                        if len(concepts) > 30 and research_area != "General":
                            logger.warning(f"Skipping excessive mapping: {conference_topic} -> {research_area} with {len(concepts)} concepts")
                            continue
                            
                        for concept in concepts:
                            # Create a unique key to avoid duplicates
                            entry_key = (conference_topic, research_area, concept)
                            if entry_key in unique_entries:
                                continue
                            unique_entries.add(entry_key)
                            
                            # Escape commas and quotes in fields
                            topic_str = conference_topic.replace('"', '""')
                            area_str = research_area.replace('"', '""')
                            concept_str = concept.replace('"', '""')
                            
                            # Add quotes if the field contains commas
                            topic_csv = f'"{topic_str}"' if ',' in topic_str else topic_str
                            area_csv = f'"{area_str}"' if ',' in area_str else area_str
                            concept_csv = f'"{concept_str}"' if ',' in concept_str else concept_str
                            
                            f.write(f"{topic_csv},{area_csv},{concept_csv}\n")
                
            logger.info(f"Hierarchical concept relationships saved to {csv_file} with {len(unique_entries)} unique entries")
            
        except Exception as e:
            logger.error(f"Failed to write CSV file: {str(e)}")

        logger.info("Concept extraction from conference topics completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 