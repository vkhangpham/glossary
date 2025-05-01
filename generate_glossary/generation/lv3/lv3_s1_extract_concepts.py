import os
import sys
import time
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
    NUM_WORKERS = 32  # Number of parallel workers
    NUM_LLM_ATTEMPTS = 3  # Use 3 LLM runs
    CONCEPT_AGREEMENT_THRESH = 2  # Concepts must appear in at least 2 responses
    KW_APPEARANCE_THRESH = 1  # Minimum frequency threshold for concepts
    MAX_CONTENT_WORDS = 10000  # Maximum words per content chunk
    COOLDOWN_PERIOD = 1  # Seconds between batches
    COOLDOWN_FREQUENCY = 10  # Number of batches before cooldown

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Get or initialize LLM with specified provider (process-local)"""
    if not hasattr(_process_local, 'llm'):
        # logger.info(f"Initializing LLM with provider: {provider} for process {os.getpid()}")
        _process_local.llm = init_llm(provider, model)
    return _process_local.llm

def init_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Initialize LLM with specified provider and model"""
    if not provider:
        provider = Provider.GEMINI  # Default to Gemini
        
    # Choose model based on parameters
    if model == "default":
        selected_model = OPENAI_MODELS["default"] if provider == Provider.OPENAI else GEMINI_MODELS["pro"]
    else:  # mini
        selected_model = OPENAI_MODELS["mini"] if provider == Provider.OPENAI else GEMINI_MODELS["default"]
    
    return LLMFactory.create_llm(
        provider=provider,
        model=selected_model,
        temperature=0.3
    )

def get_random_llm_config() -> Tuple[str, str]:
    """Get a random LLM provider and model configuration"""
    provider = random.choice([Provider.OPENAI, Provider.GEMINI])
    model = random.choice(["default", "mini"])
    return provider, model

class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""
    source: str = Field(description="Conference/journal topic from which concepts are extracted")
    concepts: List[str] = Field(description="List of extracted concepts")
    research_area: str = Field(description="Research area associated with the source topic")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")

SYSTEM_PROMPT = """
You are an expert academic concept extractor specializing in identifying fundamental, standardized academic concepts and research fields from text describing **conference topics, journal special issues, or workshop themes**.

**CORE TASK:** Extract ONLY well-established, recognized academic disciplines, sub-disciplines, or research fields explicitly mentioned in the provided text.

**CRITICAL EXTRACTION GUIDELINES:**

**1. Focus on Established Academic Fields:**
    *   **EXTRACT:** Standardized academic disciplines (e.g., "computer science", "molecular biology"), sub-disciplines ("cognitive psychology"), and broad research areas ("machine learning", "quantum physics", "artificial intelligence").
    *   **DO NOT EXTRACT:**
        *   Specific, narrow research topics (e.g., "ecological effects of anthropogenic nutrient pulses"). Extract the *broader field* if mentioned (e.g., "ecology").
        *   Workshop, session, or track titles if too specific or administrative (e.g., "workshop on advanced accounting", "machine learning track I-IV").
        *   Degree-related concepts (e.g., "Ph.D. program in data science").
        *   Project or paper titles if too specific.

**2. Strict Explicit Mention Rule:**
    *   Extract ONLY concepts DIRECTLY and EXPLICITLY mentioned in the input text.
    *   **DO NOT INFER:** Do not add related concepts, subfields, or specializations that are *not* present verbatim in the text. (e.g., If text says "machine learning", do NOT add "neural networks" or "deep learning").

**3. MANDATORY Compound Term Decomposition:**
    *   **ALWAYS Split on "and":** If two potential concepts are joined by "and", ALWAYS split them into separate concepts.
        *   Example: "Deep Learning and Natural Language Processing" -> "deep learning", "natural language processing".
        *   Example: "Explainable AI and Ethics" -> "explainable ai", "ethics".
        *   Example: "Smart Cities and Internet of Things" -> "smart cities", "internet of things".
        *   Example: "Blockchain and Supply Chain Management" -> "blockchain", "supply chain management".
    *   **Handle Shared Prefixes/Modifiers AFTER Splitting:** If terms joined by "and" share a common prefix/modifier, preserve the modifier/prefix for BOTH resulting terms after splitting.
        *   Example: "Aerospace sciences and engineering" -> "aerospace sciences", "aerospace engineering".
        *   Example: "Cognitive and behavioral neuroscience" -> "cognitive neuroscience", "behavioral neuroscience".
        *   Example: "Theoretical and applied physics" -> "theoretical physics", "applied physics".
        *   Example: "Quantum physics and computing" -> "quantum physics", "quantum computing".
    *   **Single Concepts:** If a term does *not* contain "and" connecting potential sub-concepts, extract it as a single unit.
        *   Example: "Quantum Computing" -> "quantum computing".
        *   Example: "Cognitive Computing" -> "cognitive computing".

**4. Strict Exclusions List - DO NOT EXTRACT ANY OF THESE:**
    *   **Conference/Journal/Event Identifiers:** "special issue on", "workshop on", "symposium for", "track on", "session about", "conference on".
    *   **Administrative/Organizational Terms:** "committee", "presentation", "submission", "review", "deadline", "call for papers", "agenda", "schedule".
    *   **Generic/Qualitative Terms:** "advances in", "recent developments", "emerging trends", "novel approaches", "state-of-the-art", "introduction to", "advanced", "fundamentals of".
    *   **Proper Nouns:** Names of people, specific places, journals, or conferences unless they *are* the standardized name of the concept itself.
    *   **Acronyms:** Unless the acronym is the standard, universally recognized name of the field (e.g., "NLP", "AI" are often acceptable, but prefer full names if available).
    *   **Action/Process Words:** "understanding", "application", "development", "analysis", "methods".
    *   **Navigation/Website Terms:** "home", "about", "contact", "back", "next".
    *   **Degree Designations:** "B.S.", "M.S.", "M.A.", "Ph.D.", "bachelor's", "master's", "doctoral", "certificate".
    *   **Part/Level Identifiers:** Roman numerals (I, II, III), part numbers.
    *   **ANY concepts that do not appear verbatim in the input text.**

**5. Output Standardization:**
    *   Output all concepts in **lowercase**.
    *   Prefer **singular forms** where appropriate (e.g., "algorithm" not "algorithms"), but maintain established plural forms (e.g., "data structures").

**EXAMPLES:**

1.  **Input:** "Special Issue on Machine Learning for Healthcare"
    **Concepts:** `["machine learning", "healthcare"]` (Rule 1: Established fields; Rule 4: Exclude identifier)
    **INVALID:** `["medical informatics"]` (Rule 2: Not explicitly mentioned)

2.  **Input:** "Workshop on Explainable AI and Ethics"
    **Concepts:** `["explainable ai", "ethics"]` (Rule 3: Mandatory split on "and"; Rule 4: Exclude identifier)
    **INVALID:** `["interpretable machine learning"]` (Rule 2: Not explicitly mentioned)

3.  **Input:** "Advances in Blockchain for Supply Chain"
    **Concepts:** `["blockchain", "supply chain"]` (Rule 1: Established fields; Rule 4: Exclude generic term)
    **INVALID:** `["distributed ledger"]` (Rule 2: Not explicitly mentioned)

4.  **Input:** "Cognitive Computing"
    **Concepts:** `["cognitive computing"]` (Rule 3: Single concept, no "and")
    **INVALID:** `["artificial intelligence"]` (Rule 2: Not explicitly mentioned)

5.  **Input:** "Smart Cities and Internet of Things"
    **Concepts:** `["smart cities", "internet of things"]` (Rule 3: Mandatory split on "and")
    **INVALID:** `["urban computing"]` (Rule 2: Not explicitly mentioned)

6.  **Input:** "Computational methods in theoretical and applied physics"
    **Concepts:** `["computational methods", "theoretical physics", "applied physics"]` (Rule 3: Mandatory split + shared modifier)

7.  **Input:** "Track on Ecological Effects of Anthropogenic Nutrient Pulses"
    **Concepts:** `["ecology"]` (Rule 1: Too specific topic, extract broader field; Rule 4: Exclude identifier)

8.  **Input:** "Workshop on Advanced Accounting Parts I-IV"
    **Concepts:** `["accounting"]` (Rule 1: Extract broader field; Rule 4: Exclude identifier, level, parts)

9.  **Input:** "Ph.D. Session on Natural Language Processing Applications"
    **Concepts:** `["natural language processing"]` (Rule 1 & 4: Exclude degree, session, generic 'applications')

**Output Format:**
Return ONLY the list of extracted concepts for each source topic in the specified JSON format. If a topic yields no valid concepts based on these strict rules, return an empty list `[]` for its `concepts` field. Ensure the overall output is valid JSON.
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
    # Since entries are now grouped by research area, they should all have the same area
    # Extract the research area from the first entry
    research_area = entries[0].get("research_area", "") if entries else ""

    # Format list of conference topics/content
    entries_str = "\n".join([f"- {entry.get('conference_topic', entry.get('content', ''))}" for entry in entries])

    return f"""Extract academic concepts and research fields related to the research area of **{research_area}**.

**Critical Instructions:**
1.  **Focus on Core Concepts:** Extract ONLY concepts that are core subfields or direct components of the main **{research_area}** discipline. Do NOT extract concepts that belong primarily to *other* disciplines, even if mentioned in a topic.
    *   Example: If the research area is 'Databases', and a topic mentions 'applying machine learning to query optimization', extract 'query optimization' (if relevant as a database concept) but NOT 'machine learning' itself.
2.  Follow all extraction rules specified in the initial System Prompt (regarding established fields, explicit mentions, decomposition, exclusions, etc.).
3.  Output MUST be valid JSON in the specified format.

**Output Format:**
{{
    "extractions": [
        {{
            "source": "conference_topic_name_from_list_below",
            "concepts": ["concept1", "concept2"],
            "research_area": "{research_area}"
        }}
        # ... one entry for each topic processed
    ]
}}

**Input Conference Topics/Themes for the Research Area '{research_area}':**
{entries_str}

Provide the JSON output containing the extracted concepts based *only* on the rules and the provided topics for the **{research_area}** research area."""

def process_batch_worker(args: tuple) -> List[List[ConceptExtraction]]:
    """Worker function for parallel processing"""
    batch, num_attempts = args
    try:
        prompt = build_prompt(batch)
        
        # Run multiple attempts with different providers/models
        all_extractions = []
        
        for attempt in range(num_attempts):
            provider, model = get_random_llm_config()
            logger.info(f"Attempt {attempt+1}/{num_attempts} using {provider}/{model} in process {os.getpid()}")
            
            try:
                # Get process-local LLM instance with specific provider/model
                llm = init_llm(provider, model)
                
                logger.info(f"Processing batch with {len(batch)} items")
                
                try:
                    logger.info("Attempting structured extraction with Pydantic model")
                    response = llm.infer(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        response_model=ConceptExtractionList,
                    )
                    logger.info(f"Successfully extracted {len(response.text.extractions)} concepts with structured validation")
                    all_extractions.append(response.text.extractions)
                    
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
                                            all_extractions.append(extractions)
                                            break
                                except Exception as json_error:
                                    logger.debug(f"Failed to parse JSON object {i+1}: {str(json_error)}")
                                    continue
                                    
                            # If still unsuccessful, add an empty list for this attempt
                            if len(all_extractions) <= attempt:
                                all_extractions.append([])
                        
                        except Exception as manual_error:
                            logger.error(f"Failed manual extraction attempt: {str(manual_error)}")
                            # Add an empty list to maintain attempt count
                            all_extractions.append([])
                    
                    # If all attempts fail for this try, add an empty list
                    if len(all_extractions) <= attempt:
                        all_extractions.append([])
            
            except Exception as e:
                logger.error(f"LLM run failed: {str(e)}")
                # Add an empty list to maintain attempt count
                all_extractions.append([])
        
        return all_extractions
            
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return []

def combine_extractions(extractions_list: List[List[ConceptExtraction]]) -> Dict[str, Dict[str, Any]]:
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
            research_area = extraction.research_area
            
            # Initialize source in combined results if not present
            if source not in combined_results:
                combined_results[source] = {
                    "research_area": research_area,
                    "concepts": {}
                }
                
            # Update concept counts for this source
            for concept in extraction.concepts:
                concept_lower = concept.lower()
                if concept_lower not in combined_results[source]["concepts"]:
                    combined_results[source]["concepts"][concept_lower] = 0
                combined_results[source]["concepts"][concept_lower] += 1
    
    return combined_results

def filter_concepts_by_agreement(
    combined_results: Dict[str, Dict], 
    agreement_threshold: int
) -> Dict[str, Dict[str, Any]]:
    """
    Filter concepts based on agreement threshold across multiple LLM runs
    
    Args:
        combined_results: Dictionary mapping sources to research areas and concepts with counts
        agreement_threshold: Minimum number of times a concept must appear
        
    Returns:
        Dictionary mapping sources to filtered data
    """
    filtered_results = {}
    
    for source, data in combined_results.items():
        research_area = data["research_area"]
        concepts_with_counts = data["concepts"]
        
        # Keep only concepts that meet the agreement threshold
        filtered_concepts = {
            concept for concept, count in concepts_with_counts.items() 
            if count >= agreement_threshold
        }
        
        # Add filtered concepts to results
        if filtered_concepts:
            filtered_results[source] = {
                "research_area": research_area,
                "concepts": filtered_concepts
            }
    
    return filtered_results

def process_batches_parallel(
    batches: List[List[Dict[str, Any]]],
    num_attempts: int = Config.NUM_LLM_ATTEMPTS
) -> Dict[str, Dict[str, Any]]:
    """Process batches in parallel using ProcessPoolExecutor and filter results"""
    all_results = {}
    
    with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_batch_worker, (batch, num_attempts))
            for batch in batches
        ]
        
        for i, future in enumerate(tqdm(futures, desc="Processing batches")):
            try:
                # Get extractions from all LLM runs for this batch
                batch_extractions = future.result()
                
                if batch_extractions:
                    # Combine and filter the extractions
                    combined_results = combine_extractions(batch_extractions)
                    filtered_results = filter_concepts_by_agreement(
                        combined_results, Config.CONCEPT_AGREEMENT_THRESH
                    )
                    
                    # Merge into all_results
                    for source, data in filtered_results.items():
                        research_area = data["research_area"]
                        concepts = data["concepts"]
                        
                        if source not in all_results:
                            all_results[source] = {
                                "research_area": research_area,
                                "concepts": set()
                            }
                        
                        all_results[source]["concepts"].update(concepts)
                
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
        logger.info("Starting concept extraction from conference topics")

        # Prepare input data from the step 0 output
        entries = prepare_entries_from_conference_topics(Config.INPUT_FILE, Config.META_FILE_STEP0)
        logger.info(f"Prepared {len(entries)} entries from conference topics")

        # Group entries by research area
        entries_by_research_area = {}
        for entry in entries:
            research_area = entry["research_area"]
            if research_area not in entries_by_research_area:
                entries_by_research_area[research_area] = []
            entries_by_research_area[research_area].append(entry)
            
        # Create batches for each research area separately
        batches = []
        for research_area, area_entries in entries_by_research_area.items():
            area_batches = chunk(area_entries, Config.BATCH_SIZE)
            batches.extend(area_batches)
        logger.info(f"Split into {len(batches)} batches of size up to {Config.BATCH_SIZE}, grouped by research area")
        
        # Process batches in parallel and get filtered results
        extracted_data = process_batches_parallel(batches, Config.NUM_LLM_ATTEMPTS)
        
        # Initialize mappings
        source_concept_mapping = {}  # Map sources to concepts
        conference_topic_concept_mapping = {}  # Map conference topics to concepts
        research_area_conference_topic_mapping = {}  # Map research areas to conference topics
        research_area_concept_mapping = {}  # Map research areas to concepts
        
        # Build mappings from the filtered results
        for source, data in extracted_data.items():
            research_area = data["research_area"]
            concepts = data["concepts"]
            
            # Map source to concepts
            if source not in source_concept_mapping:
                source_concept_mapping[source] = set()
            source_concept_mapping[source].update(concepts)
            
            # Map conference topic to concepts
            if source not in conference_topic_concept_mapping:
                conference_topic_concept_mapping[source] = set()
            conference_topic_concept_mapping[source].update(concepts)
            
            # Map research area to conference topic
            if research_area not in research_area_conference_topic_mapping:
                research_area_conference_topic_mapping[research_area] = set()
            research_area_conference_topic_mapping[research_area].add(source)
            
            # Map research area to concepts
            if research_area not in research_area_concept_mapping:
                research_area_concept_mapping[research_area] = set()
            research_area_concept_mapping[research_area].update(concepts)

        # Apply frequency threshold to all concepts
        all_concepts = [
            concept.lower()
            for concepts in source_concept_mapping.values()
            for concept in concepts
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
                        "llm_attempts": Config.NUM_LLM_ATTEMPTS,
                        "concept_agreement_threshold": Config.CONCEPT_AGREEMENT_THRESH,
                        "concept_frequency_threshold": Config.KW_APPEARANCE_THRESH,
                        "providers_and_models": "random selection of OpenAI default/mini and Gemini default/mini",
                        "temperature": 0.3,
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
                
                # Process data to write CSV
                for source, concepts in source_concept_mapping.items():
                    # Find research areas for this source
                    research_areas = []
                    for area, topics in research_area_conference_topic_mapping.items():
                        if source in topics:
                            research_areas.append(area)
                    
                    # If no research area found, use "General"
                    if not research_areas:
                        research_areas = ["General"]
                        
                    for research_area in research_areas:
                        for concept in concepts:
                            # Create a unique key to avoid duplicates
                            entry_key = (source, research_area, concept)
                            if entry_key in unique_entries:
                                continue
                            unique_entries.add(entry_key)
                            
                            # Escape commas and quotes in fields
                            topic_str = source.replace('"', '""')
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