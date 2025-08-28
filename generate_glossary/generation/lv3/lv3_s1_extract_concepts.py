import os
import sys
import time
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import threading

# Fix import path - Adjust based on the new file location (lv3)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
from generate_glossary.utils.llm_simple import infer_structured, infer_text, get_random_llm_config

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv3.s1") # Changed logger name

# Get the base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# Process-local storage for LLM instances
_process_local = threading.local()

# Use centralized configuration
LEVEL = 3
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Get or initialize LLM with specified provider (process-local)"""
    if not hasattr(_process_local, 'llm'):
        _process_local.llm = init_llm(provider, model)
    return _process_local.llm

# init_llm function removed - using direct LLM calls

# get_random_llm_config function removed - using centralized version

# --- Updated Pydantic Models ---
class ConceptExtraction(BaseModel):
    """Base model for concept extraction from a journal topic"""
    raw_topic: str = Field(description="Raw topic from which concepts are extracted")
    concepts: List[str] = Field(description="List of extracted fundamental academic concepts or fields")
    journal: str = Field(description="The Lv2 journal associated with the raw topic")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions from journal topics")

# --- Updated System Prompt ---
SYSTEM_PROMPT = """
You are an expert academic concept extractor specializing in identifying the **fundamental, standardized academic concepts, disciplines, or research fields** that are **directly represented** by specific **journal topics**.

**CORE TASK:** Analyze the provided journal topic names (associated with a specific journal: '{journal}') and extract ONLY the underlying, established academic disciplines, sub-disciplines, or recognized research areas explicitly mentioned or directly implied by the topic's wording.

**CRITICAL EXTRACTION GUIDELINES:**

**1. Focus on Directly Represented Academic Fields:**
    *   **EXTRACT:** Standardized academic disciplines (e.g., "computer science", "molecular biology"), sub-disciplines ("cognitive psychology"), and recognized research areas ("machine learning", "quantum physics", "condensed matter physics", "economic policy") that are **clearly and directly represented** by the topic name.
    *   **DO NOT EXTRACT:**
        *   The topic names themselves unless they *are* already a standardized field name (e.g., if a topic is just "Artificial Intelligence", extract "artificial intelligence").
        *   Fields that are only tangentially related or require excessive inference. Stick closely to what the topic describes. (e.g., For "Novel Photonic Metamaterials", extract "photonics", "metamaterials". Do not jump to just "physics" unless the topic is very broad).
        *   Organizers, session numbers, dates, or administrative details (e.g., "Track 3", "(Monday)").
        *   Vague terms like "applications", "theory", "methods", "systems", "issues", "studies", "challenges" unless part of a standardized field name (e.g., "control theory", "systems biology", "security studies").

**2. Identify Standard Fields Within Topics:**
    *   Extract the established field names implied by the specific terminology in the topic. The goal is precision based on the topic's focus.
    *   Example: Topic "Compiler Optimizations for Heterogeneous Architectures" -> Extract: `["compilers", "computer architecture"]` (These fields are directly represented).
    *   Example: Topic "Deep Reinforcement Learning for Robotic Manipulation" -> Extract: `["deep learning", "reinforcement learning", "robotics", "machine learning"]` (All standard fields mentioned or directly implied).
    *   Example: Topic "Stellar Astrophysics and Galactic Dynamics" -> Extract: `["astrophysics", "galaxy dynamics"]` (Directly represented fields).

**3. MANDATORY Compound Term Decomposition:**
    *   **Split on Conjunctions ("and", "&", "/"):** If a topic explicitly lists multiple fields joined by conjunctions, split them into separate concepts *first*. Then evaluate each resulting term against other rules.
        *   Example: "Economics and Finance" -> Split "Economics", "Finance". Extract: `["economics", "finance"]`.
        *   Example: "Signal Processing / Machine Learning" -> Split "Signal Processing", "Machine Learning". Extract: `["signal processing", "machine learning"]`.
        *   Example: "Condensed Matter & Materials Physics" -> Split "Condensed Matter", "Materials Physics". Extract: `["condensed matter physics", "materials physics"]`.
        *   Example: "Reproductive justice and abortion rights" -> Split "Reproductive justice", "abortion rights". Extract: `["reproductive justice", "abortion rights"]`.
        *   Example: "Moral conflict and abortion ethics" -> Split "Moral conflict", "abortion ethics". Extract: `["moral conflict", "abortion ethics"]`.
    *   **Handle Shared Modifiers:** If terms joined by a conjunction share a common prefix/modifier, preserve the modifier for BOTH resulting terms *if it forms a standard field name for both*. Apply context (Rule 6) after splitting.
        *   Example: "Computational Fluid Dynamics and Heat Transfer" -> Split "Computational Fluid Dynamics", "Heat Transfer". Extract: `["computational fluid dynamics", "heat transfer"]`.
        *   Example: "Organic and Biological Chemistry" -> Split "Organic Chemistry", "Biological Chemistry". Extract: `["organic chemistry", "biological chemistry"]`.
        *   Example: "Environmental Economics and Policy" -> Split "Environmental Economics", "Environmental Policy". Extract: `["environmental economics", "environmental policy"]`.
    *   **Decomposition within terms:** Identify and extract distinct standard fields packed within a topic name even without explicit conjunctions.
        *   Example: "Human-Computer Interaction Design" -> Extract: `["human-computer interaction", "interaction design"]`.
        *   Example: "Software Verification Techniques" -> Extract: `["software verification"]`.

**4. Strict Exclusions List - DO NOT EXTRACT ANY OF THESE:**
    *   General methodological terms (e.g., "analysis", "modeling" unless part of a field like "financial modeling" or disambiguated by Rule 6).
    *   Administrative/organizational terms: "track", "session", "workshop", "symposium", "conference", "proceedings", "call for papers", "cfp".
    *   Generic terms: "research", "studies", "topics", "areas", "advances in", "emerging", "applications of", "foundations of", "special", "issues", "aspects", "perspectives".
    *   Proper nouns (people, places, specific projects) unless they are part of a standard field name (e.g., "Boolean Algebra").
    *   Acronyms unless universally recognized *as the field name* (e.g., "AI" for "artificial intelligence" is acceptable, but prefer full name if available in the topic).

**5. Output Standardization:**
    *   Output all concepts in **lowercase**.
    *   Use **standard, recognized forms** of field names.
    *   Prefer **singular forms** where appropriate (e.g., "algorithm"), but maintain established plural forms (e.g., "data structures", "complex systems").
    *   Normalize variations (e.g., "computation" -> "computing" or "computational science"; be consistent).

**6. MANDATORY Contextual Disambiguation:**
   *   **CRITICAL:** If an extracted concept term (after any splitting in Rule 3) is common and could have different meanings (e.g., "history", "languages", "literature", "policy", "law", "studies", "imaging", "modeling", "development", "memory"), you **MUST** attempt to disambiguate it.
   *   **Use Context:** Prepend a relevant modifier derived from:
        1.  **Adjectives/Nouns within the topic itself:** (e.g., Topic "African History" -> "african history").
        2.  **The Journal (`{journal}`):** If the topic term is generic but the journal provides clear context (e.g., Journal: "Journal of Health Policy", Topic: "Policy Analysis" -> "health policy analysis").
   *   **No Ambiguity Allowed:** Do **NOT** output ambiguous common terms without context if relevant context is available either in the topic or the journal name.
   *   **Apply Consistently After Split:** If context is applied to one part of a split term, apply the *same relevant context* to other parts if they also need disambiguation.
   *   Examples:
       *   Topic: "Urban Policy and Planning" (Journal: General Social Science Journal) -> Extract: `["urban policy", "urban planning"]` (Context from topic).
       *   Topic: "Policy Analysis" (Journal: Journal of Health Policy) -> Extract: `["health policy analysis"]` (Context from journal name).
       *   Topic: "Neural Imaging Techniques" -> Extract: `["neural imaging"]` (Context from topic. "Techniques" excluded by Rule 4).
       *   Topic: "Development and Aging" (Journal: Neuroscience Journal) -> Split "Development", "Aging". Apply context -> `["neural development", "neural aging"]`.
       *   Topic: "Literary Theory" (Journal: Journal on East Asian Studies) -> Extract: `["east asian literary theory"]` (or potentially just "literary theory" if "East Asian" isn't standardly prefixed, but good to add context). Prefer `["east asian literature", "literary theory"]`.
       *   Topic: "History of Science" (Journal: General History Journal) -> Extract: `["history of science"]` (Specific enough).
       *   Topic: "Memory Studies" (Journal: Journal on Holocaust Studies) -> Extract: `["holocaust memory studies"]` (or `["holocaust studies", "memory studies"]`).

**EXAMPLES:**

1.  **Journal:** (Example Journal: General Physics)
    **Topic:** "Finite and infinite groups"
    **Concepts:** `["group theory"]` (Standard field encompassing the topic)

2.  **Journal:** (Example Journal: Environmental Science)
    **Topic:** "Natural resource management and land use"
    **Concepts:** `["natural resource management", "land use"]` (Split 'and')

3.  **Journal:** (Example Journal: Nuclear Physics Symposium)
    **Topic:** "Theory of hot matter and relativistic heavy-ion collisions (thor)"
    **Concepts:** `["nuclear theory", "heavy-ion collisions"]` (Core fields represented, acronym/jargon excluded)

4.  **Journal:** (Example Journal: Communications Technology)
    **Topic:** "Low power wireless networks"
    **Concepts:** `["wireless networks", "low-power electronics"]` (Directly represented fields)

5.  **Journal:** (Example Journal: Civil Engineering Expo)
    **Topic:** "Structural health monitoring and diagnostics"
    **Concepts:** `["structural health monitoring", "structural diagnostics"]` (Split 'and', shared modifier applied)

6.  **Journal:** (Example Journal: Urban Planning Summit)
    **Topic:** "Urban design | landscape| planning"
    **Concepts:** `["urban design", "landscape architecture", "urban planning"]` (Split '|', expanded 'landscape')

7.  **Journal:** (Example Journal: AI and Language)
    **Topic:** "Spoken language generation"
    **Concepts:** `["natural language generation", "speech synthesis"]` (Standard related fields)

8.  **Journal:** (Example Journal: Logistics Tech)
    **Topic:** "Data mining in logistics"
    **Concepts:** `["data mining", "logistics"]` (Directly mentioned standard fields)

9.  **Journal:** (Example Journal: Infectious Diseases Update)
    **Topic:** "Infectious disease mechanisms"
    **Concepts:** `["infectious diseases"]` ('Mechanisms' is too generic)

10. **Journal:** (Example Journal: Media & Culture Studies)
    **Topic:** "Ethical issues of fashion journalism and consumer culture"
    **Concepts:** `["journalism ethics", "fashion studies", "consumer culture"]` (Split, extracted core concepts/ethics)


**Output Format:**
Return ONLY the list of extracted concepts for each source topic in the specified JSON format. If a topic yields no valid concepts based on these strict rules, return an empty list `[]` for its `concepts` field. Ensure the overall output is valid JSON.
"""

def preprocess_content(content: str) -> str:
    """Preprocess content (conference topic list)"""
    # Minimal preprocessing needed for topic lists
    content = content.strip()
    return content

# --- Updated Prompt Builder ---
def build_prompt(entries: List[Dict[str, Any]]) -> str:
    """Build prompt for concept extraction from journal topics"""
    # Entries are grouped by journal name
    journal = entries[0].get("journal", "Unknown Journal") if entries else "Unknown Journal"

    # Format list of journal topics, ensuring proper escaping for the f-string
    topics_str = "\n".join([f"- {entry.get('raw_topic', '')}" for entry in entries if entry.get('raw_topic')])

    # Use a multi-line f-string with triple quotes for clarity
    prompt = f"""Extract the fundamental academic concepts or research fields represented by the following topics from the journal: **{journal}**.

**Critical Instructions:**
1.  **Focus on Underlying Fields:** Identify the core academic disciplines/sub-disciplines/research areas related to each topic. Do *not* simply echo the topic name unless it's already a standard field.
2.  **Infer Broader Concepts:** Infer the established fields from specific topic names (e.g., "Compiler Optimizations..." -> "compilers", "computer architecture").
3.  Follow all extraction rules from the System Prompt (decomposition, exclusions, standardization).
4.  **CRITICAL REMINDER (Rule 6):** Apply contextual modifiers ONLY if an extracted term is highly ambiguous AND the journal name provides clear context (e.g., Topic 'Modeling' from 'Financial Systems Journal' -> 'financial modeling'). Prioritize core field extraction.
5.  Output MUST be valid JSON in the specified format.

**Output Format:**
```json
{{
    "extractions": [
        {{
            "raw_topic": "topic_name_from_list_below",
            "concepts": ["concept1", "concept2", ...],
            "journal": "{journal}"
        }}
        # ... one entry for each topic processed
    ]
}}
```

**Input Topics from {journal}:**
{topics_str}

Provide the JSON output containing the extracted fundamental concepts based *only* on the rules and the provided topics for **{journal}**. If a topic yields no concepts, return an empty list for its `concepts`.
"""
    return prompt

# --- Updated Worker Function ---
def process_batch_worker(args: tuple) -> List[List[ConceptExtraction]]:
    """Worker function for parallel processing conference topics"""
    batch, num_attempts = args
    if not batch:
        return []

    # Ensure the batch has entries before proceeding
    if not any(batch):
        logger.warning("Received an empty batch, skipping.")
        return []

    try:
        # Assuming all entries in a batch belong to the same conference/journal
        journal = batch[0].get("journal", "Unknown Journal")
        prompt = build_prompt(batch)

        all_attempt_results = [] # Store results from each attempt [[extraction1, extraction2], [extraction1, extraction2], ...]

        for attempt in range(num_attempts):
            provider, model_type = get_random_llm_config()
            logger.debug(f"Attempt {attempt+1}/{num_attempts} using {provider}/{model_type} for '{journal}' in process {os.getpid()}")

            try:
                # Get process-local LLM instance with specific provider/model
        
                try:
        response = infer_structured(
            provider=provider or "openai",
            prompt=prompt,
            response_model=ConceptExtractionList,
            system_prompt=SYSTEM_PROMPT
        )
                    # Ensure response.text is the Pydantic model instance
                    if isinstance(response.text, ConceptExtractionList):
                        attempt_extractions = response.text.extractions
                        logger.debug(f"Attempt {attempt+1} succeeded with {len(attempt_extractions)} extractions for '{journal}' using structured parsing.")
                        all_attempt_results.append(attempt_extractions)
                    else:
                         logger.error(f"Attempt {attempt+1} for '{journal}' returned unexpected type: {type(response.text)}. Expected ConceptExtractionList.")
                         all_attempt_results.append([]) # Add empty list for this failed attempt

                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed for '{journal}' with provider {provider}/{model_type}. Error: {str(e)}", exc_info=False) # Less verbose logging for common errors
                    # Handle specific errors like quota
                    if "insufficient_quota" in str(e).lower() or "rate limit" in str(e).lower():
                        logger.error(f"API quota/rate limit exceeded for provider {provider}. Raising error.")
                        # Optionally raise to stop, or just log and append empty
                        # raise QuotaExceededError(f"API quota exceeded for provider {provider}")
                        all_attempt_results.append([])
                    # Handle JSON parsing/validation errors (might try raw extraction if needed)
                    elif "model_validate_json" in str(e) or "jsondecodeerror" in str(e).lower():
                         logger.warning(f"JSON validation/parsing error in attempt {attempt+1} for '{journal}'. Trying raw extraction fallback (not implemented yet).")
                         # TODO: Implement raw extraction fallback if needed
                         all_attempt_results.append([]) # Append empty for now
                    else:
                        all_attempt_results.append([]) # Append empty list for other errors

            except Exception as llm_init_e:
                logger.error(f"LLM initialization failed in attempt {attempt+1} for {provider}/{model_type}: {str(llm_init_e)}")
                all_attempt_results.append([]) # Append empty list if LLM init fails

        return all_attempt_results # Return list of lists of extractions

    except Exception as e:
        # Log error for the entire batch processing
        conf_name = batch[0].get("journal", "Unknown Journal") if batch else "Empty Batch"
        logger.error(f"Error processing batch for '{conf_name}': {str(e)}", exc_info=True)
        # Return a structure indicating failure for all attempts in this batch run
        return [[] for _ in range(num_attempts)]


# --- Updated Combination and Filtering ---
def combine_extractions(all_attempt_results: List[List[ConceptExtraction]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Combine extractions from multiple LLM runs and count concept frequencies per source topic and conference.

    Args:
        all_attempt_results: List where each element is a list of ConceptExtraction objects from one LLM attempt.
                             e.g., [[attempt1_ext1, attempt1_ext2], [attempt2_ext1, attempt2_ext2]]

    Returns:
        Dictionary mapping (source_topic, conference_name) tuples to concept frequency dictionaries {'concept': count}.
    """
    combined_counts = {} # Key: (source_topic, conference_name), Value: {'concept': count}

    for attempt_extractions in all_attempt_results:
        if not isinstance(attempt_extractions, list):
             logger.warning(f"Unexpected data type in attempt results: {type(attempt_extractions)}. Skipping this attempt.")
             continue

        for extraction in attempt_extractions:
             # Validate extraction object
             if not isinstance(extraction, ConceptExtraction):
                  logger.warning(f"Unexpected item type in extraction list: {type(extraction)}. Skipping.")
                  continue

             raw_topic = extraction.raw_topic
             conf_name = extraction.journal
             key = (raw_topic, conf_name)

             if key not in combined_counts:
                 combined_counts[key] = {} # Initialize concept counter for this source

             for concept in extraction.concepts:
                 if not isinstance(concept, str) or not concept.strip():
                      continue # Skip empty or invalid concepts
                 concept_lower = concept.lower().strip()
                 combined_counts[key][concept_lower] = combined_counts[key].get(concept_lower, 0) + 1

    return combined_counts


def filter_concepts_by_agreement(
    combined_counts: Dict[Tuple[str, str], Dict[str, int]],
    agreement_threshold: int
) -> Dict[Tuple[str, str], Set[str]]:
    """
    Filter concepts based on agreement threshold across multiple LLM runs.

    Args:
        combined_counts: Dictionary from combine_extractions. Key: (source_topic, conf_name), Value: {'concept': count}.
        agreement_threshold: Minimum number of times a concept must appear for a given source.

    Returns:
        Dictionary mapping (source_topic, conference_name) tuples to sets of agreed-upon concepts.
    """
    filtered_results = {} # Key: (source_topic, conf_name), Value: {agreed_concept1, agreed_concept2}

    for key, concept_freq_dict in combined_counts.items():
        agreed_concepts = {
            concept for concept, count in concept_freq_dict.items()
            if count >= agreement_threshold
        }
        if agreed_concepts: # Only store if there are concepts meeting the threshold
            filtered_results[key] = agreed_concepts

    return filtered_results


# --- Updated Parallel Processing Orchestrator ---
def process_batches_parallel(
    batches: List[List[Dict[str, Any]]],
    num_attempts: int = processing_config.llm_attempts,
    agreement_threshold: int = processing_config.concept_agreement_threshold
) -> Dict[Tuple[str, str], Set[str]]:
    """Process batches in parallel using ProcessPoolExecutor and filter results by agreement."""
    final_agreed_concepts = {} # Key: (source_topic, conf_name), Value: {agreed_concept1, ...}

    with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        # Map batches to worker function
        futures = [
            executor.submit(process_batch_worker, (batch, num_attempts))
            for batch in batches if batch # Ensure batch is not empty
        ]

        logger.info(f"Submitted {len(futures)} batch jobs to ProcessPoolExecutor.")

        for i, future in enumerate(tqdm(futures, desc="Processing batches")):
            try:
                # result is List[List[ConceptExtraction]]
                batch_attempt_results = future.result()

                if batch_attempt_results:
                    # Combine results from multiple attempts for this batch
                    combined_batch_counts = combine_extractions(batch_attempt_results)

                    # Filter based on agreement threshold
                    filtered_batch_results = filter_concepts_by_agreement(
                        combined_batch_counts, agreement_threshold
                    )

                    # Merge filtered results into the final dictionary
                    final_agreed_concepts.update(filtered_batch_results)

                # Apply cooldown periodically
                if i > 0 and i % processing_config.cooldown_frequency == 0:
                    logger.debug(f"Applying cooldown period of {processing_config.cooldown_period}s after batch {i+1}")
                    time.sleep(processing_config.cooldown_period)

            except Exception as e:
                logger.error(f"Error processing result from future {i}: {str(e)}", exc_info=True)
                # Decide whether to continue or stop on error
                continue

    logger.info(f"Finished processing all batches. Found agreed-upon concepts for {len(final_agreed_concepts)} (topic, conference) pairs.")
    return final_agreed_concepts


# --- Updated Input Preparation ---
def prepare_entries_from_raw_topics(topics_file: str, metadata_file: str) -> List[Dict[str, Any]]:
    """
    Prepare entries for concept extraction from raw topics file and Lv3 metadata.

    Args:
        topics_file: Path to the file containing raw topics (lv3_s0_conference_topics.txt).
        metadata_file: Path to the Lv3 metadata file (lv3_s0_metadata.json).

    Returns:
        List of entries, where each entry represents a raw topic
        and includes its associated journal name(s).
        Format: [{"raw_topic": "topic_name", "journal": "journal_name"}, ...]
    """
    entries = []
    try:
        # Read raw topics
        with open(topics_file, 'r', encoding='utf-8') as f:
            raw_topics = {line.strip() for line in f if line.strip()}
        logger.info(f"Read {len(raw_topics)} unique raw topics from {topics_file}")

        # Read Lv3 metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_file}")

        # Get the mapping from topic back to its source Lv2 journal(s)
        # Adjust key based on actual metadata structure from lv3_s0
        topic_to_journals_mapping = metadata.get('conference_topic_level2_sources', {})
        if not topic_to_journals_mapping:
             # Check alternative nesting
             metadata_nested = metadata.get("metadata", {})
             topic_to_journals_mapping = metadata_nested.get('conference_topic_level2_sources', {})

        if not topic_to_journals_mapping:
            logger.warning("Could not find 'conference_topic_level2_sources' mapping in metadata. Cannot link topics to journals.")
            # Fallback: create entries with unknown journal name
            for topic in raw_topics:
                entries.append({
                    "raw_topic": topic,
                    "journal": "Unknown Journal"
                })
            return entries

        # Create entries, linking each topic to its source journal(s)
        topics_processed = set()
        for topic, sources in topic_to_journals_mapping.items():
            if topic not in raw_topics:
                continue # Skip topics from metadata not in the main topics file (consistency check)

            if isinstance(sources, list) and sources:
                # If a topic came from multiple journals, create an entry for each pair
                for journal in sources:
                    entries.append({
                        "raw_topic": topic,
                        "journal": journal
                    })
            elif isinstance(sources, str): # Handle case where source is just a string
                 entries.append({
                        "raw_topic": topic,
                        "journal": sources
                 })
            topics_processed.add(topic)

        # Add topics from the file that might not be in the metadata mapping (shouldn't happen ideally)
        missing_topics = raw_topics - topics_processed
        if missing_topics:
             logger.warning(f"Found {len(missing_topics)} topics in {topics_file} that were not in the metadata mapping. Adding with 'Unknown' source.")
             for topic in missing_topics:
                  entries.append({
                       "raw_topic": topic,
                       "journal": "Unknown Journal"
                  })

        logger.info(f"Prepared {len(entries)} (topic, journal) pair entries for concept extraction.")

    except FileNotFoundError:
        logger.error(f"Input file not found: {topics_file} or {metadata_file}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from metadata file: {metadata_file}")
        raise
    except Exception as e:
        logger.error(f"Error preparing entries: {e}", exc_info=True)
        raise

    return entries


# --- Main Execution Logic ---
def main():
    """Main execution function for Lv3 Concept Extraction"""
    try:
        logger.info("--- Starting Lv3 Concept Extraction from Raw Topics ---")

        # Create output directories if needed
        for path in [level_config.get_step_output_file(1), level_config.get_step_metadata_file(1)]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare input data: List of {"raw_topic": ..., "journal": ...}
        entries = prepare_entries_from_raw_topics(level_config.get_step_input_file(1), level_config.get_step_metadata_file(1)_STEP0)
        if not entries:
            logger.error("No input entries prepared. Exiting.")
            return
        logger.info(f"Prepared {len(entries)} (topic, journal) pairs for processing.")

        # Group entries by journal name for potentially better contextual batching
        entries_by_journal = {}
        for entry in entries:
            journal = entry["journal"]
            if journal not in entries_by_journal:
                entries_by_journal[journal] = []
            entries_by_journal[journal].append(entry)
        logger.info(f"Grouped entries by {len(entries_by_journal)} unique journal names.")

        # Create batches, ensuring entries in a batch share the same journal name
        batches = []
        for journal, journal_entries in entries_by_journal.items():
            journal_batches = chunk(journal_entries, processing_config.batch_size)
            batches.extend(journal_batches)
        logger.info(f"Split into {len(batches)} batches of max size {processing_config.batch_size}, grouped by journal.")

        # Process batches in parallel and get filtered results
        # extracted_data: Key: (raw_topic, journal), Value: {agreed_concept1, ...}
        extracted_data = process_batches_parallel(
            batches,
            processing_config.llm_attempts,
            processing_config.concept_agreement_threshold
        )

        if not extracted_data:
            logger.warning("No concepts extracted after parallel processing and filtering. Exiting.")
            return

        # --- Aggregate and Build Mappings ---
        all_concepts_found = set()
        raw_topic_to_concept_mapping = {} # topic -> {concept1, concept2}
        journal_to_concept_mapping = {} # journal -> {concept1, concept2}
        concept_to_raw_topic_mapping = {} # concept -> {topic1, topic2}
        concept_to_journal_mapping = {} # concept -> {journal1, journal2}

        for (topic, journal), concepts in extracted_data.items():
            all_concepts_found.update(concepts)

            # Topic -> Concept
            if topic not in raw_topic_to_concept_mapping:
                raw_topic_to_concept_mapping[topic] = set()
            raw_topic_to_concept_mapping[topic].update(concepts)

            # Journal -> Concept
            if journal not in journal_to_concept_mapping:
                journal_to_concept_mapping[journal] = set()
            journal_to_concept_mapping[journal].update(concepts)

            # Concept -> Topic & Concept -> Journal
            for concept in concepts:
                if concept not in concept_to_raw_topic_mapping:
                    concept_to_raw_topic_mapping[concept] = set()
                concept_to_raw_topic_mapping[concept].add(topic)

                if concept not in concept_to_journal_mapping:
                    concept_to_journal_mapping[concept] = set()
                concept_to_journal_mapping[concept].add(journal)

        logger.info(f"Aggregated concepts. Found {len(all_concepts_found)} unique concepts initially.")

        # Optional: Apply overall frequency threshold if needed (processing_config.keyword_appearance_threshold)
        if processing_config.keyword_appearance_threshold > 1:
            logger.info(f"Applying overall frequency threshold: {processing_config.keyword_appearance_threshold}")
            # Recalculate concept counts across all sources
            all_concepts_list = [
                concept
                for concepts_set in extracted_data.values()
                for concept in concepts_set
            ]
            concept_counts = Counter(all_concepts_list)
            verified_concepts_final = {
                concept for concept, count in concept_counts.items()
                if count >= processing_config.keyword_appearance_threshold
            }
            logger.info(f"Filtered concepts by frequency threshold. Keeping {len(verified_concepts_final)} concepts.")
        else:
            verified_concepts_final = all_concepts_found

        logger.info(f"Total {len(verified_concepts_final)} unique Lv3 concepts verified.")

        # --- Save Results ---

        # Save unique verified concepts list
        final_concepts_list_sorted = sorted(list(verified_concepts_final))
        with open(level_config.get_step_output_file(1), "w", encoding="utf-8") as f:
            for concept in final_concepts_list_sorted:
                f.write(f"{concept}\n")
        logger.info(f"Saved {len(final_concepts_list_sorted)} verified concepts to {level_config.get_step_output_file(1)}")

        # --- Prepare and Save Metadata ---

        # Filter mappings to only include verified concepts
        def filter_mapping_by_concepts(mapping_dict, verified_set):
            filtered_map = {}
            for key, concepts_set in mapping_dict.items():
                filtered_concepts = concepts_set.intersection(verified_set)
                if filtered_concepts:
                    filtered_map[key] = sorted(list(filtered_concepts))
            return filtered_map

        def filter_reverse_mapping_by_concepts(mapping_dict, verified_set):
             filtered_map = {}
             for concept, sources_set in mapping_dict.items():
                 if concept in verified_set and sources_set:
                     filtered_map[concept] = sorted(list(sources_set))
             return filtered_map

        metadata_payload = {
            "metadata": {
                "input_topics_count": len({e["raw_topic"] for e in entries}),
                "input_conference_journal_count": len(entries_by_journal),
                "input_topic_conference_pairs": len(entries),
                "output_unique_concepts_count": len(verified_concepts_final),
                "batch_size": processing_config.batch_size,
                "num_workers": Config.NUM_WORKERS,
                "llm_attempts": processing_config.llm_attempts,
                "concept_agreement_threshold": processing_config.concept_agreement_threshold,
                "overall_concept_frequency_threshold": processing_config.keyword_appearance_threshold,
                "llm_providers_models": "Random selection from OpenAI/Gemini (pro, default, mini)",
                "llm_temperature": 0.2,
            },
            "conference_topic_to_concept_mapping": filter_mapping_by_concepts(
                raw_topic_to_concept_mapping, verified_concepts_final
            ),
            "conference_journal_to_concept_mapping": filter_mapping_by_concepts(
                journal_to_concept_mapping, verified_concepts_final
            ),
             "concept_to_conference_topic_mapping": filter_reverse_mapping_by_concepts(
                concept_to_raw_topic_mapping, verified_concepts_final
            ),
             "concept_to_conference_journal_mapping": filter_reverse_mapping_by_concepts(
                concept_to_journal_mapping, verified_concepts_final
            ),
            # Add concept frequencies if threshold > 1
            "final_concept_frequencies": {
                concept: count for concept, count in concept_counts.items()
                if concept in verified_concepts_final
            } if processing_config.keyword_appearance_threshold > 1 else "N/A (Threshold = 1)"
        }

        with open(level_config.get_step_metadata_file(1), "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=4, ensure_ascii=False)
        logger.info(f"Metadata saved to {level_config.get_step_metadata_file(1)}")


        # --- Optional: Save Hierarchical CSV ---
        csv_file = os.path.join(Path(level_config.get_step_metadata_file(1)).parent, "lv3_s1_hierarchical_concepts.csv")
        try:
            unique_csv_entries = set()
            with open(csv_file, "w", encoding="utf-8", newline='') as f_csv:
                 import csv
                 writer = csv.writer(f_csv)
                 # Write header
                 writer.writerow(["conference_journal", "conference_topic", "extracted_concept"])

                 # Iterate through the original extracted_data before potential frequency filtering
                 for (topic, conf_name), concepts in extracted_data.items():
                      for concept in concepts:
                           # Only write concepts that passed the final verification
                           if concept in verified_concepts_final:
                                entry_key = (conf_name, topic, concept)
                                if entry_key not in unique_csv_entries:
                                     writer.writerow([conf_name, topic, concept])
                                     unique_csv_entries.add(entry_key)

            logger.info(f"Hierarchical data saved to {csv_file} with {len(unique_csv_entries)} unique (Conf, Topic, Concept) rows.")

        except Exception as e:
            logger.error(f"Failed to write hierarchical CSV file: {str(e)}")


        logger.info("--- Lv3 Concept Extraction completed successfully ---")

    except Exception as e:
        logger.error(f"An error occurred during Lv3 Concept Extraction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Apply nest_asyncio if available (might not be needed for ProcessPoolExecutor)
    try:
        import nest_asyncio
        nest_asyncio.apply()
        logger.info("Applied nest_asyncio patch.")
    except ImportError:
        logger.debug("nest_asyncio not found, skipping patch.")
    except RuntimeError as e:
         if "cannot apply patch" not in str(e).lower():
              logger.warning(f"Could not apply nest_asyncio patch: {e}")

    main() 