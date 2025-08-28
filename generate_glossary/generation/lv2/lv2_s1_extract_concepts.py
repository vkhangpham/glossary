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
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv2.s1")

# Get the base directory
BASE_DIR = os.getcwd()  # Get the current working directory, which should be the project root

# Process-local storage for LLM instances
_process_local = threading.local()

def normalize_department_name(dept_name: str) -> str:
    """
    Normalize department name by stripping 'Department:' prefix if present
    
    Args:
        dept_name: Department name to normalize
        
    Returns:
        Normalized department name
    """
    if dept_name.lower().startswith("department:"):
        # Strip "Department:" prefix and any leading whitespace
        return dept_name[len("Department:"):].strip()
    return dept_name

# Use centralized configuration
LEVEL = 2
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
        # logger.info(f"Initializing LLM with provider: {provider} for process {os.getpid()}")
        _process_local.llm = init_llm(provider, model)
    return _process_local.llm

def init_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Initialize LLM with specified provider and model"""
    if not provider:
        provider = Provider.GEMINI  # Default to Gemini
    
    selected_model = OPENAI_MODELS[model] if provider == Provider.OPENAI else GEMINI_MODELS[model]
        
    return LLMFactory.create_llm(
        provider=provider,
        model=selected_model,
        temperature=0.3
    )

def get_random_llm_config() -> Tuple[str, str]:
    """Get a random LLM provider and model configuration"""
    provider = random.choice([Provider.OPENAI, Provider.GEMINI])
    model = random.choice(["pro", "default", "mini"])
    return provider, model

class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""
    source: str = Field(description="Topic area (research area) from which concepts are extracted")
    concepts: List[str] = Field(description="List of extracted concepts")
    department: str = Field(description="Department associated with the source topic area")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")

SYSTEM_PROMPT = """
You are an expert academic concept extractor specializing in identifying fundamental, standardized academic concepts and research fields from text describing university departments, research areas, or courses.

**CORE TASK:** Extract ONLY well-established, recognized academic disciplines, sub-disciplines, or research fields explicitly mentioned in the provided text.

**CRITICAL EXTRACTION GUIDELINES:**

**1. Focus on Established Academic Fields:**
    *   **EXTRACT:** Standardized academic disciplines (e.g., "computer science", "molecular biology"), sub-disciplines ("cognitive psychology"), and broad research areas ("machine learning", "quantum physics").
    *   **DO NOT EXTRACT:**
        *   Specific, narrow research topics (e.g., "ecological effects of anthropogenic nutrient pulses"). Extract the *broader field* if mentioned (e.g., "ecology").
        *   Course designations/titles (e.g., "accounting workshop", "biology seminar", "chemistry I-IV", "advanced topics in machine learning").
        *   Degree paths/types (e.g., "B.S. finance", "Ph.D. economics", "master's program", "certificate").
        *   Project or paper titles if too specific.

**2. Strict Explicit Mention Rule:**
    *   Extract ONLY concepts DIRECTLY and EXPLICITLY mentioned in the input text.
    *   **DO NOT INFER:** Do not add related concepts, subfields, or specializations that are *not* present verbatim in the text. (e.g., If text says "electrical engineering", do NOT add "electronics" or "power systems").

**3. MANDATORY Compound Term Decomposition:**
    *   **ALWAYS Split on "and":** If two potential concepts are joined by "and", ALWAYS split them into separate concepts.
        *   Example: "Developmental and stem cell biology" -> "developmental biology", "stem cell biology".
        *   Example: "Business and government relations" -> "business relations", "government relations" (assuming context implies split), OR simply "business", "government relations" if "business relations" isn't standard.
        *   Example: "Children and the law" -> "children", "law" (or more specific fields if mentioned).
        *   Example: "Machine learning and artificial intelligence" -> "machine learning", "artificial intelligence".
        *   Example: "Quantum mechanics and relativity" -> "quantum mechanics", "relativity".
    *   **Handle Shared Prefixes/Modifiers AFTER Splitting:** If terms joined by "and" share a common prefix/modifier, preserve the modifier for BOTH resulting terms after splitting.
        *   Example: "Aerospace sciences and engineering" -> "aerospace sciences", "aerospace engineering".
        *   Example: "Planetary science and exploration" -> "planetary science", "planetary exploration".
        *   Example: "Cognitive and behavioral neuroscience" -> "cognitive neuroscience", "behavioral neuroscience".
        *   Example: "Environmental economics and policy" -> "environmental economics", "environmental policy".
    *   **Apply Context After Splitting:** After splitting terms joined by "and", evaluate *each* resulting term against Rule 6 (Contextual Disambiguation). **This step is MANDATORY.** If a contextual modifier (e.g., geographical, thematic like 'african' from 'in africa' or department context) applies to the original phrase, this modifier **MUST** be applied to *all* resulting terms that require disambiguation according to Rule 6.
        *   Example: "public policy & development in africa" (context: African Studies) -> Split to "public policy" and "development in africa". **MUST** apply context -> "african public policy", "african development".
        *   Example: "memory and history" (context: African American Studies) -> Split to "memory" and "history". **MUST** apply context -> "african american memory", "african american history".
    *   **Single Concepts:** If a term does *not* contain "and" connecting potential sub-concepts, extract it as a single unit.
        *   Example: "Quantum mechanics" -> "quantum mechanics".
        *   Example: "Biochemistry" -> "biochemistry".

**4. Strict Exclusions List - DO NOT EXTRACT ANY OF THESE:**
    *   **General Methodological Terms:** "research methodology", "experimental design", "survey methods", "qualitative analysis", "quantitative methods", "statistical analysis", "data collection", "field research", "laboratory techniques".
    *   **Course/Level Identifiers:** "introduction to", "advanced", "fundamentals of", roman numerals (I, II, III), course numbers (e.g., "CS 101"), specific levels ("graduate", "undergraduate").
    *   **Degree Designations:** "B.S.", "M.S.", "M.A.", "Ph.D.", "bachelor's", "master's", "doctoral", "certificate".
    *   **Organizational/Institutional Terms:** "department", "faculty", "laboratory", "lab", "institute", "center", "program", "group", "university", "college". (Exception: Extract the field *name* if preceded by these, e.g., "Department of Electrical Engineering" -> "electrical engineering").
    *   **Generic/Administrative Terms:** "research", "studies", "topics", "areas", "course", "degree", "credits", "syllabus", "prerequisite", "assignment", "description", "overview", "topics in".
    *   **Proper Nouns:** Names of people, specific places, or branded initiatives, unless they *are* the standardized name of the concept itself (rare).
    *   **Acronyms:** Unless the acronym is the standard, universally recognized name of the field (e.g., "AI" *might* be acceptable if context strongly confirms, but generally prefer the full name like "artificial intelligence").
    *   **Action/Process Words:** "understanding", "application", "development", "analysis", "methods".
    *   **Navigation/Website Terms:** "home", "about", "contact", "back", "next", "click here".
    *   **Qualifiers/Vague Terms:** "general", "special", "various", "other".
    *   **ANY concepts that do not appear verbatim in the input text.**

**5. Output Standardization:**
    *   Output all concepts in **lowercase**.
    *   Prefer **singular forms** where appropriate (e.g., "algorithm" not "algorithms"), but maintain established plural forms (e.g., "data structures").

**6. MANDATORY Contextual Disambiguation:**
   *   **CRITICAL:** If an extracted concept term (whether a single term or one resulting from a split in Rule 3) is common and could have different meanings (e.g., "arts", "history", "languages", "literature", "policy", "law", "studies", "imaging", "modeling", "aging", "memory"), you **MUST** prepend a relevant modifier to clarify its specific meaning. This modifier should be derived from the most specific context available in the source text (e.g., **explicit phrases like 'in [Region/Topic]'**, the department name, the broader research area, or related adjectives like 'biomedical' or 'neural' if present).
   *   **No Ambiguity Allowed:** Do **NOT** output ambiguous common terms without context. If context is available, it **MUST** be applied.
   *   **Consistency After Splitting:** When applying context to terms resulting from a split (Rule 3), ensure the *same relevant context* is applied consistently to all parts needing disambiguation.
   *   Examples:
       *   Input context: "Department of Asian Studies", text mentions "literature" -> Extract: `["asian literature"]` (NOT `["literature"]`)
       *   Input context: "Research in African Art History", text mentions "history" and "art" -> Extract: `["african art history"]` (or `["african history", "african art"]`)
       *   Input context: "Focus on European languages" -> Extract: `["european languages"]` (NOT `["languages"]`)
       *   Input context: "Studies in Economic Policy" -> Extract: `["economic policy"]` (NOT `["policy"]`)
       *   Input context: "neural development and aging" -> Split "neural development", "aging". Apply context -> `["neural development", "neural aging"]` (NOT `["aging"]`)
       *   Input context: "Imaging & Medical Devices" (Dept: Biomedical Engineering) -> Split "imaging", "medical devices". Apply context -> `["biomedical imaging", "medical devices"]` (NOT `["imaging"]`)
   *   Aim for concise but clear modifiers. If the term is already specific (e.g., "quantum mechanics", "molecular biology"), no additional modifier is needed.

**EXAMPLES:**

1.  **Input:** "Data science and machine learning"
    **Concepts:** `["data science", "machine learning"]` (Rule 3: Mandatory split on "and")

2.  **Input:** "Quantum mechanics"
    **Concepts:** `["quantum mechanics"]` (Rule 3: Single concept, no "and")

3.  **Input:** "Department of Electrical Engineering"
    **Concepts:** `["electrical engineering"]` (Rule 4: Remove organizational term)
    **INVALID:** `["electronics", "power systems"]` (Rule 2: Not explicitly mentioned)

4.  **Input:** "Artificial intelligence in computer science"
    **Concepts:** `["artificial intelligence", "computer science"]` (Rule 2: Explicitly mentioned)
    **INVALID:** `["machine learning", "neural networks"]` (Rule 2: Not explicitly mentioned)

5.  **Input:** "Biology and ecology"
    **Concepts:** `["biology", "ecology"]` (Rule 3: Mandatory split on "and")
    **INVALID:** `["cellular biology", "ecosystem analysis"]` (Rule 2: Not explicitly mentioned)

6.  **Input:** "Aerospace sciences and engineering"
    **Concepts:** `["aerospace sciences", "aerospace engineering"]` (Rule 3: Mandatory split + shared prefix)

7.  **Input:** "B.S. in Computer Science and Engineering"
    **Concepts:** `["computer science", "engineering"]` (Rule 4: Remove degree; Rule 3: Mandatory split)

8.  **Input:** "Advanced Accounting Workshop Parts I-IV"
    **Concepts:** `[]` (Rule 1 & 4: Course designation with level/part number)

9.  **Input:** "Ecological Effects of Anthropogenic Nutrient Pulses"
    **Concepts:** `["ecology"]` (Rule 1: Too specific topic, extract broader field. Assuming 'ecology' was implicitly or explicitly mentioned)

10. **Input:** "Planetary science and exploration"
    **Concepts:** `["planetary science", "planetary exploration"]` (Rule 3: Mandatory split + shared modifier)

11. **Input:** "Research methodology in social sciences"
    **Concepts:** `["social sciences"]` (Rule 4: Exclude methodological term)

12. **Input:** "Business and government relations"
    **Concepts:** `["business relations", "government relations"]` (Rule 3: Mandatory split. Assumes "business relations" is the intended split concept based on context/standard usage. Alternatively, might be `["business", "government relations"]`)

13. **Input:** "Department of African Studies, research areas include arts and languages."
    **Concepts:** `["african arts", "african languages"]` (Rule 6: Contextual disambiguation)

14. **Input:** "topic: public policy & development in africa, department: department of african studies"
    **Concepts:** `["african public policy", "african development"]` (Rule 3: Split on '&'; Rule 6: Apply 'african' context from 'in africa' and department)

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
    # Since entries are now grouped by department, they should all have the same department
    # Extract the department from the first entry
    department = entries[0].get("department", "") if entries else ""
    
    # Format list of research areas/content
    entries_str = "\n".join([f"- {entry.get('research_area', entry.get('content', ''))}" for entry in entries])
  
    return f"""Extract academic concepts and research fields related to the department of **{department}**.

**Critical Instructions:**
1.  **Focus on Core Concepts:** Extract ONLY concepts that are core subfields or direct components of the main **{department}** discipline. Do NOT extract concepts that belong primarily to *other* disciplines, even if mentioned.
    *   Example: If the department is Aeronautics and the text mentions "artificial intelligence", do NOT extract "artificial intelligence", as it's a Computer Science concept, not an Aeronautics subfield.
2.  Follow all extraction rules specified in the initial System Prompt (regarding established fields, explicit mentions, decomposition, exclusions, etc.).
3.  **CRITICAL REMINDER (Rule 6):** You MUST apply contextual modifiers (like the department name or related adjectives) to disambiguate common terms (e.g., history, policy, imaging, modeling). Do NOT output ambiguous terms if context is available.
4.  Output MUST be valid JSON in the specified format.

**Output Format:**
{{
    "extractions": [
        {{
            "source": "topic_name_from_list_below",
            "concepts": ["concept1", "concept2"],
            "department": "{department}"
        }}
        # ... one entry for each topic processed
    ]
}}

**Input Topics from the Department of {department}:**
{entries_str}

Provide the JSON output containing the extracted concepts based *only* on the rules and the provided topics for the **{department}** department."""

def process_batch_worker(args: tuple) -> List[List[ConceptExtraction]]:
    """Worker function for parallel processing"""
    batch, num_attempts = args
    try:
        prompt = build_prompt(batch)
        
        # Run multiple LLM attempts with different providers/models
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
                                                    department=item.get('department', '')
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
            department = extraction.department
            
            # Initialize source in combined results if not present
            if source not in combined_results:
                combined_results[source] = {
                    "department": department,
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
        combined_results: Dictionary mapping sources to departments and concepts with counts
        agreement_threshold: Minimum number of times a concept must appear
        
    Returns:
        Dictionary mapping sources to filtered data
    """
    filtered_results = {}
    
    for source, data in combined_results.items():
        department = data["department"]
        concepts_with_counts = data["concepts"]
        
        # Keep only concepts that meet the agreement threshold
        filtered_concepts = {
            concept for concept, count in concepts_with_counts.items() 
            if count >= agreement_threshold
        }
        
        # Add filtered concepts to results
        if filtered_concepts:
            filtered_results[source] = {
                "department": department,
                "concepts": filtered_concepts
            }
    
    return filtered_results

def process_batches_parallel(
    batches: List[List[Dict[str, Any]]],
    num_attempts: int = processing_config.llm_attempts
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
                        combined_results, processing_config.concept_agreement_threshold
                    )
                    
                    # Merge into all_results
                    for source, data in filtered_results.items():
                        department = data["department"]
                        concepts = data["concepts"]
                        
                        if source not in all_results:
                            all_results[source] = {
                                "department": department,
                                "concepts": set()
                            }
                        
                        all_results[source]["concepts"].update(concepts)
                
                # Apply cooldown periodically
                if i > 0 and i % processing_config.cooldown_frequency == 0:
                    time.sleep(processing_config.cooldown_period)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue
                
    return all_results

def prepare_entries_from_research_areas(research_areas_file: str, metadata_file: str) -> List[Dict[str, Any]]:
    """
    Prepare entries for concept extraction from research areas file and metadata
    
    Args:
        research_areas_file: Path to research areas file
        metadata_file: Path to metadata file
        
    Returns:
        List of entries ready for concept extraction
    """
    # Read research areas
    with open(research_areas_file, 'r', encoding='utf-8') as f:
        research_areas = [line.strip() for line in f if line.strip()]
    
    # Read metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Extract level1_to_research_areas mapping from metadata
    level1_to_research_areas = metadata.get('level1_to_research_areas_mapping', {})
    
    # Create entries that match the expected format
    entries = []
    
    for level1_term, areas in level1_to_research_areas.items():
        if not areas:
            continue
            
        # Create an entry for EACH research area individually to preserve the mapping
        for area in areas:
            entry = {
                "department": level1_term,
                "content": area,
                "research_area": area  # Track the specific research area
            }
            entries.append(entry)
    
    # If there are research areas without level1 mapping, add them as a general entry
    unmapped_areas = []
    for area in research_areas:
        mapped = False
        for areas_list in level1_to_research_areas.values():
            if area in areas_list:
                mapped = True
                break
        if not mapped:
            unmapped_areas.append(area)
    
    for area in unmapped_areas:
        entry = {
            "department": "Research Areas",
            "content": area,
            "research_area": area  # Track the specific research area
        }
        entries.append(entry)
    
    return entries

def load_college_department_mapping():
    """
    Function kept for backward compatibility, but returns an empty dictionary
    """
    logger.info("College mapping is no longer used in this step")
    return {}

def main():
    """Main execution function"""
    try:
        logger.info("Starting concept extraction")

        # Prepare input data from the new step 0 format
        entries = prepare_entries_from_research_areas(level_config.get_step_input_file(1), level_config.get_step_metadata_file(1)_STEP0)
        logger.info(f"Prepared {len(entries)} entries from research areas")

        # Load the college-department mapping from level 1
        college_department_mapping = load_college_department_mapping()
        logger.info(f"Loaded {len(college_department_mapping)} department-college mappings")

        # Process in batches
        all_extracted_concepts = []
        source_concept_mapping = {}  # Map sources to concepts
        research_area_concept_mapping = {}  # Map research areas to concepts
        department_research_area_mapping = {}  # Map departments to research areas
        department_concept_mapping = {}  # Map departments to concepts
        
        # Group entries by department
        entries_by_department = {}
        for entry in entries:
            department = entry["department"]
            if department not in entries_by_department:
                entries_by_department[department] = []
            entries_by_department[department].append(entry)

        # Create batches for each department separately
        batches = []
        for department, dept_entries in entries_by_department.items():
            dept_batches = chunk(dept_entries, processing_config.batch_size)
            batches.extend(dept_batches)
        logger.info(f"Split into {len(batches)} batches of size up to {processing_config.batch_size}, grouped by department")
        
        # Process batches in parallel and get filtered results
        extracted_data = process_batches_parallel(batches, processing_config.llm_attempts)
        
        # Build mappings from the filtered results
        for source, data in extracted_data.items():
            department = data["department"]
            concepts = data["concepts"]
            
            # Update source mapping
            if source not in source_concept_mapping:
                source_concept_mapping[source] = set()
            source_concept_mapping[source].update(concepts)
            
            # Format department name
            full_dept_name = department
            if department and not department.lower().startswith("department of "):
                full_dept_name = f"department of {department}"
                
            # Map research area to concept
            if source not in research_area_concept_mapping:
                research_area_concept_mapping[source] = set()
            research_area_concept_mapping[source].update(concepts)
            
            # Map department to research area
            if full_dept_name not in department_research_area_mapping:
                department_research_area_mapping[full_dept_name] = set()
            department_research_area_mapping[full_dept_name].add(source)
            
            # Map department to concept
            if full_dept_name not in department_concept_mapping:
                department_concept_mapping[full_dept_name] = set()
            department_concept_mapping[full_dept_name].update(concepts)

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
                if count >= processing_config.keyword_appearance_threshold
            ]
        )

        logger.info(f"Extracted {len(verified_concepts)} verified concepts")

        # Create output directories if needed
        for path in [level_config.get_step_output_file(1), level_config.get_step_metadata_file(1)]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(level_config.get_step_output_file(1), "w", encoding="utf-8") as f:
            for concept in verified_concepts:
                f.write(f"{concept}\n")

        # Update mappings that use department names to add "department of" prefix
        department_research_area_mapping_prefixed = {}
        for dept, areas in department_research_area_mapping.items():
            formatted_dept = dept
            if not formatted_dept.lower().startswith("department of "):
                formatted_dept = f"department of {dept}"
            department_research_area_mapping_prefixed[formatted_dept] = areas
        
        # Also update department_concept_mapping with prefixed departments
        department_concept_mapping_prefixed = {}
        for dept, concepts in department_concept_mapping.items():
            formatted_dept = dept
            if not formatted_dept.lower().startswith("department of "):
                formatted_dept = f"department of {dept}"
            department_concept_mapping_prefixed[formatted_dept] = concepts
            
        # Create clean, consolidated mappings for metadata
        source_concept_mapping_clean = {
            source: sorted(list(concepts))
            for source, concepts in source_concept_mapping.items()
        }
        
        research_area_concept_mapping_clean = {
            area: sorted(list(concepts))
            for area, concepts in research_area_concept_mapping.items()
        }
        
        department_research_area_mapping_clean = {
            dept: sorted(list(areas))
            for dept, areas in department_research_area_mapping_prefixed.items()
        }
        
        department_concept_mapping_clean = {
            dept: sorted(list(concepts))
            for dept, concepts in department_concept_mapping_prefixed.items()
        }
        
        with open(level_config.get_step_metadata_file(1), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "input_count": len(entries),
                        "output_count": len(verified_concepts),
                        "batch_size": processing_config.batch_size,
                        "num_workers": Config.NUM_WORKERS,
                        "llm_attempts": processing_config.llm_attempts,
                        "concept_agreement_threshold": processing_config.concept_agreement_threshold,
                        "concept_frequency_threshold": processing_config.keyword_appearance_threshold,
                        "providers_and_models": "random selection of OpenAI default/mini and Gemini default/mini",
                        "temperature": 0.3,
                    },
                    "source_concept_mapping": source_concept_mapping_clean,
                    "research_area_concept_mapping": research_area_concept_mapping_clean,
                    "department_research_area_mapping": department_research_area_mapping_clean,
                    "department_concept_mapping": department_concept_mapping_clean,
                    "concept_frequencies": {
                        concept: count 
                        for concept, count in concept_counts.items()
                        if count >= processing_config.keyword_appearance_threshold
                    }
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        # Create a CSV file with the complete hierarchy
        csv_file = os.path.join(Path(level_config.get_step_metadata_file(1)).parent, "lv2_s1_hierarchical_concepts.csv")
        try:
            with open(csv_file, "w", encoding="utf-8") as f:
                # Write header with topic, department, and concept only
                f.write("topic,department,concept\n")
                
                # Create a set to track unique entries and avoid duplicates
                unique_entries = set()
                
                # Process data to write CSV
                for source, concepts in source_concept_mapping.items():
                    # Find departments for this source
                    departments = []
                    for dept, areas in department_research_area_mapping.items():
                        if source in areas:
                            departments.append(dept)
                    
                    # If no department found, use "General"
                    if not departments:
                        departments = ["General"]
                        
                    for department in departments:
                        for concept in concepts:
                            # Create a unique key to avoid duplicates
                            entry_key = (source, department, concept)
                            if entry_key in unique_entries:
                                continue
                            unique_entries.add(entry_key)
                            
                            # Escape commas and quotes in fields
                            topic_str = source.replace('"', '""')
                            dept_str = department.replace('"', '""')
                            concept_str = concept.replace('"', '""')
                            
                            # Add quotes if the field contains commas
                            topic_csv = f'"{topic_str}"' if ',' in topic_str else topic_str
                            dept_csv = f'"{dept_str}"' if ',' in dept_str else dept_str
                            concept_csv = f'"{concept_str}"' if ',' in concept_str else concept_str
                            
                            f.write(f"{topic_csv},{dept_csv},{concept_csv}\n")
                
            logger.info(f"Hierarchical concept relationships saved to {csv_file} with {len(unique_entries)} unique entries")
            
        except Exception as e:
            logger.error(f"Failed to write CSV file: {str(e)}")

        logger.info("Concept extraction completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
