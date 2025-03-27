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

class Config:
    """Configuration for concept extraction"""
    # Updated to use the new step 0 output files
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s0_research_areas.txt")
    META_FILE_STEP0 = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s0_metadata.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s1_metadata.json")
    BATCH_SIZE = 10  # Increased for better parallelization
    NUM_WORKERS = 64  # Number of parallel workers
    KW_APPEARANCE_THRESH = 1
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
    source: str = Field(description="Topic area (research area) from which concepts are extracted")
    concepts: List[str] = Field(description="List of extracted concepts")
    department: str = Field(description="Department associated with the source topic area")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")

SYSTEM_PROMPT = """
You are an expert in extracting academic concepts from research area descriptions.

IMPORTANT GUIDELINES:
- Extract ONLY concepts that are EXPLICITLY mentioned in the text of academic research area descriptions. If you receive an input that is a course, please only extract the concepts that are related to the course. Do not include course numbers or other identifiers.
- Decompose a compound concept into its individual components. Example: "Developmental and stem cell biology" should be decomposed into "developmental biology" and "stem cell biology".
- Do NOT infer or add related concepts that are not directly mentioned in the text.
- Do NOT extract generic terms (e.g., "course", "program", "degree", "credits").
- Do NOT extract administrative terms (e.g., "syllabus", "prerequisite", "assignment").
- Do NOT extract organizational descriptors (e.g., "department", "faculty", "laboratory").
- Do NOT include acronyms, proper nouns, or location-specific terms.
- Do NOT include navigation elements like "back", "next", "home", "click here".
- Do NOT include administrative topics like "course description", "how to apply".
- Do NOT include any non-academic content.

Examples:
1. Research Area: "Data science: machine learning"
   INCORRECT: "data science", "machine learning", "artificial intelligence", "neural networks", "algorithms"
   Explanation: "artificial intelligence", "neural networks", and "algorithms" are not explicitly mentioned.
   CORRECT: "data science", "machine learning"

2. Research Area: "Environ 362LS Aquatic Field Ecology (F)"
   INCORRECT: "aquatic ecology", "field ecology", "aquatic ecosystems", "aquatic organisms"
   Explanation: Only "aquatic field ecology" is mentioned, not the additional inferred concepts.
   CORRECT: "aquatic field ecology"
   
3. Research Area: "Introduction to Computer Science I"
   INCORRECT: "introduction to computer science", "computer science I"
   Explanation: "introduction to computer science" or "computer science I" contains the course number and other identifiers.
   CORRECT: "computer science"
   
4. Research Area: "Biochemistry understanding how molecules function and malfunction in living systems."
   INCORRECT: "biochemistry understanding", "molecules function and malfunction"
   Explanation: "biochemistry understanding" is not explicitly a research area, and "molecules function and malfunction" should be decomposed into "molecular function" and "molecular malfunction".
   CORRECT: "biochemistry", "molecular function", "molecular malfunction"

Please only extract the concepts without any additional explanation. 
Note that some input might not contain any concepts, in which case you should return an empty list.
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
        department = entry.get("department", "")
        content = entry.get("content", "")
        research_area = entry.get("research_area", content)  # Use research_area if available, otherwise content

        if content:
            entries_str += f"\nDepartment: {department}\nTopic: {research_area}\nContent:\n{content}\n---\n"

    return f"""Extract research topics and concepts from these academic topic areas.
Return the concepts in this exact JSON format:
{{
    "extractions": [
        {{
            "source": "topic_name",
            "concepts": ["concept1", "concept2"],
            "department": "department name"
        }}
    ]
}}

IMPORTANT: Ensure your response is valid JSON. All strings must be properly quoted and all objects must be properly closed.
Double-check your JSON before returning it.

Topic areas to process:
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
                                            department=item.get('department', '')
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
                            department = entry.get("department", "")
                            content = entry.get("content", "")
                            
                            if not content:
                                continue
                                
                            simple_prompt = f"""Extract research topics and concepts from this academic department description.
Return ONLY a comma-separated list of concepts, nothing else.

Department: {department}
Content:
{content[:5000]}  # Limit content length
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
                                    source = entry.get("research_area", entry.get("content", ""))
                                    extraction = ConceptExtraction(
                                        source=source,
                                        concepts=concepts,
                                        department=department
                                    )
                                    all_extractions.append(extraction)
                                    logger.info(f"Extracted {len(concepts)} concepts from {source} using simplified approach")
                            except Exception as simple_error:
                                logger.error(f"Failed simplified extraction for {department}: {str(simple_error)}")
                        
                        if all_extractions:
                            logger.info(f"Successfully extracted concepts for {len(all_extractions)} sources using simplified approach")
                            return all_extractions
                
                except Exception as manual_error:
                    logger.error(f"Failed manual extraction attempt: {str(manual_error)}")
                    
                # Last resort: try to extract concepts directly from the content without JSON
                try:
                    logger.info("Attempting direct concept extraction as last resort")
                    all_extractions = []
                    
                    for entry in batch:
                        department = entry.get("department", "")
                        content = entry.get("content", "")
                        
                        if not content:
                            continue
                            
                        # Extract keywords directly from content
                        import re
                        from collections import Counter
                        
                        # Simple keyword extraction based on frequency and patterns
                        words = re.findall(r'\b[a-z][a-z-]{2,}\b', content.lower())
                        word_counts = Counter(words)
                        
                        # Filter common words and get top keywords
                        common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'are', 'from', 'have', 'has', 'our', 'their', 'your', 'its'}
                        keywords = [word for word, count in word_counts.most_common(30) if word not in common_words and count > 1]
                        
                        # Look for noun phrases (simple pattern matching)
                        noun_phrases = re.findall(r'\b[a-z][a-z-]+ (?:of|and|in|for) [a-z][a-z-]+\b', content.lower())
                        noun_phrase_counts = Counter(noun_phrases)
                        top_phrases = [phrase for phrase, count in noun_phrase_counts.most_common(10) if count > 1]
                        
                        # Combine keywords and phrases
                        concepts = list(set(keywords + top_phrases))[:20]  # Limit to 20 concepts
                        
                        if concepts:
                            source = entry.get("research_area", entry.get("content", ""))
                            extraction = ConceptExtraction(
                                source=source,
                                concepts=concepts,
                                department=department
                            )
                            all_extractions.append(extraction)
                            logger.info(f"Extracted {len(concepts)} concepts from {source} using direct extraction")
                    
                    if all_extractions:
                        logger.info(f"Successfully extracted concepts for {len(all_extractions)} sources using direct extraction")
                        return all_extractions
                
                except Exception as direct_error:
                    logger.error(f"Failed direct extraction attempt: {str(direct_error)}")
            
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
    level1_to_research_areas = metadata.get('level1_to_research_areas', {})
    
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
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")
            
        logger.info("Starting concept extraction")

        # Prepare input data from the new step 0 format
        entries = prepare_entries_from_research_areas(Config.INPUT_FILE, Config.META_FILE_STEP0)
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
                    # The source is now the topic area / research area
                    source = extraction.source
                    
                    # Find the corresponding entry to get research area and department
                    matching_entries = [
                        entry for entry in entries 
                        if entry.get("research_area", entry.get("content", "")) == source
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
                        research_area = entry.get("research_area", entry.get("content", ""))
                        department = entry.get("department", "")
                        
                        # Format department name
                        full_dept_name = department
                        if department and not department.lower().startswith("department of "):
                            full_dept_name = f"department of {department}"
                            
                        # Map research area to concept
                        if research_area not in research_area_concept_mapping:
                            research_area_concept_mapping[research_area] = set()
                        research_area_concept_mapping[research_area].update(extraction.concepts)
                        
                        # Map department to research area
                        if full_dept_name not in department_research_area_mapping:
                            department_research_area_mapping[full_dept_name] = set()
                        department_research_area_mapping[full_dept_name].add(research_area)
                        
                        # Map department to concept
                        if full_dept_name not in department_concept_mapping:
                            department_concept_mapping[full_dept_name] = set()
                        department_concept_mapping[full_dept_name].update(extraction.concepts)

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
                    "research_area_concept_mapping": research_area_concept_mapping_clean,
                    "department_research_area_mapping": department_research_area_mapping_clean,
                    "department_concept_mapping": department_concept_mapping_clean,
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
        csv_file = os.path.join(Path(Config.META_FILE).parent, "lv2_s1_hierarchical_concepts.csv")
        try:
            with open(csv_file, "w", encoding="utf-8") as f:
                # Write header with topic, department, and concept only
                f.write("topic,department,concept\n")
                
                # Create a set to track unique entries and avoid duplicates
                unique_entries = set()
                
                # Validate mappings before writing
                validated_mappings = {}  # research_area -> (concepts, departments)
                
                # Process each extraction to validate the relationships
                for extraction in all_extracted_concepts:
                    source = extraction.source
                    concepts = extraction.concepts
                    department = extraction.department
                    
                    # Skip if source is empty or too generic
                    if not source or source in ["Research Areas", "General"]:
                        continue
                    
                    # Format department name
                    department = department or "General"
                    if not department.lower().startswith("department of ") and department != "General":
                        department = f"department of {department}"
                    
                    # Add to validated mappings
                    if source not in validated_mappings:
                        validated_mappings[source] = (set(concepts), {department})
                    else:
                        existing_concepts, existing_departments = validated_mappings[source]
                        existing_concepts.update(concepts)
                        existing_departments.add(department)
                
                # Write the validated mappings to CSV
                for research_area, (concepts, departments) in validated_mappings.items():
                    for dept in departments:
                        # Check if this is a valid relationship
                        # Skip excessively broad department-concept associations
                        if len(concepts) > 30 and dept != "General":
                            logger.warning(f"Skipping excessive mapping: {research_area} -> {dept} with {len(concepts)} concepts")
                            continue
                            
                        for concept in concepts:
                            # Check if concept is related to the research area
                            # Simple validation by checking if key terms from research area appear in concept or vice versa
                            research_terms = set(research_area.lower().split())
                            concept_terms = set(concept.lower().split())
                            
                            # Check for a minimum overlap or if the concept is clearly within the research area
                            common_terms = research_terms.intersection(concept_terms)
                            if (not common_terms and not any(term in research_area.lower() for term in concept.lower().split())
                                and not any(term in concept.lower() for term in research_area.lower().split())):
                                # Only log this, don't skip - allows for valid non-obvious relationships
                                logger.debug(f"Potentially unrelated: {research_area} -> {concept}")
                            
                            # Create a unique key to avoid duplicates
                            entry_key = (research_area, dept, concept)
                            if entry_key in unique_entries:
                                continue
                            unique_entries.add(entry_key)
                            
                            # Escape commas and quotes in fields
                            topic_str = research_area.replace('"', '""')
                            dept_str = dept.replace('"', '""')
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
