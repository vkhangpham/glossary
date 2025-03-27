import os
import sys
import time
import json
import asyncio
import re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk
from tqdm import tqdm

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM, InferenceResult
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv1.s1")

class Config:
    """Configuration for concept extraction"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_department_names.txt")
    METADATA_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_metadata.json")  # Add metadata file path
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s1_metadata.json")
    BATCH_SIZE = 20
    NUM_LLM_ATTEMPTS = 1
    KW_APPEARANCE_THRESH = 1
    MAX_WORKERS = 4  # Number of parallel workers
    CHUNK_SIZE = 100  # Size of chunks for parallel processing
    MIN_DEPT_LENGTH = 5  # Minimum length of a valid department name
    MAX_DEPT_LENGTH = 100  # Maximum length of a valid department name

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

def infer_college_from_department(dept_name: str) -> str:
    """
    Attempt to infer college from department name based on keywords
    
    Args:
        dept_name: Department name to analyze
        
    Returns:
        Inferred college name or "Unknown" if no match
    """
    dept_lower = dept_name.lower()
    
    # Define keyword mappings for colleges
    college_keywords = {
        "business": ["business", "management", "finance", "accounting", "marketing", "economics"],
        "engineering": ["engineering", "computer science", "electrical", "mechanical", "civil", "biomedical"],
        "education": ["education", "teaching", "curriculum", "instruction", "educational", "pedagogy"],
        "arts": ["art", "music", "theater", "design", "creative", "fine arts", "performing arts"],
        "humanities": ["humanities", "philosophy", "history", "literature", "language", "culture", "religion"],
        "medicine": ["medicine", "medical", "clinical", "doctor", "surgery", "physician"],
        "nursing": ["nursing", "nurse", "healthcare"],
        "health sciences": ["health", "nutrition", "dietetics", "kinesiology", "exercise", "physical therapy", "occupational therapy"],
        "law": ["law", "legal", "justice", "criminal justice", "paralegal"],
        "natural sciences": ["biology", "chemistry", "physics", "mathematics", "geology", "astronomy"],
        "social sciences": ["social", "psychology", "sociology", "anthropology", "political science", "economics"],
        "public health": ["public health", "epidemiology", "biostatistics", "environmental health"]
    }
    
    # Check for keyword matches
    for college, keywords in college_keywords.items():
        if any(keyword in dept_lower for keyword in keywords):
            return "college of " + college
            
    # Special cases for majors that don't contain obvious keywords
    if "pharmacy" in dept_lower:
        return "college of pharmacy"
    if "fire science" in dept_lower or "paramedicine" in dept_lower:
        return "college of health sciences"
    if "child development" in dept_lower:
        return "college of education"
    
    return "Unknown"

class ConceptExtraction(BaseModel):
    """Base model for concept extraction"""
    source: str = Field(description="Source text being processed (in format 'college of {department}')")
    concepts: List[str] = Field(description="List of extracted concepts")

class ConceptExtractionList(BaseModel):
    """Model for batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConceptExtractionList":
        """Create model from JSON string with better error handling"""
        try:
            # Clean markdown code blocks from the input
            cleaned_json = json_str.strip()
            
            # Remove markdown code block markers if present
            if cleaned_json.startswith("```"):
                # Extract content between triple backticks
                match = re.search(r'```(?:\w+)?\n(.*?)```', cleaned_json, re.DOTALL)
                if match:
                    cleaned_json = match.group(1).strip()
                else:
                    # Try simpler extraction (just remove starting/ending backticks)
                    cleaned_json = re.sub(r'^```\w*\n?', '', cleaned_json)
                    cleaned_json = re.sub(r'```$', '', cleaned_json)
                    cleaned_json = cleaned_json.strip()
            
            # Now parse the cleaned JSON
            return cls.model_validate_json(cleaned_json)
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            logger.debug(f"Problematic JSON: {json_str}")
            raise

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
of research domains, scientific disciplines, and academic departments.

Your task is to extract research topics and concepts from academic department names.
IMPORTANT: ONLY extract concepts from entries that are legitimate academic departments or fields of study. 
IGNORE entries that are not academic departments (such as "Home", "Back", "Links", administrative offices, etc).

Guidelines for extraction:
1. FIRST determine if the input is an actual academic department or discipline. If not, return an empty list of concepts.
2. For valid departments, extract specific research domains that represent academic focus areas
3. Include both broad fields and specialized subfields
4. Decompose compound terms into individual concepts where appropriate
5. Ensure each concept is a well-defined academic or research term

DO NOT include:
- Generic terms (e.g., studies, research)
- Administrative terms (e.g., department, school)
- Organizational descriptors (e.g., center, institute)
- Acronyms or abbreviations
- Proper nouns or names
- Location-specific terms
- Navigation elements (e.g., "Back", "Home", "Links")
- Administrative units (e.g., "Dean's Office", "Student Services")
- Non-academic programs (e.g., "Student Loans", "Career Services")

IMPORTANT: When returning JSON, do not wrap it in markdown code blocks with triple backticks. 
Return only the raw JSON object.

Example valid departments:
"college of communication sciences and disorders"
Valid concepts:
- communication sciences
- communication disorders
- speech pathology
- audiology

"college of electrical and computer engineering"
Valid concepts:
- electrical engineering
- computer engineering
- electronics
- digital systems

Example invalid entries (return empty concept list for these):
"college of back"
"college of home"
"college of student loans"
"college of dean's office"
"college of 100%"
"college of academic advisors"
"college of visit the college"
"""

def is_valid_department(dept: str) -> bool:
    """
    Pre-filter obviously invalid department names before sending to LLM
    
    Args:
        dept: Department name to check
        
    Returns:
        Boolean indicating if department is potentially valid
    """
    if not dept or len(dept) < Config.MIN_DEPT_LENGTH or len(dept) > Config.MAX_DEPT_LENGTH:
        return False
        
    # List of terms that indicate the entry is not an academic department
    invalid_terms = [
        "back", "home", "links", "100%", "welcome", "dean's office", "student services",
        "contact us", "click here", "learn more", "find us", "directions to", "login", "intranet",
        "academic advisors", "visit", "quick links", "faculty directory", "staff directory",
        "online portal", "register", "tuition", "apply", "admission", "commencement"
    ]
    
    # Check if department name contains any invalid terms
    dept_lower = dept.lower()
    if any(term in dept_lower for term in invalid_terms):
        return False
        
    # Check if department name is too short after cleaning
    words = re.sub(r'[^\w\s]', '', dept).split()
    if len(words) < 2:
        return False
        
    return True

def build_prompt(sources: List[Dict[str, str]]) -> str:
    """Build prompt for concept extraction"""
    sources_str = "\n".join(f"- Department: {source['department']}\n  College: {source['college']}" for source in sources)

    return f"""Extract research topics and concepts from these academic department names.
For each department name, FIRST determine if it's an actual academic department or field of study.
If it's not a legitimate academic department (like administrative offices, navigation elements, etc.), 
return an empty list of concepts.

Consider the college context when extracting concepts - the concepts should be relevant to both the department and its parent college.

Return the concepts in this exact JSON format WITHOUT any markdown formatting or code blocks:
{{
    "extractions": [
        {{"source": "department_name", "concepts": ["concept1", "concept2"]}},
        {{"source": "another_department", "concepts": []}}  // Empty list for non-departments
    ]
}}

IMPORTANT: DO NOT use markdown formatting with triple backticks. Return ONLY the raw JSON object.

Departments to process:
{sources_str}"""

async def process_batch_async(
    sources: List[Dict[str, str]], 
    provider: Optional[str] = None,
    num_attempts: int = 1
) -> List[ConceptExtractionList]:
    """
    Process a batch of sources asynchronously
    
    Args:
        sources: List of dicts with 'department' and 'college' keys
        provider: Optional provider name
        num_attempts: Number of attempts to make
        
    Returns:
        List of ConceptExtractionList objects
    """
    llm = init_llm(provider)
    responses = []
    
    for attempt in range(num_attempts):
        try:
            # Build prompt with department and college context
            prompt = build_prompt(sources)
            
            # Get response from LLM
            logger.debug(f"Sending batch of {len(sources)} sources to LLM (attempt {attempt+1}/{num_attempts})")
            response = llm.infer(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT
            )
            
            # Parse response
            try:
                # Use our helper method for better error handling
                parsed = ConceptExtractionList.from_json(response.text)
                
                # Verify that all sources are accounted for and match original departments
                # Map extraction sources back to original department names
                original_sources = {s["department"]: s for s in sources}
                for extraction in parsed.extractions:
                    # Make sure the source matches one of our original departments
                    # This maintains the mapping between extracted concepts and original department names
                    for dept in original_sources:
                        if extraction.source.lower() == dept.lower():
                            extraction.source = dept  # Use the original formatting
                            break
                
                responses.append(parsed)
                logger.info(f"Successfully parsed response with {len(parsed.extractions)} extractions")
                break  # If successful, no need for further attempts
            except Exception as e:
                logger.warning(f"Failed to parse response on attempt {attempt+1}: {str(e)}")
                
                # Try fallback parsing on last attempt
                if attempt == num_attempts - 1:
                    logger.info("Attempting fallback parsing")
                    try:
                        # First try to extract JSON from markdown code blocks
                        if "```" in response.text:
                            # Extract content between triple backticks
                            match = re.search(r'```(?:\w+)?\n(.*?)```', response.text, re.DOTALL)
                            if match:
                                json_str = match.group(1).strip()
                                try:
                                    data = json.loads(json_str)
                                    if 'extractions' in data:
                                        # Map extraction sources back to original department names
                                        for item in data['extractions']:
                                            source_text = item.get('source', '')
                                            for dept in original_sources:
                                                if source_text.lower() == dept.lower():
                                                    item['source'] = dept  # Use the original formatting
                                                    break
                                        
                                        parsed = ConceptExtractionList(
                                            extractions=[
                                                ConceptExtraction(
                                                    source=item.get('source', ''),
                                                    concepts=item.get('concepts', [])
                                                )
                                                for item in data['extractions']
                                            ]
                                        )
                                        responses.append(parsed)
                                        logger.info(f"Markdown JSON parsing successful with {len(parsed.extractions)} extractions")
                                        break
                                except Exception as json_error:
                                    logger.debug(f"Failed to parse extracted markdown JSON: {str(json_error)}")
                        
                        # If we get here, try RegEx-based extraction as fallback
                        # Look for JSON-like patterns
                        json_pattern = r'\{[\s\S]*?"extractions"[\s\S]*?\}'
                        json_matches = re.findall(json_pattern, response.text)
                        
                        if json_matches:
                            for json_str in json_matches:
                                try:
                                    data = json.loads(json_str)
                                    if 'extractions' in data:
                                        # Map extraction sources back to original department names
                                        for item in data['extractions']:
                                            source_text = item.get('source', '')
                                            for dept in original_sources:
                                                if source_text.lower() == dept.lower():
                                                    item['source'] = dept  # Use the original formatting
                                                    break
                                                    
                                        parsed = ConceptExtractionList(
                                            extractions=[
                                                ConceptExtraction(
                                                    source=item.get('source', ''),
                                                    concepts=item.get('concepts', [])
                                                )
                                                for item in data['extractions']
                                            ]
                                        )
                                        responses.append(parsed)
                                        logger.info(f"Regex JSON parsing successful with {len(parsed.extractions)} extractions")
                                        break
                                except Exception as json_error:
                                    logger.debug(f"Failed to parse regex-extracted JSON: {str(json_error)}")
                                    continue
                        
                        # If still no success, try to extract a list of dictionaries
                        if not responses:
                            # Look for list-like patterns
                            list_pattern = r'\[\s*\{[^\[\]]*\}\s*(?:,\s*\{[^\[\]]*\}\s*)*\]'
                            list_matches = re.findall(list_pattern, response.text)
                            
                            if list_matches:
                                for list_str in list_matches:
                                    try:
                                        items = json.loads(list_str)
                                        if isinstance(items, list) and all('source' in item and 'concepts' in item for item in items):
                                            # Map extraction sources back to original department names
                                            for item in items:
                                                source_text = item.get('source', '')
                                                for dept in original_sources:
                                                    if source_text.lower() == dept.lower():
                                                        item['source'] = dept  # Use the original formatting
                                                        break
                                                        
                                            parsed = ConceptExtractionList(
                                                extractions=[
                                                    ConceptExtraction(
                                                        source=item.get('source', ''),
                                                        concepts=item.get('concepts', [])
                                                    )
                                                    for item in items
                                                ]
                                            )
                                            responses.append(parsed)
                                            logger.info(f"List extraction parsing successful with {len(parsed.extractions)} extractions")
                                            break
                                    except Exception:
                                        continue
                    except Exception as fallback_error:
                        logger.error(f"Fallback parsing failed: {str(fallback_error)}")
                
        except Exception as e:
            logger.warning(f"LLM inference failed on attempt {attempt+1}: {str(e)}")
            
    return responses

def process_chunk(chunk_data: tuple[List[Dict[str, str]], str, int]) -> Dict[str, Dict[str, Any]]:
    """
    Process a chunk of sources in parallel
    
    Args:
        chunk_data: Tuple of (sources, provider, num_attempts) where sources is a list of dicts with 'department' and 'college' keys
        
    Returns:
        Dictionary mapping department names to their data (concepts and college)
    """
    sources, provider, num_attempts = chunk_data
    source_data = {}
    
    # Pre-filter sources to remove obviously invalid departments
    filtered_sources = [src for src in sources if is_valid_department(src["department"])]
    logger.debug(f"Pre-filtered from {len(sources)} to {len(filtered_sources)} potential departments")
    
    # Create mapping of departments to colleges for this chunk
    dept_to_college = {}
    for src in filtered_sources:
        dept = src["department"]
        # Store mapping for both original and normalized department names
        if dept not in dept_to_college:
            dept_to_college[dept] = []
        
        # Add college(s) to the mapping
        if "all_colleges" in src and src["all_colleges"]:
            dept_to_college[dept].extend(src["all_colleges"])
        else:
            dept_to_college[dept].append(src["college"])
            
        # Also add normalized mapping
        normalized_dept = normalize_department_name(dept)
        if normalized_dept != dept:
            if normalized_dept not in dept_to_college:
                dept_to_college[normalized_dept] = []
            if "all_colleges" in src and src["all_colleges"]:
                dept_to_college[normalized_dept].extend(src["all_colleges"])
            else:
                dept_to_college[normalized_dept].append(src["college"])
    
    # Process batches within the chunk
    batches = chunk(filtered_sources, Config.BATCH_SIZE)
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
                for extraction in response.extractions:
                    # Skip sources with empty concept lists (non-departments)
                    if not extraction.concepts:
                        continue
                        
                    # Get department name and college
                    dept_name = extraction.source
                    # Also try normalized department name for college lookup
                    normalized_dept = normalize_department_name(dept_name)
                    
                    # Try to find colleges
                    colleges = dept_to_college.get(dept_name, [])
                    
                    # Try with normalized name if original not found
                    if not colleges and normalized_dept != dept_name:
                        colleges = dept_to_college.get(normalized_dept, [])
                    
                    # If still no college found, try to infer from department name
                    if not colleges:
                        inferred_college = infer_college_from_department(dept_name)
                        if inferred_college != "Unknown":
                            colleges = [inferred_college]
                        else:
                            colleges = ["Unknown"]
                        
                    # Use first college as primary
                    primary_college = colleges[0] if colleges else "Unknown"
                    
                    if dept_name not in source_data:
                        source_data[dept_name] = {
                            "concepts": set(),
                            "college": primary_college,
                            "all_colleges": colleges
                        }
                    source_data[dept_name]["concepts"].update(extraction.concepts)
                    
        finally:
            loop.close()
            
    return source_data

async def main_async():
    """Async main execution function"""
    try:
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")

        logger.info("Starting concept extraction")

        # Read input sources and metadata
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            raw_sources = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(raw_sources)} department names")
        
        # Read metadata to get college-department mapping
        with open(Config.METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        level0_to_departments = metadata.get("level0_to_departments", {})
        
        # Create department to college mapping
        # Store a list of colleges for each department to handle multi-college departments
        department_to_college = {}
        for college, departments in level0_to_departments.items():
            for dept in departments:
                if dept not in department_to_college:
                    department_to_college[dept] = []
                department_to_college[dept].append(college)
        
        # Also create normalized department mappings
        normalized_department_to_college = {}
        for dept, colleges in department_to_college.items():
            norm_dept = normalize_department_name(dept)
            if norm_dept != dept:
                if norm_dept not in normalized_department_to_college:
                    normalized_department_to_college[norm_dept] = []
                normalized_department_to_college[norm_dept].extend(colleges)
        
        # Merge the normalized mappings back into the main mapping
        for dept, colleges in normalized_department_to_college.items():
            if dept not in department_to_college:
                department_to_college[dept] = []
            department_to_college[dept].extend(colleges)
        
        # Create a mapping from normalized form to original form to keep the best representation
        normalized_to_original = {}
        for source in raw_sources:
            if not source:  # Skip empty lines
                continue
                
            # Normalize the text
            norm_source = normalize_text(source)
            
            # If this normalized form already exists, keep the shorter or more readable version
            if norm_source in normalized_to_original:
                current = normalized_to_original[norm_source]
                # Prefer versions without parentheses or special characters if lengths are similar
                if (len(source) < len(current) * 0.8 or 
                    (len(source) < len(current) * 1.2 and 
                     sum(c in "()[]{}:" for c in current) > sum(c in "()[]{}:" for c in source))):
                    normalized_to_original[norm_source] = source
            else:
                normalized_to_original[norm_source] = source
        
        # Get the unique original forms with their college context
        sources = []
        for dept in normalized_to_original.values():
            # Try to find college using normalized department name for lookup
            normalized_dept = normalize_department_name(dept)
            colleges = None
            
            # First try with original name
            if dept in department_to_college:
                colleges = department_to_college[dept]
            # Then try with normalized name
            elif normalized_dept in department_to_college:
                colleges = department_to_college[normalized_dept]
            else:
                # Try to infer college from department name
                inferred_college = infer_college_from_department(dept)
                if inferred_college != "Unknown":
                    colleges = [inferred_college]
                else:
                    colleges = ["Unknown"]
                
            # For multiple colleges, use the first one for simplicity
            # The full list will be preserved in the metadata
            primary_college = colleges[0] if colleges else "Unknown"
            
            sources.append({
                "department": dept,
                "college": primary_college,
                "all_colleges": colleges
            })
        logger.info(f"Normalized and deduplicated to {len(sources)} unique department names with college context")
        
        # Perform initial filtering
        pre_filtered = [s for s in sources if is_valid_department(s["department"])]
        logger.info(f"Pre-filtered to {len(pre_filtered)} potential departments (removed {len(sources) - len(pre_filtered)} invalid entries)")

        # Split sources into chunks for parallel processing
        source_chunks = list(chunk(pre_filtered, Config.CHUNK_SIZE))
        chunk_data = [(chunk, provider, Config.NUM_LLM_ATTEMPTS) for chunk in source_chunks]
        
        # Process chunks in parallel
        department_data = {}
        with ProcessPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_chunk, data)
                for data in chunk_data
            ]
            
            # Collect results
            for future in tqdm(futures, desc="Processing chunks"):
                chunk_results = future.result()
                department_data.update(chunk_results)

        # Create a clean, consolidated source concept mapping
        source_concept_mapping = {
            dept: data["concepts"]
            for dept, data in department_data.items()
        }
        
        # Make sure we have the correct college for each department
        for dept, data in department_data.items():
            # Try to find the college using both original and normalized department names
            normalized_dept = normalize_department_name(dept)
            
            if dept in department_to_college:
                # Use the original mapping if available
                data["college"] = department_to_college[dept]
            elif normalized_dept in department_to_college:
                # Use normalized name for lookup
                data["college"] = department_to_college[normalized_dept]
            elif data["college"] == "Unknown":
                # Try to infer college from department name
                inferred_college = infer_college_from_department(dept)
                if inferred_college != "Unknown":
                    data["college"] = inferred_college
            # Keep existing college value if neither matches

        # Extract all concepts and apply frequency threshold
        all_concepts = [
            concept.lower()
            for dept_data in department_data.values()
            for concept in dept_data["concepts"]
        ]
        concept_counts = Counter(all_concepts)
        verified_concepts = sorted([
            concept
            for concept, count in concept_counts.items()
            if count >= Config.KW_APPEARANCE_THRESH
        ])
        
        logger.info(f"Extracted {len(verified_concepts)} verified concepts")
        logger.info(f"From {len(department_data)} valid academic departments")

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
                        "input_count": len(raw_sources),
                        "normalized_count": len(sources),
                        "pre_filtered_count": len(pre_filtered),
                        "valid_departments_count": len(department_data),
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
                    },
                    "college_concept_mapping": {
                        college: sorted(list(set(
                            concept
                            for dept, data in department_data.items()
                            if data["college"] == college  # This assumes college is a string
                            for concept in data["concepts"]
                        )))
                        # Get unique primary colleges as strings (not lists)
                        for college in set(
                            data["college"] if isinstance(data["college"], str) else data["college"][0] if data["college"] else "Unknown" 
                            for data in department_data.values()
                        )
                    },
                    # Add department_to_college mapping to preserve relationship
                    "department_to_college": {
                        dept: data["college"] if isinstance(data["college"], str) else data["college"][0] if data["college"] else "Unknown"
                        for dept, data in department_data.items()
                    },
                    # Add full mapping with all colleges
                    "department_to_all_colleges": {
                        dept: data.get("all_colleges", [data["college"]]) if isinstance(data.get("all_colleges"), list) else [data["college"]]
                        for dept, data in department_data.items()
                    },
                    # Save full department data with concepts and college
                    "department_data": {
                        dept: {
                            "college": data["college"],
                            "concepts": sorted(list(data["concepts"]))
                        }
                        for dept, data in department_data.items()
                    }
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        # Create an additional CSV file with all the department-concept-college relationships
        csv_file = os.path.join(Path(Config.META_FILE).parent, "lv1_s1_department_concepts.csv")
        try:
            with open(csv_file, "w", encoding="utf-8") as f:
                # Write header
                f.write("department,college,concept\n")
                
                # Write data rows
                for dept, data in department_data.items():
                    college = data["college"]
                    # Make sure college is a string, not a list
                    if isinstance(college, list):
                        college = college[0] if college else "Unknown"
                    
                    for concept in sorted(list(data["concepts"])):
                        # Escape commas and quotes in fields
                        dept_str = dept.replace('"', '""')
                        college_str = college.replace('"', '""')
                        concept_str = concept.replace('"', '""')
                        
                        # Add quotes if the field contains commas
                        dept_csv = f'"{dept_str}"' if ',' in dept else dept_str
                        college_csv = f'"{college_str}"' if ',' in college else college_str
                        concept_csv = f'"{concept_str}"' if ',' in concept else concept_str
                        
                        f.write(f"{dept_csv},{college_csv},{concept_csv}\n")
        
            logger.info(f"Department-concept-college relationships saved to {csv_file}")
        except Exception as e:
            logger.error(f"Failed to write CSV file: {str(e)}")

        logger.info("Concept extraction completed successfully")

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
