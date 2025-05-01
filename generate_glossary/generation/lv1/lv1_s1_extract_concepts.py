import os
import sys
import time
import json
import re
from numpy.random import choice
import threading
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydash import chunk
from tqdm import tqdm
from functools import lru_cache

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM, InferenceResult
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Load environment variables
load_dotenv()

def setup_logging(name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with the specified level, name, and optional file output."""
    # Create logger with the given name
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    # Silence verbose logs from underlying libraries
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("google.cloud.aiplatform").setLevel(logging.WARNING)
    logging.getLogger('google_genai.models').setLevel(logging.WARNING)
    logging.getLogger('google_genai').setLevel(logging.WARNING)
    logging.getLogger("vertexai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger

# Initialize logger
logger = setup_logging("lv1.s1")

# Process-local storage for LLM instances
_process_local = threading.local()

class Config:
    """Configuration for concept extraction"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_department_names.txt")
    METADATA_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_metadata.json")  # Add metadata file path
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s1_extracted_concepts.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s1_metadata.json")
    BATCH_SIZE = 20
    NUM_LLM_ATTEMPTS = 5
    CONCEPT_AGREEMENT_THRESH = 3
    NUM_WORKERS = 8  # Increased number of parallel workers
    CHUNK_SIZE = 100  # Size of chunks for parallel processing
    MIN_DEPT_LENGTH = 5  # Minimum length of a valid department name
    MAX_DEPT_LENGTH = 100  # Maximum length of a valid department name
    COOLDOWN_PERIOD = 1  # Seconds between batches
    COOLDOWN_FREQUENCY = 10  # Number of batches before cooldown
    LOG_LEVEL = "INFO"  # Default log level
    LOG_FILE = None     # Default to no separate log file

class QuotaExceededError(Exception):
    """Raised when the API quota is exceeded."""
    pass

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

def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Get or initialize LLM with specified provider (process-local)"""
    if not hasattr(_process_local, 'llm'):
        # logger.info(f"Initializing LLM with provider: {provider} for process {os.getpid()}")
        _process_local.llm = init_llm(provider, model)
    return _process_local.llm

def init_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """Initialize LLM with specified provider and model"""
    if not provider:
        provider = Provider.OPENAI  # Default to OpenAI
        
    selected_model = OPENAI_MODELS[model] if provider == Provider.OPENAI else GEMINI_MODELS[model]
        
    return LLMFactory.create_llm(
        provider=provider,
        model=selected_model,
        temperature=0.3
    )

def get_random_llm_config() -> Tuple[str, str]:
    """Get a random LLM provider and model configuration"""
    provider = choice([Provider.OPENAI, Provider.GEMINI], p=[0.5, 0.5], replace=False)
    model = choice([ "pro", "default", "mini"], p=[0.2, 0.5, 0.3], replace=False)
    return provider, model

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

SYSTEM_PROMPT = """You are an expert in academic research classification with deep knowledge of research domains, scientific disciplines, and academic departments. Your task is to extract academic concepts and fields of study from university department names.

**CORE TASK:** Extract ONLY well-established, recognized academic disciplines, sub-disciplines, or broad research fields explicitly mentioned in the provided department names.

**CRITICAL GUIDELINES:**

**1. Filter for Legitimacy FIRST:**
    *   Before extracting, determine if the input string represents a legitimate academic department or field of study.
    *   **IGNORE and return an empty concept list `[]` for:** Administrative offices ("Dean's Office"), navigation elements ("Home", "Back", "Links"), non-academic services ("Student Loans", "Career Services"), generic placeholders ("100%"), irrelevant text, etc.

**2. Focus on Established Academic Fields:**
    *   **EXTRACT:** Standardized academic disciplines (e.g., "computer science", "molecular biology"), sub-disciplines ("cognitive psychology"), and broad research areas ("machine learning", "quantum physics").
    *   **DO NOT EXTRACT:**
        *   Specific, narrow research topics (e.g., "ecological effects of anthropogenic nutrient pulses").
        *   Course designations/titles (e.g., "accounting workshop", "biology seminar", "chemistry I-IV", "advanced topics in machine learning").
        *   Degree paths/types (e.g., "B.S. finance", "Ph.D. economics", "master's program", "bachelor's", "master's", "doctoral"). (Extract the *field name* itself if present, e.g., "B.S. in Computer Science" -> "computer science").
        *   Project or paper titles if too specific.
        *   Course labels with part numbers or levels ("chemistry I-IV", "physics part 2").

**3. Strict Explicit Mention Rule:**
    *   Extract ONLY concepts DIRECTLY and EXPLICITLY mentioned in the input text.
    *   **DO NOT INFER:** Do not add related concepts, subfields, or specializations that are *not* present verbatim in the text. (e.g., If text says "electrical engineering", do NOT add "electronics" or "power systems").

**4. Clean Administrative Terms:**
    *   For valid departments, remove administrative qualifiers like "department of", "school of", "college of", "program in" before extracting the core concept(s).
        *   Example: "Department of Biology" -> "biology"
        *   Example: "School of Medicine" -> "medicine"
        *   Example: "Ph.D. Program in Biomedical Engineering" -> "biomedical engineering"

**5. Preserve Exact Terms:**
    *   Extract the EXACT TERMS as they appear in the cleaned department name. Do not modify or add specializations not directly mentioned.

**6. MANDATORY Compound Term Decomposition (Split on "and"):**
    *   If a cleaned department name contains concepts joined by "and", **ALWAYS** split them into separate concepts.
    *   If the concepts joined by "and" share a common preceding word or phrase (prefix/modifier), repeat that shared prefix/modifier for **EACH** resulting concept after the split. Apply the *most relevant* shared context.
        *   Example: "department of electrical and computer engineering" -> ["electrical engineering", "computer engineering"] (Shared modifier "engineering" is implied by structure; "electrical" and "computer" are distinct specifiers)
        *   Example: "school of cognitive and behavioral neuroscience" -> ["cognitive neuroscience", "behavioral neuroscience"] (Shared modifier "neuroscience")
        *   Example: "college of aerospace sciences and engineering" -> ["aerospace sciences", "aerospace engineering"] (Shared prefix "aerospace")
        *   Example: "department of biology and chemistry" -> ["biology", "chemistry"] (No shared prefix/modifier immediately preceding 'and' to repeat)
        *   Example: "school of cognitive and behavioral sciences" -> ["cognitive sciences", "behavioral sciences"] (Shared modifier "sciences")
        *   Example: "textile engineering, chemistry and science" -> Should be decomposed considering "textile" as a modifier applying across the related terms: ["textile engineering", "textile chemistry", "textile science"]. The goal is to correctly associate the primary subject ("textile") with all relevant sub-fields listed.

**7. Strict Exclusions List - DO NOT EXTRACT ANY OF THESE:**
    *   **Generic Terms:** "studies", "research", "topics", "areas".
    *   **Administrative/Organizational Terms:** "department", "school", "college", "center", "institute", "program", "office", "services".
    *   **Degree Designations:** "B.S.", "M.S.", "M.A.", "Ph.D.", "bachelor's", "master's", "doctoral". (Remove these, but keep the field name if present).
    *   **Course Identifiers:** "introduction to", "advanced", roman numerals (I, II, III), course numbers, "workshop", "seminar", "laboratory".
    *   **Acronyms/Abbreviations:** Unless it's the universally standard name for the field.
    *   **Proper Nouns/Names:** Unless part of the standard field name.
    *   **Location-Specific Terms.**
    *   **Navigation/Website Terms:** "Back", "Home", "Links", "Visit", "Apply", "Contact".
    *   **ANY concepts that do not appear verbatim in the input text after cleaning.**

**EXAMPLES:**

*   **Input:** "college of communication sciences and disorders"
    *   **Valid Concepts:** `["communication sciences", "communication disorders"]`
    *   **INVALID:** `["speech pathology", "audiology"]` (Not explicitly mentioned)

*   **Input:** "college of electrical and computer engineering"
    *   **Valid Concepts:** `["electrical engineering", "computer engineering"]`
    *   **INVALID:** `["electronics", "digital systems"]` (Not explicitly mentioned)

*   **Input:** "department of biology and chemistry"
    *   **Valid Concepts:** `["biology", "chemistry"]`
    *   **INVALID:** `["molecular biology", "organic chemistry"]` (Not explicitly mentioned)

*   **Input:** "school of cognitive and behavioral sciences"
    *   **Valid Concepts:** `["cognitive sciences", "behavioral sciences"]` (Decomposed shared suffix)

*   **Input:** "Department of B.S. and M.S. Programs in Computer Science"
    *   **Valid Concepts:** `["computer science"]` (Degree types removed)

*   **Input:** "Ph.D. Program in Biomedical Engineering"
    *   **Valid Concepts:** `["biomedical engineering"]` (Degree type removed)

*   **Input:** "college of back" -> **Valid Concepts:** `[]` (Invalid department)
*   **Input:** "college of home" -> **Valid Concepts:** `[]` (Invalid department)
*   **Input:** "college of student loans" -> **Valid Concepts:** `[]` (Non-academic unit)
*   **Input:** "college of dean's office" -> **Valid Concepts:** `[]` (Administrative office)
*   **Input:** "college of academic advisors" -> **Valid Concepts:** `[]` (Administrative unit)

**Output Format:**
Return the results in the specified JSON format. Make sure the overall output is a valid JSON object. **DO NOT** wrap the JSON in markdown code blocks (triple backticks).

```json
{
    "extractions": [
        {"source": "college of {department_name}", "concepts": ["concept1", "concept2"]},
        {"source": "college of {another_department}", "concepts": []} // Empty list for non-departments or if no valid concepts found
    ]
}
```
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

def preprocess_content(content: str) -> str:
    """Preprocess content to reduce size and improve quality"""
    # Remove extra whitespace
    content = " ".join(content.split())
    return content

def build_prompt(sources: List[Dict[str, str]]) -> str:
    """Build prompt for concept extraction"""
    # Extract college from the first entry - all entries in a batch should have the same college
    college = sources[0].get('college', 'Unknown') if sources else 'Unknown'
    
    # Format list of department names
    departments_str = "\n".join([f"- Department: {source['department']}" for source in sources])

    return f"""Extract research topics and concepts from these academic department names from the {college}.
For each department name, FIRST determine if it's an actual academic department or field of study.
If it's not a legitimate academic department (like administrative offices, navigation elements, etc.), 
return an empty list of concepts.

Consider the college context ({college}) when extracting concepts - the concepts should be relevant to both the department and its parent college.

Return the concepts in this exact JSON format WITHOUT any markdown formatting or code blocks:
{{
    "extractions": [
        {{"source": "department_name", "concepts": ["concept1", "concept2"]}},
        {{"source": "another_department", "concepts": []}}  // Empty list for non-departments
    ]
}}

IMPORTANT: DO NOT use markdown formatting with triple backticks. Return ONLY the raw JSON object.

Departments to process:
{departments_str}"""

def process_batch_worker(args: tuple) -> List[List[ConceptExtraction]]:
    """Worker function for parallel processing"""
    batch, num_attempts = args
    try:
        prompt = build_prompt(batch)
        
        # Run multiple LLM attempts with different providers/models
        all_extractions = []
        
        for attempt in range(num_attempts):
            provider, model = get_random_llm_config()
            logger.debug(f"Attempt {attempt+1}/{num_attempts} using {provider}/{model} in process {os.getpid()}")
            
            try:
                # Get process-local LLM instance with specific provider/model
                llm = init_llm(provider, model)
                
                logger.debug(f"Processing batch with {len(batch)} items")
                
                try:
                    logger.debug("Attempting structured extraction with Pydantic model")
                    response = llm.infer(
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        response_model=ConceptExtractionList,
                    )
                    logger.debug(f"Successfully extracted {len(response.text.extractions)} concepts with structured validation")
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
                            logger.debug("Attempting to extract concepts without structured validation")
                            simple_response = llm.infer(
                                prompt=prompt,
                                system_prompt=SYSTEM_PROMPT,
                            )
                            
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Raw response: {simple_response.text[:200]}...")
                            
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
                            logger.debug(f"Found {len(matches)} potential JSON objects in response")
                            
                            for i, match in enumerate(matches):
                                try:
                                    if logger.isEnabledFor(logging.DEBUG):
                                        logger.debug(f"Attempting to parse JSON object {i+1}: {match[:100]}...")
                                    data = json.loads(match)
                                    if 'extractions' in data:
                                        # Convert to ConceptExtraction objects
                                        extractions = []
                                        for item in data['extractions']:
                                            try:
                                                extraction = ConceptExtraction(
                                                    source=item.get('source', ''),
                                                    concepts=item.get('concepts', [])
                                                )
                                                extractions.append(extraction)
                                            except Exception as item_error:
                                                logger.warning(f"Failed to parse extraction item: {str(item_error)}")
                                        
                                        if extractions:
                                            logger.debug(f"Successfully extracted {len(extractions)} concepts manually")
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
    total_extractions = 0
    
    # Process each list of extractions
    for extractions in extractions_list:
        # Process each extraction in the list
        for extraction in extractions:
            source = extraction.source
            total_extractions += len(extraction.concepts)
            
            # Initialize source in combined results if not present
            if source not in combined_results:
                combined_results[source] = {}
                
            # Update concept counts for this source
            for concept in extraction.concepts:
                concept_lower = concept.lower()
                if concept_lower not in combined_results[source]:
                    combined_results[source][concept_lower] = 0
                combined_results[source][concept_lower] += 1
    
    logger.debug(f"Combined {total_extractions} total concept extractions across all LLM runs")
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
    filtered_count = 0
    total_count = 0
    
    for source, concepts in combined_results.items():
        # Keep only concepts that meet the agreement threshold
        total_count += len(concepts)
        filtered_concepts = {
            concept for concept, count in concepts.items() 
            if count >= agreement_threshold
        }
        filtered_count += len(filtered_concepts)
        
        # Add filtered concepts to results
        if filtered_concepts:
            filtered_results[source] = filtered_concepts
    
    logger.debug(f"Filtered {filtered_count} concepts from {total_count} total (threshold={agreement_threshold})")
    return filtered_results

def process_batches_parallel(
    batches: List[List[Dict[str, Any]]],
    num_attempts: int = Config.NUM_LLM_ATTEMPTS
) -> Dict[str, Dict[str, Any]]:
    """Process batches in parallel using ProcessPoolExecutor and filter results"""
    all_results = {}
    
    logger.info(f"Processing {len(batches)} batches using {Config.NUM_WORKERS} workers")
    
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
                    for source, concepts in filtered_results.items():
                        if source not in all_results:
                            all_results[source] = {
                                "concepts": set(),
                                "college": ""  # Will be filled later
                            }
                        
                        all_results[source]["concepts"].update(concepts)
                
                # Apply cooldown periodically
                if i > 0 and i % Config.COOLDOWN_FREQUENCY == 0:
                    logger.debug(f"Applying cooldown for {Config.COOLDOWN_PERIOD} seconds after {i} batches")
                    time.sleep(Config.COOLDOWN_PERIOD)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                continue
                
    logger.info(f"Extracted concepts for {len(all_results)} sources")
    return all_results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Extract academic concepts from department names")
    
    # Logging options
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default=Config.LOG_LEVEL, help="Set logging level")
    parser.add_argument("--log-file", type=str, help="Log to specified file")
    
    # Processing options
    parser.add_argument("--workers", type=int, default=Config.NUM_WORKERS,
                        help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                        help="Batch size for processing")
    parser.add_argument("--attempts", type=int, default=Config.NUM_LLM_ATTEMPTS,
                        help="Number of LLM attempts per batch")
    
    # Input/output options
    parser.add_argument("--input", type=str, default=Config.INPUT_FILE,
                        help="Input file with department names")
    parser.add_argument("--output", type=str, default=Config.OUTPUT_FILE,
                        help="Output file for extracted concepts")
    
    return parser.parse_args()

def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Update config with command line arguments
        Config.LOG_LEVEL = args.log_level
        Config.LOG_FILE = args.log_file
        Config.NUM_WORKERS = args.workers
        Config.BATCH_SIZE = args.batch_size
        Config.NUM_LLM_ATTEMPTS = args.attempts
        Config.INPUT_FILE = args.input
        Config.OUTPUT_FILE = args.output
        
        # Derive metadata and output file paths based on input path
        input_path = Path(Config.INPUT_FILE)
        input_dir = input_path.parent
        
        # Update metadata file path
        if "s0_department_names" in input_path.name:
            Config.METADATA_FILE = os.path.join(input_dir, "lv1_s0_metadata.json")
        
        # Update output file path if necessary
        if Config.OUTPUT_FILE == os.path.join(Config.BASE_DIR, "data/lv1/raw/lv1_s1_extracted_concepts.txt"):
            # Using default path, derive from input path
            output_filename = input_path.name.replace("s0_department_names", "s1_extracted_concepts")
            Config.OUTPUT_FILE = os.path.join(input_dir, output_filename)
        
        # Update metadata output path
        Config.META_FILE = Config.OUTPUT_FILE.replace("_extracted_concepts.txt", "_metadata.json")
        
        # Configure advanced logging if needed
        if Config.LOG_FILE or Config.LOG_LEVEL != "INFO":
            global logger
            logger = setup_logging("lv1.s1", log_level=Config.LOG_LEVEL, log_file=Config.LOG_FILE)
            
        logger.info(f"Starting concept extraction with level={Config.LOG_LEVEL}, workers={Config.NUM_WORKERS}")
        logger.info(f"Input: {Config.INPUT_FILE}")
        logger.info(f"Output: {Config.OUTPUT_FILE}")

        # Read input sources and metadata
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            raw_sources = [line.strip() for line in f.readlines()]
        logger.info(f"Read {len(raw_sources)} department names")
        
        # Read metadata to get college-department mapping
        with open(Config.METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        level0_to_departments = metadata.get("level0_to_departments", {})
        
        # Create mappings:
        # 1. dept_original_form: maps normalized department names to their original form
        # 2. department_to_college: maps original department names to colleges
        # 3. norm_dept_to_college: maps normalized department names to colleges
        dept_original_form = {}
        department_to_college = {}
        norm_dept_to_college = {}
        
        # First create mappings from original department names to colleges
        for college, departments in level0_to_departments.items():
            for dept in departments:
                # Store original department to college mapping
                if dept not in department_to_college:
                    department_to_college[dept] = []
                if college not in department_to_college[dept]:
                    department_to_college[dept].append(college)
                
                # Create normalized form mapping
                norm_dept = normalize_department_name(dept.lower())
                
                # Map normalized form to original form (for reference)
                dept_original_form[norm_dept] = dept
                
                # Map normalized department to college
                if norm_dept not in norm_dept_to_college:
                    norm_dept_to_college[norm_dept] = []
                if college not in norm_dept_to_college[norm_dept]:
                    norm_dept_to_college[norm_dept].append(college)
        
        logger.info(f"Created college mappings for {len(department_to_college)} departments and {len(norm_dept_to_college)} normalized departments")
        
        # Create a mapping from normalized forms of raw sources to original forms
        # This helps capture actual department names from the raw input
        normalized_to_original = {}
        for source in raw_sources:
            if not source:  # Skip empty lines
                continue
                
            # Normalize and lowercase the source for consistent matching
            norm_source = normalize_department_name(source.lower())
            
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
        for norm_source, orig_source in normalized_to_original.items():
            colleges = []
            
            # Check if the normalized form exists in our normalized department mapping
            if norm_source in norm_dept_to_college:
                colleges = norm_dept_to_college[norm_source]
                logger.debug(f"Found college mapping for normalized '{norm_source}': {colleges}")
            
            # If not found, check if the original form exists in our direct department mapping
            elif orig_source in department_to_college:
                colleges = department_to_college[orig_source]
                logger.debug(f"Found college mapping for original '{orig_source}': {colleges}")
            
            # If still not found, check if removing "department of" prefix helps
            elif norm_source.startswith("department of "):
                shorter_norm = norm_source[len("department of "):]
                if shorter_norm in norm_dept_to_college:
                    colleges = norm_dept_to_college[shorter_norm]
                    logger.debug(f"Found college mapping for shortened '{shorter_norm}': {colleges}")
            
            # If still not found, try to infer from keywords
            if not colleges:
                inferred_college = infer_college_from_department(orig_source)
                if inferred_college != "Unknown":
                    colleges = [inferred_college]
                    logger.debug(f"Inferred college for '{orig_source}': {inferred_college}")
                else:
                    colleges = ["Unknown"]
                    logger.debug(f"No college mapping found for '{orig_source}', using 'Unknown'")
            
            # For multiple colleges, use the first one for simplicity in processing
            # The full list is preserved in all_colleges for metadata
            primary_college = colleges[0] if colleges else "Unknown"
            
            sources.append({
                "department": orig_source,
                "college": primary_college,
                "all_colleges": colleges,
                "normalized_department": norm_source  # Store normalized form for reference
            })
        
        logger.info(f"Normalized and deduplicated to {len(sources)} unique department names with college context")
        
        # Perform initial filtering to remove obviously invalid departments
        pre_filtered = [s for s in sources if is_valid_department(s["department"])]
        logger.info(f"Pre-filtered to {len(pre_filtered)} potential departments (removed {len(sources) - len(pre_filtered)} invalid entries)")

        # Group entries by college for better context
        entries_by_college = {}
        for entry in pre_filtered:
            college = entry["college"]
            if college not in entries_by_college:
                entries_by_college[college] = []
            entries_by_college[college].append(entry)
        
        # Create batches for each college separately to maintain context
        batches = []
        for college, college_entries in entries_by_college.items():
            college_batches = chunk(college_entries, Config.BATCH_SIZE)
            batches.extend(college_batches)
            logger.debug(f"Created {len(college_batches)} batches for college: {college}")
        
        logger.info(f"Split into {len(batches)} batches of size up to {Config.BATCH_SIZE}, grouped by college")
        
        # Process batches in parallel and get filtered results
        department_data = process_batches_parallel(batches, Config.NUM_LLM_ATTEMPTS)
        logger.info(f"Processed {len(department_data)} departments from {len(batches)} batches")

        # Add college information to department data, ensuring correct mapping
        for dept, data in department_data.items():
            # Normalize department name for consistent lookup
            norm_dept = normalize_department_name(dept.lower())
            
            # Find colleges using multiple lookup strategies
            colleges = []
            
            # First, try to find by looking up the normalized department name
            if norm_dept in norm_dept_to_college:
                colleges = norm_dept_to_college[norm_dept]
                logger.debug(f"Found colleges for '{dept}' using normalized lookup: {colleges}")
            
            # Next, try with original department name
            elif dept in department_to_college:
                colleges = department_to_college[dept]
                logger.debug(f"Found colleges for '{dept}' using direct lookup: {colleges}")
            
            # Try to find in our sources list (most reliable since it was used for batching)
            if not colleges:
                for source in sources:
                    if source["department"] == dept or source["normalized_department"] == norm_dept:
                        colleges = source["all_colleges"]
                        logger.debug(f"Found colleges for '{dept}' in sources list: {colleges}")
                        break
            
            # If still no college found, try with department prefix variations
            if not colleges and not norm_dept.startswith("department of "):
                prefixed_norm = "department of " + norm_dept
                if prefixed_norm in norm_dept_to_college:
                    colleges = norm_dept_to_college[prefixed_norm]
                    logger.debug(f"Found colleges for '{dept}' using prefixed lookup: {colleges}")
            
            # If still not found, try without department prefix
            if not colleges and norm_dept.startswith("department of "):
                unprefixed_norm = norm_dept[len("department of "):]
                if unprefixed_norm in norm_dept_to_college:
                    colleges = norm_dept_to_college[unprefixed_norm]
                    logger.debug(f"Found colleges for '{dept}' using unprefixed lookup: {colleges}")
            
            # Last resort - infer from keywords
            if not colleges:
                inferred_college = infer_college_from_department(dept)
                if inferred_college != "Unknown":
                    colleges = [inferred_college]
                    logger.debug(f"Inferred college for '{dept}': {inferred_college}")
                else:
                    colleges = ["Unknown"]
                    logger.debug(f"No college found for '{dept}', using 'Unknown'")
            
            # Use first college as primary for data structure
            primary_college = colleges[0] if colleges else "Unknown"
            data["college"] = primary_college
            data["all_colleges"] = colleges

        # Extract all concepts and apply frequency threshold
        all_concepts = [
            concept.lower()
            for dept_data in department_data.values()
            for concept in dept_data["concepts"]
        ]
        concept_counts = Counter(all_concepts)
        verified_concepts = sorted([concept for concept in concept_counts])
        
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

        # Create clean, consolidated source concept mapping
        source_concept_mapping = {
            dept: sorted(list(data["concepts"]))
            for dept, data in department_data.items()
        }
        
        # Create consistent college names with "college of" prefix
        def format_college_name(college):
            if not college:
                return "Unknown"
            college_lower = college.lower()
            if not college_lower.startswith("college of "):
                return f"college of {college_lower}"
            return college_lower
        
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
                        "num_workers": Config.NUM_WORKERS,
                        "llm_attempts": Config.NUM_LLM_ATTEMPTS,
                        "concept_agreement_threshold": Config.CONCEPT_AGREEMENT_THRESH,
                        "providers_and_models": "random selection of OpenAI default/mini and Gemini default/mini",
                        "temperature": 0.3,
                    },
                    "source_concept_mapping": source_concept_mapping,
                    "concept_frequencies": {
                        concept: count 
                        for concept, count in concept_counts.items()
                    },
                    # Save full department data with concepts and colleges
                    "department_data": {
                        normalize_department_name(dept): {
                            "college": [format_college_name(c) for c in (data["all_colleges"] if isinstance(data["all_colleges"], list) else [data["college"]])],
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
                
                # Create a set to track unique entries and avoid duplicates
                unique_entries = set()
                
                # Write data rows
                for dept, data in department_data.items():
                    normalized_dept = normalize_department_name(dept)
                    colleges = data["all_colleges"] if "all_colleges" in data else [data["college"]]
                    
                    for college in colleges:
                        # Format college name consistently with "college of" prefix
                        college_fmt = format_college_name(college)
                        
                        for concept in sorted(list(data["concepts"])):
                            # Create a unique key to avoid duplicates
                            entry_key = (normalized_dept, college_fmt, concept)
                            if entry_key in unique_entries:
                                continue
                            unique_entries.add(entry_key)
                            
                            # Escape commas and quotes in fields
                            dept_str = normalized_dept.replace('"', '""')
                            college_str = college_fmt.replace('"', '""')
                            concept_str = concept.replace('"', '""')
                            
                            # Add quotes if the field contains commas
                            dept_csv = f'"{dept_str}"' if ',' in dept_str else dept_str
                            college_csv = f'"{college_str}"' if ',' in college_str else college_str
                            concept_csv = f'"{concept_str}"' if ',' in concept_str else concept_str
                            
                            f.write(f"{dept_csv},{college_csv},{concept_csv}\n")
        
            logger.info(f"Department-concept-college relationships saved to {csv_file} with {len(unique_entries)} unique entries")
        except Exception as e:
            logger.error(f"Failed to write CSV file: {str(e)}")

        logger.info("Concept extraction completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
