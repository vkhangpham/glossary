import os
import sys
import asyncio
import json
import time
import re
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dotenv import load_dotenv
import aiohttp
import argparse

# Fix import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
# No longer need Provider import - using direct provider strings

# Import shared web search utilities
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists

# Load environment variables and setup logging
load_dotenv('.env')
logger = setup_logger("lv1.s0")
random.seed(42)

# ----- CONSTANTS -----
# Processing constants
MAX_SEARCH_RESULTS = 50
MAX_CONCURRENT_REQUESTS = 5
BATCH_SIZE = 100  # Process multiple level 0 terms in a single batch
MAX_SEARCH_QUERIES = 100  # Maximum number of queries to send in a single web_search_bulk call

# LLM configuration
NUM_LLM_ATTEMPTS = 3  # Run the LLM extraction 3 times
AGREEMENT_THRESHOLD = 2  # Departments must appear in at least 2 responses
DEFAULT_LLM_MODEL_TYPES = ["default", "mini", "nano"] # Default model types for attempts
DEFAULT_MIN_SCORE_FOR_LLM = 0.3

# Search queries - consolidated in one place
SEARCH_QUERIES = [
    "site:.edu college of {term} list of departments and divisions",
    "site:.edu school of {term} list of departments and divisions",
    "site:.edu faculty of {term} list of departments and divisions",
]

# Department-related keywords for enhanced filtering
DEPARTMENT_KEYWORDS = [
    "department", "school", "college", "faculty", "division", "program", 
    "institute", "center", "studies", "sciences", "arts", "research",
    "engineering", "education", "technology", "humanities", "social",
    "business", "finance", "accounting", "marketing", "management",
    "economics", "computer science", "software engineering", "physics",
    "chemistry", "biology", "mathematics", "statistics", "data science",
    "art", "music", "dance", "theater", "visual arts", "fine arts",
    "humanities", "social sciences", "psychology", "sociology", "anthropology",
    "geography", "history", "philosophy", "literature", "language", "linguistics",
    "communication", "media", "journalism", "public relations", "advertising",
    "international relations", "political science", "economics", "finance",
    "marketing", "management", "operations research", "supply chain", "logistics",
    "human resources", "labor relations", "industrial relations", "organizational behavior",
    r"[\w\s&,\'-]+ studies",
    r"[\w\s&,\'-]+ sciences"
]

# Anti-keywords indicating non-department text
NON_DEPARTMENT_KEYWORDS = [
    "login", "sign in", "register", "apply", "admission", "contact", 
    "about", "home", "sitemap", "search", "privacy", "terms", "copyright",
    "accessibility", "careers", "jobs", "employment", "staff", "faculty",
    "directory", "phone", "email", "address", "location", "directions",
    "map", "parking", "visit", "tour", "events", "news", "calendar"
]

# Department pattern matches for regex matching
DEPARTMENT_PATTERNS = [
    r"(?:department|school|college|faculty|division|program|institute) of [\w\s&,'-]+",
    r"[\w\s&,'-]+ (?:department|school|college|faculty|division|program|institute)",
    r"[\w\s&,'-]+ studies",
    r"[\w\s&,'-]+ sciences"
]

# LLM system prompt template
DEPARTMENT_VALIDATION_SYSTEM_PROMPT_TEMPLATE = """You are an expert in academic institution organization and department structures.

Your task is to analyze a provided list and extract ONLY the departments that are EXPLICITLY and DIRECTLY under the umbrella of The College of {term}.

IMPORTANT: You must be EXTREMELY STRICT about this. Generic academic subjects like "Philosophy" or language departments like "Spanish and Portuguese" should NOT be included unless they are EXPLICITLY stated to be part of The College of {term} in particular.

Instructions:
1. Return a JSON array of valid department names from the list: ["department1", "department2", ...]
2. Include ONLY departments that EXPLICITLY belong to The College of {term}
3. Exclude ALL of the following:
   - Website menu items, navigation sections, or non-relevant content
   - Generic academic departments that could exist in many colleges (unless explicitly labeled as belonging to The College of {term})
   - Departments that belong to a DIFFERENT college (not The College of {term})
   - Sub-departments or research areas that are too specific (not direct departments)
   - Generic categories, administrative sections, or non-department items
   - Programs, centers, or institutes that are not academic departments

Guidelines:
- Think about what departments would logically be part of a College of {term}
- Check if the item sounds like a proper academic department name
- Only include departments that are plausibly part of the College of {term} based on subject matter
- In case of ambiguity, be conservative and EXCLUDE items rather than including them

Examples:

Example 1 - For The College of Engineering:
Input List: ["Civil Engineering", "Mechanical Engineering", "Electrical Engineering", "Physics", "Chemistry", "Biomechanics Lab", "Undergraduate Programs", "Admissions", "Faculty Resources", "Chemical Engineering", "Research Opportunities"]
Output: ["Civil Engineering", "Mechanical Engineering", "Electrical Engineering", "Chemical Engineering"]
Explanation: Only included engineering departments, excluded science departments, labs, administrative sections, and generic programs.

Example 2 - For The College of Arts:
Input List: ["Theater Arts", "Music", "Fine Arts", "Dance", "Visual Arts", "Philosophy", "Spanish and Portuguese", "News", "Events", "Apply Now", "Art History", "Student Resources", "Performing Arts Center"]
Output: ["Theater Arts", "Music", "Fine Arts", "Dance", "Visual Arts", "Art History"]
Explanation: Only included arts-specific departments; excluded generic humanities, language departments, administrative items, and non-department entities.

Example 3 - For The College of Business:
Input List: ["Marketing", "Finance", "Accounting", "Management", "Economics", "Computer Science", "Software Engineering", "Faculty Resources", "MBA Program", "Business Analytics", "International Business", "Student Organizations", "Career Services"]
Output: ["Marketing", "Finance", "Accounting", "Management", "Economics", "Business Analytics", "International Business"]
Explanation: Only included business departments, excluded computing departments and administrative/student service sections.

THIS IS CRITICAL: Return ONLY the JSON array with valid departments, nothing else. If no valid departments are found, return an empty array [].
Do not include explanations, introductions, or other text.

Remember: Be SELECTIVE. It is better to exclude a department if there's any doubt about whether it belongs specifically to The College of {term}."""

# Use centralized configuration
LEVEL = 1
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

def read_level0_terms(input_path: str) -> List[str]:
    """Read level 0 terms from input file"""
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            terms = [line.strip() for line in file.readlines() if line.strip()]
        logger.info(f"Successfully read {len(terms)} level 0 terms")
        return terms
    except Exception as e:
        logger.error(f"Failed to read level 0 terms: {str(e)}", exc_info=True)
        raise


def clean_department_name(item: str) -> str:
    """Clean a department name"""
    # Remove common prefixes like "Department of", "School of"
    item = item.strip()
    item = re.sub(r'^(Department|School|College|Faculty|Division|Program|Institute) of ', '', item, flags=re.IGNORECASE)
    # Remove trailing numbers, parenthetical info
    item = re.sub(r'\s*\(\d+\).*$', '', item)
    item = re.sub(r'\s*\d+\s*$', '', item)
    # Remove URLs
    item = re.sub(r'http\S+', '', item)
    # Clean whitespace
    item = ' '.join(item.split())
    return item


def score_department_list(items: List[str], metadata: Dict[str, Any], context_term: str) -> float:
    """Score a department list based on various heuristics"""
    # Custom department scoring logic with weights
    weights = {
        "keyword": 0.3,       # Higher weight for keywords
        "structure": 0.15,
        "pattern": 0.25,      # Higher weight for patterns
        "non_term": 0.15,
        "consistency": 0.05,
        "size": 0.05,
        "html_type": 0.05
    }
    
    # Use the common scoring function from list_extractor
    return score_list(
        items=items,
        metadata=metadata,
        context_term=context_term,
        keywords=DEPARTMENT_KEYWORDS,
        scoring_weights=weights
    )


def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        level_config.data_dir / "cache",
        level_config.data_dir / "raw_search",
        level_config.data_dir / "raw_results",
        level_config.data_dir / "detailed_meta",
        os.path.dirname(level_config.get_step_output_file(0)),
        os.path.dirname(level_config.get_step_metadata_file(0))
    ]
    
    logger.info(f"BASE_DIR: {str(level_config.data_dir.parent.parent)}")
    logger.info(f"LV0_INPUT_FILE: {str(get_level_config(0).get_final_file())}")
    logger.info(f"OUTPUT_FILE: {level_config.get_step_output_file(0)}")
    logger.info(f"META_FILE: {level_config.get_step_metadata_file(0)}")
    logger.info(f"CACHE_DIR: {level_config.data_dir / "cache"}")
    logger.info(f"RAW_SEARCH_DIR: {level_config.data_dir / "raw_search"}")
    logger.info(f"RAW_RESULTS_DIR: {level_config.data_dir / "raw_results"}")
    logger.info(f"DETAILED_META_DIR: {level_config.data_dir / "detailed_meta"}")
    
    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise


def save_raw_url_results(level0_term: str, url_to_lists: Dict[str, List[List[str]]]):
    """Save raw URL results for a term to a JSON file
    
    Args:
        level0_term: The level 0 term
        url_to_lists: Dictionary mapping URLs to lists of lists of strings
    """
    try:
        # Sanitize filename
        safe_filename = re.sub(r'[\/:*?"<>|]', '_', level0_term) + "_url_lists.json"
        output_path = os.path.join(level_config.data_dir / "raw_results", safe_filename)
        
        # Convert data to a serializable format
        serializable_data = {
            "level0_term": level0_term,
            "urls": {}
        }
        
        # Organize by URL for better readability
        for url, lists in url_to_lists.items():
            serializable_data["urls"][url] = lists
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)
            
        logger.info(f"Saved raw URL results for '{level0_term}' to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save raw URL results for '{level0_term}': {str(e)}", exc_info=True)


async def run_multiple_llm_extractions(
    all_extracted_lists_raw: List[Dict[str, Any]],
    level0_term: str,
    filter_config: FilterConfig,
    num_attempts: int = NUM_LLM_ATTEMPTS,
    agreement_threshold: int = AGREEMENT_THRESHOLD,
    logger: Optional[Any] = None,
    model_types: List[str] = DEFAULT_LLM_MODEL_TYPES
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[List[str]]]:
    """
    Run multiple LLM extractions and select departments that appear in multiple responses.
    Each attempt uses a randomly selected provider (Gemini/OpenAI) and model type.

    Args:
        all_extracted_lists_raw: Raw extracted lists to process
        level0_term: The level 0 term being processed
        filter_config: Configuration for filtering
        num_attempts: Number of LLM extraction attempts
        agreement_threshold: Minimum number of appearances required for a department
        logger: Optional logger
        model_types: List of model types to use for each attempt

    Returns:
        Tuple containing:
        - final_lists: Combined lists with items that meet the agreement threshold
        - llm_candidates: The candidates sent to the LLM (from the first run)
        - llm_results: The consolidated results from multiple LLM runs
    """
    if not all_extracted_lists_raw:
        return [], [], []

    logger.info(f"Running multiple LLM extractions ({num_attempts}) for '{level0_term}'")

    all_results = []
    all_candidates = []
    all_raw_results = []

    # Ensure model_types list has enough entries for num_attempts
    if len(model_types) < num_attempts:
        model_types = model_types * (num_attempts // len(model_types) + 1)
    current_model_types = model_types[:num_attempts]
    available_providers = [Provider.GEMINI, Provider.OPENAI] # Define available providers

    for attempt in range(num_attempts):
        # Randomly select provider and model type for this attempt
        current_provider = random.choice(available_providers)
        current_model_type = random.choice(model_types) # Randomly choose from the list

        # Create a new filter config with the current provider and model type
        current_filter_config = FilterConfig(
            scoring_fn=filter_config.scoring_fn,
            clean_item_fn=filter_config.clean_item_fn,
            provider=current_provider,
            use_llm_validation=filter_config.use_llm_validation,
            binary_llm_decision=filter_config.binary_llm_decision,
            binary_system_prompt=filter_config.binary_system_prompt, # Use the existing prompt
            min_score_for_llm=filter_config.min_score_for_llm,
            model_type=current_model_type # Pass specific model type for this attempt
        )

        logger.info(f"Attempt {attempt+1}/{num_attempts} using RANDOM provider: {current_provider}, model: {current_model_type}")

        try:
            # Run the standard filter_lists on the raw lists
            # Make sure to use the same raw list input for each attempt
            final_lists, llm_candidates, llm_results = await filter_lists(
                all_extracted_lists_raw, level0_term, current_filter_config, logger
            )

            # Store the results from this attempt
            all_results.append(final_lists)
            if attempt == 0:
                # Store candidates from first attempt only
                all_candidates = llm_candidates
            # Accumulate results from all attempts
            all_raw_results.extend(llm_results)

            logger.info(f"Attempt {attempt+1} found {len(final_lists)} verified lists/departments")

        except Exception as e:
            logger.error(f"Error in extraction attempt {attempt+1} for '{level0_term}': {str(e)}", exc_info=True)
            # Continue with other attempts

    # Consolidate results from multiple runs
    all_items = []
    for final_lists_attempt in all_results:
        for lst in final_lists_attempt:
             # Handle cases where lst might be a list of strings or a single string
             if isinstance(lst, list):
                 all_items.extend([item for item in lst if isinstance(item, str)])
             elif isinstance(lst, str):
                 all_items.append(lst)

    # Count occurrences of each item (case-insensitive)
    item_counts = {}
    original_casing = {}
    for item in all_items:
        item_lower = item.lower()
        if item_lower not in item_counts:
            item_counts[item_lower] = 0
            original_casing[item_lower] = item # Store original casing
        item_counts[item_lower] += 1

    # Filter items by agreement threshold, restoring original casing
    agreed_items = [original_casing[item_lower] for item_lower, count in item_counts.items() if count >= agreement_threshold]

    logger.info(f"Found {len(agreed_items)} departments meeting agreement threshold ({agreement_threshold}) for '{level0_term}'")

    # Format results for return - create a single list containing all agreed items
    final_consolidated_list = [agreed_items] if agreed_items else []

    # Return the candidates from the first run and all raw results collected
    return final_consolidated_list, all_candidates, all_raw_results


async def process_level0_term(level0_term: str,
                              provider: Optional[str] = None,
                              session: Optional[Any] = None,
                              min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                              model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                              num_llm_attempts: int = NUM_LLM_ATTEMPTS,
                              agreement_threshold: int = AGREEMENT_THRESHOLD,
                              prefetched_search_results: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Process a single level 0 term to extract department names"""
    logger.info(f"Processing level 0 term: {level0_term} (LLM Min Score: {min_score_for_llm}, Models: {model_types}, Attempts: {num_llm_attempts}, Agree: {agreement_threshold})")
    
    term_details = {
        "level0_term": level0_term,
        "all_urls": [],
        "raw_extracted_lists": [],
        "llm_io_pairs": [],
        "final_consolidated_departments": [],
        "error": None
    }
    
    try:
        # Create configurations for the shared utilities
        search_config = WebSearchConfig(
            base_dir=str(level_config.data_dir.parent.parent),
            raw_search_dir=level_config.data_dir / "raw_search"
        )
        
        html_config = HTMLFetchConfig(
            cache_dir=level_config.data_dir / "cache"
        )
        
        list_config = ListExtractionConfig(
            keywords=DEPARTMENT_KEYWORDS,
            anti_keywords=NON_DEPARTMENT_KEYWORDS,
            patterns=DEPARTMENT_PATTERNS
        )
        
        # Use the template to generate a system prompt for this specific term
        validation_system_prompt = DEPARTMENT_VALIDATION_SYSTEM_PROMPT_TEMPLATE.format(term=level0_term)
        
        filter_config = FilterConfig(
            scoring_fn=score_department_list,
            clean_item_fn=clean_department_name,
            provider=provider,
            use_llm_validation=True,
            binary_llm_decision=False,
            binary_system_prompt=validation_system_prompt,
            min_score_for_llm=min_score_for_llm,
        )
        
        # Use prefetched search results if provided, otherwise perform search
        search_results = prefetched_search_results
        if search_results is None:
            # Use SEARCH_QUERIES constant with formatting
            queries = [query.format(term=level0_term) for query in SEARCH_QUERIES]
            
            # Perform web search with multiple queries
            logger.info(f"Searching with {len(queries)} different queries for '{level0_term}'")
            search_results = web_search_bulk(queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)
        
        if not search_results or not search_results.get("data"):
            logger.warning(f"No search results for '{level0_term}'")
            return {
                "level0_term": level0_term,
                "departments": [],
                "count": 0,
                "url_sources": {},
                "quality_scores": {},
                "verified": False,
                "num_urls": 0,
                "num_lists": 0
            }
        
        # Process search results
        try:
            # Extract URLs from all queries and combine them
            all_urls = []
            for query_index, query_data in enumerate(search_results.get("data", [])):
                query_urls = [r.get("url") for r in query_data.get("results", [])]
                query_urls = [url for url in query_urls if url]
                logger.debug(f"Query {query_index+1} returned {len(query_urls)} URLs")
                all_urls.extend(query_urls)
            
            # Remove duplicate URLs while preserving order
            seen_urls = set()
            urls = [url for url in all_urls if not (url in seen_urls or seen_urls.add(url))]
            
            if not urls:
                logger.warning(f"No URLs found in search results for '{level0_term}'")
                return {
                    "level0_term": level0_term,
                    "departments": [],
                    "count": 0,
                    "url_sources": {},
                    "quality_scores": {},
                    "verified": False,
                    "num_urls": 0,
                    "num_lists": 0
                }
            
            logger.info(f"Found {len(urls)} URLs for '{level0_term}'")
            term_details["all_urls"] = urls
            
            # Configure semaphore for concurrent requests
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            
            all_extracted_lists_raw = []
            url_to_lists = {}
            
            # If session is provided, use it, otherwise create a new one
            if session is None:
                async with aiohttp.ClientSession() as new_session:
                    # Fetch webpages
                    fetch_tasks = [
                        fetch_webpage(url, new_session, semaphore, html_config, level0_term, logger) 
                        for url in urls[:MAX_SEARCH_RESULTS]
                    ]
                    html_contents = await asyncio.gather(*fetch_tasks)
                    
                    # Process each webpage
                    for url, html_content in zip(urls[:MAX_SEARCH_RESULTS], html_contents):
                        if not html_content:
                            continue
                    
                        # Extract lists from the webpage
                        extracted_lists = extract_lists_from_html(html_content, list_config)
                        
                        if extracted_lists:
                            all_extracted_lists_raw.extend(extracted_lists)
                            
                            # Store only the actual string items in url_to_lists, not the full dictionary
                            url_items_lists = []
                            for list_data in extracted_lists:
                                if isinstance(list_data, dict) and "items" in list_data:
                                    # Extract only the string items, not the dictionary structure
                                    items = list_data["items"]
                                    if isinstance(items, list):
                                        # Make sure all items are strings
                                        clean_items = [str(item) for item in items if item]
                                        if clean_items:
                                            url_items_lists.append(clean_items)
                                elif isinstance(list_data, list):
                                    # If it's already a list, ensure all items are strings
                                    clean_items = [str(item) for item in list_data if item]
                                    if clean_items:
                                        url_items_lists.append(clean_items)
                            
                            # Only store if we have valid lists
                            if url_items_lists:
                                url_to_lists[url] = url_items_lists
                            
                        logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
            else:
                # Use the provided session
                fetch_tasks = [
                    fetch_webpage(url, session, semaphore, html_config, level0_term, logger) 
                    for url in urls[:MAX_SEARCH_RESULTS]
                ]
                html_contents = await asyncio.gather(*fetch_tasks)
                
                # Process each webpage
                for url, html_content in zip(urls[:MAX_SEARCH_RESULTS], html_contents):
                    if not html_content:
                        continue
                    
                    # Extract lists from the webpage
                    extracted_lists = extract_lists_from_html(html_content, list_config)
                    
                    if extracted_lists:
                        all_extracted_lists_raw.extend(extracted_lists)
                        
                        # Store only the actual string items in url_to_lists, not the full dictionary
                        url_items_lists = []
                        for list_data in extracted_lists:
                            if isinstance(list_data, dict) and "items" in list_data:
                                # Extract only the string items, not the dictionary structure
                                items = list_data["items"]
                                if isinstance(items, list):
                                    # Make sure all items are strings
                                    clean_items = [str(item) for item in items if item]
                                    if clean_items:
                                        url_items_lists.append(clean_items)
                            elif isinstance(list_data, list):
                                # If it's already a list, ensure all items are strings
                                clean_items = [str(item) for item in list_data if item]
                                if clean_items:
                                    url_items_lists.append(clean_items)
                        
                        # Only store if we have valid lists
                        if url_items_lists:
                            url_to_lists[url] = url_items_lists
                        
                    logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
            
            term_details["raw_extracted_lists"] = all_extracted_lists_raw
            
            # Save the raw URL results for this term (using string lists)
            save_raw_url_results(level0_term, url_to_lists)
                
            # Filter and validate lists
            if not all_extracted_lists_raw:
                logger.warning(f"No lists extracted for '{level0_term}'")
                term_details["error"] = "No lists extracted from fetched HTML"
                save_detailed_term_metadata(level0_term, term_details)
                return {
                    "level0_term": level0_term,
                    "departments": [],
                    "count": 0,
                    "url_sources": {},
                    "quality_scores": {},
                    "verified": False,
                    "num_urls": 0,
                    "num_lists": 0
                }
            
            logger.info(f"Extracted a total of {len(all_extracted_lists_raw)} raw lists for '{level0_term}'. Starting filtering...")
            
            # Filter lists using the multi-LLM voting utility
            final_llm_output_lists, llm_candidates, llm_results = await run_multiple_llm_extractions(
                all_extracted_lists_raw, level0_term, filter_config,
                num_attempts=num_llm_attempts,
                agreement_threshold=agreement_threshold,
                logger=logger,
                model_types=model_types
            )
            
            # Store LLM I/O pairs for detailed metadata
            term_details["llm_io_pairs"] = []
            if llm_candidates and llm_results:
                 # Ensure results align with candidates (should match length if consolidation worked)
                 # Note: llm_results now contains results from *all* attempts. We pair with first-run candidates.
                 num_pairs_to_log = min(len(llm_candidates), len(llm_results)) # Log what we can pair
                 if len(llm_candidates) * num_llm_attempts != len(llm_results):
                      logger.warning(f"LLM results count ({len(llm_results)}) doesn't match candidates ({len(llm_candidates)}) * attempts ({num_llm_attempts}). Logging paired results up to {num_pairs_to_log}.")
                 
                 # Log pairs based on first-run candidates and all collected results
                 # This might not perfectly align if errors occurred, but gives insight
                 candidate_index = 0
                 result_index = 0
                 while candidate_index < len(llm_candidates) and result_index < len(llm_results):
                    candidate_input = llm_candidates[candidate_index].get('items', []) if isinstance(llm_candidates[candidate_index], dict) else llm_candidates[candidate_index]
                    # Log all results potentially related to this candidate (from different attempts)
                    # This is an approximation as results aren't tagged per attempt
                    outputs_for_candidate = llm_results[result_index : result_index + num_llm_attempts]
                    term_details["llm_io_pairs"].append({
                        "input_list_to_llm": candidate_input,
                        "output_lists_from_llm_attempts": outputs_for_candidate
                    })
                    candidate_index += 1
                    result_index += num_llm_attempts # Move result index by number of attempts
            
            if not final_llm_output_lists:
                logger.warning(f"No lists passed multi-LLM filtering for '{level0_term}'")
                term_details["error"] = "No lists passed multi-LLM filtering"
                save_detailed_term_metadata(level0_term, term_details)
                return {
                    "level0_term": level0_term,
                    "departments": [],
                    "count": 0,
                    "url_sources": {},
                    "quality_scores": {},
                    "verified": False,
                    "num_urls": 0,
                    "num_lists": len(all_extracted_lists_raw)
                }
            
            logger.info(f"After multi-LLM filtering, {len(final_llm_output_lists)} consolidated list(s) remain for '{level0_term}'")
            
            # Consolidate departments from the final list(s)
            # consolidate_lists expects a list of lists
            departments = consolidate_lists(
                final_llm_output_lists,
                level0_term,
                min_frequency=1,
                min_list_appearances=1,
                similarity_threshold=0.7
            )
            term_details["final_consolidated_departments"] = departments
            
            logger.info(f"Found {len(departments)} departments for '{level0_term}' after consolidation")
            
            # Log the type of departments for debugging
            if departments:
                logger.debug(f"Type of first department: {type(departments[0])}")
                if not isinstance(departments[0], str):
                    logger.warning(f"Unexpected department type returned from consolidate_lists: {type(departments[0])}")
                
            # Replace with sanitized list
            string_departments = []
            for dept in departments:
                if isinstance(dept, str):
                    string_departments.append(dept)
                elif isinstance(dept, dict) and 'items' in dept:
                    # If it's a dictionary, extract the items
                    logger.warning(f"Found dictionary in consolidated results for '{level0_term}'. Extracting items.")
                    if isinstance(dept['items'], list):
                        string_departments.extend([item for item in dept['items'] if isinstance(item, str)])
                elif isinstance(dept, list):
                    # If it's a list, extend with its string items
                    string_departments.extend([str(item) for item in dept if isinstance(item, str)])
                else:
                    logger.warning(f"Skipping unexpected item type in consolidated results: {type(dept)}")
                
            # Replace with sanitized list
            departments = string_departments
            
            # Track URL sources for each department
            department_sources = {}
            department_quality = {}
            
            # Basic quality score (can be refined later)
            for dept in departments:
                department_quality[dept] = 1.0
            
            # Save detailed term metadata
            save_detailed_term_metadata(level0_term, term_details)

            # Simplified return structure focusing on final departments
            return {
                "level0_term": level0_term,
                "departments": departments,
                "count": len(departments),
                "url_sources": {},
                "quality_scores": department_quality,
                "verified": len(departments) > 0,
                "num_urls": len(urls),
                "num_lists": len(all_extracted_lists_raw),
            }
            
        except Exception as e:
            logger.error(f"Error processing term '{level0_term}': {str(e)}", exc_info=True)
            term_details["error"] = str(e)
            save_detailed_term_metadata(level0_term, term_details)
            return {
                "level0_term": level0_term,
                "departments": [],
                "count": 0,
                "url_sources": {},
                "quality_scores": {},
                "verified": False,
                "num_urls": 0,
                "num_lists": 0,
                "error": str(e)
            }
    except Exception as e:
        logger.error(f"Major error processing term '{level0_term}': {str(e)}", exc_info=True)
        term_details["error"] = f"Unhandled exception: {str(e)}"
        save_detailed_term_metadata(level0_term, term_details)
        return {
            "level0_term": level0_term,
            "departments": [],
            "count": 0,
            "url_sources": {},
            "quality_scores": {},
            "verified": False,
            "num_urls": 0,
            "num_lists": 0,
            "error": str(e)
        }


def save_detailed_term_metadata(level0_term: str, data: Dict[str, Any]):
    """Saves the detailed processing metadata for a single level0 term."""
    try:
        # Ensure the detailed metadata directory exists
        os.makedirs(level_config.data_dir / "detailed_meta", exist_ok=True)

        # Sanitize filename
        safe_filename = re.sub(r'[\/:*?"<>|]', '_', level0_term) + "_details.json"
        output_path = os.path.join(level_config.data_dir / "detailed_meta", safe_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            # Use default handler for non-serializable objects
            json.dump(data, f, indent=2, default=lambda o: f"<non-serializable: {type(o).__name__}>")
        logger.debug(f"Saved detailed metadata for '{level0_term}' to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save detailed metadata for '{level0_term}': {str(e)}", exc_info=True)


def perform_bulk_web_search(level0_terms: List[str], search_config: WebSearchConfig) -> Dict[str, Dict[str, Any]]:
    """
    Perform a bulk web search for multiple level 0 terms at once.
    
    Args:
        level0_terms: List of level 0 terms to search for
        search_config: Configuration for web search
        
    Returns:
        A dictionary mapping level0_term to its search results
    """
    if not level0_terms:
        return {}
    
    # Use global SEARCH_QUERIES constant
    queries_per_term = SEARCH_QUERIES
    
    # Dynamic calculation of max terms per search batch
    num_queries_per_term = len(queries_per_term)
    max_terms_per_search = MAX_SEARCH_QUERIES // num_queries_per_term
    
    # Limit the number of terms per batch to prevent exceeding API limits
    terms_per_batch = min(len(level0_terms), max_terms_per_search)
    logger.info(f"Performing bulk web search for {len(level0_terms)} terms (max {max_terms_per_search} per batch, using {num_queries_per_term} queries per term)")
    
    # Prepare all queries for all terms
    all_queries = []
    term_to_query_indices = {}  # Maps each term to its query indices in all_queries
    
    for i, term in enumerate(level0_terms):
        # Store the starting index for this term's queries
        start_idx = len(all_queries)
        
        # Add queries for this term
        current_term_queries = [query.format(term=term) for query in queries_per_term]
        all_queries.extend(current_term_queries)
        
        # Store the range of indices for this term
        term_to_query_indices[term] = (start_idx, len(all_queries))
    
    # Perform the bulk search - web_search_bulk is synchronous, not async
    search_results = web_search_bulk(all_queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)
    
    if not search_results or not search_results.get("data"):
        logger.warning(f"No search results found for any of the {len(level0_terms)} terms")
        return {}
    
    # Map results back to their respective terms
    term_to_results = {}
    
    for term, (start_idx, end_idx) in term_to_query_indices.items():
        # Extract results for this term's queries
        term_results = {
            "data": search_results.get("data", [])[start_idx:end_idx],
        }
        
        # Only add if we have data
        if term_results["data"]:
            term_to_results[term] = term_results
    
    logger.info(f"Got search results for {len(term_to_results)} out of {len(level0_terms)} terms")
    return term_to_results


async def process_level0_terms_batch(batch: List[str],
                                   provider: Optional[str] = None,
                                   session: Optional[Any] = None,
                                   min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                                   model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                                   num_llm_attempts: int = NUM_LLM_ATTEMPTS,
                                   agreement_threshold: int = AGREEMENT_THRESHOLD
                                   ) -> List[Dict[str, Any]]:
    """Process a batch of level 0 terms with optimized bulk web searching"""
    if not batch:
        return []
        
    logger.info(f"Processing batch of {len(batch)} level 0 terms")
    
    # Create search configuration
    search_config = WebSearchConfig(
        base_dir=str(level_config.data_dir.parent.parent),
        raw_search_dir=level_config.data_dir / "raw_search"
    )
    
    # Use global SEARCH_QUERIES constant for calculation
    num_queries_per_term = len(SEARCH_QUERIES)
    max_terms_per_search = MAX_SEARCH_QUERIES // num_queries_per_term
    
    # Split the batch into smaller chunks for web search
    search_results_by_term = {}
    for i in range(0, len(batch), max_terms_per_search):
        search_batch = batch[i:i + max_terms_per_search]
        logger.info(f"Performing bulk web search for {len(search_batch)} terms (batch {i//max_terms_per_search + 1})")
        
        # Perform the bulk web search for this mini-batch - perform_bulk_web_search is now synchronous
        batch_results = perform_bulk_web_search(search_batch, search_config)
        search_results_by_term.update(batch_results)
    
    # Process each term with its pre-fetched search results
    tasks = []
    for term in batch:
        # Get the pre-fetched search results for this term (if any)
        prefetched_results = search_results_by_term.get(term)
        
        # Process the term
        task = process_level0_term(
            term,
            provider,
            session,
            min_score_for_llm,
            model_types,
            num_llm_attempts,
            agreement_threshold,
            prefetched_results
        )
        tasks.append(task)
    
    # Run all tasks in parallel
    return await asyncio.gather(*tasks)


async def main_async():
    """Async main execution function"""
    try:
        # Get provider from command line args
        parser = argparse.ArgumentParser(description="Extract department names for level 0 terms.")
        parser.add_argument("--provider", help="LLM provider (e.g., gemini, openai)")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for processing terms (default: {BATCH_SIZE})")
        parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, help=f"Max concurrent term processing requests (default: {MAX_CONCURRENT_REQUESTS})")
        parser.add_argument("--min-score-for-llm", type=float, default=DEFAULT_MIN_SCORE_FOR_LLM, help=f"Minimum heuristic score to send a list to LLM (default: {DEFAULT_MIN_SCORE_FOR_LLM})")
        parser.add_argument("--input-file", default=str(get_level_config(0).get_final_file()), help=f"Path to the input file containing level 0 terms (default: {str(get_level_config(0).get_final_file())})")
        parser.add_argument("--append", action='store_true', help="Append results to existing output files instead of overwriting.")
        parser.add_argument("--llm-attempts", type=int, default=NUM_LLM_ATTEMPTS, help=f"Number of LLM extraction attempts per term (default: {NUM_LLM_ATTEMPTS})")
        parser.add_argument("--agreement-threshold", type=int, default=AGREEMENT_THRESHOLD, help=f"Minimum appearances threshold for departments (default: {AGREEMENT_THRESHOLD})")
        parser.add_argument("--llm-model-types", type=str, default=",".join(DEFAULT_LLM_MODEL_TYPES), help=f"Comma-separated LLM model types for attempts (e.g., default,pro,mini) (default: {','.join(DEFAULT_LLM_MODEL_TYPES)})")

        args = parser.parse_args()
        
        provider_arg = args.provider
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        min_score_for_llm = args.min_score_for_llm
        input_file_path = args.input_file
        append_mode = args.append
        num_llm_attempts = args.llm_attempts
        agreement_threshold = args.agreement_threshold
        llm_model_types = [mt.strip() for mt in args.llm_model_types.split(',') if mt.strip()]
        
        # Determine the single provider to use
        current_provider = provider_arg or Provider.GEMINI
        logger.info(f"Using provider: {current_provider}")
        
        if batch_size != BATCH_SIZE:
            logger.info(f"Using custom batch size: {batch_size}")
        if max_concurrent != MAX_CONCURRENT_REQUESTS:
             logger.info(f"Using custom concurrent limit: {max_concurrent}")
        if min_score_for_llm != DEFAULT_MIN_SCORE_FOR_LLM:
             logger.info(f"Using custom LLM score threshold: {min_score_for_llm}")
        if args.llm_model_types != ",".join(DEFAULT_LLM_MODEL_TYPES):
            logger.info(f"Using custom LLM model types: {llm_model_types}")
        if num_llm_attempts != NUM_LLM_ATTEMPTS:
             logger.info(f"Using custom LLM attempts: {num_llm_attempts}")
        if agreement_threshold != AGREEMENT_THRESHOLD:
             logger.info(f"Using custom agreement threshold: {agreement_threshold}")
        if input_file_path != str(get_level_config(0).get_final_file()):
             logger.info(f"Using custom input file: {input_file_path}")
        if append_mode:
             logger.info("Append mode enabled. Results will be added to existing files.")

        logger.info("Starting department names extraction using level 0 terms (Single Run with Multi-LLM Voting)")
        logger.info(f"Using optimized web search with dynamically calculated max terms per batch")
        
        # Create output directories
        ensure_dirs_exist()
        
        # Read level 0 terms
        level0_terms = read_level0_terms(input_file_path)
        
        logger.info(f"Processing {len(level0_terms)} level 0 terms with batch size {batch_size}, max {max_concurrent} concurrent terms, provider {current_provider}")
        logger.info(f"Multi-LLM Config: Attempts={num_llm_attempts}, Agreement={agreement_threshold}, Models={llm_model_types}, Score Threshold={min_score_for_llm}")

        # --- Single Run Logic ---
        current_output_file = level_config.get_step_output_file(0)
        current_meta_file = level_config.get_step_metadata_file(0)
        
        # Initialize aiohttp session for the run
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, CookieJar
        import ssl, certifi
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = TCPConnector(
            ssl=ssl_context,
            limit=max_concurrent * 3,
            limit_per_host=2,
            force_close=True,
            enable_cleanup_closed=True
        )
        cookie_jar = CookieJar(unsafe=True)
        
        all_results = []
        start_time = time.time()
        
        timeout = ClientTimeout(total=7200, connect=60)
        async with ClientSession(connector=connector, cookie_jar=cookie_jar, timeout=timeout) as session:
            for i in range(0, len(level0_terms), batch_size):
                batch = level0_terms[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(level0_terms) + batch_size - 1)//batch_size}")
                
                # Process the batch using the optimized batch processor
                batch_results = await process_level0_terms_batch(
                    batch,
                    current_provider,
                    session,
                    min_score_for_llm,
                    llm_model_types,
                    num_llm_attempts,
                    agreement_threshold
                )

                all_results.extend(batch_results)

                elapsed = time.time() - start_time
                terms_processed = min(i + batch_size, len(level0_terms))
                terms_per_second = terms_processed / max(1, elapsed)
                eta_seconds = (len(level0_terms) - terms_processed) / max(0.1, terms_per_second)
                logger.info(f"Processed {terms_processed}/{len(level0_terms)} terms ({terms_processed/len(level0_terms)*100:.1f}%) in {elapsed:.1f}s ({terms_per_second:.2f} terms/s, ETA: {eta_seconds/60:.1f}m)")
                await asyncio.sleep(2)
        
        # --- Process Results from Single Run --- 
        all_departments = []
        department_sources = {}
        department_quality_scores = {}
        level0_to_departments = {}
        verified_terms_count = 0
        total_urls_processed = 0
        total_raw_lists_found = 0
        verified_departments_count = 0
        department_counts_by_level0 = {}
        
        for result in all_results:
            if result.get("error"):
                 logger.warning(f"Skipping aggregation for term '{result['level0_term']}' due to processing error: {result['error']}")
                 continue

            level0_term = result["level0_term"]
            departments = result["departments"]
            quality_scores = result.get("quality_scores", {})
            verified = result.get("verified", False)
            num_urls = result.get("num_urls", 0)
            num_lists = result.get("num_lists", 0)
            
            total_urls_processed += num_urls
            total_raw_lists_found += num_lists
            
            if verified:
                verified_terms_count += 1
                verified_departments_count += len(departments)
                department_counts_by_level0[level0_term] = result["count"]
                
                if level0_term not in level0_to_departments:
                    level0_to_departments[level0_term] = []
                
                level0_to_departments[level0_term].extend(departments)
                all_departments.extend(departments)

                for dept in departments:
                    if dept not in department_sources:
                         department_sources[dept] = []
                    department_sources[dept].append(level0_term)

                    if dept in quality_scores:
                        if dept not in department_quality_scores:
                            department_quality_scores[dept] = quality_scores[dept]
                        else:
                            department_quality_scores[dept] = max(department_quality_scores[dept], quality_scores[dept])
        
        logger.info(f"Found {verified_terms_count} level 0 terms with verified departments using multi-LLM voting.")
        
        # Remove duplicates while preserving case
        unique_departments = []
        seen = set()
        for dept in all_departments:
            if not isinstance(dept, str):
                logger.warning(f"Skipping non-string department: {type(dept)}")
                continue
            dept_lower = dept.lower()
            if dept_lower not in seen:
                seen.add(dept_lower)
                unique_departments.append(dept)
                
        # Deduplicate level0_to_departments mapping
        for level0_term, depts in level0_to_departments.items():
            unique_level0_depts = []
            seen_level0 = set()
            for dept in depts:
                if isinstance(dept, str):
                    dept_lower = dept.lower()
                    if dept_lower not in seen_level0:
                        seen_level0.add(dept_lower)
                        unique_level0_depts.append(dept)
            level0_to_departments[level0_term] = unique_level0_depts
            
        # Shuffle final list
        final_unique_departments_list = list(seen)
        random.shuffle(final_unique_departments_list)
        
        # --- Save Results from Single Run --- 
        
        existing_unique_departments_lower = set()
        if append_mode and os.path.exists(current_output_file):
            logger.info(f"Loading existing departments from {current_output_file} for append mode.")
            try:
                with open(current_output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        existing_unique_departments_lower.add(line.strip().lower())
                logger.info(f"Loaded {len(existing_unique_departments_lower)} existing unique departments (lowercase).")
            except Exception as e:
                logger.warning(f"Could not read existing output file {current_output_file}: {e}. Starting fresh.")
                existing_unique_departments_lower = set()
                
        final_unique_departments_map = {dept.lower(): dept for dept in existing_unique_departments_lower}
        newly_added_count = 0
        for dept in final_unique_departments_list:
            dept_lower = dept.lower()
            if dept_lower not in final_unique_departments_map:
                 final_unique_departments_map[dept_lower] = dept
                 newly_added_count += 1
                
        logger.info(f"Added {newly_added_count} new unique departments.")
        
        final_unique_departments_to_write = sorted(list(final_unique_departments_map.values()), key=str.lower)
        random.shuffle(final_unique_departments_to_write)
        
        with open(level_config.get_step_output_file(0), "w", encoding="utf-8") as f:
            for dept in final_unique_departments_to_write:
                f.write(f"{dept}\n")
        
        # --- Load and Merge Metadata (if append mode) --- 
        metadata = {}
        if append_mode and os.path.exists(current_meta_file):
            logger.info(f"Loading existing metadata from {current_meta_file} for append mode.")
            try:
                with open(current_meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info("Existing metadata loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not read/parse existing metadata file {current_meta_file}: {e}. Creating new metadata.")
                metadata = {}
                
        # Prepare data for metadata (using original casing for keys where appropriate, lowercase for comparisons/sets)
        lowercase_level0_to_departments = {term: [d.lower() for d in depts if isinstance(d, str)] for term, depts in level0_to_departments.items()}
        lowercase_department_sources = {dept.lower(): sources for dept, sources in department_sources.items() if isinstance(dept, str)}
        lowercase_department_quality_scores = {dept.lower(): score for dept, score in department_quality_scores.items() if isinstance(dept, str)}

        # Merge data if in append mode
        if append_mode:
            existing_level0_map = metadata.get("level0_to_departments", {})
            for term, depts in lowercase_level0_to_departments.items():
                if term not in existing_level0_map:
                    existing_level0_map[term] = []
                existing_depts_set = set(existing_level0_map[term])
                for dept in depts:
                    if dept not in existing_depts_set:
                        existing_level0_map[term].append(dept)
                        existing_depts_set.add(dept)
            metadata["level0_to_departments"] = existing_level0_map
            
            existing_dept_sources = metadata.get("department_sources", {})
            for dept, sources in department_sources.items():
                dept_lower = dept.lower()
                existing_key = next((k for k in existing_dept_sources if k.lower() == dept_lower), dept)
                if existing_key not in existing_dept_sources:
                    existing_dept_sources[existing_key] = []
                existing_sources_set = set(existing_dept_sources[existing_key])
                for source in sources:
                    if source not in existing_sources_set:
                        existing_dept_sources[existing_key].append(source)
                        existing_sources_set.add(source)
            metadata["department_sources"] = existing_dept_sources
            
            existing_quality_scores = metadata.get("department_quality_scores", {})
            new_quality_scores_map = {}
            for dept, score in department_quality_scores.items():
                 dept_lower = dept.lower()
                 existing_key = next((k for k in existing_quality_scores if k.lower() == dept_lower), dept)
                 current_score = existing_quality_scores.get(existing_key, 0)
                 new_quality_scores_map[existing_key] = max(current_score, score)
            metadata["department_quality_scores"] = new_quality_scores_map

            meta_details = metadata["metadata"]
            meta_details["total_departments"] = len(final_unique_departments_to_write)
            meta_details["level0_terms_processed_this_run"] = len(level0_terms)
            meta_details["total_level0_terms_processed"] = meta_details.get("total_level0_terms_processed", 0) + len(level0_terms)
            meta_details["verified_level0_terms_this_run"] = verified_terms_count
            meta_details["total_verified_level0_terms"] = meta_details.get("total_verified_level0_terms", 0) + verified_terms_count
            meta_details["total_urls_processed"] = meta_details.get("total_urls_processed", 0) + total_urls_processed
            meta_details["total_raw_lists_extracted"] = meta_details.get("total_raw_lists_extracted", 0) + total_raw_lists_found
            meta_details["verified_departments_count_this_run"] = verified_departments_count
            meta_details["total_verified_departments"] = meta_details.get("total_verified_departments", 0) + verified_departments_count
            meta_details.setdefault("department_counts_by_level0", {}).update(department_counts_by_level0)
            meta_details["provider"] = current_provider
            meta_details["max_concurrent"] = max_concurrent
            meta_details["batch_size"] = batch_size
            meta_details["min_score_for_llm"] = min_score_for_llm
            meta_details["num_llm_attempts"] = num_llm_attempts
            meta_details["agreement_threshold"] = agreement_threshold
            meta_details["llm_model_types_used"] = llm_model_types[:num_llm_attempts]

            total_terms_ever = meta_details["total_level0_terms_processed"]
            total_verified_ever = meta_details["total_verified_level0_terms"]
            total_verified_depts_ever = meta_details["total_verified_departments"]
            meta_details["processing_stats"] = {
                "avg_urls_per_term": meta_details["total_urls_processed"] / max(1, total_terms_ever),
                "avg_raw_lists_per_term": meta_details["total_raw_lists_extracted"] / max(1, total_terms_ever),
                "avg_departments_per_verified_term": total_verified_depts_ever / max(1, total_verified_ever)
            }
            
        else:
            metadata = {
                "level0_to_departments": level0_to_departments,
                "department_sources": department_sources,
                "department_quality_scores": department_quality_scores,
                "metadata": {
                    "execution_time": f"{time.time() - start_time:.2f} seconds",
                    "total_departments": len(final_unique_departments_to_write),
                    "level0_terms_processed": len(level0_terms),
                    "verified_level0_terms": verified_terms_count,
                    "department_counts_by_level0": department_counts_by_level0,
                    "provider": current_provider,
                    "max_concurrent": max_concurrent,
                    "batch_size": batch_size,
                    "min_score_for_llm": min_score_for_llm,
                    "num_llm_attempts": num_llm_attempts,
                    "agreement_threshold": agreement_threshold,
                    "llm_model_types_used": llm_model_types[:num_llm_attempts],
                    "total_urls_processed": total_urls_processed,
                    "total_raw_lists_extracted": total_raw_lists_found,
                    "verified_departments_count": verified_departments_count,
                    "processing_stats": {
                        "avg_urls_per_term": total_urls_processed / max(1, len(level0_terms)),
                        "avg_raw_lists_per_term": total_raw_lists_found / max(1, len(level0_terms)),
                        "avg_departments_per_verified_term": verified_departments_count / max(1, verified_terms_count)
                    }
                }
            }
            
        with open(current_meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully extracted {len(final_unique_departments_to_write)} unique departments from {verified_terms_count} verified level 0 terms using multi-LLM voting.")
        logger.info(f"Department names saved to {current_output_file}")
        logger.info(f"Metadata saved to {current_meta_file}")
        logger.info(f"Detailed per-term metadata for this run saved in {level_config.data_dir / "detailed_meta"}")

    except Exception as e:
        logger.error(f"An error occurred in main_async: {str(e)}", exc_info=True)


def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
