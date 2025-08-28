import os
import sys
import random
import json
import time
import asyncio
import aiohttp
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter
import ssl
import certifi

# Fix import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
# No longer need Provider import - using direct provider strings

# Import shared web search utilities
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists, init_llm

# Load environment variables and setup logging
load_dotenv('.env')
logger = setup_logger("lv2.s0")
random.seed(42)

# Constants
MAX_SEARCH_RESULTS = 30
MAX_CONCURRENT_REQUESTS = 8
BATCH_SIZE = 10  # Reduced from 5 to prevent resource issues when sending more lists to LLM
MAX_SEARCH_QUERIES = 50  # Maximum number of queries to send in a single web_search_bulk call
# Maximum terms per search is calculated dynamically as MAX_SEARCH_QUERIES // number_of_queries_per_term

# Research-related keywords for enhanced filtering
RESEARCH_KEYWORDS = [
    "research", "study", "analysis", "theory", "methodology", "approach", 
    "experiment", "investigation", "application", "development", "model", 
    "framework", "technology", "technique", "method", "assessment", "evaluation",
    "innovation", "discovery", "implementation", "design", "exploration",
    "characterization", "measurement", "computation", "simulation", "modeling",
    "theory", "algorithm", "system", "process", "protocol", "strategy",
    "laboratory", "lab", "project", "program", "initiative", "paradigm",
    "analytics", "data", "clinical", "optimization", "engineering",
    r"[\w\s&,\'-]+ (?:algorithm|system|technique|method)"
]

# Anti-keywords indicating non-research text
NON_RESEARCH_KEYWORDS = [
    "login", "sign in", "register", "apply", "admission", "contact", 
    "about", "home", "sitemap", "search", "privacy", "terms", "copyright",
    "accessibility", "careers", "jobs", "employment", "staff", "faculty",
    "directory", "phone", "email", "address", "location", "directions",
    "map", "parking", "visit", "tour", "events", "news", "calendar",
    "library", "bookstore", "housing", "dining", "athletics", "sports",
    "recreation", "student", "alumni", "giving", "donate", "support",
    "request", "form", "apply now", "learn more", "read more", "view",
    "download", "upload", "submit", "send", "share", "follow", "like",
    "tweet", "post", "comment", "subscribe", "newsletter", "blog",
    "click here", "link", "back", "next", "previous", "continue"
]

# Research pattern matches for regex matching
RESEARCH_PATTERNS = [
    r"(?:research|study|analysis|investigation) (?:of|on|in) [\w\s&,'-]+",
    r"[\w\s&,'-]+ (?:theory|methodology|approach|framework)",
    r"[\w\s&,'-]+ (?:analysis|modeling|simulation)",
    r"[\w\s&,'-]+ (?:development|implementation|application)",
    r"[\w\s&,'-]+ (?:algorithm|system|technique|method)"
]

# Search queries - consolidated in one place
SEARCH_QUERIES = [
    "site:.edu department of {term} (research areas | research topics)",
    "{term} (topics | methods | tasks)",
]

# LLM system prompt template for research area validation
RESEARCH_VALIDATION_SYSTEM_PROMPT_TEMPLATE = """You are a highly meticulous academic research assistant specializing in identifying ONLY the most specific and directly relevant research areas within university departments.

Task: Analyze the provided list and extract ONLY items that are DIRECTLY and UNQUESTIONABLY research areas, research groups, research labs, or teaching courses that fall STRICTLY under the umbrella of **The Department of {term}**.

Input: A list of potential research areas/topics/courses/other text.
Output: A Python-style list `[...]` containing ONLY the items that are DEFINITIVELY and DIRECTLY related to **The Department of {term}**. Return ONLY the verified items, preserving their original phrasing.

EXTREMELY STRICT Exclusion Criteria - DO NOT INCLUDE:
1. **ANYTHING not specifically and directly related to {term}**: If there is ANY doubt about relevance, EXCLUDE it.
2. **Adjacent or neighboring fields**: EXCLUDE items from adjacent, related, or overlapping disciplines. ONLY include items that are CORE to {term} itself.
3. **Interdisciplinary areas**: EXCLUDE interdisciplinary items unless they are PRIMARILY housed within {term} and the {term} aspect is the DOMINANT component.
4. **General university/administrative items**: EXCLUDE ALL general programs, centers, resources, or administrative text.
5. **Vague or generic terms**: EXCLUDE ANY term that could apply to multiple departments. Only include SPECIFIC research areas.
6. **People or organizations**: EXCLUDE faculty names, student groups, or professional associations.
7. **Navigational/website elements**: EXCLUDE ALL menu items, headers, footers, and navigation elements.
8. **Broader categories**: If an item represents a category that contains {term} rather than a subcategory OF {term}, EXCLUDE it.

Guidelines:
- Be EXTREMELY SELECTIVE. When in doubt, EXCLUDE.
- ONLY include items that would be recognized by experts in {term} as proper subfields, topics, or courses.
- For courses, they MUST be specifically teaching {term} content, not just mentioning or using {term}.
- Return an empty list `[]` if NO items meet these strict criteria.
- Output ONLY the Python-style list, nothing else.

Example (Department of Biology):
Input List: ["Molecular Biology", "Genetics", "Ecology and Evolution", "University Research Opportunities", "BIOL 101: Introduction to Biology", "Physics Department", "Contact Us", "Neuroscience Program (Interdisciplinary)", "Cell Biology Lab"]
Output: ["Molecular Biology", "Genetics", "Ecology and Evolution", "BIOL 101: Introduction to Biology", "Cell Biology Lab"]

Example (Department of Computer Science):
Input List: ["Data Science", "Artificial Intelligence", "Machine Learning", "Student Resources", "CS 101: Introduction to Programming", "Mathematics Department", "Quantum Computing Lab", "Contact Us", "Faculty Profiles", "Software Engineering", "Bioinformatics"]
Output: ["Artificial Intelligence", "Machine Learning", "CS 101: Introduction to Programming", "Software Engineering"]

I will ONLY include items that are DEFINITIVELY and DIRECTLY related to **The Department of {term}**. Any item with even slight uncertainty about its direct relevance to {term} will be excluded.

Analyze the following list with EXTREME STRICTNESS:"""

DEFAULT_MIN_SCORE_FOR_LLM = 0.65 # Default minimum score to send list to LLM (Higher for Lv2)
DEFAULT_LLM_MODEL_TYPES = ["default", "mini", "nano"] # Default model types for attempts

# Use centralized configuration
LEVEL = 2
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

def read_level1_terms(input_path: str) -> List[str]:
    """Read level 1 terms from input file"""
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            terms = [line.strip() for line in file.readlines() if line.strip()]
        logger.info(f"Successfully read {len(terms)} level 1 terms")
        return terms
    except Exception as e:
        logger.error(f"Failed to read level 1 terms: {str(e)}", exc_info=True)
        raise


def clean_research_area(item: str) -> str:
    """Clean a research area"""
            # Remove common prefixes like "Research on", "Studies in"
    item = item.strip()
    item = re.sub(r'^(Research|Studies|Topics|Focus|Areas|Interests) (on|in|of) ', '', item, flags=re.IGNORECASE)
    # Remove trailing numbers, parenthetical info
    item = re.sub(r'\s*\(\d+\).*$', '', item)
    item = re.sub(r'\s*\d+\s*$', '', item)
    # Remove URLs
    item = re.sub(r'http\S+', '', item)
    # Clean whitespace
    item = ' '.join(item.split())
    return item


def score_research_area_list(items: List[str], metadata: Dict[str, Any], context_term: str) -> float:
    """Score a research area list based on various heuristics"""
    # Adjust weights to prioritize content relevance over structure/size
    weights = {
        "keyword": 0.35,      # Increased: Presence of research keywords
        "structure": 0.05,     # Decreased: Less weight on specific HTML tag structure
        "pattern": 0.25,      # Increased: Consistency and relevance of item naming patterns
        "non_term": 0.25,     # Increased: Penalty for non-relevant/navigational terms
        "consistency": 0.05,  # Decreased: General formatting consistency (less critical than content)
        "size": 0.0,          # Removed: List size is less important if content is good
        "html_type": 0.05     # Decreased: HTML element type less critical than content relevance
    }
    
    # Ensure weights sum to 1 (or close enough)
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6: 
        # Basic normalization if they don't sum to 1
        weights = {k: v / total_weight for k, v in weights.items()}

    # Use the common scoring function from list_extractor, passing the adjusted weights
    # Note: Assumes score_list function in list_extractor accepts 'scoring_weights'
    try:
        from generate_glossary.utils.web_search.list_extractor import score_list
        return score_list(
            items=items, # Pass the cleaned items
            metadata=metadata,
            context_term=context_term,
            keywords=RESEARCH_KEYWORDS, # Use specific research keywords
            scoring_weights=weights
        )
    except ImportError:
        # Fallback if score_list is not available
        logger.debug("score_list function not found, using fallback metadata scoring.")
        # Simplified fallback based on metadata only
        keyword_ratio = metadata.get("keyword_ratio", 0)
        pattern_ratio = metadata.get("pattern_ratio", 0)
        non_term_ratio = metadata.get("non_term_ratio", 1)
        nav_score = metadata.get("structure_analysis", {}).get("nav_score", 1)
        # Apply weights similar to above
        score = (keyword_ratio * weights["keyword"] + 
                 pattern_ratio * weights["pattern"] + 
                 (1 - non_term_ratio) * weights["non_term"] + 
                 (1 - nav_score) * weights["structure"]) # Approximation
        return min(max(score, 0.0), 1.0)


# def save_raw_url_results(level1_term: str, url_to_lists: Dict[str, List[List[str]]]):
#     \"\"\"Save raw URL results for a term to a JSON file
    
#     Args:
#         level1_term: The level 1 term
#         url_to_lists: Dictionary mapping URLs to lists of lists of strings
#     \"\"\"
#     try:
#         # Sanitize filename
#         safe_filename = re.sub(r'[\/:*?"<>|]', '_', level1_term) + "_url_lists.json"
#         output_path = os.path.join(level_config.data_dir / "raw_search", safe_filename)
        
#         # Convert data to a serializable format
#         serializable_data = {
#             "level1_term": level1_term,
#             "urls": {}
#         }
        
#         # Organize by URL for better readability
#         for url, lists in url_to_lists.items():
#             serializable_data["urls"][url] = lists
            
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(serializable_data, f, indent=2)
            
#         logger.info(f"Saved raw URL results for '{level1_term}' to {output_path}")
#     except Exception as e:
#         logger.error(f"Failed to save raw URL results for '{level1_term}': {str(e)}", exc_info=True)


async def run_multiple_llm_extractions(
    all_extracted_lists_raw: List[Dict[str, Any]],
    level1_term: str,
    filter_config: FilterConfig,
    num_attempts: int = processing_config.llm_attempts,
    agreement_threshold: int = processing_config.concept_agreement_threshold,
    logger: Optional[Any] = None,
    model_types: List[str] = DEFAULT_LLM_MODEL_TYPES
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[List[str]]]:
    """
    Run multiple LLM extractions and select research areas that appear in multiple responses.
    Each attempt uses a randomly selected provider (Gemini/OpenAI) and model type.
    
    Args:
        all_extracted_lists_raw: Raw extracted lists to process
        level1_term: The level 1 term being processed
        filter_config: Configuration for filtering
        num_attempts: Number of LLM extraction attempts
        agreement_threshold: Minimum number of appearances required for a research area
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
    
    logger.info(f"Running multiple LLM extractions ({num_attempts}) for '{level1_term}'")
    
    # First, score all lists so we can select the top ones if needed
    scored_lists = []
    # Run scoring for all the raw lists
    for idx, list_dict in enumerate(all_extracted_lists_raw):
        # Make sure each list has items
        if not isinstance(list_dict, dict) or 'items' not in list_dict or not list_dict['items']:
            continue
            
        # Ensure we have metadata for scoring
        if 'metadata' not in list_dict:
            list_dict['metadata'] = {}
            
        # Calculate the score
        score = filter_config.scoring_fn(list_dict['items'], list_dict.get('metadata', {}), level1_term)
        
        # Store the scored list
        scored_lists.append({
            'index': idx,  # Keep track of original index
            'list_dict': list_dict,
            'score': score
        })
        
    # Sort by score, highest first
    scored_lists.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter lists based on the min_score_for_llm threshold
    filtered_lists = [sl for sl in scored_lists if sl['score'] >= filter_config.min_score_for_llm]
    logger.info(f"Found {len(filtered_lists)} lists that passed score threshold ({filter_config.min_score_for_llm})")
    
    # Ensure we have at least 10 lists for the LLM
    MIN_LISTS_FOR_LLM = 10
    if len(filtered_lists) < MIN_LISTS_FOR_LLM and scored_lists:
        # Add more lists to reach minimum (take from highest scored lists that didn't pass threshold)
        lists_to_add = min(MIN_LISTS_FOR_LLM - len(filtered_lists), len(scored_lists) - len(filtered_lists))
        if lists_to_add > 0:
            # Get indices of lists that didn't pass threshold but are highest scoring
            remaining_lists = [sl for sl in scored_lists if sl not in filtered_lists]
            remaining_lists.sort(key=lambda x: x['score'], reverse=True)  # Sort by score
            
            # Add the top scoring lists that didn't pass threshold
            top_remaining = remaining_lists[:lists_to_add]
            filtered_lists.extend(top_remaining)
            
            logger.info(f"Added {len(top_remaining)} additional lists to reach minimum of {MIN_LISTS_FOR_LLM} lists for LLM")
            
            # Re-sort all lists by score to maintain order
            filtered_lists.sort(key=lambda x: x['score'], reverse=True)
    
    # Reconstruct the list of raw dictionaries for filtering from the filtered_lists
    filtered_raw_lists = [sl['list_dict'] for sl in filtered_lists]
    
    # If we still don't have any lists, return empty results
    if not filtered_raw_lists:
        logger.warning(f"No lists available for LLM processing after filtering for '{level1_term}'")
        return [], [], []
    
    logger.info(f"Running LLM extraction on {len(filtered_raw_lists)} lists (including fallback lists)")
    
    # Run multiple extraction attempts
    all_results = []
    all_candidates = []
    all_raw_results = []
    
    # Ensure model_types list has enough entries for num_attempts
    if len(model_types) < num_attempts:
        # If we need more providers than we have, duplicate them
        model_types = model_types * (num_attempts // len(model_types) + 1)
    current_model_types = model_types[:num_attempts]
    available_providers = [Provider.GEMINI, Provider.OPENAI] # Define available providers
    
    for attempt in range(num_attempts):
        # Use a different provider for each attempt if possible
        current_provider = random.choice(available_providers)
        current_model_type = random.choice(model_types) # Randomly choose from the list
        
        # Create a new filter config with the current provider
        current_filter_config = FilterConfig(
            scoring_fn=filter_config.scoring_fn,
            clean_item_fn=filter_config.clean_item_fn,
            provider=current_provider,
            use_llm_validation=filter_config.use_llm_validation,
            binary_llm_decision=filter_config.binary_llm_decision,
            binary_system_prompt=filter_config.binary_system_prompt,
            min_score_for_llm=0.0,  # Set to 0 because we've already filtered the lists
            model_type=current_model_type # Pass specific model type for this attempt
        )
        
        logger.info(f"Attempt {attempt+1}/{num_attempts} using RANDOM provider: {current_provider}, model: {current_model_type}")
        
        try:
            # Pass the filtered lists directly to filter_lists
            # Since we've already filtered based on scores, we set min_score_for_llm to 0
            # to ensure all lists are processed by the LLM
            final_lists, llm_candidates, llm_results = await filter_lists(
                filtered_raw_lists, level1_term, current_filter_config, logger
            )
            
            # Store the results from this attempt
            all_results.append(final_lists)
            if attempt == 0:
                # Store candidates from first attempt only (they should be the same each time)
                all_candidates = llm_candidates
            all_raw_results.extend(llm_results)
            
            logger.info(f"Attempt {attempt+1} found {len(final_lists)} verified lists")
            
        except Exception as e:
            logger.error(f"Error in extraction attempt {attempt+1}: {str(e)}")
            # Continue with other attempts
    
    # Consolidate results from multiple runs
    # First, flatten all extracted lists
    all_items = []
    for final_lists_attempt in all_results:
        for lst in final_lists_attempt:
            all_items.extend(lst)
    
    # Count occurrences of each item
    item_counts = {}
    for item in all_items:
        item_lower = item.lower()
        if item_lower not in item_counts:
            item_counts[item_lower] = 0
        item_counts[item_lower] += 1
    
    # Filter items by agreement threshold
    agreed_items = [item for item, count in item_counts.items() if count >= agreement_threshold]
    
    logger.info(f"Found {len(agreed_items)} research areas that meet the agreement threshold")
    
    # Format results for return - create a single list containing all agreed items
    final_consolidated_list = [agreed_items] if agreed_items else []
    
    return final_consolidated_list, all_candidates, all_raw_results


async def process_level1_term(level1_term: str,
                              provider: Optional[str] = None,
                              session: Optional[Any] = None,
                              general_semaphore: Optional[asyncio.Semaphore] = None,
                              browser_semaphore: Optional[asyncio.Semaphore] = None,
                              min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                              model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                              num_llm_attempts: int = processing_config.llm_attempts,
                              agreement_threshold: int = processing_config.concept_agreement_threshold,
                              prefetched_search_results: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Process a single level1 term to extract research areas and save detailed metadata"""
    logger.info(f"Processing level 1 term: {level1_term} (LLM Min Score: {min_score_for_llm}, Models: {model_types}, Attempts: {num_llm_attempts}, Agree: {agreement_threshold})")
    
    # Initialize structures for detailed metadata
    term_details = {
        "level1_term": level1_term,
        "all_urls": [],
        "raw_extracted_lists": [],
        "llm_io_pairs": [],
        "final_consolidated_areas": [],
        "error": None
    }
    
    # Initialize structure for raw URL lists
    url_to_raw_lists = {} # Moved initialization here
    
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
            keywords=RESEARCH_KEYWORDS,
            anti_keywords=NON_RESEARCH_KEYWORDS,
            patterns=RESEARCH_PATTERNS
        )
        
        filter_config = FilterConfig(
            scoring_fn=score_research_area_list,
            clean_item_fn=clean_research_area,
            provider=provider,
            use_llm_validation=True,
            binary_llm_decision=False,
            binary_system_prompt=RESEARCH_VALIDATION_SYSTEM_PROMPT_TEMPLATE.format(term=level1_term),
            min_score_for_llm=min_score_for_llm, # Pass threshold
            # model_type will be set within run_multiple_llm_extractions
        )
        
        # Use prefetched search results if provided, otherwise perform search
        search_results = prefetched_search_results
        if search_results is None:
            # Use SEARCH_QUERIES constant with formatting
            queries = [query.format(term=level1_term) for query in SEARCH_QUERIES]
            
            # Perform web search with multiple queries
            logger.info(f"Searching with {len(queries)} different queries for '{level1_term}'")
            search_results = web_search_bulk(queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)
        
        if not search_results or not search_results.get("data"):
            logger.warning(f"No search results for '{level1_term}'")
            term_details["error"] = "No search results"
            # Save partial details before returning -> NO! Return the details instead.
            # save_detailed_metadata(level1_term, term_details)
            return {
                "level1_term": level1_term, 
                "research_areas": [], 
                "count": 0, 
                "verified": False, 
                "num_urls": 0, 
                "num_lists": 0,
                "term_details": term_details, # Return details
                "url_to_raw_lists": {} # Return empty raw lists
            }
        
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
        term_details["all_urls"] = urls
    
        if not urls:
            logger.warning(f"No URLs found in search results for '{level1_term}'")
            term_details["error"] = "No URLs found in search results"
            # save_detailed_metadata(level1_term, term_details)
            return {
                "level1_term": level1_term, 
                "research_areas": [], 
                "count": 0, 
                "verified": False, 
                "num_urls": 0, 
                "num_lists": 0,
                "term_details": term_details, # Return details
                "url_to_raw_lists": {} # Return empty raw lists
            }
            
        logger.info(f"Found {len(urls)} URLs for '{level1_term}'")
            
        # Configure semaphore for concurrent requests
        # Use the provided general_semaphore if available, otherwise create one
        # (Note: creating a new semaphore here might not be ideal if called directly,
        #  but it ensures it exists when called from the batch processor)
        semaphore_to_use = general_semaphore or asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        all_extracted_lists_raw = [] # Store raw list dicts
        # url_to_raw_lists = {} # Moved initialization higher up
        
        # If session is provided, use it, otherwise create a new one
        if session is None:
            # Create custom connector with proper connection management
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context(),
                limit=MAX_CONCURRENT_REQUESTS,
                limit_per_host=2,
                force_close=True  # Force close connections to prevent descriptor reuse
            )
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=60),
                raise_for_status=False
            ) as new_session:
                # Fetch webpages
                fetch_tasks = [
                    fetch_webpage(url, new_session, semaphore_to_use, browser_semaphore, html_config, level1_term, logger)
                    for url in urls[:MAX_SEARCH_RESULTS]
                ]
                
                # Use gather with return_exceptions=True to continue even if some requests fail
                html_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                
                # Process each webpage
                for i, (url, result) in enumerate(zip(urls[:MAX_SEARCH_RESULTS], html_contents)):
                    # Skip if the result is an exception
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching content from {url}: {str(result)}")
                        continue
                        
                    html_content = result
                    if not html_content:
                        continue
                        
                    # Extract lists from the webpage
                    extracted_lists = extract_lists_from_html(html_content, list_config)
            
                    if extracted_lists:
                        all_extracted_lists_raw.extend(extracted_lists) # Append raw dicts
                        
                        # Store only string items from the extracted lists, not the full dictionary
                        url_lists = []
                        for list_data in extracted_lists:
                            if isinstance(list_data, dict) and "items" in list_data:
                                items = list_data["items"]
                                if isinstance(items, list):
                                    # Make sure all items are strings
                                    clean_items = [str(item) for item in items if item]
                                    if clean_items:
                                        url_lists.append(clean_items)
                            elif isinstance(list_data, list):
                                # If it's already a list, ensure all items are strings
                                clean_items = [str(item) for item in list_data if item]
                                if clean_items:
                                    url_lists.append(clean_items)
                        
                        # Only store if we have valid lists
                        if url_lists:
                            url_to_raw_lists[url] = url_lists # Store raw lists per URL
                        
                        logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
        else:
            # Use the provided session
            fetch_tasks = [
                fetch_webpage(url, session, semaphore_to_use, browser_semaphore, html_config, level1_term, logger)
                for url in urls[:MAX_SEARCH_RESULTS]
            ]
            
            # Use gather with return_exceptions=True to continue even if some requests fail
            html_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # Process each webpage
            for i, (url, result) in enumerate(zip(urls[:MAX_SEARCH_RESULTS], html_contents)):
                # Skip if the result is an exception
                if isinstance(result, Exception):
                    logger.error(f"Error fetching content from {url}: {str(result)}")
                    continue
                    
                html_content = result
                if not html_content:
                    continue
                    
                # Extract lists from the webpage
                extracted_lists = extract_lists_from_html(html_content, list_config)
                
                if extracted_lists:
                    all_extracted_lists_raw.extend(extracted_lists) # Append raw dicts
                    
                    # Store only string items from the extracted lists, not the full dictionary
                    url_lists = []
                    for list_data in extracted_lists:
                        if isinstance(list_data, dict) and "items" in list_data:
                            items = list_data["items"]
                            if isinstance(items, list):
                                # Make sure all items are strings
                                clean_items = [str(item) for item in items if item]
                                if clean_items:
                                    url_lists.append(clean_items)
                        elif isinstance(list_data, list):
                            # If it's already a list, ensure all items are strings
                            clean_items = [str(item) for item in list_data if item]
                            if clean_items:
                                url_lists.append(clean_items)
                    
                    # Only store if we have valid lists
                    if url_lists:
                        url_to_raw_lists[url] = url_lists # Store raw lists per URL
                    
                    logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
            
        term_details["raw_extracted_lists"] = all_extracted_lists_raw

        # Filter and validate lists
        if not all_extracted_lists_raw:
            logger.warning(f"No lists extracted for '{level1_term}'")
            term_details["error"] = "No lists extracted from fetched HTML"
            # save_detailed_metadata(level1_term, term_details)
            return {
                "level1_term": level1_term, 
                "research_areas": [], 
                "count": 0, 
                "verified": False, 
                "num_urls": len(urls), 
                "num_lists": 0,
                "term_details": term_details, # Return details
                "url_to_raw_lists": url_to_raw_lists # Return raw lists collected so far
            }
            
        logger.info(f"Extracted a total of {len(all_extracted_lists_raw)} raw lists for '{level1_term}'. Starting filtering...")
        
        # Filter lists using the shared filtering utility - Capture necessary returned values
        final_llm_output_lists, llm_candidates, llm_results = await run_multiple_llm_extractions(
            all_extracted_lists_raw, level1_term, filter_config, 
            num_attempts=num_llm_attempts,
            agreement_threshold=agreement_threshold,
            logger=logger,
            model_types=model_types # Pass model types here
        )
        
        # Ensure proper handling of filter_lists output
        cleaned_final_lists = []
        
        if isinstance(final_llm_output_lists, tuple) and len(final_llm_output_lists) == 3:
            # In case filter_lists returns nested tuple structure
            final_lists_part = final_llm_output_lists[0]
            if isinstance(final_lists_part, list):
                for lst in final_lists_part:
                    if isinstance(lst, list):
                        # Add list of strings
                        cleaned_final_lists.append([str(item) for item in lst if isinstance(item, str) and item])
                    elif isinstance(lst, str):
                        # Single string item becomes a list with one item
                        cleaned_final_lists.append([lst])
        elif isinstance(final_llm_output_lists, list):
            # Normal case - list of lists or list of strings
            for lst in final_llm_output_lists:
                if isinstance(lst, list):
                    # Handle list of strings
                    cleaned_final_lists.append([str(item) for item in lst if isinstance(item, str) and item])
                elif isinstance(lst, str):
                    # Single string becomes a list with one item
                    cleaned_final_lists.append([lst])
                elif isinstance(lst, dict) and 'items' in lst:
                    # Handle dictionary with 'items' field
                    items = lst['items']
                    if isinstance(items, list):
                        cleaned_final_lists.append([str(item) for item in items if isinstance(item, str) and item])
        
        # Ensure final_llm_output_lists is in the expected format for consolidate_lists
        final_llm_output_lists = cleaned_final_lists
        
        # Store intermediate results in term_details in the new format
        term_details["llm_io_pairs"] = []
        if llm_candidates and llm_results:
             # Ensure results align with candidates (should match length)
             num_pairs = min(len(llm_candidates), len(llm_results))
             if len(llm_candidates) != len(llm_results):
                  logger.warning(f"Mismatch between LLM candidates ({len(llm_candidates)}) and results ({len(llm_results)}) for {level1_term}. Pairing up to {num_pairs}.")
             
             for i in range(num_pairs):
                  # Ensure candidate has 'items' field
                  candidate_input = llm_candidates[i].get('items', []) if isinstance(llm_candidates[i], dict) else llm_candidates[i]
                  llm_output = llm_results[i]
                  term_details["llm_io_pairs"].append({
                       "input_list_to_llm": candidate_input,
                       "output_list_from_llm": llm_output
                  })
        
        if not final_llm_output_lists:
            logger.warning(f"No lists passed filtering/LLM validation for '{level1_term}'")
            term_details["error"] = "No lists passed filtering/LLM validation"
            # save_detailed_metadata(level1_term, term_details)
            return {
                "level1_term": level1_term, 
                "research_areas": [], 
                "count": 0, 
                "verified": False, 
                "num_urls": len(urls), 
                "num_lists": len(all_extracted_lists_raw),
                "term_details": term_details, # Return details
                "url_to_raw_lists": url_to_raw_lists # Return raw lists collected so far
            }
            
        logger.info(f"After filtering/LLM, {len(final_llm_output_lists)} lists/sub-lists remain for '{level1_term}'")
        
        # Consolidate research areas
        research_areas = consolidate_lists(
            final_llm_output_lists, 
            level1_term, 
            min_frequency=1,
            min_list_appearances=1,
            similarity_threshold=0.7
        )
        term_details["final_consolidated_areas"] = research_areas
        
        logger.info(f"Consolidated to {len(research_areas)} unique research areas for '{level1_term}'")
        
        # Simplified source/quality tracking
        research_area_quality = {area: 1.0 for area in research_areas}

        # Save detailed metadata for this term -> NO! Return it instead.
        # save_detailed_metadata(level1_term, term_details)

        # Save the raw URL results for this term -> NO! Return it instead.
        # save_raw_url_results(level1_term, url_to_raw_lists)

        # Return main results needed for aggregation PLUS the data to be saved later
        return {
            "level1_term": level1_term,
            "research_areas": research_areas,
            "count": len(research_areas),
            "url_sources": {}, # Simplified
            "quality_scores": research_area_quality, # Simplified
            "verified": len(research_areas) > 0,
            "num_urls": len(urls),
            "num_lists": len(all_extracted_lists_raw), # Report raw list count
            "term_details": term_details, # Return details for later saving
            "url_to_raw_lists": url_to_raw_lists # Return raw lists for later saving
        }
            
    except Exception as e:
        logger.error(f"Error processing term '{level1_term}': {str(e)}", exc_info=True)
        term_details["error"] = f"Unhandled exception: {str(e)}"
        # save_detailed_metadata(level1_term, term_details) # Attempt to save details on error -> NO! Return it instead.
        # Return error structure including details
        return {
            "level1_term": level1_term, 
            "research_areas": [], 
            "count": 0, 
            "verified": False, 
            "num_urls": len(term_details.get('all_urls', [])), # Use details if available
            "num_lists": len(term_details.get('raw_extracted_lists', [])), # Use details if available
            "error": str(e),
            "term_details": term_details, # Return details even on error
            "url_to_raw_lists": url_to_raw_lists # Return any collected raw lists
        }

# def save_detailed_metadata(level1_term: str, data: Dict[str, Any]):
#     \"\"\"Saves the detailed processing metadata for a single level1 term to a JSON file.\"\"\"
#     try:
#         # Sanitize filename
#         safe_filename = re.sub(r'[\/:*?"<>|]', '_', level1_term) + "_details.json"
#         output_path = os.path.join(level_config.data_dir / "detailed_meta", safe_filename)
        
#         # Ensure the directory exists
#         os.makedirs(level_config.data_dir / "detailed_meta", exist_ok=True)
        
#         with open(output_path, "w", encoding="utf-8") as f:
#             # Use default handler for non-serializable objects if any slip through (e.g., sets)
#             json.dump(data, f, indent=2, default=lambda o: f"<non-serializable: {type(o).__name__}>")
#         logger.info(f"Saved detailed metadata for '{level1_term}' to {output_path}")
#     except Exception as e:
#         logger.error(f"Failed to save detailed metadata for '{level1_term}': {str(e)}", exc_info=True)


def perform_bulk_web_search(level1_terms: List[str], search_config: WebSearchConfig) -> Dict[str, Dict[str, Any]]:
    """
    Perform a bulk web search for multiple level 1 terms at once.
    
    Args:
        level1_terms: List of level 1 terms to search for
        search_config: Configuration for web search
        
    Returns:
        A dictionary mapping level1_term to its search results
    """
    if not level1_terms:
        return {}
    
    # Use global SEARCH_QUERIES constant
    queries_per_term = SEARCH_QUERIES
    
    # Dynamic calculation of max terms per search batch
    num_queries_per_term = len(queries_per_term)
    max_terms_per_search = MAX_SEARCH_QUERIES // num_queries_per_term
    
    # Limit the number of terms per batch to prevent exceeding API limits
    terms_per_batch = min(len(level1_terms), max_terms_per_search)
    logger.info(f"Performing bulk web search for {len(level1_terms)} terms (max {max_terms_per_search} per batch, using {num_queries_per_term} queries per term)")
    
    # Prepare all queries for all terms
    all_queries = []
    term_to_query_indices = {}  # Maps each term to its query indices in all_queries
    
    for i, term in enumerate(level1_terms):
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
        logger.warning(f"No search results found for any of the {len(level1_terms)} terms")
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
    
    logger.info(f"Got search results for {len(term_to_results)} out of {len(level1_terms)} terms")
    return term_to_results


async def process_level1_terms_batch(batch: List[str],
                                   provider: Optional[str] = None,
                                   session: Optional[Any] = None,
                                   general_semaphore: Optional[asyncio.Semaphore] = None,
                                   browser_semaphore: Optional[asyncio.Semaphore] = None,
                                   min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                                   model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                                   num_llm_attempts: int = processing_config.llm_attempts,
                                   agreement_threshold: int = processing_config.concept_agreement_threshold
                                   ) -> List[Dict[str, Any]]:
    """Process a batch of level 1 terms with optimized bulk web searching"""
    if not batch:
        return []
        
    logger.info(f"Processing batch of {len(batch)} level 1 terms")
    
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
        
        # Perform the bulk web search for this mini-batch
        batch_results = perform_bulk_web_search(search_batch, search_config)
        search_results_by_term.update(batch_results)
    
    # Process each term with its pre-fetched search results
    tasks = []
    for term in batch:
        # Get the pre-fetched search results for this term (if any)
        prefetched_results = search_results_by_term.get(term)
        
        # Process the term
        task = process_level1_term(
            term,
            provider,
            session,
            general_semaphore,
            browser_semaphore,
            min_score_for_llm,
            model_types,
            num_llm_attempts,
            agreement_threshold,
            prefetched_results
        )
        tasks.append(task)
    
    # Run all tasks in parallel
    return await asyncio.gather(*tasks)


def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        level_config.data_dir / "cache",
        level_config.data_dir / "raw_search",
        level_config.data_dir / "detailed_meta",
        os.path.dirname(level_config.get_step_output_file(0)),
        os.path.dirname(level_config.get_step_metadata_file(0))
    ]
    
    logger.info(f"BASE_DIR: {str(level_config.data_dir.parent.parent)}")
    logger.info(f"LV1_INPUT_FILE: {str(get_level_config(1).get_final_file())}")
    logger.info(f"OUTPUT_FILE: {level_config.get_step_output_file(0)}")
    logger.info(f"META_FILE: {level_config.get_step_metadata_file(0)}")
    logger.info(f"CACHE_DIR: {level_config.data_dir / "cache"}")
    logger.info(f"RAW_SEARCH_DIR: {level_config.data_dir / "raw_search"}")
    logger.info(f"DETAILED_META_DIR: {level_config.data_dir / "detailed_meta"}")
    
    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise


async def main_async():
    """Async main execution function"""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Extract research areas for level 1 terms.")
        parser.add_argument("--provider", help="LLM provider (e.g., gemini, openai)")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for processing terms (default: {BATCH_SIZE})")
        parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, help=f"Max concurrent term processing requests (default: {MAX_CONCURRENT_REQUESTS})")
        parser.add_argument("--input-file", default=str(get_level_config(1).get_final_file()), help=f"Path to the input file containing level 1 terms (default: {str(get_level_config(1).get_final_file())})")
        parser.add_argument("--append", action='store_true', help="Append results to existing output files instead of overwriting.")
        parser.add_argument("--llm-attempts", type=int, default=processing_config.llm_attempts, help=f"Number of LLM extraction attempts per term (default: {processing_config.llm_attempts})")
        parser.add_argument("--agreement-threshold", type=int, default=processing_config.concept_agreement_threshold, help=f"Minimum appearances threshold for research areas (default: {processing_config.concept_agreement_threshold})")
        parser.add_argument("--min-score-for-llm", type=float, default=DEFAULT_MIN_SCORE_FOR_LLM, help=f"Minimum heuristic score to send a list to LLM (default: {DEFAULT_MIN_SCORE_FOR_LLM})")
        parser.add_argument("--llm-model-types", type=str, default=",".join(DEFAULT_LLM_MODEL_TYPES), help=f"Comma-separated LLM model types for attempts (e.g., default,pro,mini) (default: {','.join(DEFAULT_LLM_MODEL_TYPES)})")
        
        args = parser.parse_args()
        
        provider = args.provider
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        input_file_path = args.input_file
        append_mode = args.append
        num_llm_attempts = args.llm_attempts
        agreement_threshold = args.agreement_threshold
        min_score_for_llm = args.min_score_for_llm
        llm_model_types = [mt.strip() for mt in args.llm_model_types.split(',') if mt.strip()]
        
        # Update configuration with command line arguments if provided
        # These are now passed directly to functions, no need to update Config class vars
        # processing_config.llm_attempts = num_llm_attempts 
        # processing_config.concept_agreement_threshold = agreement_threshold
        
        if provider:
            logger.info(f"Using provider: {provider}")
        if batch_size != BATCH_SIZE:
            logger.info(f"Using custom batch size: {batch_size}")
        if max_concurrent != MAX_CONCURRENT_REQUESTS:
             logger.info(f"Using custom concurrent limit: {max_concurrent}")
        if min_score_for_llm != DEFAULT_MIN_SCORE_FOR_LLM:
            logger.info(f"Using custom LLM score threshold: {min_score_for_llm}")
        if args.llm_model_types != ",".join(DEFAULT_LLM_MODEL_TYPES):
            logger.info(f"Using custom LLM model types: {llm_model_types}")
        if input_file_path != str(get_level_config(1).get_final_file()):
             logger.info(f"Using custom input file: {input_file_path}")
        if append_mode:
             logger.info("Append mode enabled. Results will be added to existing files.")
        
        logger.info(f"Using {num_llm_attempts} LLM extraction attempts with agreement threshold of {agreement_threshold}")
        logger.info("Starting research areas extraction using level 1 terms")
        logger.info(f"Using optimized web search with dynamically calculated max terms per batch")
        
        # Create output directories
        ensure_dirs_exist()
        
        # Read level 1 terms from the specified input file
        level1_terms = read_level1_terms(input_file_path)
        
        logger.info(f"Processing {len(level1_terms)} level 1 terms from '{input_file_path}' with batch size {batch_size} and max {max_concurrent} concurrent terms")
        
        # Use single output file and metadata
        output_file = level_config.get_step_output_file(0)
        meta_file = level_config.get_step_metadata_file(0)
        current_provider = provider or Provider.GEMINI  # Default to Gemini if not specified
        
        logger.info(f"Using provider: {current_provider} with {num_llm_attempts} LLM attempts (models: {llm_model_types}), agreement threshold {agreement_threshold}, and score threshold {min_score_for_llm}")
        
        # Initialize aiohttp session for the entire run
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, CookieJar
        # Import browser semaphore limit from html_fetch
        from generate_glossary.utils.web_search.html_fetch import MAX_CONCURRENT_BROWSERS

        # Create a default SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Improved connector configuration to handle file descriptor issues
        connector = TCPConnector(
            ssl=ssl_context,
            limit=max_concurrent,  # Ensures this uses the potentially overridden value
            limit_per_host=2,
            force_close=True,  # Force close connections to prevent descriptor reuse
            enable_cleanup_closed=True  # Enable cleanup of closed connections
        )
        
        cookie_jar = CookieJar(unsafe=True)  # Allow unsafe cookies to be more permissive
        
        all_results = []
        start_time = time.time()
        
        # Create the semaphores
        general_semaphore = asyncio.Semaphore(max_concurrent) # General fetch semaphore
        browser_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BROWSERS) # Specific browser semaphore
        logger.info(f"Using general fetch semaphore limit: {max_concurrent}")
        logger.info(f"Using headless browser semaphore limit: {MAX_CONCURRENT_BROWSERS}")

        # Process in batches
        timeout = ClientTimeout(total=3600, connect=30)  # 1 hour total timeout, 30s connect timeout
        async with ClientSession(
            connector=connector,
            cookie_jar=cookie_jar, 
            timeout=timeout,
            raise_for_status=False
        ) as session:
            for i in range(0, len(level1_terms), batch_size):
                batch = level1_terms[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(level1_terms) + batch_size - 1)//batch_size}")
                
                # Process the batch using the optimized batch processor
                batch_results = await process_level1_terms_batch(
                    batch,
                    current_provider,
                    session,
                    general_semaphore,
                    browser_semaphore,
                    min_score_for_llm,
                    llm_model_types,
                    num_llm_attempts,
                    agreement_threshold
                )
                
                all_results.extend(batch_results)
                
                # Log progress after each batch
                elapsed = time.time() - start_time
                terms_processed = min(i + batch_size, len(level1_terms))
                terms_per_second = terms_processed / max(1, elapsed)
                eta_seconds = (len(level1_terms) - terms_processed) / max(0.1, terms_per_second)
                
                logger.info(f"Processed {terms_processed}/{len(level1_terms)} terms "
                            f"({terms_processed/len(level1_terms)*100:.1f}%) in {elapsed:.1f}s "
                            f"({terms_per_second:.2f} terms/s, ETA: {eta_seconds/60:.1f}m)")
                
                # Add a small delay between batches to avoid overloading
                await asyncio.sleep(2)  # Increased delay to reduce pressure on resources
        
        # Collect all research areas and prepare for saving
        found_research_areas = []
        verified_terms_count = 0
        processed_stats = {
            "total_urls_processed": 0,
            "total_lists_found": 0,
            "verified_research_areas_count": 0
        }
        level1_term_result_counts = {}
        level1_to_research_areas = {}
        research_area_sources = {}
        research_area_quality_scores = {}
        
        # --- Data to be saved later ---
        all_term_details = []
        all_raw_url_lists = {}
        
        for result in all_results:
            level1_term = result["level1_term"]
            
            # Store details and raw lists for saving later (but don't save them individually)
            # if "term_details" in result:
            #     all_term_details.append(result["term_details"])
            # if "url_to_raw_lists" in result and result["url_to_raw_lists"]:
            #     all_raw_url_lists[level1_term] = result["url_to_raw_lists"]
                
            if result.get("error"):
                logger.error(f"Skipping aggregation for term '{result['level1_term']}' due to processing error: {result['error']}")
                continue
                
            research_areas = result["research_areas"]
            quality_scores = result.get("quality_scores", {})
            verified = result.get("verified", False)
            num_urls = result.get("num_urls", 0)
            num_lists = result.get("num_lists", 0)
            
            processed_stats["total_urls_processed"] += num_urls
            processed_stats["total_lists_found"] += num_lists # Total raw lists found
            
            if verified:
                verified_terms_count += 1
                processed_stats["verified_research_areas_count"] += len(research_areas)
                
                if level1_term not in level1_to_research_areas:
                    level1_to_research_areas[level1_term] = []
                level1_to_research_areas[level1_term].extend(research_areas)
                found_research_areas.extend(research_areas)
                
                level1_term_result_counts[level1_term] = len(research_areas)
                
                for area in research_areas:
                    if area not in research_area_sources:
                        research_area_sources[area] = []
                    research_area_sources[area].append(level1_term)
                    
                    # Store quality scores
                    if area in quality_scores:
                        research_area_quality_scores[area] = quality_scores[area]
                    else:
                        research_area_quality_scores[area] = 1.0  # Default quality score
        
        logger.info(f"Consolidated results from {len(all_results)} processed terms.")
        logger.info(f"Found {verified_terms_count} verified terms.")
        
        # --- Saving detailed metadata for each term ---
        # logger.info(f"Saving detailed metadata for {len(all_term_details)} terms (DISABLED)")
        # for term_detail in all_term_details:
        #     if "level1_term" in term_detail:
        #         term = term_detail["level1_term"]
        #         # save_detailed_metadata(term, term_detail)
        
        # --- Saving raw URL results for each term ---
        # logger.info(f"Saving raw URL results for {len(all_raw_url_lists)} terms (DISABLED)")
        # for term, url_lists in all_raw_url_lists.items():
        #     # save_raw_url_results(term, url_lists)
        
        # Explicitly close any remaining resources to avoid "Too many open files" error
        logger.info("⚠️ Resource management: Explicitly closing any remaining open resources")
        try:
            # Python's GC should handle most resource cleanup, but we can help it along
            # by clearing large collections and running GC
            import gc
            logger.debug("Clearing memory and running garbage collection...")
            all_term_details.clear()
            all_raw_url_lists.clear()
            gc.collect()
            logger.debug("Memory cleanup complete")
        except Exception as cleanup_error:
            logger.warning(f"Non-critical error during cleanup: {cleanup_error}")
        
        # --- File Saving Logic (Final Output and Metadata) ---
        
        existing_unique_areas = set()
        if append_mode and os.path.exists(output_file):
            logger.info(f"Loading existing research areas from {output_file} for append mode.")
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        existing_unique_areas.add(line.strip())
                logger.info(f"Loaded {len(existing_unique_areas)} existing unique areas.")
            except Exception as e:
                logger.warning(f"Could not read existing output file {output_file}: {e}. Starting fresh.", exc_info=True)
                existing_unique_areas = set() # Start fresh if read fails
        
        # Add newly found unique areas
        final_unique_areas_set = set(existing_unique_areas)
        newly_added_count = 0
        for area in found_research_areas:
            if area not in final_unique_areas_set:
                final_unique_areas_set.add(area)
                newly_added_count += 1
        
        logger.info(f"Added {newly_added_count} new unique research areas.")
        
        # Save output file (unique areas)
        final_unique_areas_list = sorted(list(final_unique_areas_set))
        random.shuffle(final_unique_areas_list) # Keep randomization if desired
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for area in final_unique_areas_list:
                    # Write research areas in lowercase for consistency
                    f.write(f"{area.lower()}\n")
            logger.info(f"Saved {len(final_unique_areas_list)} total unique research areas to {output_file}")
        except Exception as e:
            logger.error(f"Failed to write output file {output_file}: {e}", exc_info=True)

        # Save metadata file
        
        # Load existing metadata if in append mode
        metadata = {}
        if append_mode and os.path.exists(meta_file):
            logger.info(f"Loading existing metadata from {meta_file} for append mode.")
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info("Existing metadata loaded successfully.")
            except Exception as e:
                 logger.warning(f"Could not read or parse existing metadata file {meta_file}: {e}. Creating new metadata.", exc_info=True)
                 metadata = {} # Start fresh if read/parse fails
        
        # Update metadata fields
        metadata["execution_time"] = f"{time.time() - start_time:.2f} seconds"
        metadata["total_unique_research_areas"] = len(final_unique_areas_list) # Update with final count
        
        # Update counts and mappings, merging carefully
        metadata["total_level1_terms_processed"] = metadata.get("total_level1_terms_processed", 0) + len(level1_terms) # Increment total processed
        metadata["verified_level1_terms_count"] = metadata.get("verified_level1_terms_count", 0) + verified_terms_count # Increment verified
        
        # Merge result counts
        existing_result_counts = metadata.get("level1_term_result_counts", {})
        existing_result_counts.update(level1_term_result_counts) # Add/overwrite counts for newly processed terms
        metadata["level1_term_result_counts"] = existing_result_counts
        
        # Merge research area mappings
        existing_mapping = metadata.get("level1_to_research_areas_mapping", {})
        for term, areas in level1_to_research_areas.items():
            if term not in existing_mapping:
                existing_mapping[term] = []
            # Add only unique new areas for this term
            existing_areas_set = set(existing_mapping[term])
            for area in areas:
                 if area not in existing_areas_set:
                      existing_mapping[term].append(area)
                      existing_areas_set.add(area)
        metadata["level1_to_research_areas_mapping"] = existing_mapping
        
        # Merge research area sources
        existing_sources = metadata.get("research_area_level1_sources", {})
        for area, sources in research_area_sources.items():
             if area not in existing_sources:
                 existing_sources[area] = []
             # Add unique new source terms for this area
             existing_source_set = set(existing_sources[area])
             for source in sources:
                  if source not in existing_source_set:
                       existing_sources[area].append(source)
                       existing_source_set.add(source)
        metadata["research_area_level1_sources"] = existing_sources
        
        # Update quality scores
        existing_quality_scores = metadata.get("research_area_quality_scores", {})
        for area, score in research_area_quality_scores.items():
            if area not in existing_quality_scores:
                existing_quality_scores[area] = score
            else:
                # Take the maximum score if the area already exists
                existing_quality_scores[area] = max(existing_quality_scores[area], score)
        metadata["research_area_quality_scores"] = existing_quality_scores

        # Update provider, concurrent, batch size, and LLM attempts information
        metadata["provider"] = current_provider
        metadata["max_concurrent"] = max_concurrent
        metadata["batch_size"] = batch_size
        metadata["num_llm_attempts"] = num_llm_attempts
        metadata["agreement_threshold"] = agreement_threshold
        metadata["min_score_for_llm"] = min_score_for_llm
        metadata["llm_model_types_used"] = llm_model_types[:num_llm_attempts] # Record actual models used

        # Update processing stats (merge/add)
        metadata["total_urls_processed"] = metadata.get("total_urls_processed", 0) + processed_stats["total_urls_processed"]
        metadata["total_raw_lists_extracted"] = metadata.get("total_raw_lists_extracted", 0) + processed_stats["total_lists_found"]
        
        # Recalculate averages based on updated totals
        total_terms_ever_processed = metadata["total_level1_terms_processed"]
        total_verified_ever = metadata["verified_level1_terms_count"]
        total_verified_areas_ever = sum(len(areas) for areas in metadata["level1_to_research_areas_mapping"].values()) # More accurate way to count total verified areas

        metadata["processing_stats"] = {
             "avg_urls_per_term": metadata["total_urls_processed"] / max(1, total_terms_ever_processed),
             "avg_raw_lists_per_term": metadata["total_raw_lists_extracted"] / max(1, total_terms_ever_processed),
             "avg_final_areas_per_verified_term": total_verified_areas_ever / max(1, total_verified_ever)
         }

        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({
                    # 1. level1_to_research_areas mapping
                    "level1_to_research_areas_mapping": {
                        term: [area.lower() for area in areas] for term, areas in existing_mapping.items()
                    },
                    
                    # 2. research area sources mapping
                    "research_area_level1_sources": {
                        area.lower(): sources for area, sources in existing_sources.items()
                    },
                    
                    # 3. Research area quality scores
                    "research_area_quality_scores": {
                        area.lower(): score for area, score in existing_quality_scores.items()
                    },
                    
                    # 4. All other metadata
                    "metadata": {
                        "execution_time": f"{time.time() - start_time:.2f} seconds",
                        "total_unique_research_areas": len(final_unique_areas_list),
                        "total_level1_terms_processed": metadata.get("total_level1_terms_processed", 0) + len(level1_terms),
                        "verified_level1_terms_count": metadata.get("verified_level1_terms_count", 0) + verified_terms_count,
                        "level1_term_result_counts": existing_result_counts,
                        "provider": current_provider,
                        "max_concurrent": max_concurrent,
                        "batch_size": batch_size,
                        "num_llm_attempts": num_llm_attempts,
                        "agreement_threshold": agreement_threshold,
                        "min_score_for_llm": min_score_for_llm,
                        "llm_model_types_used": llm_model_types[:num_llm_attempts],
                        "total_urls_processed": metadata.get("total_urls_processed", 0) + processed_stats["total_urls_processed"],
                        "total_raw_lists_extracted": metadata.get("total_raw_lists_extracted", 0) + processed_stats["total_lists_found"],
                        "processing_stats": {
                            "avg_urls_per_term": metadata["total_urls_processed"] / max(1, total_terms_ever_processed),
                            "avg_raw_lists_per_term": metadata["total_raw_lists_extracted"] / max(1, total_terms_ever_processed),
                            "avg_final_areas_per_verified_term": total_verified_areas_ever / max(1, total_verified_ever)
                        }
                    }
                }, f, indent=2)
            logger.info(f"Final aggregated metadata saved to {meta_file}")
        except Exception as e:
            logger.error(f"Failed to write metadata file {meta_file}: {e}", exc_info=True)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Research areas extraction completed")

def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 