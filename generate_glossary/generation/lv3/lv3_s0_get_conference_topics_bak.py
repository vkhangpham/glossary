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
# Add import for ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor

# Fix import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import Provider

# Import shared web search utilities
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage, MAX_CONCURRENT_BROWSERS
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, consolidate_lists, init_llm
from generate_glossary.utils.web_search.filtering import filter_lists as original_filter_lists

# Add imports for model constants and random model selection
from generate_glossary.utils.llm import OPENAI_MODELS, GEMINI_MODELS

# Load environment variables and setup logging
load_dotenv('.env')
logger = setup_logger("lv3.s0")
# Prevent log propagation to avoid duplicate log messages
logger.propagate = False
random.seed(42)

# Constants
MAX_SEARCH_RESULTS = 30
MAX_CONCURRENT_REQUESTS = 8
BATCH_SIZE = 10  
MAX_SEARCH_QUERIES = 50

# Conference/journal-related keywords for enhanced filtering
TOPIC_KEYWORDS = [
    "topic", "track", "workshop", "symposium", "session", "theme", 
    "special issue", "call for papers", "cfp", "submission", "paper", 
    "research", "area", "program", "panel", "tutorial", "keynote",
    "presentation", "publication", "proceedings", "accepted", "contribution",
    "article", "manuscript", "work", "study", "investigation", "analysis",
    "focus", "interest", "perspective", "approach", "methodology", "framework",
    r"[\w\s&,\'-]+ (?:topics|papers|submissions)"
]

# Anti-keywords indicating non-topic text
NON_TOPIC_KEYWORDS = [
    "login", "sign in", "register", "apply", "contact", 
    "about", "home", "sitemap", "search", "privacy", "terms", "copyright",
    "registration fee", "accommodation", "venue", "travel", "sponsors",
    "deadline", "date", "schedule", "time", "location", "place", "map",
    "committee", "chair", "organizer", "editor", "reviewer", "attendee",
    "download", "upload", "submit", "send", "share", "follow", "like",
    "tweet", "post", "comment", "subscribe", "newsletter", "blog",
    "click here", "link", "back", "next", "previous", "continue"
]

# Topic pattern matches for regex matching
TOPIC_PATTERNS = [
    r"(?:topics|tracks|workshops|sessions) (?:of|on|in|for) [\w\s&,'-]+",
    r"(?:special issues?|special sections?) (?:on|in|about) [\w\s&,'-]+",
    r"(?:call for papers|cfp) (?:on|in|about) [\w\s&,'-]+",
    r"[\w\s&,'-]+ (?:track|workshop|session|symposium|panel)",
    r"[\w\s&,'-]+ (?:topics|papers|submissions)"
]

# Search queries - consolidated in one place
SEARCH_QUERIES = [
    "(journal of | conference on) {term} research topics",
    "(journal of | conference on) {term} call for papers",
    "{term} (topics | methods | tasks)",
]

# Conference topic validation system prompt template
CONFERENCE_VALIDATION_SYSTEM_PROMPT_TEMPLATE = """You are a highly meticulous academic research assistant specializing in identifying conference topics, journal special issues, and workshop themes.

Task: Carefully analyze the provided list of items. Extract ONLY the items that represent specific, legitimate conference topics, workshop themes, journal special issues, or symposium topics related to or within the field of **{term}**.

Input: A list of potential conference/journal topics.
Output: A Python-style list `[...]` containing ONLY the verified items from the input list that are DEFINITIVELY specific conference topics or journal special issues related to **{term}**.

EXTREMELY STRICT Exclusion Criteria - DO NOT INCLUDE:
1. **Navigation/administrative items:** Reject 'Home', 'About', 'Contact', 'Registration', 'Venue', 'Important Dates', 'Committees'.
2. **Generic terms without specific focus:** Reject vague terms like 'Research', 'Papers', 'Program' on their own without specific topic context.
3. **People or roles:** Reject 'Keynote Speakers', 'Program Committee', 'Chairs', 'Organizers', 'Editors'.
4. **Unrelated topics:** Reject topics clearly not related to **{term}** - if there is ANY doubt about relevance, EXCLUDE it.
5. **Logistics:** Reject 'Registration Fee', 'Accommodation', 'Travel Information'.
6. **Publication types without topics:** Reject generic 'Proceedings', 'Journal', 'Conference Papers' without specific research focus.
7. **Adjacent or neighboring fields:** EXCLUDE items from adjacent, related, or overlapping disciplines. ONLY include items that are CORE to {term} itself.
8. **Interdisciplinary areas:** EXCLUDE interdisciplinary items unless they are PRIMARILY related to {term} and the {term} aspect is the DOMINANT component.
9. **General conference/administrative text:** EXCLUDE ALL general programs, tracks, resources, or administrative text.

Guidelines:
- Be EXTREMELY SELECTIVE. When in doubt, EXCLUDE.
- Return an empty list `[]` if NO items in the input list meet these strict criteria.
- Accept ONLY specific research topics, paper categories, special issue themes, workshop focuses that would be recognized by experts in {term}.
- Focus on SUBJECT MATTER, not organizational structure or event logistics.
- Include topics that are sub-areas or specialized aspects of **{term}**.
- If a topic is slightly broader than **{term}** but would clearly include it as a core sub-field, you may include it.
- Output ONLY the Python-style list, with no extra text, explanation, or markdown formatting.

Example 1 (Machine Learning):
Input List: ["Home", "Reinforcement Learning Applications", "Registration", "Deep Learning for Computer Vision", "Program Committee", "Neural Network Architectures", "Important Dates", "Call for Papers", "Venue", "Machine Learning Ethics and Governance", "Contact Us", "Interpretable AI", "NLP Transformer Models"]
Output: ["Reinforcement Learning Applications", "Deep Learning for Computer Vision", "Neural Network Architectures", "Machine Learning Ethics and Governance", "Interpretable AI", "NLP Transformer Models"]

Example 2 (Quantum Physics):
Input List: ["Quantum Entanglement", "Conference Schedule", "Quantum Computing Algorithms", "Registration Information", "Quantum Field Theory", "Keynote Speakers", "Quantum Error Correction", "Topological Quantum Matter", "Submission Guidelines", "Quantum Sensing and Metrology", "Contact", "About", "Committee"]
Output: ["Quantum Entanglement", "Quantum Computing Algorithms", "Quantum Field Theory", "Quantum Error Correction", "Topological Quantum Matter", "Quantum Sensing and Metrology"]

Example 3 (Climate Science):
Input List: ["Climate Modeling and Predictions", "Registration", "Earth System Dynamics", "Conference Venue", "Extreme Weather Events", "Climate Change Mitigation Strategies", "Carbon Cycle Science", "Important Dates", "Organizers", "Climate Policy and Governance", "Ocean-Atmosphere Interactions", "Submit Abstract", "Contact Us"]
Output: ["Climate Modeling and Predictions", "Earth System Dynamics", "Extreme Weather Events", "Climate Change Mitigation Strategies", "Carbon Cycle Science", "Climate Policy and Governance", "Ocean-Atmosphere Interactions"]

I will ONLY include items that are DEFINITIVELY and DIRECTLY related to conference topics, journal special issues, or workshop themes in the field of **{term}**. Any item with even slight uncertainty about its direct relevance to {term} will be excluded.

Analyze the following list with EXTREME STRICTNESS:"""

DEFAULT_MIN_SCORE_FOR_LLM = 0.65
DEFAULT_LLM_MODEL_TYPES = ["default", "mini", "nano"]

class Config:
    """Configuration for the extraction pipeline"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    
    # Default level 2 input file path
    DEFAULT_LV2_INPUT_FILE = os.path.join(BASE_DIR, "data/lv2/lv2_final.txt")
    
    # Output file path (single run with multiple LLM attempts)
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s0_conference_topics.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s0_metadata.json")
    
    # Cache directory for web search results
    SEARCH_CACHE_DIR = os.path.join(BASE_DIR, "data/lv3/cache")
    
    # Detailed metadata for each term (for debugging and analysis)
    DETAILED_META_DIR = os.path.join(BASE_DIR, "data/lv3/detailed_metadata")
    
    # Raw search results directory
    RAW_SEARCH_DIR = os.path.join(BASE_DIR, "data/lv3/raw_search_results")
    
    # Configuration for multiple LLM attempts
    NUM_LLM_ATTEMPTS = 3  # Run the LLM extraction 3 times
    AGREEMENT_THRESHOLD = 2  # Conference topics must appear in at least 2 responses

def read_level2_terms(input_path: str) -> List[str]:
    """Read level 2 terms from input file"""
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            terms = [line.strip() for line in file.readlines() if line.strip()]
        logger.info(f"Successfully read {len(terms)} level 2 terms")
        return terms
    except Exception as e:
        logger.error(f"Failed to read level 2 terms: {str(e)}", exc_info=True)
        raise


def clean_conference_topic(item: str) -> str:
    """Clean a conference/journal topic"""
    # Remove common prefixes like "Topics on", "Call for papers on"
    item = item.strip()
    # item = re.sub(r'^(Topics|Tracks|Workshops|Sessions|Special Issues?|Call for Papers|CFP|Areas|Themes?) (on|in|of|for|about) ', '', item, flags=re.IGNORECASE)
    # Remove trailing numbers, parenthetical info
    item = re.sub(r'\s*\(\d+\).*$', '', item)
    item = re.sub(r'\s*\d+\s*$', '', item)
    # Remove URLs
    item = re.sub(r'http\S+', '', item)
    # Clean whitespace
    item = ' '.join(item.split())
    return item


def score_conference_topic_list(items: List[str], metadata: Dict[str, Any], context_term: str) -> float:
    """Score a conference topic list based on various heuristics"""
    # Adjust weights to prioritize content relevance over structure/size
    weights = {
        "keyword": 0.35,      # Presence of conference/journal keywords
        "structure": 0.05,    # HTML tag structure
        "pattern": 0.25,      # Consistency and relevance of item naming patterns
        "non_term": 0.25,     # Penalty for non-relevant/navigational terms
        "consistency": 0.05,  # General formatting consistency
        "size": 0.0,         # List size is less important
        "html_type": 0.05     # HTML element type
    }
    
    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6: 
        weights = {k: v / total_weight for k, v in weights.items()}

    # Use the common scoring function from list_extractor
    try:
        return score_list(
            items=items,
            metadata=metadata,
            context_term=context_term,
            keywords=TOPIC_KEYWORDS,
            scoring_weights=weights
        )
    except ImportError:
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
                 (1 - nav_score) * weights["structure"])
        return min(max(score, 0.0), 1.0)


def preprocess_html_content(html_content: str) -> str:
    """
    Remove code blocks and other problematic content from HTML before extraction
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Preprocessed HTML content with problematic sections removed
    """
    # Remove markdown code blocks
    html_content = re.sub(r'```[\w]*\n[\s\S]*?\n```', '', html_content)
    
    # Remove inline code blocks
    html_content = re.sub(r'`[^`\n]*`', '', html_content)
    
    # Remove script tags and their contents
    html_content = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html_content)
    
    # Remove style tags and their contents
    html_content = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html_content)
    
    return html_content


# Define a wrapper function for filter_lists that handles markdown in LLM responses
async def filter_lists(raw_lists, context_term, config, logger):
    """
    Wrapper around original filter_lists that handles markdown formatting in LLM responses
    
    Args:
        raw_lists: Raw lists to filter
        context_term: Context term for filtering
        config: Filter configuration
        logger: Logger instance
        
    Returns:
        Same return values as original filter_lists but with cleaned LLM responses
    """
    # Run the original filter_lists function
    final_llm_output_lists, llm_candidates, llm_results = await original_filter_lists(
        raw_lists, context_term, config, logger
    )
    
    # If LLM validation is not enabled, return original results
    if not config.use_llm_validation:
        return final_llm_output_lists, llm_candidates, llm_results
    
    # Clean LLM results to remove markdown formatting
    cleaned_llm_results = []
    for result in llm_results:
        # If result is a string, clean it
        if isinstance(result, str):
            # Handle multiple code block formats (markdown, python, json, etc)
            cleaned_result = result
            
            # Try to handle code blocks with language specified
            if re.search(r'^```[\w]*\n', cleaned_result):
                # Remove opening code block marker with language
                cleaned_result = re.sub(r'^```[\w]*\n', '', cleaned_result)
                cleaned_result = re.sub(r'\n```$', '', cleaned_result)
            
            # Try to handle generic code blocks
            if re.search(r'^```\n', cleaned_result):
                cleaned_result = re.sub(r'^```\n', '', cleaned_result)
                cleaned_result = re.sub(r'\n```$', '', cleaned_result)
            
            # Check for entire text wrapped in code block without newlines
            if re.search(r'^```[\w]*', cleaned_result):
                cleaned_result = re.sub(r'^```[\w]*', '', cleaned_result)
                cleaned_result = re.sub(r'```$', '', cleaned_result)
            
            # Handle single backtick format code
            cleaned_result = re.sub(r'^`', '', cleaned_result)
            cleaned_result = re.sub(r'`$', '', cleaned_result)
            
            # Final cleanup
            cleaned_result = cleaned_result.strip()
            
            # Log what was changed for debugging
            if cleaned_result != result:
                logger.info(f"Cleaned markdown from LLM result: '{result[:30]}...' -> '{cleaned_result[:30]}...'")
            
            cleaned_llm_results.append(cleaned_result)
        else:
            # If result is already parsed (list or dict), keep as is
            cleaned_llm_results.append(result)
    
    logger.info(f"Processed {len(cleaned_llm_results)} LLM results to handle markdown formatting")
    
    # Return the same structure but with cleaned results
    return final_llm_output_lists, llm_candidates, cleaned_llm_results


async def run_multiple_llm_extractions(
    all_extracted_lists_raw: List[Dict[str, Any]],
    level2_term: str,
    filter_config: FilterConfig,
    num_attempts: int = Config.NUM_LLM_ATTEMPTS,
    agreement_threshold: int = Config.AGREEMENT_THRESHOLD,
    logger: Optional[Any] = None,
    model_types: List[str] = DEFAULT_LLM_MODEL_TYPES
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[List[str]]]:
    """
    Run multiple LLM extractions and select conference topics that appear in multiple responses
    Each attempt uses a randomly selected provider (Gemini/OpenAI) and model type.
    
    Args:
        all_extracted_lists_raw: Raw extracted lists to process
        level2_term: The level 2 term being processed
        filter_config: Configuration for filtering
        num_attempts: Number of LLM extraction attempts
        agreement_threshold: Minimum number of appearances required for a conference topic
        logger: Optional logger
        model_types: List of model types to use for attempts
    
    Returns:
        Tuple containing:
        - final_lists: Combined lists with items that meet the agreement threshold
        - llm_candidates: The candidates sent to the LLM (from the first run)
        - llm_results: The consolidated results from multiple LLM runs
    """
    if not all_extracted_lists_raw:
        return [], [], []
    
    logger.info(f"Running multiple LLM extractions ({num_attempts}) for '{level2_term}'")
    
    # Run multiple extraction attempts
    all_results = []
    all_candidates = []
    all_raw_results = []
    
    # Ensure model_types list has enough entries for num_attempts
    if len(model_types) < num_attempts:
        model_types = model_types * (num_attempts // len(model_types) + 1)
    current_model_types = model_types[:num_attempts]
    available_providers = [Provider.GEMINI, Provider.OPENAI] # Define available providers
    
    for attempt in range(num_attempts):
        # Use a different provider for each attempt if possible
        current_provider = random.choice(available_providers)
        current_model_type = current_model_types[attempt] # Randomly choose from the list
        
        # Randomly select a model type
        # current_model_type = random.choice(model_types) # Model type is now passed in
        
        # Create a new filter config with the current provider
        current_filter_config = FilterConfig(
            scoring_fn=filter_config.scoring_fn,
            clean_item_fn=filter_config.clean_item_fn,
            provider=current_provider,
            use_llm_validation=filter_config.use_llm_validation,
            binary_llm_decision=filter_config.binary_llm_decision,
            binary_system_prompt=filter_config.binary_system_prompt,
            min_score_for_llm=filter_config.min_score_for_llm,
            model_type=current_model_type
        )
        
        logger.info(f"Attempt {attempt+1}/{num_attempts} using RANDOM provider: {current_provider}, model type: {current_model_type}")
        
        try:
            # Patch init_llm to use the selected model for this attempt
            original_init_llm = init_llm
            
            # Define the patched function - it needs to accept model_type even if unused
            def patched_init_llm(provider: Optional[str] = None, model_type: Optional[str] = None):
                logger.debug(f"Using patched init_llm for provider '{provider}' with pre-selected model '{current_model_type}' (ignoring passed model_type: {model_type})")
                # Call the original function but force the selected model type
                return original_init_llm(provider, current_model_type)

            # Apply the patch
            import generate_glossary.utils.web_search.filtering as filtering
            filtering.init_llm = patched_init_llm
            
            # Run the standard filter_lists on the raw lists
            final_lists, llm_candidates, llm_results = await filter_lists(
                all_extracted_lists_raw, level2_term, current_filter_config, logger
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
        finally:
            # Ensure the original function is restored regardless of success or failure
            if 'original_init_llm' in locals() and 'filtering' in locals():
                 filtering.init_llm = original_init_llm
                 logger.debug(f"Restored original init_llm after attempt {attempt+1}")
    
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
            item_counts[item_lower] = {"count": 0, "original": item}
        item_counts[item_lower]["count"] += 1
    
    # Filter items by agreement threshold
    agreed_items = [item_data["original"] for item_lower, item_data in item_counts.items() 
                   if item_data["count"] >= agreement_threshold]
    
    logger.info(f"Found {len(agreed_items)} conference topics that meet the agreement threshold")
    
    # Format results for return - create a single list containing all agreed items
    final_consolidated_list = [agreed_items] if agreed_items else []
    
    return final_consolidated_list, all_candidates, all_raw_results


async def process_level2_term(level2_term: str,
                              provider: Optional[str] = None,
                              session: Optional[Any] = None,
                              general_semaphore: Optional[asyncio.Semaphore] = None,
                              browser_semaphore: Optional[asyncio.Semaphore] = None,
                              min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                              model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                              num_llm_attempts: int = Config.NUM_LLM_ATTEMPTS,
                              agreement_threshold: int = Config.AGREEMENT_THRESHOLD,
                              prefetched_search_results: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Process a single level2 term to extract conference/journal topics and save detailed metadata"""
    logger.info(f"Term '{level2_term}': START processing (LLM Min Score: {min_score_for_llm}, Models: {model_types}, Attempts: {num_llm_attempts}, Agree: {agreement_threshold})")

    # Initialize structures for detailed metadata
    term_details = {
        "level2_term": level2_term,
        "all_urls": [],
        "raw_extracted_lists": [],
        "llm_io_pairs": [],
        "final_consolidated_topics": [],
        "error": None
    }
    
    # Initialize structure for raw URL lists
    url_to_raw_lists = {}
    
    try:
        # Create configurations for the shared utilities
        search_config = WebSearchConfig(
            base_dir=Config.BASE_DIR,
            raw_search_dir=Config.RAW_SEARCH_DIR
        )
        
        html_config = HTMLFetchConfig(
            cache_dir=Config.SEARCH_CACHE_DIR
        )
        
        list_config = ListExtractionConfig(
            keywords=TOPIC_KEYWORDS,
            anti_keywords=NON_TOPIC_KEYWORDS,
            patterns=TOPIC_PATTERNS
        )
        
        filter_config = FilterConfig(
            scoring_fn=score_conference_topic_list,
            clean_item_fn=clean_conference_topic,
            provider=provider,
            use_llm_validation=True,
            binary_llm_decision=False,
            binary_system_prompt=CONFERENCE_VALIDATION_SYSTEM_PROMPT_TEMPLATE.format(term=level2_term),
            min_score_for_llm=min_score_for_llm,
            model_type=model_types[0] if model_types else "default"
        )
        
        # Use prefetched search results if provided, otherwise perform search
        search_results = prefetched_search_results
        if search_results is None:
            # Use the general semaphore to limit concurrent API calls
            if general_semaphore:
                async with general_semaphore:
                    # Use SEARCH_QUERIES constant with formatting
                    queries = [query.format(term=level2_term) for query in SEARCH_QUERIES]
                    
                    # Perform web search with multiple queries
                    logger.info(f"Searching with {len(queries)} different queries for '{level2_term}'")
                    search_results = web_search_bulk(queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)
            else:
                # No semaphore provided, run without concurrency control
                queries = [query.format(term=level2_term) for query in SEARCH_QUERIES]
                logger.info(f"Searching with {len(queries)} different queries for '{level2_term}' (no semaphore)")
                search_results = web_search_bulk(queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)
        
        if not search_results or not search_results.get("data"):
            logger.warning(f"No search results for '{level2_term}'")
            term_details["error"] = "No search results"
            # Do NOT save details here, return them instead
            return {
                "level2_term": level2_term, 
                "conference_topics": [], 
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
            logger.warning(f"No URLs found in search results for '{level2_term}'")
            term_details["error"] = "No URLs found in search results"
            # Do NOT save details here, return them instead
            return {
                "level2_term": level2_term, 
                "conference_topics": [], 
                "count": 0, 
                "verified": False, 
                "num_urls": 0, 
                "num_lists": 0,
                "term_details": term_details, # Return details
                "url_to_raw_lists": {} # Return empty raw lists
            }
            
        logger.info(f"Found {len(urls)} URLs for '{level2_term}'")
            
        # Use the provided semaphore or create one for this process
        # For browsers (HTML fetching)
        semaphore_to_use = browser_semaphore if browser_semaphore else asyncio.Semaphore(MAX_CONCURRENT_BROWSERS)
        # For headless browser fetching - if needed
        browser_semaphore_to_use = browser_semaphore if browser_semaphore else asyncio.Semaphore(2)  # Limit headless browser concurrency
    
        all_extracted_lists_raw = [] # Store raw list dicts
        url_to_raw_lists = {}
        
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
                    fetch_webpage(url, new_session, semaphore_to_use, browser_semaphore_to_use, 
                                 html_config, level2_term, logger) 
                    for url in urls[:MAX_SEARCH_RESULTS]
                ]
                
                logger.info(f"Term '{level2_term}': Starting HTML fetching for {len(fetch_tasks)} URLs (new session)...")
                # Use gather with return_exceptions=True to continue even if some requests fail
                html_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                logger.info(f"Term '{level2_term}': Finished HTML fetching (new session).")
                
                # Process each webpage
                for i, (url, result) in enumerate(zip(urls[:MAX_SEARCH_RESULTS], html_contents)):
                    # Skip if the result is an exception
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching content from {url}: {str(result)}")
                        continue
                        
                    html_content = result
                    if not html_content:
                        continue
                        
                    # Preprocess HTML content to remove problematic sections
                    preprocessed_html = preprocess_html_content(html_content)
                        
                    # Extract lists from the webpage
                    extracted_lists = extract_lists_from_html(preprocessed_html, list_config)
            
                    if extracted_lists:
                        all_extracted_lists_raw.extend(extracted_lists) # Append raw dicts
                        url_to_raw_lists[url] = extracted_lists # Store raw dicts per URL
                        logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
        else:
            # Use the provided session
            fetch_tasks = [
                fetch_webpage(url, session, semaphore_to_use, browser_semaphore_to_use, 
                             html_config, level2_term, logger) 
                for url in urls[:MAX_SEARCH_RESULTS]
            ]
            
            logger.info(f"Term '{level2_term}': Starting HTML fetching for {len(fetch_tasks)} URLs (provided session)...")
            # Use gather with return_exceptions=True to continue even if some requests fail
            html_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            logger.info(f"Term '{level2_term}': Finished HTML fetching (provided session).")
            
            # Process each webpage
            for i, (url, result) in enumerate(zip(urls[:MAX_SEARCH_RESULTS], html_contents)):
                # Skip if the result is an exception
                if isinstance(result, Exception):
                    logger.error(f"Error fetching content from {url}: {str(result)}")
                    continue
                    
                html_content = result
                if not html_content:
                    continue
                    
                # Preprocess HTML content to remove problematic sections
                preprocessed_html = preprocess_html_content(html_content)
                    
                # Extract lists from the webpage
                extracted_lists = extract_lists_from_html(preprocessed_html, list_config)
                
                if extracted_lists:
                    all_extracted_lists_raw.extend(extracted_lists) # Append raw dicts
                    url_to_raw_lists[url] = extracted_lists # Store raw dicts per URL
                    
                logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
            
        # Do NOT save raw results here, return them instead
        # save_raw_url_results(level2_term, url_to_raw_lists)
        
        term_details["raw_extracted_lists"] = all_extracted_lists_raw

        # Filter and validate lists
        if not all_extracted_lists_raw:
            logger.warning(f"No lists extracted for '{level2_term}'")
            term_details["error"] = "No lists extracted from fetched HTML"
            logger.info(f"Term '{level2_term}': Returning early - no lists extracted.") # Add log
            # Do NOT save details here, return them instead
            return {
                "level2_term": level2_term, 
                "conference_topics": [], 
                "count": 0, 
                "verified": False, 
                "num_urls": len(urls), 
                "num_lists": 0,
                "term_details": term_details, # Return details
                "url_to_raw_lists": url_to_raw_lists # Return raw lists
            }
            
        logger.info(f"Term '{level2_term}': Extracted {len(all_extracted_lists_raw)} raw lists. Starting LLM processing...")
        
        # Use multiple LLM extractions with agreement threshold
        final_llm_output_lists, llm_candidates, llm_results = await run_multiple_llm_extractions(
            all_extracted_lists_raw, level2_term, filter_config, num_llm_attempts, agreement_threshold, logger, model_types
        )
        logger.info(f"Term '{level2_term}': Finished LLM processing.")
        
        # Store intermediate results in term_details in the new format
        term_details["llm_io_pairs"] = []
        if llm_candidates and llm_results:
             # Ensure results align with candidates (should match length)
             num_pairs = min(len(llm_candidates), len(llm_results))
             if len(llm_candidates) != len(llm_results):
                  logger.warning(f"Mismatch between LLM candidates ({len(llm_candidates)}) and results ({len(llm_results)}) for {level2_term}. Pairing up to {num_pairs}.")
             
             for i in range(num_pairs):
                  # Ensure candidate has 'items' field
                  candidate_input = llm_candidates[i].get('items', []) if isinstance(llm_candidates[i], dict) else llm_candidates[i]
                  llm_output = llm_results[i]
                  term_details["llm_io_pairs"].append({
                       "input_list_to_llm": candidate_input,
                       "output_list_from_llm": llm_output
                  })
        
        if not final_llm_output_lists:
            logger.warning(f"No lists passed filtering/LLM validation for '{level2_term}'")
            term_details["error"] = "No lists passed filtering/LLM validation"
            logger.info(f"Term '{level2_term}': Returning early - no lists passed LLM validation.") # Add log
            # Do NOT save details here, return them instead
            return {
                "level2_term": level2_term, 
                "conference_topics": [], 
                "count": 0, 
                "verified": False, 
                "num_urls": len(urls), 
                "num_lists": len(all_extracted_lists_raw),
                "term_details": term_details, # Return details
                "url_to_raw_lists": url_to_raw_lists # Return raw lists
            }
            
        logger.info(f"Term '{level2_term}': After filtering/LLM, {len(final_llm_output_lists)} lists/sub-lists remain. Starting consolidation...")
        
        # Consolidate conference topics
        conference_topics = consolidate_lists(
            final_llm_output_lists, 
            level2_term, 
            min_frequency=1,
            min_list_appearances=1,
            similarity_threshold=0.7
        )
        term_details["final_consolidated_topics"] = conference_topics
        logger.info(f"Term '{level2_term}': Finished consolidation. Found {len(conference_topics)} unique topics.")
        
        # Simplified source/quality tracking
        topic_quality = {topic: 1.0 for topic in conference_topics}

        # Do NOT save detailed metadata here, return it instead
        # save_detailed_metadata(level2_term, term_details)

        # Return main results needed for aggregation PLUS the data to be saved later
        return {
            "level2_term": level2_term,
            "conference_topics": conference_topics,
            "count": len(conference_topics),
            "url_sources": {}, # Simplified
            "quality_scores": topic_quality, # Simplified
            "verified": len(conference_topics) > 0,
            "num_urls": len(urls),
            "num_lists": len(all_extracted_lists_raw), # Report raw list count
            "provider": provider or "multiple",
            "max_concurrent": MAX_CONCURRENT_REQUESTS,
            "batch_size": BATCH_SIZE,
            "num_llm_attempts": num_llm_attempts,
            "agreement_threshold": agreement_threshold,
            "min_score_for_llm": min_score_for_llm,
            "llm_model_types_used": model_types[:num_llm_attempts], # Record actual models used
            "term_details": term_details, # Return details for later saving
            "url_to_raw_lists": url_to_raw_lists # Return raw lists for later saving
        }
            
    except Exception as e:
        logger.error(f"Term '{level2_term}': Error processing term: {str(e)}", exc_info=True)
        term_details["error"] = f"Unhandled exception: {str(e)}"
        logger.info(f"Term '{level2_term}': Returning error due to exception.") # Add log
        # Do NOT save details here, return them instead
        # Return error structure
        return {
            "level2_term": level2_term, 
            "conference_topics": [], 
            "count": 0, 
            "verified": False, 
            "num_urls": len(term_details['all_urls']), 
            "num_lists": len(term_details['raw_extracted_lists']), 
            "error": str(e),
            "term_details": term_details, # Return details for later saving
            "url_to_raw_lists": url_to_raw_lists # Return raw lists for later saving
        }

def perform_bulk_web_search(level2_terms: List[str], search_config: WebSearchConfig) -> Dict[str, Dict[str, Any]]:
    """
    Perform a bulk web search for multiple level 2 terms at once.
    
    Args:
        level2_terms: List of level 2 terms to search for
        search_config: Configuration for web search
        
    Returns:
        A dictionary mapping level2_term to its search results
    """
    if not level2_terms:
        return {}
    
    # Use global SEARCH_QUERIES constant
    queries_per_term = SEARCH_QUERIES
    
    # Dynamic calculation of max terms per search batch
    num_queries_per_term = len(queries_per_term)
    max_terms_per_search = MAX_SEARCH_QUERIES // num_queries_per_term
    
    # Limit the number of terms per batch to prevent exceeding API limits
    terms_per_batch = min(len(level2_terms), max_terms_per_search)
    logger.info(f"Performing bulk web search for {len(level2_terms)} terms (max {max_terms_per_search} per batch, using {num_queries_per_term} queries per term)")
    
    # Prepare all queries for all terms
    all_queries = []
    term_to_query_indices = {}  # Maps each term to its query indices in all_queries
    
    for i, term in enumerate(level2_terms):
        # Store the starting index for this term's queries
        start_idx = len(all_queries)
        
        # Add queries for this term
        current_term_queries = [query.format(term=term) for query in queries_per_term]
        all_queries.extend(current_term_queries)
        
        # Store the range of indices for this term
        term_to_query_indices[term] = (start_idx, len(all_queries))
    
    # Perform the bulk search
    search_results = web_search_bulk(all_queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)
    
    if not search_results or not search_results.get("data"):
        logger.warning(f"No search results found for any of the {len(level2_terms)} terms")
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
    
    logger.info(f"Got search results for {len(term_to_results)} out of {len(level2_terms)} terms")
    return term_to_results


async def process_level2_terms_batch(batch: List[str],
                                      provider: Optional[str] = None,
                                      session: Optional[Any] = None,
                                      general_semaphore: Optional[asyncio.Semaphore] = None,
                                      browser_semaphore: Optional[asyncio.Semaphore] = None,
                                      min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                                      model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                                      num_llm_attempts: int = Config.NUM_LLM_ATTEMPTS,
                                      agreement_threshold: int = Config.AGREEMENT_THRESHOLD
                                      ) -> List[Dict[str, Any]]:
    """Process a batch of level 2 terms with optimized bulk web searching"""
    if not batch:
        return []
        
    logger.info(f"Processing batch of {len(batch)} level 2 terms")
    
    # Create search configuration
    search_config = WebSearchConfig(
        base_dir=Config.BASE_DIR,
        raw_search_dir=Config.RAW_SEARCH_DIR
    )
    
    # Use global SEARCH_QUERIES constant for calculation
    num_queries_per_term = len(SEARCH_QUERIES)
    max_terms_per_search = MAX_SEARCH_QUERIES // num_queries_per_term
    
    # Get the current event loop
    loop = asyncio.get_event_loop()
    # Create a ThreadPoolExecutor to run synchronous search
    executor = ThreadPoolExecutor()
    
    # Split the batch into smaller chunks for web search
    search_results_by_term = {}
    search_tasks = []
    
    for i in range(0, len(batch), max_terms_per_search):
        search_batch = batch[i:i + max_terms_per_search]
        logger.info(f"Preparing bulk web search task for {len(search_batch)} terms (batch {i//max_terms_per_search + 1})")
        # Run the synchronous function in an executor
        task = loop.run_in_executor(executor, perform_bulk_web_search, search_batch, search_config)
        search_tasks.append(task)
        
    # Await all search tasks concurrently
    logger.info(f"Awaiting {len(search_tasks)} bulk web search tasks...")
    completed_searches = await asyncio.gather(*search_tasks)
    logger.info("Bulk web searches completed.")
    
    # Combine results from all search tasks
    for result_dict in completed_searches:
        if result_dict:
            search_results_by_term.update(result_dict)
    
    # Shutdown the executor
    executor.shutdown()

    # Process each term with its pre-fetched search results
    tasks = []
    for term in batch:
        # Get the pre-fetched search results for this term (if any)
        prefetched_results = search_results_by_term.get(term)
        
        # Process the term with semaphores
        task = process_level2_term(
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
    
    # Run all term processing tasks in parallel
    logger.info(f"Awaiting {len(tasks)} term processing tasks for batch...")
    results = await asyncio.gather(*tasks)
    logger.info(f"Term processing tasks for batch completed.")
    return results


def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        Config.SEARCH_CACHE_DIR,
        Config.RAW_SEARCH_DIR,
        Config.DETAILED_META_DIR,
        os.path.dirname(Config.OUTPUT_FILE),
        os.path.dirname(Config.META_FILE)
    ]
    
    logger.info(f"BASE_DIR: {Config.BASE_DIR}")
    logger.info(f"LV2_INPUT_FILE: {Config.DEFAULT_LV2_INPUT_FILE}")
    logger.info(f"OUTPUT_FILE: {Config.OUTPUT_FILE}")
    logger.info(f"META_FILE: {Config.META_FILE}")
    logger.info(f"CACHE_DIR: {Config.SEARCH_CACHE_DIR}")
    logger.info(f"RAW_SEARCH_DIR: {Config.RAW_SEARCH_DIR}")
    logger.info(f"DETAILED_META_DIR: {Config.DETAILED_META_DIR}")
    
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
        parser = argparse.ArgumentParser(description="Extract conference/journal topics for level 2 terms.")
        parser.add_argument("--provider", help="LLM provider (e.g., gemini, openai)")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for processing terms (default: {BATCH_SIZE})")
        parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, help=f"Max concurrent term processing requests (default: {MAX_CONCURRENT_REQUESTS})")
        parser.add_argument("--input-file", default=Config.DEFAULT_LV2_INPUT_FILE, help=f"Path to the input file containing level 2 terms (default: {Config.DEFAULT_LV2_INPUT_FILE})")
        parser.add_argument("--append", action='store_true', help="Append results to existing output files instead of overwriting.")
        parser.add_argument("--num-attempts", type=int, default=Config.NUM_LLM_ATTEMPTS, help=f"Number of LLM extraction attempts per term (default: {Config.NUM_LLM_ATTEMPTS})")
        parser.add_argument("--agreement-threshold", type=int, default=Config.AGREEMENT_THRESHOLD, help=f"Minimum appearances threshold for conference topics (default: {Config.AGREEMENT_THRESHOLD})")
        parser.add_argument("--min-score-for-llm", type=float, default=DEFAULT_MIN_SCORE_FOR_LLM, help=f"Minimum heuristic score to send a list to LLM (default: {DEFAULT_MIN_SCORE_FOR_LLM})")
        parser.add_argument("--llm-model-types", type=str, default=",".join(DEFAULT_LLM_MODEL_TYPES), help=f"Comma-separated LLM model types for attempts (e.g., default,pro,mini) (default: {','.join(DEFAULT_LLM_MODEL_TYPES)})")
        
        args = parser.parse_args()
        
        provider = args.provider
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        input_file_path = args.input_file
        append_mode = args.append
        num_llm_attempts = args.num_attempts
        agreement_threshold = args.agreement_threshold
        min_score_for_llm = args.min_score_for_llm
        llm_model_types = [mt.strip() for mt in args.llm_model_types.split(',') if mt.strip()]
        
        if provider:
            logger.info(f"Using provider: {provider}")
        if batch_size != BATCH_SIZE:
            logger.info(f"Using custom batch size: {batch_size}")
        if max_concurrent != MAX_CONCURRENT_REQUESTS:
             logger.info(f"Using custom concurrent limit: {max_concurrent}")
        if input_file_path != Config.DEFAULT_LV2_INPUT_FILE:
             logger.info(f"Using custom input file: {input_file_path}")
        if append_mode:
             logger.info("Append mode enabled. Results will be added to existing files.")
        if num_llm_attempts != Config.NUM_LLM_ATTEMPTS:
             logger.info(f"Using custom number of LLM attempts: {num_llm_attempts}")
        if agreement_threshold != Config.AGREEMENT_THRESHOLD:
             logger.info(f"Using custom agreement threshold: {agreement_threshold}")
        if min_score_for_llm != DEFAULT_MIN_SCORE_FOR_LLM:
             logger.info(f"Using custom LLM score threshold: {min_score_for_llm}")
        if args.llm_model_types != ",".join(DEFAULT_LLM_MODEL_TYPES):
             logger.info(f"Using custom LLM model types: {llm_model_types}")
        
        logger.info("=" * 80)
        logger.info(f"Starting conference/journal topics extraction using level 2 terms")
        logger.info(f"Using optimized web search with dynamically calculated max terms per batch")
        
        # Create output directories
        ensure_dirs_exist()
        
        # Read level 2 terms from the specified input file
        level2_terms = read_level2_terms(input_file_path)
        
        logger.info(f"Processing {len(level2_terms)} level 2 terms from '{input_file_path}'")
        logger.info(f"Configuration:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Max concurrent requests: {max_concurrent}")
        logger.info(f"  - LLM provider: {provider or 'auto-select'}")
        logger.info(f"  - LLM attempts: {num_llm_attempts}")
        logger.info(f"  - LLM model types: {llm_model_types[:num_llm_attempts]}")
        logger.info(f"  - Agreement threshold: {agreement_threshold}")
        logger.info(f"  - Score threshold: {min_score_for_llm}")
        logger.info(f"  - Append mode: {append_mode}")
        logger.info("=" * 80)
        
        # Initialize aiohttp session for the entire run
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, CookieJar
        
        # Create semaphores for concurrency control
        general_semaphore = asyncio.Semaphore(max_concurrent)  # Use max_concurrent
        browser_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BROWSERS)  # Use specific browser limit (matches lv2)
        logger.info(f"Using general fetch semaphore limit: {max_concurrent}")
        logger.info(f"Using headless browser semaphore limit: {MAX_CONCURRENT_BROWSERS}")
        
        all_results = []
        start_time = time.time()
        
        # Process in batches
        timeout = ClientTimeout(total=3600, connect=30)  # 1 hour total timeout, 30s connect timeout
        logger.info("Attempting to start aiohttp ClientSession...")
        async with ClientSession(
            connector=TCPConnector(
                ssl=ssl.create_default_context(cafile=certifi.where()),
                limit=max_concurrent,  # Match lv2's connector limit
                limit_per_host=2,
                force_close=True,  # Force close connections to prevent descriptor reuse
                enable_cleanup_closed=True  # Enable cleanup of closed connections
            ),
            cookie_jar=CookieJar(unsafe=True), 
            timeout=timeout,
            raise_for_status=False
        ) as session:
            for i in range(0, len(level2_terms), batch_size):
                batch = level2_terms[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(level2_terms) + batch_size - 1)//batch_size}")
                
                # Process the batch using the optimized batch processor with semaphores
                batch_results = await process_level2_terms_batch(
                    batch,
                    provider,
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
                terms_processed = min(i + batch_size, len(level2_terms))
                terms_per_second = terms_processed / max(1, elapsed)
                eta_seconds = (len(level2_terms) - terms_processed) / max(0.1, terms_per_second)
                
                logger.info(f"Processed {terms_processed}/{len(level2_terms)} terms "
                            f"({terms_processed/len(level2_terms)*100:.1f}%) in {elapsed:.1f}s "
                            f"({terms_per_second:.2f} terms/s, ETA: {eta_seconds/60:.1f}m)")
        logger.info("Finished aiohttp ClientSession context manager.")
        
        # Collect all conference topics and prepare for saving
        found_topics = []
        verified_terms_count = 0
        processed_stats = {
            "total_urls_processed": 0,
            "total_lists_found": 0,
            "verified_topics_count": 0
        }
        level2_term_result_counts = {}
        level2_to_conference_topics = {}
        topic_sources = {}
        
        # --- Data to be saved later ---
        all_term_details = []
        all_raw_url_lists = {}
        
        for result in all_results:
            level2_term = result["level2_term"]
            
            # Store details and raw lists for saving later
            if "term_details" in result:
                all_term_details.append(result["term_details"])
            if "url_to_raw_lists" in result and result["url_to_raw_lists"]:
                all_raw_url_lists[level2_term] = result["url_to_raw_lists"]
            
            if result.get("error"):
                logger.error(f"Skipping aggregation for term '{result['level2_term']}' due to processing error: {result['error']}")
                continue
                
            conference_topics = result["conference_topics"]
            verified = result.get("verified", False)
            num_urls = result.get("num_urls", 0)
            num_lists = result.get("num_lists", 0)
            
            processed_stats["total_urls_processed"] += num_urls
            processed_stats["total_lists_found"] += num_lists
            
            if verified:
                verified_terms_count += 1
                processed_stats["verified_topics_count"] += len(conference_topics)
                
                if level2_term not in level2_to_conference_topics:
                    level2_to_conference_topics[level2_term] = []
                level2_to_conference_topics[level2_term].extend(conference_topics)
                found_topics.extend(conference_topics)
                
                level2_term_result_counts[level2_term] = len(conference_topics)
                
                for topic in conference_topics:
                    if topic not in topic_sources:
                        topic_sources[topic] = []
                    topic_sources[topic].append(level2_term)
        
        logger.info(f"Consolidated results from {len(all_results)} processed terms.")
        logger.info(f"Found {verified_terms_count} verified terms with topics.")
        
        # --- Saving detailed metadata for each term ---
        # logger.info(f"Saving detailed metadata for {len(all_term_details)} terms (DISABLED)")
        # for term_detail in all_term_details:
        #     if "level2_term" in term_detail:
        #         term = term_detail["level2_term"]
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
        
        # --- File Saving Logic ---
        
        existing_unique_topics = set()
        if append_mode and os.path.exists(Config.OUTPUT_FILE):
            logger.info(f"Loading existing conference topics from {Config.OUTPUT_FILE} for append mode.")
            try:
                with open(Config.OUTPUT_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        existing_unique_topics.add(line.strip())
                logger.info(f"Loaded {len(existing_unique_topics)} existing unique topics.")
            except Exception as e:
                logger.warning(f"Could not read existing output file {Config.OUTPUT_FILE}: {e}. Starting fresh.", exc_info=True)
                existing_unique_topics = set() # Start fresh if read fails
        
        # Add newly found unique topics
        final_unique_topics_set = set(existing_unique_topics)
        newly_added_count = 0
        for topic in found_topics:
            if topic not in final_unique_topics_set:
                final_unique_topics_set.add(topic)
                newly_added_count += 1
        
        logger.info(f"Added {newly_added_count} new unique conference topics.")
        
        # Save output file (unique topics)
        final_unique_topics_list = sorted(list(final_unique_topics_set))
        random.shuffle(final_unique_topics_list) # Keep randomization if desired
        
        try:
            with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
                for topic in final_unique_topics_list:
                    # Write conference topics in lowercase for consistency
                    f.write(f"{topic.lower()}\n")
            logger.info(f"Saved {len(final_unique_topics_list)} total unique conference topics to {Config.OUTPUT_FILE}")
        except Exception as e:
            logger.error(f"Failed to write output file {Config.OUTPUT_FILE}: {e}", exc_info=True)
    
        # Load existing metadata if in append mode
        metadata = {}
        if append_mode and os.path.exists(Config.META_FILE):
            logger.info(f"Loading existing metadata from {Config.META_FILE} for append mode.")
            try:
                with open(Config.META_FILE, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info("Existing metadata loaded successfully.")
            except Exception as e:
                 logger.warning(f"Could not read or parse existing metadata file {Config.META_FILE}: {e}. Creating new metadata.", exc_info=True)
                 metadata = {} # Start fresh if read/parse fails
        
        # Extract existing mappings for the top-level keys (or initialize them if they don't exist)
        level2_to_conference_topics_mapping = metadata.get("level2_to_conference_topics_mapping", {})
        conference_topic_level2_sources = metadata.get("conference_topic_level2_sources", {})
        
        # Update mappings with new data
        for term, topics in level2_to_conference_topics.items():
            if term not in level2_to_conference_topics_mapping:
                level2_to_conference_topics_mapping[term] = []
            # Add only unique new topics for this term
            existing_topics_set = set(level2_to_conference_topics_mapping[term])
            for topic in topics:
                 if topic not in existing_topics_set:
                      level2_to_conference_topics_mapping[term].append(topic)
                      existing_topics_set.add(topic)
        
        # Update conference topic sources
        for topic, sources in topic_sources.items():
             if topic not in conference_topic_level2_sources:
                 conference_topic_level2_sources[topic] = []
             # Add unique new source terms for this topic
             existing_source_set = set(conference_topic_level2_sources[topic])
             for source in sources:
                  if source not in existing_source_set:
                       conference_topic_level2_sources[topic].append(source)
                       existing_source_set.add(source)
        
        # Prepare the metadata
        current_metadata = {
            "level2_to_conference_topics_mapping": level2_to_conference_topics_mapping,
            "conference_topic_level2_sources": conference_topic_level2_sources,
            "metadata": {
                "execution_time": f"{time.time() - start_time:.2f} seconds",
                "total_unique_conference_topics": len(final_unique_topics_list),
                "total_level2_terms_processed": metadata.get("total_level2_terms_processed", 0) + len(level2_terms),
                "verified_level2_terms_count": metadata.get("verified_level2_terms_count", 0) + verified_terms_count,
                "total_urls_processed": metadata.get("total_urls_processed", 0) + processed_stats["total_urls_processed"],
                "total_raw_lists_extracted": metadata.get("total_raw_lists_extracted", 0) + processed_stats["total_lists_found"],
                "provider": provider or "multiple",
                "max_concurrent": max_concurrent,
                "batch_size": batch_size,
                "num_llm_attempts": num_llm_attempts,
                "agreement_threshold": agreement_threshold,
                "min_score_for_llm": min_score_for_llm,
                "llm_model_types_used": llm_model_types[:num_llm_attempts], # Record actual models used
                "level2_term_result_counts": level2_term_result_counts
            }
        }
        
        # Calculate and add averages to metadata
        total_terms_ever_processed = current_metadata["metadata"]["total_level2_terms_processed"]
        total_verified_ever = current_metadata["metadata"]["verified_level2_terms_count"]
        total_verified_topics_ever = sum(len(topics) for topics in level2_to_conference_topics_mapping.values())
        
        current_metadata["metadata"]["avg_urls_per_term"] = current_metadata["metadata"]["total_urls_processed"] / max(1, total_terms_ever_processed)
        current_metadata["metadata"]["avg_raw_lists_per_term"] = current_metadata["metadata"]["total_raw_lists_extracted"] / max(1, total_terms_ever_processed)
        current_metadata["metadata"]["avg_final_topics_per_verified_term"] = total_verified_topics_ever / max(1, total_verified_ever)

        try:
            with open(Config.META_FILE, "w", encoding="utf-8") as f:
                json.dump(current_metadata, f, indent=2)
            logger.info(f"Updated metadata saved to {Config.META_FILE}")
        except Exception as e:
            logger.error(f"Failed to write metadata file {Config.META_FILE}: {e}", exc_info=True)
            
        logger.info(f"Detailed per-term metadata saving is DISABLED.")
        # logger.info(f"Detailed per-term metadata saved in {Config.DETAILED_META_DIR}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("=" * 80)
    logger.info("Conference/journal topics extraction completed")
    logger.info("=" * 80)

def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 