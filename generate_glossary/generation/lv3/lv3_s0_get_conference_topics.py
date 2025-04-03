import os
import sys
import random
import json
import time
import asyncio
import aiohttp
import re
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter
import ssl

# Fix import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import Provider

# Import shared web search utilities
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, consolidate_lists, init_llm
from generate_glossary.utils.web_search.filtering import filter_lists as original_filter_lists

# Just after imports, add a helper function for LLM output cleaning
from generate_glossary.utils.web_search.filtering import filter_lists as original_filter_lists

# Load environment variables and setup logging
load_dotenv('.env')
logger = setup_logger("lv3.s0")
random.seed(42)

# Constants
MAX_SEARCH_RESULTS = 10
MAX_CONCURRENT_REQUESTS = 32
BATCH_SIZE = 100  # Process multiple level 2 terms in a single batch

# Conference/journal-related keywords for enhanced filtering
TOPIC_KEYWORDS = [
    "topic", "track", "workshop", "symposium", "session", "theme", 
    "special issue", "call for papers", "cfp", "submission", "paper", 
    "research", "area", "program", "panel", "tutorial", "keynote",
    "presentation", "publication", "proceedings", "accepted", "contribution",
    "article", "manuscript", "work", "study", "investigation", "analysis",
    "focus", "interest", "perspective", "approach", "methodology", "framework"
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

class Config:
    """Configuration for conference/journal topic extraction from level 2 terms"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    DEFAULT_LV2_INPUT_FILE = os.path.join(BASE_DIR, "data/final/lv2/lv2_final.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s0_conference_topics.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv3/raw/lv3_s0_metadata.json")
    CACHE_DIR = os.path.join(BASE_DIR, "data/lv3/cache")
    RAW_SEARCH_DIR = os.path.join(BASE_DIR, "data/lv3/raw_search_results")
    DETAILED_META_DIR = os.path.join(BASE_DIR, "data/lv3/detailed_metadata")

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
    item = re.sub(r'^(Topics|Tracks|Workshops|Sessions|Special Issues?|Call for Papers|CFP|Areas|Themes?) (on|in|of|for|about) ', '', item, flags=re.IGNORECASE)
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


async def process_level2_term(level2_term: str, provider: Optional[str] = None, session: Optional[Any] = None) -> Dict[str, Any]:
    """Process a single level2 term to extract conference/journal topics and save detailed metadata"""
    logger.info(f"Processing level 2 term: {level2_term}")
    
    # Initialize structures for detailed metadata
    term_details = {
        "level2_term": level2_term,
        "all_urls": [],
        "raw_extracted_lists": [],
        "llm_io_pairs": [],
        "final_consolidated_topics": [],
        "error": None
    }
    
    try:
        # Create configurations for the shared utilities
        search_config = WebSearchConfig(
            base_dir=Config.BASE_DIR,
            raw_search_dir=Config.RAW_SEARCH_DIR
        )
        
        html_config = HTMLFetchConfig(
            cache_dir=Config.CACHE_DIR
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
            binary_system_prompt=f"""You are a highly meticulous academic research assistant focused on identifying conference topics, journal special issues, and workshop themes.

Task: Carefully analyze the provided list of items. Extract ONLY the items that represent specific, legitimate conference topics, workshop themes, journal special issues, or symposium topics related to or within the field of **{level2_term}**.

Input: A list of potential conference/journal topics.
Output: A Python-style list `[...]` containing ONLY the verified items from the input list that are specific conference topics or journal special issues related to **{level2_term}**.

CRITICAL Exclusion Criteria - DO NOT INCLUDE:
1. **Navigation/administrative items:** Reject 'Home', 'About', 'Contact', 'Registration', 'Venue', 'Important Dates', 'Committees'.
2. **Generic terms without specific focus:** Reject vague terms like 'Research', 'Papers', 'Program' on their own without specific topic context.
3. **People or roles:** Reject 'Keynote Speakers', 'Program Committee', 'Chairs', 'Organizers', 'Editors'.
4. **Unrelated topics:** Reject topics clearly not related to **{level2_term}**.
5. **Logistics:** Reject 'Registration Fee', 'Accommodation', 'Travel Information'.

Guidelines:
- Return an empty list `[]` if NO items in the input list meet the criteria for conference/journal topics related to **{level2_term}**.
- Accept specific research topics, paper categories, special issue themes, workshop focuses.
- Focus on SUBJECT MATTER, not organizational structure or event logistics.
- Output ONLY the Python-style list, with no extra text, explanation, or markdown formatting.

Example:
Input List for "Machine Learning":
["Home", "Reinforcement Learning Applications", "Registration", "Deep Learning for Computer Vision", "Program Committee", "Neural Network Architectures", "Important Dates", "Call for Papers", "Venue", "Machine Learning Ethics and Governance", "Contact Us"]
Output: ["Reinforcement Learning Applications", "Deep Learning for Computer Vision", "Neural Network Architectures", "Machine Learning Ethics and Governance"]
"""
        )
        
        # Construct search query
        query = f"(journal of | conference on) {level2_term} (topics | call for papers | tracks | workshops | programs | events)"
        
        # Perform web search
        search_results = web_search_bulk([query], search_config, logger=logger)
        
        if not search_results or not search_results.get("data"):
            logger.warning(f"No search results for '{level2_term}'")
            term_details["error"] = "No search results"
            # Save partial details before returning
            save_detailed_metadata(level2_term, term_details)
            return {"level2_term": level2_term, "conference_topics": [], "count": 0, "verified": False, "num_urls": 0, "num_lists": 0}
        
        # Extract URLs
        urls = [r.get("url") for r in search_results.get("data", [])[0].get("results", [])]
        urls = [url for url in urls if url]
        term_details["all_urls"] = urls
    
        if not urls:
            logger.warning(f"No URLs found in search results for '{level2_term}'")
            term_details["error"] = "No URLs found in search results"
            save_detailed_metadata(level2_term, term_details)
            return {"level2_term": level2_term, "conference_topics": [], "count": 0, "verified": False, "num_urls": 0, "num_lists": 0}
            
        logger.info(f"Found {len(urls)} URLs for '{level2_term}'")
            
        # Configure semaphore for concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
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
                    fetch_webpage(url, new_session, semaphore, html_config, level2_term, logger) 
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
                fetch_webpage(url, session, semaphore, html_config, level2_term, logger) 
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
                    
                # Preprocess HTML content to remove problematic sections
                preprocessed_html = preprocess_html_content(html_content)
                    
                # Extract lists from the webpage
                extracted_lists = extract_lists_from_html(preprocessed_html, list_config)
                
                if extracted_lists:
                    all_extracted_lists_raw.extend(extracted_lists) # Append raw dicts
                    url_to_raw_lists[url] = extracted_lists # Store raw dicts per URL
                    
                logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
            
        term_details["raw_extracted_lists"] = all_extracted_lists_raw

        # Filter and validate lists
        if not all_extracted_lists_raw:
            logger.warning(f"No lists extracted for '{level2_term}'")
            term_details["error"] = "No lists extracted from fetched HTML"
            save_detailed_metadata(level2_term, term_details)
            return {"level2_term": level2_term, "conference_topics": [], "count": 0, "verified": False, "num_urls": len(urls), "num_lists": 0}
            
        logger.info(f"Extracted a total of {len(all_extracted_lists_raw)} raw lists for '{level2_term}'. Starting filtering...")
        
        # Filter lists using the shared filtering utility - Capture necessary returned values
        final_llm_output_lists, llm_candidates, llm_results = await filter_lists(
            all_extracted_lists_raw, level2_term, filter_config, logger
        )
        
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
            save_detailed_metadata(level2_term, term_details)
            return {"level2_term": level2_term, "conference_topics": [], "count": 0, "verified": False, "num_urls": len(urls), "num_lists": len(all_extracted_lists_raw)}
            
        logger.info(f"After filtering/LLM, {len(final_llm_output_lists)} lists/sub-lists remain for '{level2_term}'")
        
        # Consolidate conference topics
        conference_topics = consolidate_lists(
            final_llm_output_lists, 
            level2_term, 
            min_frequency=1,
            min_list_appearances=1,
            similarity_threshold=0.7
        )
        term_details["final_consolidated_topics"] = conference_topics
        
        logger.info(f"Consolidated to {len(conference_topics)} unique conference topics for '{level2_term}'")
        
        # Simplified source/quality tracking
        topic_quality = {topic: 1.0 for topic in conference_topics}

        # Save detailed metadata for this term
        save_detailed_metadata(level2_term, term_details)

        # Return main results needed for aggregation
        return {
            "level2_term": level2_term,
            "conference_topics": conference_topics,
            "count": len(conference_topics),
            "url_sources": {}, # Simplified
            "quality_scores": topic_quality, # Simplified
            "verified": len(conference_topics) > 0,
            "num_urls": len(urls),
            "num_lists": len(all_extracted_lists_raw) # Report raw list count
        }
            
    except Exception as e:
        logger.error(f"Error processing term '{level2_term}': {str(e)}", exc_info=True)
        term_details["error"] = f"Unhandled exception: {str(e)}"
        save_detailed_metadata(level2_term, term_details) # Attempt to save details on error
        # Return error structure
        return {"level2_term": level2_term, "conference_topics": [], "count": 0, "verified": False, "num_urls": 0, "num_lists": 0, "error": str(e)}

def save_detailed_metadata(level2_term: str, data: Dict[str, Any]):
    """Saves the detailed processing metadata for a single level2 term to a JSON file."""
    try:
        # Sanitize filename
        safe_filename = re.sub(r'[\/:*?"<>|]', '_', level2_term) + "_details.json"
        output_path = os.path.join(Config.DETAILED_META_DIR, safe_filename)
        
        # Ensure the directory exists
        os.makedirs(Config.DETAILED_META_DIR, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            # Use default handler for non-serializable objects if any slip through (e.g., sets)
            json.dump(data, f, indent=2, default=lambda o: f"<non-serializable: {type(o).__name__}>")
        logger.info(f"Saved detailed metadata for '{level2_term}' to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save detailed metadata for '{level2_term}': {str(e)}", exc_info=True)


async def process_level2_terms_batch(batch: List[str], provider: Optional[str] = None, session: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Process a batch of level 2 terms"""
    if session:
        # If session is provided, use it for each term
        tasks = [process_level2_term(term, provider, session) for term in batch]
    else:
        # For backward compatibility
        tasks = [process_level2_term(term, provider) for term in batch]
    return await asyncio.gather(*tasks)


def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        Config.CACHE_DIR,
        Config.RAW_SEARCH_DIR,
        Config.DETAILED_META_DIR,
        os.path.dirname(Config.OUTPUT_FILE),
        os.path.dirname(Config.META_FILE)
    ]
    
    logger.info(f"BASE_DIR: {Config.BASE_DIR}")
    logger.info(f"LV2_INPUT_FILE: {Config.DEFAULT_LV2_INPUT_FILE}")
    logger.info(f"OUTPUT_FILE: {Config.OUTPUT_FILE}")
    logger.info(f"META_FILE: {Config.META_FILE}")
    logger.info(f"CACHE_DIR: {Config.CACHE_DIR}")
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
        
        args = parser.parse_args()
        
        provider = args.provider
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        input_file_path = args.input_file
        append_mode = args.append
        
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
        
        logger.info("Starting conference/journal topics extraction using level 2 terms")
        
        # Create output directories
        ensure_dirs_exist()
        
        # Read level 2 terms from the specified input file
        level2_terms = read_level2_terms(input_file_path)
        
        logger.info(f"Processing {len(level2_terms)} level 2 terms from '{input_file_path}' with batch size {batch_size} and max {max_concurrent} concurrent terms")
        
        # Initialize aiohttp session for the entire run
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, CookieJar
        import ssl, certifi
        
        # Create a default SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Improved connector configuration to handle file descriptor issues
        connector = TCPConnector(
            ssl=ssl_context,
            limit=max_concurrent * 3,  # Increase connection limit beyond concurrent requests
            limit_per_host=2,
            force_close=True,  # Force close connections to prevent descriptor reuse
            enable_cleanup_closed=True  # Enable cleanup of closed connections
        )
        
        cookie_jar = CookieJar(unsafe=True)  # Allow unsafe cookies to be more permissive
        
        # Create a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_term_with_throttling(term):
            """Process a term with throttling to limit concurrent execution"""
            async with semaphore:
                return await process_level2_term(term, provider, session)
        
        all_results = []
        start_time = time.time()
        
        # Process in batches
        timeout = ClientTimeout(total=3600, connect=30)  # 1 hour total timeout, 30s connect timeout
        async with ClientSession(
            connector=connector,
            cookie_jar=cookie_jar, 
            timeout=timeout,
            raise_for_status=False
        ) as session:
            for i in range(0, len(level2_terms), batch_size):
                batch = level2_terms[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(level2_terms) + batch_size - 1)//batch_size}")
                
                # Process terms in batch concurrently with throttling
                tasks = [process_term_with_throttling(term) for term in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions in batch results
                processed_batch_results = []
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing term '{batch[j]}': {str(result)}")
                        # Add a placeholder result for failed terms
                        processed_batch_results.append({
                            "level2_term": batch[j],
                            "conference_topics": [],
                            "count": 0,
                            "url_sources": {},
                            "quality_scores": {},
                            "verified": False,
                            "num_urls": 0,
                            "num_lists": 0
                        })
                    else:
                        processed_batch_results.append(result)
                
                all_results.extend(processed_batch_results)
                
                # Log progress after each batch
                elapsed = time.time() - start_time
                terms_processed = min(i + batch_size, len(level2_terms))
                terms_per_second = terms_processed / max(1, elapsed)
                eta_seconds = (len(level2_terms) - terms_processed) / max(0.1, terms_per_second)
                
                logger.info(f"Processed {terms_processed}/{len(level2_terms)} terms "
                            f"({terms_processed/len(level2_terms)*100:.1f}%) in {elapsed:.1f}s "
                            f"({terms_per_second:.2f} terms/s, ETA: {eta_seconds/60:.1f}m)")
                
                # Add a small delay between batches to avoid overloading
                await asyncio.sleep(2)  # Increased delay to reduce pressure on resources
        
        # Collect all conference topics and prepare for saving
        newly_found_topics = []
        newly_verified_terms_count = 0
        newly_processed_stats = {
            "total_urls_processed": 0,
            "total_lists_found": 0,
            "verified_topics_count": 0
        }
        new_level2_term_result_counts = {}
        new_level2_to_conference_topics = {}
        new_topic_sources = {}
        
        for result in all_results:
            if result.get("error"):
                logger.error(f"Skipping aggregation for term '{result['level2_term']}' due to processing error: {result['error']}")
                continue
                
            level2_term = result["level2_term"]
            conference_topics = result["conference_topics"]
            verified = result.get("verified", False)
            num_urls = result.get("num_urls", 0)
            num_lists = result.get("num_lists", 0)
            
            newly_processed_stats["total_urls_processed"] += num_urls
            newly_processed_stats["total_lists_found"] += num_lists
            
            if verified:
                newly_verified_terms_count += 1
                newly_processed_stats["verified_topics_count"] += len(conference_topics)
                
                if level2_term not in new_level2_to_conference_topics:
                    new_level2_to_conference_topics[level2_term] = []
                new_level2_to_conference_topics[level2_term].extend(conference_topics)
                newly_found_topics.extend(conference_topics)
                
                new_level2_term_result_counts[level2_term] = len(conference_topics)
                
                for topic in conference_topics:
                    if topic not in new_topic_sources:
                        new_topic_sources[topic] = []
                    new_topic_sources[topic].append(level2_term)
        
        logger.info(f"Consolidated results from {len(all_results)} newly processed terms.")
        logger.info(f"Found {newly_verified_terms_count} newly verified terms.")
        
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
        for topic in newly_found_topics:
            if topic not in final_unique_topics_set:
                final_unique_topics_set.add(topic)
                newly_added_count += 1
        
        logger.info(f"Added {newly_added_count} new unique conference topics.")
        
        # Save global output file (unique topics)
        final_unique_topics_list = sorted(list(final_unique_topics_set))
        random.shuffle(final_unique_topics_list) # Keep randomization if desired
        
        try:
            with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
                for topic in final_unique_topics_list:
                    f.write(f"{topic}\n")
            logger.info(f"Saved {len(final_unique_topics_list)} total unique conference topics to {Config.OUTPUT_FILE}")
        except Exception as e:
            logger.error(f"Failed to write output file {Config.OUTPUT_FILE}: {e}", exc_info=True)

        # Save global aggregate metadata file
        
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
        
        # Update metadata fields
        metadata["execution_time"] = f"{time.time() - start_time:.2f} seconds (current run)" # Indicate this is for the current run
        metadata["total_unique_conference_topics"] = len(final_unique_topics_list) # Update with final count
        
        # Update counts and mappings, merging carefully
        metadata["total_level2_terms_processed"] = metadata.get("total_level2_terms_processed", 0) + len(level2_terms) # Increment total processed
        metadata["verified_level2_terms_count"] = metadata.get("verified_level2_terms_count", 0) + newly_verified_terms_count # Increment verified
        
        # Merge result counts
        existing_result_counts = metadata.get("level2_term_result_counts", {})
        existing_result_counts.update(new_level2_term_result_counts) # Add/overwrite counts for newly processed terms
        metadata["level2_term_result_counts"] = existing_result_counts
        
        # Merge conference topic mappings
        existing_mapping = metadata.get("level2_to_conference_topics_mapping", {})
        for term, topics in new_level2_to_conference_topics.items():
            if term not in existing_mapping:
                existing_mapping[term] = []
            # Add only unique new topics for this term
            existing_topics_set = set(existing_mapping[term])
            for topic in topics:
                 if topic not in existing_topics_set:
                      existing_mapping[term].append(topic)
                      existing_topics_set.add(topic)
        metadata["level2_to_conference_topics_mapping"] = existing_mapping
        
        # Merge conference topic sources
        existing_sources = metadata.get("conference_topic_level2_sources", {})
        for topic, sources in new_topic_sources.items():
             if topic not in existing_sources:
                 existing_sources[topic] = []
             # Add unique new source terms for this topic
             existing_source_set = set(existing_sources[topic])
             for source in sources:
                  if source not in existing_source_set:
                       existing_sources[topic].append(source)
                       existing_source_set.add(source)
        metadata["conference_topic_level2_sources"] = existing_sources

        # Update provider, concurrent, batch size if they changed (or set initially)
        metadata["provider"] = provider or metadata.get("provider", "gemini")
        metadata["max_concurrent"] = max_concurrent
        metadata["batch_size"] = batch_size

        # Update processing stats (merge/add)
        existing_stats = metadata.get("processing_stats", {})
        metadata["total_urls_processed"] = metadata.get("total_urls_processed", 0) + newly_processed_stats["total_urls_processed"]
        metadata["total_raw_lists_extracted"] = metadata.get("total_raw_lists_extracted", 0) + newly_processed_stats["total_lists_found"]
        
        # Recalculate averages based on updated totals
        total_terms_ever_processed = metadata["total_level2_terms_processed"]
        total_verified_ever = metadata["verified_level2_terms_count"]
        total_verified_topics_ever = sum(len(topics) for topics in metadata["level2_to_conference_topics_mapping"].values()) # More accurate way to count total verified topics

        metadata["processing_stats"] = {
             "avg_urls_per_term": metadata["total_urls_processed"] / max(1, total_terms_ever_processed),
             "avg_raw_lists_per_term": metadata["total_raw_lists_extracted"] / max(1, total_terms_ever_processed),
             "avg_final_topics_per_verified_term": total_verified_topics_ever / max(1, total_verified_ever)
         }

        try:
            with open(Config.META_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Updated global aggregate metadata saved to {Config.META_FILE}")
        except Exception as e:
            logger.error(f"Failed to write metadata file {Config.META_FILE}: {e}", exc_info=True)
            
        logger.info(f"Detailed per-term metadata for this run saved in {Config.DETAILED_META_DIR}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Conference/journal topics extraction completed")

def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 