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
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists, init_llm

# Load environment variables and setup logging
load_dotenv('.env')
logger = setup_logger("lv2.s0")
random.seed(42)

# Constants
MAX_SEARCH_RESULTS = 30
MAX_CONCURRENT_REQUESTS = 32
BATCH_SIZE = 100  # Process multiple level 1 terms in a single batch

# Research-related keywords for enhanced filtering
RESEARCH_KEYWORDS = [
    "research", "study", "analysis", "theory", "methodology", "approach", 
    "experiment", "investigation", "application", "development", "model", 
    "framework", "technology", "technique", "method", "assessment", "evaluation",
    "innovation", "discovery", "implementation", "design", "exploration",
    "characterization", "measurement", "computation", "simulation", "modeling",
    "theory", "algorithm", "system", "process", "protocol", "strategy",
    "laboratory", "lab", "project", "program", "initiative", "paradigm",
    "analytics", "data", "clinical", "optimization", "engineering"
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

class Config:
    """Configuration for research areas extraction from level 1 terms"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    DEFAULT_LV1_INPUT_FILE = os.path.join(BASE_DIR, "data/lv1/lv1_final.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s0_research_areas.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s0_metadata.json")
    CACHE_DIR = os.path.join(BASE_DIR, "data/lv2/cache")
    RAW_SEARCH_DIR = os.path.join(BASE_DIR, "data/lv2/raw_search_results")
    DETAILED_META_DIR = os.path.join(BASE_DIR, "data/lv2/detailed_metadata")

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


async def process_level1_term(level1_term: str, provider: Optional[str] = None, session: Optional[Any] = None) -> Dict[str, Any]:
    """Process a single level1 term to extract research areas and save detailed metadata"""
    logger.info(f"Processing level 1 term: {level1_term}")
    
    # Initialize structures for detailed metadata
    term_details = {
        "level1_term": level1_term,
        "all_urls": [],
        "raw_extracted_lists": [],
        "llm_io_pairs": [],
        "final_consolidated_areas": [],
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
            binary_system_prompt=f"""You are a highly meticulous academic research assistant focused on identifying specific research areas and courses within university departments.

Task: Carefully analyze the provided list of items. Extract ONLY the items that represent specific, legitimate research areas, research groups, research labs, or teaching courses offered by or *highly relevant to the curriculum and research focus of* **The Department of {level1_term}**.

Input: A list of potential research areas/topics/courses/other text.
Output: A Python-style list `[...]` containing ONLY the verified items from the input list that are specific to and directly belong to or are highly relevant to **The Department of {level1_term}**. Preserve the original phrasing.

CRITICAL Exclusion Criteria - DO NOT INCLUDE:
1.  **Items from *clearly distinct* departments/fields:** Reject anything clearly belonging to a *significantly different academic field* (e.g., if analyzing 'English', reject 'Organic Chemistry', 'Microeconomics', 'Calculus').
2.  **University-wide or College-wide items:** Reject general university/college programs, centers, initiatives, or resources unless they are EXPLICITLY and PRIMARILY housed within **The Department of {level1_term}**.
3.  **Administrative/Navigational items:** Reject 'About Us', 'Contact', 'Faculty Directory', 'Admissions', 'Apply Now', 'News', 'Events', 'Home', 'Sitemap', 'Login'.
4.  **Overly Generic Terms:** Reject vague terms like 'Research', 'Studies', 'Graduate Programs', 'Undergraduate Programs', 'Curriculum', 'Resources', 'Facilities' unless part of a specific research area name (e.g., 'Research in Network Security' is okay, just 'Research' is not).
5.  **People/Organizations:** Reject specific faculty names, student groups, or associations.
6.  **Non-academic Content:** Reject irrelevant text, copyright notices, addresses, etc.

Guidelines:
- Return an empty list `[]` if NO items in the input list meet the criteria for **The Department of {level1_term}**.
- Focus on SPECIFIC research fields, sub-disciplines, labs, groups, or course titles directly associated with **The Department of {level1_term}**.
- **Accept relevant course titles/research areas even if a prefix (e.g., 'ENG', 'LIT', 'HIST') doesn't exactly match the department name, as long as the *topic* fits within the scope of {level1_term}.** Judge based on topical relevance to **{level1_term}**.
- Be conservative: If an item's relevance to **The Department of {level1_term}** is unclear or seems too broad, EXCLUDE it.
- Output ONLY the Python-style list, with no extra text, explanation, or markdown formatting.

Example 1 (Department of Biology):
Input List: ["Molecular Biology", "Genetics", "Ecology and Evolution", "University Research Opportunities", "BIOL 101: Introduction to Biology", "Physics Department", "Contact Us", "Neuroscience Program (Interdisciplinary)", "Cell Biology Lab"]
Output: ["Molecular Biology", "Genetics", "Ecology and Evolution", "BIOL 101: Introduction to Biology", "Cell Biology Lab"]

Example 2 (Department of Cultural Studies):
Input List: ["CUL 201 Intro to Cultural Theory", "ENG 3010 Modern Criticism", "HIST 350 Social History", "ANTH 210 Anthropology of Media", "PHYS 101 General Physics", "About the Department"]
Output: ["CUL 201 Intro to Cultural Theory", "ENG 3010 Modern Criticism", "HIST 350 Social History", "ANTH 210 Anthropology of Media"]
(Explanation: Includes relevant ENG, HIST, ANTH courses due to topical overlap, excludes Physics and admin item).

Analyze the following list STRICTLY for **The Department of {level1_term}** and return the verified items in the specified Python list format:"""
        )
        
        # Construct search query
        query = f"site:.edu department of {level1_term} (research areas | research topics | teaching courses)"
        
        # Perform web search
        search_results = web_search_bulk([query], search_config, logger=logger)
        
        if not search_results or not search_results.get("data"):
            logger.warning(f"No search results for '{level1_term}'")
            term_details["error"] = "No search results"
            # Save partial details before returning
            save_detailed_metadata(level1_term, term_details)
            return {"level1_term": level1_term, "research_areas": [], "count": 0, "verified": False, "num_urls": 0, "num_lists": 0}
        
        # Extract URLs
        urls = [r.get("url") for r in search_results.get("data", [])[0].get("results", [])]
        urls = [url for url in urls if url]
        term_details["all_urls"] = urls
    
        if not urls:
            logger.warning(f"No URLs found in search results for '{level1_term}'")
            term_details["error"] = "No URLs found in search results"
            save_detailed_metadata(level1_term, term_details)
            return {"level1_term": level1_term, "research_areas": [], "count": 0, "verified": False, "num_urls": 0, "num_lists": 0}
            
        logger.info(f"Found {len(urls)} URLs for '{level1_term}'")
            
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
                    fetch_webpage(url, new_session, semaphore, html_config, level1_term, logger) 
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
                        url_to_raw_lists[url] = extracted_lists # Store raw dicts per URL
                        logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
        else:
            # Use the provided session
            fetch_tasks = [
                fetch_webpage(url, session, semaphore, html_config, level1_term, logger) 
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
                    url_to_raw_lists[url] = extracted_lists # Store raw dicts per URL
                    
                logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
            
        term_details["raw_extracted_lists"] = all_extracted_lists_raw

        # Filter and validate lists
        if not all_extracted_lists_raw:
            logger.warning(f"No lists extracted for '{level1_term}'")
            term_details["error"] = "No lists extracted from fetched HTML"
            save_detailed_metadata(level1_term, term_details)
            return {"level1_term": level1_term, "research_areas": [], "count": 0, "verified": False, "num_urls": len(urls), "num_lists": 0}
            
        logger.info(f"Extracted a total of {len(all_extracted_lists_raw)} raw lists for '{level1_term}'. Starting filtering...")
        
        # Filter lists using the shared filtering utility - Capture necessary returned values
        final_llm_output_lists, llm_candidates, llm_results = await filter_lists(
            all_extracted_lists_raw, level1_term, filter_config, logger
        )
        
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
            save_detailed_metadata(level1_term, term_details)
            return {"level1_term": level1_term, "research_areas": [], "count": 0, "verified": False, "num_urls": len(urls), "num_lists": len(all_extracted_lists_raw)}
            
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

        # Save detailed metadata for this term
        save_detailed_metadata(level1_term, term_details)

        # Return main results needed for aggregation
        return {
            "level1_term": level1_term,
            "research_areas": research_areas,
            "count": len(research_areas),
            "url_sources": {}, # Simplified
            "quality_scores": research_area_quality, # Simplified
            "verified": len(research_areas) > 0,
            "num_urls": len(urls),
            "num_lists": len(all_extracted_lists_raw) # Report raw list count
        }
            
    except Exception as e:
        logger.error(f"Error processing term '{level1_term}': {str(e)}", exc_info=True)
        term_details["error"] = f"Unhandled exception: {str(e)}"
        save_detailed_metadata(level1_term, term_details) # Attempt to save details on error
        # Return error structure
        return {"level1_term": level1_term, "research_areas": [], "count": 0, "verified": False, "num_urls": 0, "num_lists": 0, "error": str(e)}

def save_detailed_metadata(level1_term: str, data: Dict[str, Any]):
    """Saves the detailed processing metadata for a single level1 term to a JSON file."""
    try:
        # Sanitize filename
        safe_filename = re.sub(r'[\/:*?"<>|]', '_', level1_term) + "_details.json"
        output_path = os.path.join(Config.DETAILED_META_DIR, safe_filename)
        
        # Ensure the directory exists
        os.makedirs(Config.DETAILED_META_DIR, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            # Use default handler for non-serializable objects if any slip through (e.g., sets)
            json.dump(data, f, indent=2, default=lambda o: f"<non-serializable: {type(o).__name__}>")
        logger.info(f"Saved detailed metadata for '{level1_term}' to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save detailed metadata for '{level1_term}': {str(e)}", exc_info=True)


async def process_level1_terms_batch(batch: List[str], provider: Optional[str] = None, session: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Process a batch of level 1 terms"""
    if session:
        # If session is provided, use it for each term
        tasks = [process_level1_term(term, provider, session) for term in batch]
    else:
        # For backward compatibility
        tasks = [process_level1_term(term, provider) for term in batch]
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
    logger.info(f"LV1_INPUT_FILE: {Config.DEFAULT_LV1_INPUT_FILE}")
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
        parser = argparse.ArgumentParser(description="Extract research areas for level 1 terms.")
        parser.add_argument("--provider", help="LLM provider (e.g., gemini, openai)")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for processing terms (default: {BATCH_SIZE})")
        parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, help=f"Max concurrent term processing requests (default: {MAX_CONCURRENT_REQUESTS})")
        parser.add_argument("--input-file", default=Config.DEFAULT_LV1_INPUT_FILE, help=f"Path to the input file containing level 1 terms (default: {Config.DEFAULT_LV1_INPUT_FILE})")
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
        if input_file_path != Config.DEFAULT_LV1_INPUT_FILE:
             logger.info(f"Using custom input file: {input_file_path}")
        if append_mode:
             logger.info("Append mode enabled. Results will be added to existing files.")
        
        logger.info("Starting research areas extraction using level 1 terms")
        
        # Create output directories
        ensure_dirs_exist()
        
        # Read level 1 terms from the specified input file
        level1_terms = read_level1_terms(input_file_path)
        
        logger.info(f"Processing {len(level1_terms)} level 1 terms from '{input_file_path}' with batch size {batch_size} and max {max_concurrent} concurrent terms")
        
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
                return await process_level1_term(term, provider, session)
        
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
            for i in range(0, len(level1_terms), batch_size):
                batch = level1_terms[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(level1_terms) + batch_size - 1)//batch_size}")
                
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
                            "level1_term": batch[j],
                            "research_areas": [],
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
                terms_processed = min(i + batch_size, len(level1_terms))
                terms_per_second = terms_processed / max(1, elapsed)
                eta_seconds = (len(level1_terms) - terms_processed) / max(0.1, terms_per_second)
                
                logger.info(f"Processed {terms_processed}/{len(level1_terms)} terms "
                            f"({terms_processed/len(level1_terms)*100:.1f}%) in {elapsed:.1f}s "
                            f"({terms_per_second:.2f} terms/s, ETA: {eta_seconds/60:.1f}m)")
                
                # Add a small delay between batches to avoid overloading
                await asyncio.sleep(2)  # Increased delay to reduce pressure on resources
        
        # Collect all research areas and prepare for saving
        newly_found_research_areas = []
        newly_verified_terms_count = 0
        newly_processed_stats = {
            "total_urls_processed": 0,
            "total_lists_found": 0,
            "verified_research_areas_count": 0
        }
        new_level1_term_result_counts = {}
        new_level1_to_research_areas = {}
        new_research_area_sources = {}
        
        for result in all_results:
            if result.get("error"):
                logger.error(f"Skipping aggregation for term '{result['level1_term']}' due to processing error: {result['error']}")
                continue
                
            level1_term = result["level1_term"]
            research_areas = result["research_areas"]
            verified = result.get("verified", False)
            num_urls = result.get("num_urls", 0)
            num_lists = result.get("num_lists", 0)
            
            newly_processed_stats["total_urls_processed"] += num_urls
            newly_processed_stats["total_lists_found"] += num_lists # Total raw lists found
            
            if verified:
                newly_verified_terms_count += 1
                newly_processed_stats["verified_research_areas_count"] += len(research_areas)
                
                if level1_term not in new_level1_to_research_areas:
                    new_level1_to_research_areas[level1_term] = []
                new_level1_to_research_areas[level1_term].extend(research_areas)
                newly_found_research_areas.extend(research_areas)
                
                new_level1_term_result_counts[level1_term] = len(research_areas)
                
                for area in research_areas:
                    if area not in new_research_area_sources:
                        new_research_area_sources[area] = []
                    new_research_area_sources[area].append(level1_term)
        
        logger.info(f"Consolidated results from {len(all_results)} newly processed terms.")
        logger.info(f"Found {newly_verified_terms_count} newly verified terms.")
        
        # --- File Saving Logic ---
        
        existing_unique_areas = set()
        if append_mode and os.path.exists(Config.OUTPUT_FILE):
            logger.info(f"Loading existing research areas from {Config.OUTPUT_FILE} for append mode.")
            try:
                with open(Config.OUTPUT_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        existing_unique_areas.add(line.strip())
                logger.info(f"Loaded {len(existing_unique_areas)} existing unique areas.")
            except Exception as e:
                logger.warning(f"Could not read existing output file {Config.OUTPUT_FILE}: {e}. Starting fresh.", exc_info=True)
                existing_unique_areas = set() # Start fresh if read fails
        
        # Add newly found unique areas
        final_unique_areas_set = set(existing_unique_areas)
        newly_added_count = 0
        for area in newly_found_research_areas:
            if area not in final_unique_areas_set:
                final_unique_areas_set.add(area)
                newly_added_count += 1
        
        logger.info(f"Added {newly_added_count} new unique research areas.")
        
        # Save global output file (unique areas)
        final_unique_areas_list = sorted(list(final_unique_areas_set))
        random.shuffle(final_unique_areas_list) # Keep randomization if desired
        
        try:
            with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
                for area in final_unique_areas_list:
                    f.write(f"{area}\n")
            logger.info(f"Saved {len(final_unique_areas_list)} total unique research areas to {Config.OUTPUT_FILE}")
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
        metadata["total_unique_research_areas"] = len(final_unique_areas_list) # Update with final count
        
        # Update counts and mappings, merging carefully
        metadata["total_level1_terms_processed"] = metadata.get("total_level1_terms_processed", 0) + len(level1_terms) # Increment total processed
        metadata["verified_level1_terms_count"] = metadata.get("verified_level1_terms_count", 0) + newly_verified_terms_count # Increment verified
        
        # Merge result counts
        existing_result_counts = metadata.get("level1_term_result_counts", {})
        existing_result_counts.update(new_level1_term_result_counts) # Add/overwrite counts for newly processed terms
        metadata["level1_term_result_counts"] = existing_result_counts
        
        # Merge research area mappings
        existing_mapping = metadata.get("level1_to_research_areas_mapping", {})
        for term, areas in new_level1_to_research_areas.items():
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
        for area, sources in new_research_area_sources.items():
             if area not in existing_sources:
                 existing_sources[area] = []
             # Add unique new source terms for this area
             existing_source_set = set(existing_sources[area])
             for source in sources:
                  if source not in existing_source_set:
                       existing_sources[area].append(source)
                       existing_source_set.add(source)
        metadata["research_area_level1_sources"] = existing_sources

        # Update provider, concurrent, batch size if they changed (or set initially)
        metadata["provider"] = provider or metadata.get("provider", "gemini")
        metadata["max_concurrent"] = max_concurrent
        metadata["batch_size"] = batch_size

        # Update processing stats (merge/add)
        existing_stats = metadata.get("processing_stats", {})
        metadata["total_urls_processed"] = metadata.get("total_urls_processed", 0) + newly_processed_stats["total_urls_processed"]
        metadata["total_raw_lists_extracted"] = metadata.get("total_raw_lists_extracted", 0) + newly_processed_stats["total_lists_found"]
        
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
            with open(Config.META_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Updated global aggregate metadata saved to {Config.META_FILE}")
        except Exception as e:
            logger.error(f"Failed to write metadata file {Config.META_FILE}: {e}", exc_info=True)
            
        logger.info(f"Detailed per-term metadata for this run saved in {Config.DETAILED_META_DIR}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Research areas extraction completed")

def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 