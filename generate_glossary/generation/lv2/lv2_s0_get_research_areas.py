import os
import sys
import random
import json
import time
import asyncio
import aiohttp
import re
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
MAX_SEARCH_RESULTS = 50
MAX_CONCURRENT_REQUESTS = 5
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
    LV1_INPUT_FILE = os.path.join(BASE_DIR, "data/lv1/postprocessed/lv1_final.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s0_research_areas.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv2/raw/lv2_s0_metadata.json")
    CACHE_DIR = os.path.join(BASE_DIR, "data/lv2/cache")
    RAW_SEARCH_DIR = os.path.join(BASE_DIR, "data/lv2/raw_search_results")
    

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
    # Custom research area scoring logic with weights
    weights = {
        "keyword": 0.25,      # Weight for keywords
        "structure": 0.15,    # HTML structure weight
        "pattern": 0.20,      # Naming pattern weight
        "non_term": 0.15,     # Absence of non-relevant terms weight
        "consistency": 0.10,  # Consistency in formatting weight
        "size": 0.05,         # Appropriate list size weight
        "html_type": 0.10     # HTML element type appropriateness weight
    }
    
    # Use the common scoring function from list_extractor
    return score_list(
        items=items,
        metadata=metadata,
        context_term=context_term,
        keywords=RESEARCH_KEYWORDS,
        scoring_weights=weights
    )


async def process_level1_term(level1_term: str, provider: Optional[str] = None, session: Optional[Any] = None) -> Dict[str, Any]:
    """Process a single level1 term to extract research areas"""
    logger.info(f"Processing level 1 term: {level1_term}")
    
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
        binary_llm_decision=True,
        binary_system_prompt=f"""You are a research expert specializing in academic research classification and organization.

Task: Determine if a list contains legitimate research areas/topics that are DIRECTLY studied within The Department of {level1_term}.

You must return a clear YES or NO decision for each list.
- Answer YES ONLY if the list primarily contains research areas or teaching courses that belong DIRECTLY to The Department of {level1_term}
- Answer NO if:
  1. The list contains website menu items, navigation sections, or non-relevant content
  2. The list contains valid research areas, but they belong to a DIFFERENT department (not The Department of {level1_term})
  3. The list contains courses, faculty names, or items that are not research areas
  4. The list contains overly general topics without specific research focus

Examples:

For The Department of Computer Science:
- VALID (YES): "Machine Learning, Computer Vision, Algorithms, Database Systems, Cybersecurity"
- INVALID (NO): "Finance, Economics, Business Analytics" (these belong to Business departments)
- INVALID (NO): "CS101, Introduction to Programming, Advanced Algorithms" (these are courses, not research areas)
- INVALID (NO): "Faculty Directory, Student Resources, Graduate Programs" (these are website sections)

For The Department of Psychology:
- VALID (YES): "Cognitive Psychology, Behavioral Neuroscience, Clinical Psychology, Developmental Psychology"
- INVALID (NO): "Organic Chemistry, Inorganic Chemistry, Biochemistry" (these belong to Chemistry)
- INVALID (NO): "Dr. Smith Lab, Psychology Student Association" (these are people/organizations, not research areas)

For The Department of Biology:
- VALID (YES): "Molecular Biology, Genetics, Ecology, Evolutionary Biology, Cell Biology"
- INVALID (NO): "Physics, Astronomy, Mathematics" (these belong to other departments)
- INVALID (NO): "Lab Equipment, Research Publications, Conference Proceedings" (these are resources, not research areas)

THIS IS CRITICAL: Your decision must be binary (YES/NO) with no middle ground or uncertainty. The list must contain DIRECT research areas under The Department of {level1_term} to be considered valid."""
    )
    
    # Construct search query
    query = f"site:.edu department of {level1_term} (research areas | teaching course)"
    
    # Perform web search
    search_results = web_search_bulk([query], search_config, logger=logger)
    
    if not search_results or not search_results.get("data"):
        logger.warning(f"No search results for '{level1_term}'")
        return {
            "level1_term": level1_term,
            "research_areas": [],
            "count": 0,
            "url_sources": {},
            "quality_scores": {},
            "verified": False,
            "num_urls": 0,
            "num_lists": 0
        }
    
    # Process search results
    try:
        # Extract URLs from search results
        urls = [r.get("url") for r in search_results.get("data", [])[0].get("results", [])]
        urls = [url for url in urls if url]
    
        if not urls:
            logger.warning(f"No URLs found in search results for '{level1_term}'")
            return {
                "level1_term": level1_term,
                "research_areas": [],
                "count": 0,
                "url_sources": {},
                "quality_scores": {},
                "verified": False,
                "num_urls": 0,
                "num_lists": 0
            }
            
        logger.info(f"Found {len(urls)} URLs for '{level1_term}'")
            
        # Configure semaphore for concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
        all_extracted_lists = []
        url_to_lists = {}
        
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
                        all_extracted_lists.extend(extracted_lists)
                        url_to_lists[url] = [list(l["items"]) for l in extracted_lists]
                        
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
                    all_extracted_lists.extend(extracted_lists)
                    url_to_lists[url] = [list(l["items"]) for l in extracted_lists]
                    
                logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
            
        # Filter and validate lists
        if not all_extracted_lists:
            logger.warning(f"No lists extracted for '{level1_term}'")
            return {
                "level1_term": level1_term,
                "research_areas": [],
                "count": 0,
                "url_sources": {},
                "quality_scores": {},
                "verified": False,
                "num_urls": len(urls),
                "num_lists": 0
            }
            
        logger.info(f"Extracted a total of {len(all_extracted_lists)} lists for '{level1_term}'")
        
        # Filter lists using the shared filtering utility
        filtered_lists = await filter_lists(all_extracted_lists, level1_term, filter_config, logger)
        
        if not filtered_lists:
            logger.warning(f"No lists passed filtering for '{level1_term}'")
            return {
                "level1_term": level1_term,
                "research_areas": [],
                "count": 0,
                "url_sources": {},
                "quality_scores": {},
                "verified": False,
                "num_urls": len(urls),
                "num_lists": len(all_extracted_lists)
            }
            
        logger.info(f"After filtering, {len(filtered_lists)} lists remain for '{level1_term}'")
        
        # Consolidate research areas using the shared utility
        research_areas = consolidate_lists(
            filtered_lists, 
            level1_term, 
            min_frequency=1,
            min_list_appearances=1,
            similarity_threshold=0.7
        )
        
        logger.info(f"Found {len(research_areas)} research areas for '{level1_term}'")
        
        # Track URL sources for each research area
        research_area_sources = {}
        research_area_quality = {}
        
        # Map research areas to URLs they came from
        for url, lists in url_to_lists.items():
            for area_list in lists:
                for area in area_list:
                    area_lower = area.lower()
                    # Find matching research area in our final list
                    for final_area in research_areas:
                        final_area_lower = final_area.lower()
                        if area_lower == final_area_lower or area_lower in final_area_lower or final_area_lower in area_lower:
                            if final_area not in research_area_sources:
                                research_area_sources[final_area] = []
                            if url not in research_area_sources[final_area]:
                                research_area_sources[final_area].append(url)
        
        # Set quality score for each research area
        for area in research_areas:
            # Basic quality score based on number of sources
            sources = research_area_sources.get(area, [])
            research_area_quality[area] = min(1.0, len(sources) / 3)  # Scale up to 3 sources = 1.0
        
        # Save extracted lists data for analysis
        try:
            extracted_lists_file = os.path.join(Config.RAW_SEARCH_DIR, f"{level1_term}_extracted_lists.json")
            with open(extracted_lists_file, "w", encoding="utf-8") as f:
                # Calculate the number of verified lists per URL
                verified_lists_per_url = {}
                verified_lists_ids = {}
                for url, lists in url_to_lists.items():
                    # Count how many lists contain at least one research area from the final list
                    verified_list_count = 0
                    verified_ids = []
                    
                    # For each list from this URL
                    for i, area_list in enumerate(lists):
                        # Check if any research area in this list made it to the final research areas list
                        for area in area_list:
                            area_lower = area.lower()
                            if any(area_lower == final_area.lower() or 
                                area_lower in final_area.lower() or 
                                final_area.lower() in area_lower 
                                for final_area in research_areas):
                                verified_list_count += 1
                                verified_ids.append(i)
                                break
                    
                    verified_lists_per_url[url] = verified_list_count
                    verified_lists_ids[url] = verified_ids
                
                # Create URL verification status dictionary
                url_verification_status = {url: len(verified_ids) > 0 
                                        for url, verified_ids in verified_lists_ids.items()}
                
                # Calculate total lists per URL
                total_lists_per_url = {url: len(lists) for url, lists in url_to_lists.items()}
                
                json.dump({
                    "level1_term": level1_term,
                    "url_verification_status": url_verification_status,
                    "verified_lists_per_url": verified_lists_per_url,
                    "total_lists_per_url": total_lists_per_url,
                    "verified_lists_ids": verified_lists_ids,
                    "num_urls": len(urls),
                    "num_lists": len(all_extracted_lists),
                    "url_to_lists": url_to_lists,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            logger.debug(f"Saved extracted lists for '{level1_term}' to {extracted_lists_file}")
        except Exception as e:
            logger.warning(f"Failed to save extracted lists for '{level1_term}': {str(e)}")
    
        return {
            "level1_term": level1_term,
            "research_areas": research_areas,
            "count": len(research_areas),
            "url_sources": research_area_sources,
            "quality_scores": research_area_quality,
            "verified": len(research_areas) > 0,
            "num_urls": len(urls),
            "num_lists": len(all_extracted_lists),
            "verified_lists_per_url": verified_lists_per_url if 'verified_lists_per_url' in locals() else {},
            "total_lists_per_url": total_lists_per_url if 'total_lists_per_url' in locals() else {},
            "verified_lists_ids": verified_lists_ids if 'verified_lists_ids' in locals() else {}
        }
            
    except Exception as e:
        logger.error(f"Error processing term '{level1_term}': {str(e)}", exc_info=True)
        return {
            "level1_term": level1_term,
            "research_areas": [],
            "count": 0,
            "url_sources": {},
            "quality_scores": {},
            "verified": False,
            "num_urls": 0,
            "num_lists": 0
        }


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
        os.path.dirname(Config.OUTPUT_FILE),
        os.path.dirname(Config.META_FILE)
    ]
    
    logger.info(f"BASE_DIR: {Config.BASE_DIR}")
    logger.info(f"LV1_INPUT_FILE: {Config.LV1_INPUT_FILE}")
    logger.info(f"OUTPUT_FILE: {Config.OUTPUT_FILE}")
    logger.info(f"META_FILE: {Config.META_FILE}")
    logger.info(f"CACHE_DIR: {Config.CACHE_DIR}")
    logger.info(f"RAW_SEARCH_DIR: {Config.RAW_SEARCH_DIR}")
    
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
        # Get provider from command line args
        provider = None
        if len(sys.argv) > 1 and sys.argv[1] == "--provider":
            provider = sys.argv[2]
            logger.info(f"Using provider: {provider}")
        
        # Get batch size from command line args
        batch_size = BATCH_SIZE
        if len(sys.argv) > 1 and sys.argv[1] == "--batch-size":
            try:
                batch_size = int(sys.argv[2])
                logger.info(f"Using custom batch size: {batch_size}")
            except ValueError:
                logger.warning(f"Invalid batch size provided: {sys.argv[2]}. Using default: {BATCH_SIZE}")
        
        # Get concurrent processing limit from command line args
        max_concurrent = 5  # Default concurrent terms
        if len(sys.argv) > 1 and sys.argv[1] == "--max-concurrent":
            try:
                max_concurrent = int(sys.argv[2])
                logger.info(f"Using custom concurrent limit: {max_concurrent}")
            except ValueError:
                logger.warning(f"Invalid concurrent limit provided: {sys.argv[2]}. Using default: 5")
        
        logger.info("Starting research areas extraction using level 1 terms")
        
        # Create output directories
        ensure_dirs_exist()
        
        # Read level 1 terms
        level1_terms = read_level1_terms(Config.LV1_INPUT_FILE)
        
        logger.info(f"Processing {len(level1_terms)} level 1 terms with batch size {batch_size} and max {max_concurrent} concurrent terms")
        
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
        
        # Collect all research areas
        all_research_areas = []
        research_area_sources = {}  # Level1 term sources
        research_area_url_sources = {}  # URL sources
        research_area_quality_scores = {}  # Quality scores
        level1_to_research_areas = {}  # Level1 term to research areas mapping
        verified_terms_count = 0  # Count verified level1 terms
        
        # Collect statistics for each level1 term and across all terms
        total_urls_processed = 0
        total_lists_found = 0
        verified_research_areas_count = 0
        urls_per_term = {}
        lists_per_term = {}
        verified_lists_per_term = {}
        verified_lists_per_url_all = {}
        total_lists_per_url_all = {}
        verified_lists_ids_all = {}
        url_verification_status_all = {}
        
        for result in all_results:
            level1_term = result["level1_term"]
            research_areas = result["research_areas"]
            url_sources = result.get("url_sources", {})
            quality_scores = result.get("quality_scores", {})
            verified = result.get("verified", False)  # Get verification status
            
            # Collect statistics
            num_urls = result.get("num_urls", 0)
            num_lists = result.get("num_lists", 0)
            
            # Get verified lists data
            verified_lists_per_url = result.get("verified_lists_per_url", {})
            total_lists_per_url = result.get("total_lists_per_url", {})
            verified_lists_ids = result.get("verified_lists_ids", {})
            
            # Calculate number of verified lists for this term
            verified_list_count = sum(verified_lists_per_url.values())
            
            # Store statistics by term
            urls_per_term[level1_term] = num_urls
            lists_per_term[level1_term] = num_lists
            verified_lists_per_term[level1_term] = verified_list_count
            
            # Store URL-specific data in global collections
            verified_lists_per_url_all.update(verified_lists_per_url)
            total_lists_per_url_all.update(total_lists_per_url)
            verified_lists_ids_all[level1_term] = verified_lists_ids
            
            # Create URL verification status dictionary for all terms
            for url, verified_ids in verified_lists_ids.items():
                if url not in url_verification_status_all:
                    url_verification_status_all[url] = {}
                url_verification_status_all[url][level1_term] = len(verified_ids) > 0
            
            # Update totals
            total_urls_processed += num_urls
            total_lists_found += num_lists
            
            # Only include verified research areas in the mapping
            if verified:
                verified_terms_count += 1
                verified_research_areas_count += len(research_areas)
                
                # Initialize level1_to_research_areas entry if it doesn't exist
                if level1_term not in level1_to_research_areas:
                    level1_to_research_areas[level1_term] = []
                
                # Add research areas to level1_to_research_areas mapping
                level1_to_research_areas[level1_term].extend(research_areas)
                
                # Add research areas to the global collection
                all_research_areas.extend(research_areas)
                
                # Track sources and quality scores
                for area in research_areas:
                    # Track level1 term sources
                    if area not in research_area_sources:
                        research_area_sources[area] = []
                    research_area_sources[area].append(level1_term)
                    
                    # Track URL sources
                    if area not in research_area_url_sources:
                        research_area_url_sources[area] = []
                    if area in url_sources:
                        research_area_url_sources[area].extend(url_sources[area])
                    
                    # Track quality scores
                    if area in quality_scores:
                        if area not in research_area_quality_scores:
                            research_area_quality_scores[area] = quality_scores[area]
                        else:
                            research_area_quality_scores[area] = max(
                                research_area_quality_scores[area],
                                quality_scores[area]
                            )
        
        logger.info(f"Found {verified_terms_count} level 1 terms with verified research areas")
        
        # Remove duplicates while preserving case
        unique_research_areas = []
        seen = set()
        for area in all_research_areas:
            area_lower = area.lower()
            if area_lower not in seen:
                seen.add(area_lower)
                unique_research_areas.append(area)
        
        # Also deduplicate level1_to_research_areas mapping while preserving case
        for level1_term, areas in level1_to_research_areas.items():
            unique_level1_areas = []
            seen_level1_areas = set()
            for area in areas:
                area_lower = area.lower()
                if area_lower not in seen_level1_areas:
                    seen_level1_areas.add(area_lower)
                    unique_level1_areas.append(area)
            level1_to_research_areas[level1_term] = unique_level1_areas
        
        # Randomize order
        random.shuffle(unique_research_areas)
        
        # Save to output file
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for area in unique_research_areas:
                f.write(f"{area}\n")
        
        # Save metadata
        metadata = {
            "execution_time": f"{time.time() - start_time:.2f} seconds",
            "total_research_areas": len(unique_research_areas),
            "level1_terms": len(level1_terms),
            "verified_level1_terms": verified_terms_count,
            "research_area_counts_by_level1": {
                result["level1_term"]: result["count"] 
                for result in all_results 
                if result.get("verified", False)  # Only include verified terms
            },
            "level1_to_research_areas": level1_to_research_areas,  # Only contains verified entries now
            "research_area_sources": {area: sources for area, sources in research_area_sources.items()},
            "research_area_url_sources": {area: list(set(urls)) for area, urls in research_area_url_sources.items() if urls},
            "research_area_quality_scores": research_area_quality_scores,
            "provider": provider or "gemini",
            "max_concurrent": max_concurrent,
            "batch_size": batch_size,
            # Add statistics tracking
            "urls_per_term": urls_per_term,
            "lists_per_term": lists_per_term,
            "total_urls_processed": total_urls_processed,
            "total_lists_found": total_lists_found,
            "verified_research_areas_count": verified_research_areas_count,
            "processing_stats": {
                "avg_urls_per_term": total_urls_processed / max(1, len(level1_terms)),
                "avg_lists_per_term": total_lists_found / max(1, len(level1_terms)),
                "avg_research_areas_per_term": verified_research_areas_count / max(1, verified_terms_count)
            },
            "verified_lists_per_term": verified_lists_per_term,
            "verified_lists_per_url_all": verified_lists_per_url_all,
            "total_lists_per_url_all": total_lists_per_url_all,
            "verified_lists_ids_all": verified_lists_ids_all,
            "url_verification_status_all": url_verification_status_all
        }
        
        with open(Config.META_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully extracted {len(unique_research_areas)} unique research areas from {verified_terms_count} verified level 1 terms")
        logger.info(f"Research areas saved to {Config.OUTPUT_FILE}")
        logger.info(f"Metadata saved to {Config.META_FILE}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Research areas extraction completed")


def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 