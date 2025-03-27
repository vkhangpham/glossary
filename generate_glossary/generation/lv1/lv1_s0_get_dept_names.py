import os
import sys
import asyncio
import json
import time
import re
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv
import aiohttp

# Fix import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import Provider

# Import shared web search utilities
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists

# Load environment variables and setup logging
load_dotenv('.env')
logger = setup_logger("lv1.s0")
random.seed(42)

# Constants
MAX_SEARCH_RESULTS = 50
MAX_CONCURRENT_REQUESTS = 5
BATCH_SIZE = 100  # Process multiple level 0 terms in a single batch

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

class Config:
    """Configuration for department names extraction from level 0 terms"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    LV0_INPUT_FILE = os.path.join(BASE_DIR, "data/lv0/postprocessed/lv0_final.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_department_names.txt")
    META_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_metadata.json")
    CACHE_DIR = os.path.join(BASE_DIR, "data/lv1/cache")
    RAW_SEARCH_DIR = os.path.join(BASE_DIR, "data/lv1/raw_search_results")
    

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


async def process_level0_term(level0_term: str, provider: Optional[str] = None, session: Optional[Any] = None) -> Dict[str, Any]:
    """Process a single level 0 term to extract department names"""
    logger.info(f"Processing level 0 term: {level0_term}")
    
    # Create configurations for the shared utilities
    search_config = WebSearchConfig(
        base_dir=Config.BASE_DIR,
        raw_search_dir=Config.RAW_SEARCH_DIR
    )
    
    html_config = HTMLFetchConfig(
        cache_dir=Config.CACHE_DIR
    )
    
    list_config = ListExtractionConfig(
        keywords=DEPARTMENT_KEYWORDS,
        anti_keywords=NON_DEPARTMENT_KEYWORDS,
        patterns=DEPARTMENT_PATTERNS
    )
    
    # Use a default binary system prompt if none provided
    binary_system_prompt = f"""You are an expert in academic institution organization and department structures.

Your task is to evaluate whether a provided list contains departments that are DIRECTLY under the umbrella of The College of {level0_term}.

You must return a clear YES or NO decision for each list.
- Answer YES ONLY if the list primarily contains departments that belong DIRECTLY to The College of {level0_term}
- Answer NO if:
  1. The list contains website menu items, navigation sections, or non-relevant content
  2. The list contains valid departments, but they belong to a DIFFERENT college (not The College of {level0_term})
  3. The list contains sub-departments or research areas that are too specific (not direct departments)

Examples:

For The College of Engineering:
- VALID (YES): "Civil Engineering, Mechanical Engineering, Electrical Engineering, Chemical Engineering"
- INVALID (NO): "Mathematics, Physics, Chemistry, Biology" (these belong to College of Science, not Engineering)
- INVALID (NO): "Biomechanics Lab, Robotics Research Group, AI Systems" (these are research groups, not departments)
- INVALID (NO): "Undergraduate Programs, Graduate Studies, Faculty Resources" (these are website sections)

For The College of Arts:
- VALID (YES): "Theater, Music, Fine Arts, Dance, Visual Arts"
- INVALID (NO): "Microbiology, Genetics, Zoology" (these belong to College of Science)
- INVALID (NO): "Jazz Performance, Piano Studies, Sculpture Studio" (these are programs within departments, too specific)

For The College of Business:
- VALID (YES): "Marketing, Finance, Accounting, Management, Economics"
- INVALID (NO): "Computer Science, Software Engineering" (these belong to College of Computing)

THIS IS CRITICAL: Your decision must be binary (YES/NO) with no middle ground or uncertainty. The list must contain DIRECT departments under The College of {level0_term} to be considered valid."""
    
    filter_config = FilterConfig(
        scoring_fn=score_department_list,
        clean_item_fn=clean_department_name,
        provider=provider,
        use_llm_validation=True,
        binary_llm_decision=True,
        binary_system_prompt=binary_system_prompt
    )
    
    # Construct search query
    query = f"site:.edu college of {level0_term} list of departments"
    
    # Perform web search
    search_results = web_search_bulk([query], search_config, logger=logger)
    
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
        # Extract URLs from search results
        urls = [r.get("url") for r in search_results.get("data", [])[0].get("results", [])]
        urls = [url for url in urls if url]
        
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
        
        # Configure semaphore for concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        all_extracted_lists = []
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
                        all_extracted_lists.extend(extracted_lists)
                        url_to_lists[url] = [list(l["items"]) for l in extracted_lists]
                        
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
                    all_extracted_lists.extend(extracted_lists)
                    url_to_lists[url] = [list(l["items"]) for l in extracted_lists]
                    
                logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
        
        # Filter and validate lists
        if not all_extracted_lists:
            logger.warning(f"No lists extracted for '{level0_term}'")
            return {
                        "level0_term": level0_term,
                "departments": [],
                        "count": 0,
                        "url_sources": {},
                "quality_scores": {},
                        "verified": False,
                        "num_urls": len(urls),
                        "num_lists": 0
                    }
                
        logger.info(f"Extracted a total of {len(all_extracted_lists)} lists for '{level0_term}'")
        
        # Filter lists using the shared filtering utility
        filtered_lists = await filter_lists(all_extracted_lists, level0_term, filter_config, logger)
        
        if not filtered_lists:
            logger.warning(f"No lists passed filtering for '{level0_term}'")
            return {
                        "level0_term": level0_term,
                "departments": [],
                        "count": 0,
                        "url_sources": {},
                "quality_scores": {},
                        "verified": False,
                        "num_urls": len(urls),
                        "num_lists": len(all_extracted_lists)
                    }
            
        logger.info(f"After filtering, {len(filtered_lists)} lists remain for '{level0_term}'")
        
        # Consolidate departments using the shared utility
        departments = consolidate_lists(
            filtered_lists, 
            level0_term, 
            min_frequency=1,
            min_list_appearances=1,
            similarity_threshold=0.7
        )
        
        logger.info(f"Found {len(departments)} departments for '{level0_term}'")
        
        # Track URL sources for each department
        department_sources = {}
        department_quality = {}
        
        # Map departments to URLs they came from
        for url, lists in url_to_lists.items():
            for dept_list in lists:
                for dept in dept_list:
                    dept_lower = dept.lower()
                    # Find matching department in our final list
                    for final_dept in departments:
                        final_dept_lower = final_dept.lower()
                        if dept_lower == final_dept_lower or dept_lower in final_dept_lower or final_dept_lower in dept_lower:
                            if final_dept not in department_sources:
                                department_sources[final_dept] = []
                            if url not in department_sources[final_dept]:
                                department_sources[final_dept].append(url)
        
        # Set quality score for each department
        for dept in departments:
            # Basic quality score based on number of sources
            sources = department_sources.get(dept, [])
            department_quality[dept] = min(1.0, len(sources) / 3)  # Scale up to 3 sources = 1.0
        
        # Save extracted lists data for analysis
        try:
            extracted_lists_file = os.path.join(Config.RAW_SEARCH_DIR, f"{level0_term}_extracted_lists.json")
            with open(extracted_lists_file, "w", encoding="utf-8") as f:
                # Calculate the number of verified lists per URL
                verified_lists_per_url = {}
                verified_lists_ids = {}
                for url, lists in url_to_lists.items():
                    # Count how many lists contain at least one department from the final list
                    verified_list_count = 0
                    verified_ids = []
                    
                    # For each list from this URL
                    for i, dept_list in enumerate(lists):
                        # Check if any department in this list made it to the final departments list
                        for dept in dept_list:
                            dept_lower = dept.lower()
                            if any(dept_lower == final_dept.lower() or 
                                   dept_lower in final_dept.lower() or 
                                   final_dept.lower() in dept_lower 
                                   for final_dept in departments):
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
                    "level0_term": level0_term,
                    "url_verification_status": url_verification_status,
                    "verified_lists_per_url": verified_lists_per_url,
                    "total_lists_per_url": total_lists_per_url,
                    "verified_lists_ids": verified_lists_ids,
                        "num_urls": len(urls),
                        "num_lists": len(all_extracted_lists),
                        "url_to_lists": url_to_lists,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            logger.debug(f"Saved extracted lists for '{level0_term}' to {extracted_lists_file}")
        except Exception as e:
            logger.warning(f"Failed to save extracted lists for '{level0_term}': {str(e)}")
    
        return {
            "level0_term": level0_term,
            "departments": departments,
            "count": len(departments),
                "url_sources": department_sources,
                "quality_scores": department_quality,
                "verified": len(departments) > 0,
                "num_urls": len(urls),
                "num_lists": len(all_extracted_lists),
                "verified_lists_per_url": verified_lists_per_url if 'verified_lists_per_url' in locals() else {},
                "total_lists_per_url": total_lists_per_url if 'total_lists_per_url' in locals() else {},
                "verified_lists_ids": verified_lists_ids if 'verified_lists_ids' in locals() else {}
            }
            
    except Exception as e:
        logger.error(f"Error processing term '{level0_term}': {str(e)}", exc_info=True)
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


async def process_level0_terms_batch(batch: List[str], provider: Optional[str] = None, session: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Process a batch of level 0 terms"""
    if session:
        # If session is provided, use it for each term
        tasks = [process_level0_term(term, provider, session) for term in batch]
    else:
        # For backward compatibility
        tasks = [process_level0_term(term, provider) for term in batch]
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
    logger.info(f"LV0_INPUT_FILE: {Config.LV0_INPUT_FILE}")
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
        
        logger.info("Starting department names extraction using level 0 terms")
        
        # Create output directories
        ensure_dirs_exist()
        
        # Read level 0 terms
        level0_terms = read_level0_terms(Config.LV0_INPUT_FILE)
        
        logger.info(f"Processing {len(level0_terms)} level 0 terms with batch size {batch_size} and max {max_concurrent} concurrent terms")
        
        # Initialize aiohttp session for the entire run
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, CookieJar
        import ssl, certifi
        
        # Create a default SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT  # Allow legacy renegotiation
        
        # Configure connection parameters
        connector = TCPConnector(ssl=ssl_context, limit=max_concurrent * 2, limit_per_host=2)
        cookie_jar = CookieJar(unsafe=True)  # Allow unsafe cookies to be more permissive
        
        # Create a semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_term_with_throttling(term):
            """Process a term with throttling to limit concurrent execution"""
            async with semaphore:
                return await process_level0_term(term, provider, session)
        
        all_results = []
        start_time = time.time()
        
        # Process in batches
        timeout = ClientTimeout(total=3600)  # 1 hour timeout
        async with ClientSession(connector=connector, cookie_jar=cookie_jar, timeout=timeout) as session:
            for i in range(0, len(level0_terms), batch_size):
                batch = level0_terms[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(level0_terms) + batch_size - 1)//batch_size}")
                
                # Process terms in batch concurrently with throttling
                tasks = [process_term_with_throttling(term) for term in batch]
                batch_results = await asyncio.gather(*tasks)
                
                all_results.extend(batch_results)
                
                # Log progress after each batch
                elapsed = time.time() - start_time
                terms_processed = min(i + batch_size, len(level0_terms))
                terms_per_second = terms_processed / max(1, elapsed)
                eta_seconds = (len(level0_terms) - terms_processed) / max(0.1, terms_per_second)
                
                logger.info(f"Processed {terms_processed}/{len(level0_terms)} terms "
                            f"({terms_processed/len(level0_terms)*100:.1f}%) in {elapsed:.1f}s "
                            f"({terms_per_second:.2f} terms/s, ETA: {eta_seconds/60:.1f}m)")
                
                # Add a small delay between batches to avoid overloading
                await asyncio.sleep(1)
        
        # Collect all departments
        all_departments = []
        department_sources = {}  # Level0 term sources
        department_url_sources = {}  # URL sources
        department_quality_scores = {}  # Quality scores
        level0_to_departments = {}  # Level0 term to departments mapping
        verified_terms_count = 0  # Count verified level0 terms
        
        # Collect statistics for each level0 term and across all terms
        total_urls_processed = 0
        total_lists_found = 0
        verified_departments_count = 0
        urls_per_term = {}
        lists_per_term = {}
        verified_lists_per_term = {}
        verified_lists_per_url_all = {}
        total_lists_per_url_all = {}
        verified_lists_ids_all = {}
        url_verification_status_all = {}
        
        for result in all_results:
            level0_term = result["level0_term"]
            departments = result["departments"]
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
            urls_per_term[level0_term] = num_urls
            lists_per_term[level0_term] = num_lists
            verified_lists_per_term[level0_term] = verified_list_count
            
            # Store URL-specific data in global collections
            verified_lists_per_url_all.update(verified_lists_per_url)
            total_lists_per_url_all.update(total_lists_per_url)
            verified_lists_ids_all[level0_term] = verified_lists_ids
            
            # Create URL verification status dictionary for all terms
            for url, verified_ids in verified_lists_ids.items():
                if url not in url_verification_status_all:
                    url_verification_status_all[url] = {}
                url_verification_status_all[url][level0_term] = len(verified_ids) > 0
            
            # Update totals
            total_urls_processed += num_urls
            total_lists_found += num_lists
            
            # Only include verified departments in the mapping
            if verified:
                verified_terms_count += 1
                verified_departments_count += len(departments)
                
                # Initialize level0_to_departments entry if it doesn't exist
                if level0_term not in level0_to_departments:
                    level0_to_departments[level0_term] = []
                
                # Add departments to level0_to_departments mapping
                level0_to_departments[level0_term].extend(departments)
                
                # Add departments to the global collection
                all_departments.extend(departments)
                
                # Track sources and quality scores
                for dept in departments:
                    # Track level0 term sources
                    if dept not in department_sources:
                        department_sources[dept] = []
                    department_sources[dept].append(level0_term)
                    
                    # Track URL sources
                    if dept not in department_url_sources:
                        department_url_sources[dept] = []
                    if dept in url_sources:
                        department_url_sources[dept].extend(url_sources[dept])
                    
                    # Track quality scores
                    if dept in quality_scores:
                        if dept not in department_quality_scores:
                            department_quality_scores[dept] = quality_scores[dept]
                        else:
                            department_quality_scores[dept] = max(
                                department_quality_scores[dept],
                                quality_scores[dept]
                            )
        
        logger.info(f"Found {verified_terms_count} level 0 terms with verified departments")
        
        # Remove duplicates while preserving case
        unique_departments = []
        seen = set()
        for dept in all_departments:
            dept_lower = dept.lower()
            if dept_lower not in seen:
                seen.add(dept_lower)
                unique_departments.append(dept)
        
        # Also deduplicate level0_to_departments mapping while preserving case
        for level0_term, depts in level0_to_departments.items():
            unique_level0_depts = []
            seen_level0_depts = set()
            for dept in depts:
                dept_lower = dept.lower()
                if dept_lower not in seen_level0_depts:
                    seen_level0_depts.add(dept_lower)
                    unique_level0_depts.append(dept)
            level0_to_departments[level0_term] = unique_level0_depts
        
        # Randomize order
        random.shuffle(unique_departments)
        
        # Save to output file
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for dept in unique_departments:
                f.write(f"{dept}\n")
        
        # Save metadata
        metadata = {
            "execution_time": f"{time.time() - start_time:.2f} seconds",
            "total_departments": len(unique_departments),
            "level0_terms": len(level0_terms),
            "verified_level0_terms": verified_terms_count,
            "department_counts_by_level0": {
                result["level0_term"]: result["count"] 
                for result in all_results 
                if result.get("verified", False)  # Only include verified terms
            },
            "level0_to_departments": level0_to_departments,  # Only contains verified entries now
            "department_sources": {dept: sources for dept, sources in department_sources.items()},
            "department_url_sources": {dept: list(set(urls)) for dept, urls in department_url_sources.items() if urls},
            "department_quality_scores": department_quality_scores,
            "provider": provider or "gemini",
            "max_concurrent": max_concurrent,
            "batch_size": batch_size,
            # Add statistics tracking
            "urls_per_term": urls_per_term,
            "lists_per_term": lists_per_term,
            "total_urls_processed": total_urls_processed,
            "total_lists_found": total_lists_found,
            "verified_departments_count": verified_departments_count,
            "processing_stats": {
                "avg_urls_per_term": total_urls_processed / max(1, len(level0_terms)),
                "avg_lists_per_term": total_lists_found / max(1, len(level0_terms)),
                "avg_departments_per_term": verified_departments_count / max(1, verified_terms_count)
            },
            "verified_lists_per_term": verified_lists_per_term,
            "verified_lists_per_url_all": verified_lists_per_url_all,
            "total_lists_per_url_all": total_lists_per_url_all,
            "verified_lists_ids_all": verified_lists_ids_all,
            "url_verification_status_all": url_verification_status_all
        }
        
        with open(Config.META_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully extracted {len(unique_departments)} unique departments from {verified_terms_count} verified level 0 terms")
        logger.info(f"Department names saved to {Config.OUTPUT_FILE}")
        logger.info(f"Metadata saved to {Config.META_FILE}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Department names extraction completed")


def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
