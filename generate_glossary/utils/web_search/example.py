import os
import sys
import asyncio
import json
import logging
import aiohttp
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import our utility modules
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists
from generate_glossary.utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Configure logger
logger = setup_logger("web_search_example")

# Common constants for both levels
MAX_CONCURRENT_REQUESTS = 5
BATCH_SIZE = 10

# Level-specific keywords
DEPARTMENT_KEYWORDS = [
    "department", "school", "college", "faculty", "division", "program", 
    "institute", "center", "studies", "sciences", "arts", "research",
    "engineering", "education", "technology", "humanities", "social"
]

RESEARCH_KEYWORDS = [
    "research", "study", "analysis", "theory", "methodology", "approach", 
    "experiment", "investigation", "application", "development", "model",
    "framework", "technology", "technique", "method", "assessment"
]

# Anti-keywords for both levels
NON_RELEVANT_KEYWORDS = [
    "login", "sign in", "register", "apply", "admission", "contact", 
    "about", "home", "sitemap", "search", "privacy", "terms", "copyright",
    "accessibility", "careers", "jobs", "employment", "staff", "faculty"
]

# Level-specific patterns
DEPARTMENT_PATTERNS = [
    r"(?:department|school|college|faculty|division|program|institute) of [\w\s&,'-]+",
    r"[\w\s&,'-]+ (?:department|school|college|faculty|division|program|institute)",
    r"[\w\s&,'-]+ studies",
    r"[\w\s&,'-]+ sciences"
]

RESEARCH_PATTERNS = [
    r"(?:research|study|analysis|investigation) (?:of|on|in) [\w\s&,'-]+",
    r"[\w\s&,'-]+ (?:theory|methodology|approach|framework)",
    r"[\w\s&,'-]+ (?:analysis|modeling|simulation)",
    r"[\w\s&,'-]+ (?:development|implementation|application)"
]

def setup_directories():
    """Ensure all required directories exist"""
    base_dir = Path(__file__).parent.parent.parent.parent
    
    # Create directories
    dirs = [
        base_dir / "data" / "example" / "cache",
        base_dir / "data" / "example" / "raw_search_results"
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        
    return base_dir

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

def score_department_list(items: List[str], metadata: Dict[str, Any], context_term: str) -> float:
    """Score a department list"""
    # Custom department scoring logic
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

def score_research_area_list(items: List[str], metadata: Dict[str, Any], context_term: str) -> float:
    """Score a research area list"""
    # Custom research area scoring logic
    weights = {
        "keyword": 0.25,
        "structure": 0.15,
        "pattern": 0.2,
        "non_term": 0.15,
        "consistency": 0.1,
        "size": 0.05,
        "html_type": 0.1
    }
    
    # Use the common scoring function from list_extractor
    return score_list(
        items=items,
        metadata=metadata,
        context_term=context_term,
        keywords=RESEARCH_KEYWORDS,
        scoring_weights=weights
    )

async def process_level0_term(level0_term: str, base_dir: Path):
    """Process a level 0 term to extract departments (level 1)"""
    logger.info(f"Processing level 0 term: {level0_term}")
    
    # 1. Configure the components
    search_config = WebSearchConfig(
        base_dir=str(base_dir),
        raw_search_dir=str(base_dir / "data" / "example" / "raw_search_results")
    )
    
    html_config = HTMLFetchConfig(
        cache_dir=str(base_dir / "data" / "example" / "cache")
    )
    
    list_config = ListExtractionConfig(
        keywords=DEPARTMENT_KEYWORDS,
        anti_keywords=NON_RELEVANT_KEYWORDS,
        patterns=DEPARTMENT_PATTERNS
    )
    
    filter_config = FilterConfig(
        scoring_fn=score_department_list,
        clean_item_fn=clean_department_name,
        binary_system_prompt="""You are an expert in academic department classification.

Your task is to evaluate whether a provided list truly contains valid academic department names.

You must return a clear YES or NO decision for each list.
- Answer YES if the list primarily contains actual department names
- Answer NO if the list contains menu items, navigation links, non-academic content, etc."""
    )
    
    # 2. Construct search query for level 0 term
    query = f"site:.edu college of {level0_term} list of departments"
    
    # 3. Perform web search
    search_results = web_search_bulk([query], search_config, logger=logger)
    
    if not search_results or not search_results.get("data"):
        logger.warning(f"No search results for '{level0_term}'")
        return []
    
    # 4. Extract search result URLs
    try:
        urls = [r.get("url") for r in search_results.get("data", [])[0].get("results", [])]
        urls = [url for url in urls if url]
        
        if not urls:
            logger.warning(f"No URLs found in search results for '{level0_term}'")
            return []
            
        logger.info(f"Found {len(urls)} URLs for '{level0_term}'")
    except Exception as e:
        logger.error(f"Error extracting URLs: {str(e)}")
        return []
    
    # 5. Fetch and process webpages
    try:
        # Configure semaphore for concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Configure aiohttp session
        timeout = aiohttp.ClientTimeout(
            connect=15,
            sock_read=40,
            total=80
        )
        
        all_extracted_lists = []
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Fetch webpages
            fetch_tasks = [
                fetch_webpage(url, session, semaphore, html_config, level0_term, logger) 
                for url in urls[:10]  # Limit to 10 URLs for this example
            ]
            html_contents = await asyncio.gather(*fetch_tasks)
            
            # Process each webpage
            for url, html_content in zip(urls[:10], html_contents):
                if not html_content:
                    continue
                    
                # Extract lists from the webpage
                extracted_lists = extract_lists_from_html(html_content, list_config)
                all_extracted_lists.extend(extracted_lists)
                
                logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
        
        # 6. Filter and validate lists
        if not all_extracted_lists:
            logger.warning(f"No lists extracted for '{level0_term}'")
            return []
            
        logger.info(f"Extracted a total of {len(all_extracted_lists)} lists for '{level0_term}'")
        
        # Filter lists
        filtered_lists = await filter_lists(all_extracted_lists, level0_term, filter_config, logger)
        
        if not filtered_lists:
            logger.warning(f"No lists passed filtering for '{level0_term}'")
            return []
            
        logger.info(f"After filtering, {len(filtered_lists)} lists remain for '{level0_term}'")
        
        # 7. Consolidate lists to get final departments
        departments = consolidate_lists(
            filtered_lists, 
            level0_term, 
            min_frequency=1,
            min_list_appearances=1
        )
        
        logger.info(f"Found {len(departments)} departments for '{level0_term}'")
        
        return departments
        
    except Exception as e:
        logger.error(f"Error processing term '{level0_term}': {str(e)}")
        return []

async def process_level1_term(level1_term: str, base_dir: Path):
    """Process a level 1 term to extract research areas (level 2)"""
    logger.info(f"Processing level 1 term: {level1_term}")
    
    # 1. Configure the components
    search_config = WebSearchConfig(
        base_dir=str(base_dir),
        raw_search_dir=str(base_dir / "data" / "example" / "raw_search_results")
    )
    
    html_config = HTMLFetchConfig(
        cache_dir=str(base_dir / "data" / "example" / "cache")
    )
    
    list_config = ListExtractionConfig(
        keywords=RESEARCH_KEYWORDS,
        anti_keywords=NON_RELEVANT_KEYWORDS,
        patterns=RESEARCH_PATTERNS
    )
    
    filter_config = FilterConfig(
        scoring_fn=score_research_area_list,
        clean_item_fn=clean_research_area,
        binary_system_prompt="""You are an expert in academic research classification.

Your task is to evaluate whether a provided list truly contains valid research areas or topics.

You must return a clear YES or NO decision for each list.
- Answer YES if the list primarily contains actual research areas
- Answer NO if the list contains menu items, website sections, non-academic content, etc."""
    )
    
    # 2. Construct search query for level 1 term
    query = f"site:.edu department of {level1_term} (research areas | teaching course)"
    
    # 3. Perform web search
    search_results = web_search_bulk([query], search_config, logger=logger)
    
    if not search_results or not search_results.get("data"):
        logger.warning(f"No search results for '{level1_term}'")
        return []
    
    # 4. Extract search result URLs
    try:
        urls = [r.get("url") for r in search_results.get("data", [])[0].get("results", [])]
        urls = [url for url in urls if url]
        
        if not urls:
            logger.warning(f"No URLs found in search results for '{level1_term}'")
            return []
            
        logger.info(f"Found {len(urls)} URLs for '{level1_term}'")
    except Exception as e:
        logger.error(f"Error extracting URLs: {str(e)}")
        return []
    
    # 5. Fetch and process webpages
    try:
        # Configure semaphore for concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Configure aiohttp session
        timeout = aiohttp.ClientTimeout(
            connect=15,
            sock_read=40,
            total=80
        )
        
        all_extracted_lists = []
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Fetch webpages
            fetch_tasks = [
                fetch_webpage(url, session, semaphore, html_config, level1_term, logger) 
                for url in urls[:10]  # Limit to 10 URLs for this example
            ]
            html_contents = await asyncio.gather(*fetch_tasks)
            
            # Process each webpage
            for url, html_content in zip(urls[:10], html_contents):
                if not html_content:
                    continue
                    
                # Extract lists from the webpage
                extracted_lists = extract_lists_from_html(html_content, list_config)
                all_extracted_lists.extend(extracted_lists)
                
                logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")
        
        # 6. Filter and validate lists
        if not all_extracted_lists:
            logger.warning(f"No lists extracted for '{level1_term}'")
            return []
            
        logger.info(f"Extracted a total of {len(all_extracted_lists)} lists for '{level1_term}'")
        
        # Filter lists
        filtered_lists = await filter_lists(all_extracted_lists, level1_term, filter_config, logger)
        
        if not filtered_lists:
            logger.warning(f"No lists passed filtering for '{level1_term}'")
            return []
            
        logger.info(f"After filtering, {len(filtered_lists)} lists remain for '{level1_term}'")
        
        # 7. Consolidate lists to get final research areas
        research_areas = consolidate_lists(
            filtered_lists, 
            level1_term, 
            min_frequency=1,
            min_list_appearances=1
        )
        
        logger.info(f"Found {len(research_areas)} research areas for '{level1_term}'")
        
        return research_areas
        
    except Exception as e:
        logger.error(f"Error processing term '{level1_term}': {str(e)}")
        return []

async def main():
    """Main execution function"""
    # Setup directories
    base_dir = setup_directories()
    
    # Example level 0 term (to get departments)
    level0_term = "biology"
    departments = await process_level0_term(level0_term, base_dir)
    
    # Save departments to file
    if departments:
        output_path = base_dir / "data" / "example" / f"{level0_term}_departments.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "level0_term": level0_term,
                "departments": departments
            }, f, indent=2)
        logger.info(f"Saved {len(departments)} departments to {output_path}")
    
    # Example level 1 term (to get research areas)
    level1_term = "computer science"
    research_areas = await process_level1_term(level1_term, base_dir)
    
    # Save research areas to file
    if research_areas:
        output_path = base_dir / "data" / "example" / f"{level1_term}_research_areas.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "level1_term": level1_term,
                "research_areas": research_areas
            }, f, indent=2)
        logger.info(f"Saved {len(research_areas)} research areas to {output_path}")

if __name__ == "__main__":
    asyncio.run(main()) 