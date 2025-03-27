import os
import requests
import time
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union

# Constants
MAX_SEARCH_RESULTS = 40
MAX_RETRIES = 4
RATE_LIMIT_DELAY = 1  # seconds

class WebSearchConfig:
    """Configuration for web search"""
    def __init__(self, 
                 rapidapi_key: Optional[str] = None, 
                 rapidapi_host: Optional[str] = None,
                 base_dir: Optional[str] = None,
                 raw_search_dir: Optional[str] = None):
        self.rapidapi_key = rapidapi_key or os.getenv("RAPIDAPI_KEY")
        self.rapidapi_host = rapidapi_host or os.getenv("RAPIDAPI_HOST")
        self.base_dir = base_dir
        self.raw_search_dir = raw_search_dir
        
        # API configuration
        self.api_headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": self.rapidapi_host,
        }
        self.web_search_url = "https://real-time-web-search.p.rapidapi.com/search"
    
    def get_raw_search_file_path(self, term: str) -> str:
        """Get file path for raw search results for a term"""
        if not self.raw_search_dir:
            raise ValueError("raw_search_dir not set in WebSearchConfig")
            
        # Sanitize the term for use in a filename
        safe_term = re.sub(r'[^\w\-\.]', '_', term)
        return os.path.join(self.raw_search_dir, f"{safe_term}_search_results.json")


def web_search_bulk(queries: List[str], 
                   config: WebSearchConfig,
                   limit: int = MAX_SEARCH_RESULTS,
                   logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Perform bulk web search for multiple queries using RapidAPI
    
    Args:
        queries: List of search queries
        config: WebSearchConfig object with API settings
        limit: Maximum number of results per query
        logger: Optional logger to use (if None, no logging)
        
    Returns:
        Dictionary containing search results
    """
    payload = {
        "queries": queries,
        "limit": str(limit)
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            if logger:
                logger.info(f"Performing bulk search for {len(queries)} queries (attempt {attempt+1})")
            
            response = requests.post(config.web_search_url, json=payload, headers=config.api_headers)
            response.raise_for_status()
            results = response.json()
            
            if not results.get("data"):
                if logger:
                    logger.warning("No data in response")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_DELAY)
                    continue
                return {"data": []}
                
            # Verify each query has results
            for item in results["data"]:
                if not item.get("results") and logger:
                    logger.warning(f"No results for query: {item.get('query', 'unknown')}")
            
            # Extract and save term-specific search results if applicable
            if len(queries) == 1 and queries[0]:
                save_term_search_results(queries[0], results, config, logger)
            
            return results
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to perform bulk web search (attempt {attempt+1}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RATE_LIMIT_DELAY * (attempt + 1))
            else:
                if logger:
                    logger.error(f"All {MAX_RETRIES} attempts failed for queries: {queries}")
                return {"data": []}


def save_term_search_results(query: str, 
                            results: Dict[str, Any], 
                            config: WebSearchConfig,
                            logger: Optional[logging.Logger] = None) -> None:
    """
    Extract term from query and save raw search results
    
    Args:
        query: The search query with embedded term
        results: The search results to save
        config: Configuration object
        logger: Optional logger
    """
    # First try pattern for level 1 (department names)
    match = re.search(r"college of (.*?) list", query)
    term_type = "level0"
    
    if not match:
        # Try pattern for level 2 (research areas)
        match = re.search(r"department of (.*?) \(", query)
        term_type = "level1"
    
    if not match:
        # Generic fallback to extract any term
        match = re.search(r'site:.edu (?:.*?)([a-zA-Z ]+)(?:.*?)(?:list|research|course)', query)
        term_type = "unknown"
    
    if match:
        term = match.group(1).strip()
        try:
            raw_results_file = config.get_raw_search_file_path(term)
            with open(raw_results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            if logger:
                logger.debug(f"Saved raw search results for '{term}' ({term_type}) to {raw_results_file}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to save raw search results for '{term}': {str(e)}") 