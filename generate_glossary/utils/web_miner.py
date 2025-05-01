"""Unified web content mining module with support for general web and Wikipedia content."""

import os
import json
import asyncio
import aiohttp
import argparse
import subprocess
import time  # Add time module import
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from urllib.parse import urlparse, urlunparse
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import trafilatura
from asyncio import Semaphore
from tqdm import tqdm
from aiohttp import ClientTimeout, ClientError
from .llm import (
    LLMFactory,
    InferenceResult,
    BaseLLM,
    Provider,
    OPENAI_MODELS,
    GEMINI_MODELS,
)
from .verification_utils import (
    get_educational_score_async,
    verify_content_async,
    verify_batch_content_async,
    is_wikipedia_url,
    convert_numpy_types
)
import logging
from dotenv import load_dotenv
from generate_glossary.utils.web_scraper import fetch_page, parse_html
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
# Simplify logging format to just the message
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter('%(message)s'))

# Constants
MAX_CONCURRENT_REQUESTS = 30  # Increased from 20 for better parallelism
RETRY_ATTEMPTS = 3  # Number of retry attempts for API calls
RATE_LIMIT_DELAY = 0.5  # Reduced from 1 second to 0.5 seconds
MAX_CONTENT_LENGTH = 8000  # Maximum content length to process
BATCH_SIZE = 100  # Default batch size for processing
CONNECT_TIMEOUT = 10  # Connection timeout in seconds
READ_TIMEOUT = 30  # Socket read timeout in seconds
TOTAL_TIMEOUT = 60  # Total operation timeout in seconds
MAX_RESULTS_PER_QUERY = 5  # Maximum results per query
MAX_WORKERS = max(4, cpu_count() - 1)  # Use most cores but leave one for system
MAX_EXTRACTION_BATCH = 15  # Increased from 10 to 15
MAX_CONCURRENT_PROCESSING = 10  # Doubled from 5 to 10

# API Configuration
RAPIDAPI_HEADERS = {
    "content-type": "application/json",
    "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
    "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com",
}

WEB_SEARCH_URL = "https://real-time-web-search.p.rapidapi.com/search"

# Common headers for web requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Skip domains that are likely to be irrelevant or inaccessible
SKIP_DOMAINS = {
    "latimes.com",
    "nytimes.com",
    "wsj.com",
    "bloomberg.com",
    "facebook.com",
    "twitter.com",
    "linkedin.com",
    "youtube.com",
    "instagram.com",
    "google.com",
}

# Prioritized domains
PRIORITY_DOMAINS = {
    "wikipedia.org": 1.0,  # Highest priority
    "arxiv.org": 0.9,
    "github.com": 0.8,
    "stackoverflow.com": 0.7,
    "docs.python.org": 0.7,
    "developer.mozilla.org": 0.7,
}

class WebContent(BaseModel):
    """Model for storing web content information"""
    url: str
    title: str = Field(default="")
    snippet: str = Field(default="")
    raw_content: str = Field(default="")
    processed_content: str = Field(default="")
    score: float = Field(default=0.5)  # Default score for unknown domains
    is_verified: bool = False
    verification_reason: str = ""
    educational_score: Optional[float] = None  # Added field for final verification score
    query: Optional[str] = None  # Store the query that found this URL

class SearchSettings(BaseModel):
    """Settings for web content search and verification"""
    min_score: float = Field(default=0.7)
    max_concurrent_requests: int = Field(default=MAX_CONCURRENT_REQUESTS)
    batch_size: int = Field(default=BATCH_SIZE)  # Use the constant instead of hardcoded value
    provider: Optional[str] = Field(default=None)
    system_prompt: Optional[str] = Field(
        default="""You are an expert in extractive summarization for academic and technical concepts.
Your task is to identify and extract ONLY the most relevant sentences from the source text that directly explain the concept.
DO NOT paraphrase, rewrite, or generate new content.
DO NOT add your own explanations or interpretations.
ONLY extract 1-3 of the most relevant sentences from the original text that best define or explain the concept.
If no relevant sentences exist in the text, respond with "No relevant information found."
"""
    )
    # Note: We always generate both regular and Wikipedia-focused queries for each term
    use_rapidapi: bool = Field(default=True)  # Whether to use RapidAPI search
    show_progress: bool = Field(default=False)  # Whether to show progress bars
    skip_verification: bool = Field(default=False)  # Whether to skip content verification step
    max_workers: int = Field(default=MAX_WORKERS)  # Use the constant instead of repeating the formula
    use_cache: bool = Field(default=True)  # Whether to cache LLM results
    use_batch_llm: bool = Field(default=True)  # Whether to use batch processing for LLM requests
    log_level: str = Field(default="ERROR")  # Logging level to use
    safe_mode: bool = Field(default=True)  # Whether to use safer thread limits
    parallel_extraction: bool = Field(default=True)  # Whether to extract content in parallel
    skip_low_quality: bool = Field(default=True)  # Skip low quality content early
    content_threshold: float = Field(default=1.3)  # Changed from 1.5 to 1.3
    # Add context CSV arguments
    context_csv_file: Optional[str] = Field(default=None)
    concept_column: Optional[str] = Field(default="concept")
    context_column: Optional[str] = Field(default=None)
    # Add skip summarization flag
    skip_summarization: bool = Field(default=False)  # Whether to skip LLM summarization
    
    def __init__(self, **data):
        super().__init__(**data)
        self._configure_logging()
        
    def _configure_logging(self):
        """Configure logging based on the settings"""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        # Get numeric log level with fallback to ERROR
        numeric_level = levels.get(self.log_level.upper(), logging.ERROR)
        
        # Configure root logger if not already configured
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=numeric_level,
                format='%(message)s'  # Simplified format without timestamps and levels
            )
        else:
            # Otherwise just set the level
            root_logger.setLevel(numeric_level)
            
        # Set logging level for our logger
        logger.setLevel(numeric_level)
        
        # Set levels for other modules
        for module in ["aiohttp", "urllib3", "asyncio", "trafilatura", "transformers"]:
            logging.getLogger(module).setLevel(logging.ERROR)
            
        # Set level for verification utils
        from .verification_utils import logger as verification_logger
        verification_logger.setLevel(numeric_level)

def get_domain_priority(url: str) -> float:
    """Get priority score for a domain."""
    domain = urlparse(url).netloc.lower()
    # Check each priority domain
    for priority_domain, score in PRIORITY_DOMAINS.items():
        if priority_domain in domain:
            return score
    return 0.5  # Default priority for unknown domains

def get_text_variations(text: str) -> Set[str]:
    """Get singular/plural variations of text"""
    variations = {text}

    # Handle common plural endings
    if text.endswith("ies"):
        variations.add(text[:-3] + "y")  # laboratories -> laboratory
    elif text.endswith("s"):
        variations.add(text[:-1])  # journals -> journal
    else:
        variations.add(text + "s")  # journal -> journals

    return variations

def standardize_url(url: str) -> str:
    """Standardize URL format by removing fragments and query parameters"""
    parsed_url = urlparse(url)
    path = parsed_url.path.split("#")[0]  # Remove fragment
    return urlunparse((parsed_url.scheme, parsed_url.netloc, path, "", "", ""))

def should_skip_domain(url: str) -> bool:
    """Check if a domain should be skipped"""
    domain = urlparse(url).netloc.lower()
    return any(skip_domain in domain for skip_domain in SKIP_DOMAINS)

def is_likely_binary_file(url: str) -> bool:
    """Check if URL likely points to a binary file that can't be processed as HTML"""
    # Check file extension
    path = urlparse(url).path.lower()
    binary_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
                         '.zip', '.rar', '.gz', '.tar', '.7z', '.exe', '.bin', '.dat']
    return any(path.endswith(ext) for ext in binary_extensions)

def is_wikipedia_url(url: str) -> bool:
    """Check if a URL is from Wikipedia"""
    domain = urlparse(url).netloc.lower()
    return 'wikipedia.org' in domain

def standardize_wikipedia_url(url: str) -> str:
    """Standardize Wikipedia URL format"""
    parsed_url = urlparse(url)
    path = parsed_url.path.split("#")[0].split(":")[-1]
    return urlunparse(("https", "en.wikipedia.org", path, "", "", ""))

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def search_rapidapi_batch(
    queries: List[str],
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    show_progress: bool = False
) -> Dict[str, Any]:
    """Search using RapidAPI with rate limiting and retries"""
    assert len(queries) <= 100, "Maximum 100 queries per batch allowed"

    # Format queries according to RapidAPI docs
    payload = {
        "queries": queries,
        "limit": str(MAX_RESULTS_PER_QUERY * 2)  # Request more to ensure we get enough after filtering
    }

    async with semaphore:  # Control concurrent requests
        try:
            async with session.post(
                WEB_SEARCH_URL, json=payload, headers=RAPIDAPI_HEADERS
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"RapidAPI error response: {error_text}")
                    response.raise_for_status()
                await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                return await response.json()
        except Exception as e:
            logger.error(f"Error during RapidAPI batch search: {e}")
            raise  # Let retry handle the error

async def process_rapidapi_results(
    search_results: Dict[str, Any],
    show_progress: bool = False
) -> List[Dict[str, str]]:
    """Process RapidAPI search results into a standardized format"""
    results = []
    
    if not search_results:
        logger.warning("Empty search results received from RapidAPI")
        return results
    
    # Extract results from the response
    data = search_results.get("data", [])
    if not data:
        logger.warning("No data found in RapidAPI response")
        return results
    
    # Create tasks for processing all query results in parallel
    all_tasks = []
    query_result_map = {}  # Map to track which query each task belongs to
    
    for query_data in data:
        query = query_data.get("query", "")
        query_results = query_data.get("results", [])
        
        # Create tasks for processing each result
        for item in query_results:
            task = asyncio.create_task(process_result(item, query))
            all_tasks.append(task)
            query_result_map[task] = query
    
    # Process all results in parallel
    processed_results_by_query = {}
    
    if show_progress:
        # Fixed implementation for progress tracking
        pending = set(all_tasks)
        with tqdm(total=len(all_tasks), desc="Processing results") as pbar:
            while pending:
                # Use asyncio.wait with timeout to avoid blocking
                done, pending = await asyncio.wait(
                    pending, 
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.1
                )
                
                for task in done:
                    result = task.result()
                    if result:
                        query = query_result_map[task]
                        if query not in processed_results_by_query:
                            processed_results_by_query[query] = []
                        processed_results_by_query[query].append(result)
                    pbar.update(1)
    else:
        # Process all tasks and organize by query
        processed_results = await asyncio.gather(*all_tasks)
        
        for task, result in zip(all_tasks, processed_results):
            if result:
                query = query_result_map[task]
                if query not in processed_results_by_query:
                    processed_results_by_query[query] = []
                processed_results_by_query[query].append(result)
    
    # Process each query's results
    for query, query_results in processed_results_by_query.items():
        # Sort by score (highest first)
        query_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Find Wikipedia results
        wiki_results = [r for r in query_results if is_wikipedia_url(r["url"])]
        other_results = [r for r in query_results if not is_wikipedia_url(r["url"])]
        
        # Combine results ensuring a mix of Wikipedia and non-Wikipedia results
        final_results = []
        
        # Always include at least one Wikipedia result if available
        if wiki_results and len(wiki_results) > 0:
            final_results.append(wiki_results[0])
        
        # Add non-Wikipedia results (at least 2 if available)
        non_wiki_to_add = min(len(other_results), 2)
        if non_wiki_to_add > 0:
            final_results.extend(other_results[:non_wiki_to_add])
        
        # Fill remaining slots with a mix of Wikipedia and other results
        remaining_slots = MAX_RESULTS_PER_QUERY - len(final_results)
        if remaining_slots > 0:
            # Add more Wikipedia results if available
            wiki_idx = 1  # Start from second Wikipedia result
            other_idx = non_wiki_to_add  # Start from after already added non-Wiki results
            
            for _ in range(remaining_slots):
                # Alternate between Wikipedia and other results
                if wiki_idx < len(wiki_results):
                    final_results.append(wiki_results[wiki_idx])
                    wiki_idx += 1
                elif other_idx < len(other_results):
                    final_results.append(other_results[other_idx])
                    other_idx += 1
                else:
                    # No more results to add
                    break
        
        # Log the distribution of results
        wiki_count = sum(1 for r in final_results if is_wikipedia_url(r["url"]))
        other_count = len(final_results) - wiki_count
        logger.info(f"Query '{query}': {wiki_count} Wikipedia results, {other_count} other results")
        
        results.extend(final_results)
    
    logger.info(f"Processed {len(results)} valid results from RapidAPI")
    return results

async def process_result(item: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    """Process a single search result item"""
    if not isinstance(item, dict):
        logger.warning(f"Invalid result format: {type(item)}")
        return None
        
    url = item.get("url", "")
    if not url:
        logger.warning("Result missing URL")
        return None
    
    # Calculate priority score
    priority = get_domain_priority(url)
    position_score = 1.0 - (item.get("position", 0) / 10)  # Position-based score
    final_score = (priority * 0.7) + (position_score * 0.3)  # Weighted combination
    
    return {
        "url": url,
        "title": item.get("title", ""),
        "snippet": item.get("snippet", ""),
        "query": query,
        "score": final_score
    }

def process_content_with_trafilatura(html_content: str) -> Optional[str]:
    """Process HTML content with trafilatura in a separate process"""
    try:
        content = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=False,
            no_fallback=True,
            include_links=False,
            include_images=False,
            include_formatting=False,
            with_metadata=False,
            output_format="txt"
        )
        return content
    except Exception:
        return None

# Global process executor for trafilatura to reuse
trafilatura_executor = None

def initialize_trafilatura_executor(max_workers: int = None):
    """Initialize the global process executor for trafilatura."""
    global trafilatura_executor
    if trafilatura_executor is None:
        try:
            # Use a conservative number of workers - fewer is safer
            if max_workers is None:
                # Default to a very conservative number to avoid resource exhaustion
                max_workers = min(4, max(1, cpu_count() // 4))
                
            # Limit to maximum of 4 workers to avoid OpenMP thread issues
            workers = min(4, max_workers)
            logger.info(f"Initializing trafilatura executor with {workers} workers")
            trafilatura_executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        except Exception as e:
            # Fallback to even fewer workers if we encounter errors
            fallback_workers = 1
            logger.warning(f"Error initializing process pool with {max_workers} workers: {e}")
            logger.warning(f"Falling back to {fallback_workers} worker")
            trafilatura_executor = concurrent.futures.ProcessPoolExecutor(max_workers=fallback_workers)
    return trafilatura_executor

async def extract_web_content(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    show_progress: bool = False,
    max_workers: int = None
) -> Optional[str]:
    """Extract content from a webpage using trafilatura with web_scraper as fallback"""
    global trafilatura_executor
    
    # Skip known binary file types that can't be parsed as HTML
    if is_likely_binary_file(url):
        logger.warning(f"Skipping likely binary file: {url}")
        return None
    
    # Initialize the executor if not already done
    if trafilatura_executor is None:
        try:
            trafilatura_executor = initialize_trafilatura_executor(max_workers)
        except Exception as e:
            logger.error(f"Failed to initialize trafilatura executor: {e}")
            # If we can't initialize the executor, fall back to in-process extraction
            trafilatura_executor = None
    
    async with semaphore:  # Control concurrent requests
        try:
            # Configure timeouts
            timeout = ClientTimeout(
                connect=CONNECT_TIMEOUT,    # Connection timeout
                sock_read=READ_TIMEOUT,     # Socket read timeout
                total=TOTAL_TIMEOUT        # Total operation timeout
            )
            
            # First try with trafilatura
            async with session.get(url, headers=HEADERS, timeout=timeout) as response:
                if response.status != 200:
                    logger.warning(f"HTTP error {response.status} for URL: {url}")
                    return None
                
                # Check content type to avoid binary files
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type and 'application/json' not in content_type:
                    if any(binary_type in content_type for binary_type in ['pdf', 'octet-stream', 'binary', 'excel', 'word', 'powerpoint']):
                        logger.warning(f"Skipping binary content: {url} (Content-Type: {content_type})")
                        return None
                
                try:
                    # Try to decode with UTF-8 first
                    html_content = await response.text(encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        # If UTF-8 fails, try with Latin-1 which accepts any byte value
                        logger.info(f"UTF-8 decoding failed for {url}, trying latin-1")
                        html_content = await response.text(encoding='latin-1')
                    except Exception as e:
                        logger.warning(f"Failed to decode content from {url}: {e}")
                        return None
                
                # Process with trafilatura in a shared process pool to avoid overhead
                # of creating a new pool for each URL
                try:
                    if trafilatura_executor is not None:
                        loop = asyncio.get_event_loop()
                        content = await loop.run_in_executor(
                            trafilatura_executor, 
                            process_content_with_trafilatura, 
                            html_content
                        )
                    else:
                        # Fall back to in-process extraction if executor is None
                        content = process_content_with_trafilatura(html_content)
                except (RuntimeError, OSError) as e:
                    if "Thread" in str(e) and "unavailable" in str(e):
                        # This is likely a thread creation error
                        logger.warning(f"Thread creation error: {e}. Falling back to in-process extraction.")
                        # Clear the executor so it will be recreated with fewer workers
                        trafilatura_executor = None
                        # Fall back to in-process extraction
                        content = process_content_with_trafilatura(html_content)

                # If trafilatura fails, try web_scraper as fallback
                if not content:
                    logger.info(f"Trafilatura failed for {url}, trying web_scraper fallback")
                    parsed_content = parse_html(html_content)
                    if parsed_content:
                        content = parsed_content
                    else:
                        logger.warning(f"Web scraper fallback also failed for {url}")
                        return None

                await asyncio.sleep(RATE_LIMIT_DELAY * 0.1)  # Further reduced delay
                return content

        except asyncio.TimeoutError:
            logger.warning(f"Timeout error while fetching {url} (connect={CONNECT_TIMEOUT}s, read={READ_TIMEOUT}s)")
            return None
        except UnicodeDecodeError as e:
            logger.warning(f"Encoding error for {url}: {str(e)}")
            return None
        except ClientError as e:
            logger.warning(f"Network error for {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            # Try web_scraper as fallback on exception
            try:
                logger.info(f"Trying web_scraper fallback after error for {url}")
                # Use the same timeout configuration for fallback
                async with session.get(url, headers=HEADERS, timeout=timeout) as response:
                    try:
                        html_content = await response.text(encoding='utf-8')
                    except UnicodeDecodeError:
                        # If UTF-8 fails, try with Latin-1
                        logger.info(f"Fallback UTF-8 decoding failed for {url}, trying latin-1")
                        html_content = await response.text(encoding='latin-1')
                    
                    content = parse_html(html_content)
                    if content:
                        return content
            except Exception as e2:
                logger.warning(f"Web scraper fallback also failed with error: {str(e2)}")
            return None

def truncate_content(text: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Truncate text to maximum length while keeping whole sentences."""
    if not text or len(text) <= max_length:
        return text
    
    # Find the last sentence boundary before max_length
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    if last_period > 0:
        return text[:last_period + 1]
    return truncated

async def process_content_with_verification(
    content: WebContent,
    term: str,
    min_score: float,
    skip_verification: bool = False
) -> Dict[str, Any]:
    """Process a single content item with verification.
    
    This function always returns content regardless of score, but marks
    it as verified only if it meets the minimum score threshold.
    
    Args:
        content: WebContent object to process
        term: The term this content belongs to
        min_score: Minimum score threshold for verification
        skip_verification: Whether to skip verification step for speed
        
    Returns:
        Dictionary with processed content and verification info
    """
    try:
        # Truncate content before processing
        truncated_content = truncate_content(content.processed_content)
        
        # Skip verification if requested
        if skip_verification:
            # For websites with high domain priority, consider them verified automatically
            is_verified = True
            reason = "Verification skipped, content accepted"
            score = content.score  # Use initial score
        else:
            # Verify content quality based on processed content
            is_verified, reason, score = (await verify_content_async(
                content.url,
                truncated_content,
                min_score=min_score
            ))
        
        # Include all content with verification info
        content_dict = content.model_dump()
        # Truncate raw and processed content
        content_dict["raw_content"] = truncate_content(content_dict.get("raw_content", ""))
        content_dict["processed_content"] = truncated_content
        content_dict["is_verified"] = bool(is_verified)  # Convert numpy.bool_ to Python bool
        content_dict["verification_reason"] = reason
        content_dict["score"] = float(score)  # Convert numpy.float32 to Python float
        content_dict["term"] = term  # Add the term this content belongs to
        
        return content_dict
    except Exception as e:
        logger.error(f"Error processing content for {content.url}: {e}")
        # Return content with error info
        content_dict = content.model_dump()
        content_dict["is_verified"] = False
        content_dict["verification_reason"] = f"Error during verification: {str(e)}"
        content_dict["score"] = 0.0
        content_dict["term"] = term
        return content_dict

async def process_content_batch(
    contents: List[WebContent],
    term: str,
    min_score: float,
    skip_verification: bool = False
) -> List[Dict[str, Any]]:
    """Process a batch of content items with verification.
    
    Args:
        contents: List of WebContent objects to process
        term: The term these contents belong to
        min_score: Minimum score threshold for verification
        skip_verification: Whether to skip verification
        
    Returns:
        List of dictionaries with processed content and verification info
    """
    tasks = [process_content_with_verification(content, term, min_score, skip_verification) for content in contents]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]  # Filter out failed items

def write_results(results: Dict[str, List[Dict[str, Any]]], output_path: str) -> None:
    """
    Write final results to various output formats.
    
    Args:
        results: Dictionary mapping terms to lists of content items
        output_path: Base path for output files
    """
    # Check if there are any results to write
    if not results:
        logger.warning("No results to write.")
        print("No results to write.")
        return
        
    # Check if any of the results contain content
    has_content = False
    for contents in results.values():
        if contents:
            has_content = True
            break
    
    if not has_content:
        msg = "No content found for any terms, skipping file write"
        logger.warning(msg)
        print(msg)
        # Create empty JSON file anyway to indicate the process ran
        try:
            json_path = f"{output_path}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4)  # Use 4-space indentation
            logger.info(f"Created empty JSON file: {json_path}")
            print(f"✓ Created empty JSON file: {json_path}")
        except Exception as e:
            logger.error(f"Error creating empty JSON file: {e}")
            print(f"ERROR: Failed to create empty JSON file: {e}")
        return
        
    # Build text mappings for basic output
    text_lines = []
    for term, contents in results.items():
        # Skip terms with no content
        if not contents:
            continue
            
        # Get verified content
        verified_content = [c for c in contents if c.get("is_verified", False)]
        
        # If no verified content, use all content
        content_to_use = verified_content if verified_content else contents
        
        # Add term and URLs to text lines
        text_lines.append(f"{term}:")
        for c in content_to_use:
            text_lines.append(f"  {c.get('url', 'No URL')}")
        text_lines.append("")  # Empty line between terms
    
    # Determine output file paths
    text_path = f"{output_path}.txt"
    json_path = f"{output_path}.json"
    summary_path = f"{output_path}_summary.json"
    
    # Write text file with mappings
    try:
        logger.info(f"Writing text mappings to {text_path}")
        print(f"Writing text mappings to {text_path}")
        
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))
        
        logger.info(f"Successfully wrote text mappings to {text_path}")
        print(f"✓ Successfully wrote text mappings to {text_path}")
    except Exception as e:
        logger.error(f"Error writing text file: {e}")
        print(f"ERROR: Failed to write text file: {e}")
    
    # Calculate summary statistics
    summary = calculate_result_statistics(results)
    
    # Write summary to JSON
    try:
        logger.info(f"Writing summary statistics to {summary_path}")
        print(f"Writing summary statistics to {summary_path}")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, cls=NumpyJSONEncoder)  # Use 4-space indentation
        
        logger.info(f"Successfully wrote summary to {summary_path}")
        print(f"✓ Successfully wrote summary to {summary_path}")
    except Exception as e:
        logger.error(f"Error writing summary file: {e}")
        print(f"ERROR: Failed to write summary file: {e}")
    
    # Write full JSON results
    try:
        logger.info(f"Writing full results to {json_path}")
        print(f"Writing full results to {json_path}")
        
        # Use the custom encoder to handle numpy types
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyJSONEncoder)  # Use 4-space indentation
        
        logger.info(f"Successfully wrote full results to {json_path}")
        logger.info(f"File size: {os.path.getsize(json_path) / (1024 * 1024):.2f} MB")
        print(f"✓ Successfully wrote full results to {json_path}")
    except Exception as e:
        logger.error(f"Error writing JSON file: {e}")
        print(f"ERROR: Failed to write JSON file: {e}")

def calculate_result_statistics(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Calculate statistics for the results."""
    total_terms = len(results)
    total_content = sum(len(contents) for contents in results.values())
    terms_with_content = sum(1 for contents in results.values() if contents)
    
    total_verified = 0
    total_urls = set()
    verified_urls = set()
    domain_counts = {}
    
    for contents in results.values():
        for content in contents:
            url = content.get("url", "")
            is_verified = content.get("is_verified", False)
            
            if url:
                total_urls.add(url)
                
                if is_verified:
                    total_verified += 1
                    verified_urls.add(url)
                
                # Count domains
                try:
                    domain = urlparse(url).netloc
                    if domain:
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                except:
                    pass
    
    # Sort domains by count
    sorted_domains = [
        {"domain": domain, "count": count}
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    
    # Calculate percentages
    terms_with_content_pct = (terms_with_content / total_terms * 100) if total_terms > 0 else 0
    verified_content_pct = (total_verified / total_content * 100) if total_content > 0 else 0
    
    return {
        "total_terms": total_terms,
        "terms_with_content": terms_with_content,
        "terms_with_content_pct": terms_with_content_pct,
        "total_content": total_content,
        "verified_content": total_verified,
        "verified_content_pct": verified_content_pct,
        "unique_urls": len(total_urls),
        "unique_verified_urls": len(verified_urls),
        "top_domains": sorted_domains[:20],  # Top 20 domains
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and Pydantic models."""
    def default(self, obj):
        # Use the convert_numpy_types function to handle NumPy types
        obj = convert_numpy_types(obj)
        
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
            
        return super().default(obj)

# Cache for processed content to avoid duplicate LLM calls
content_cache = {}

# Define a shared queue for LLM batch processing - moved to function-level creation
MAX_LLM_QUEUE_SIZE = 20  # Increased from BATCH_SIZE // 3

async def batch_llm_processor(llm: BaseLLM, system_prompt: str, use_cache: bool = True, queue: asyncio.Queue = None):
    """Process LLM requests in batches to reduce API overhead.
    
    This function should be run as a background task that continuously processes
    the queue of LLM requests.
    """
    # Queue must be created in the same event loop that's running this processor
    processing_queue = queue or asyncio.Queue()
    batch = []
    batch_futures = []
    
    while True:
        try:
            # Get the next item from the queue with a timeout
            try:
                item = await asyncio.wait_for(processing_queue.get(), timeout=0.3)  # Reduced timeout
                prompt, cache_key, future = item
                batch.append((prompt, cache_key, future))
            except asyncio.TimeoutError:
                # If timeout and we have items in the batch, process them
                if batch:
                    await process_batch(llm, system_prompt, batch, use_cache)
                    batch = []
                    batch_futures = []
                continue
            
            # If batch is full or queue is empty, process the batch
            if len(batch) >= MAX_LLM_QUEUE_SIZE or processing_queue.empty():
                await process_batch(llm, system_prompt, batch, use_cache)
                batch = []
                batch_futures = []
            
            # Mark task as done
            processing_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in batch LLM processor: {e}")
            # For any items in the batch that weren't processed, set an error result
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(e)
            batch = []
            batch_futures = []
            # Continue processing the queue despite errors
            await asyncio.sleep(0.2)  # Reduced delay
    
    return processing_queue  # Return queue for use by other functions

async def process_batch(llm: BaseLLM, system_prompt: str, batch, use_cache: bool):
    """Process a batch of LLM requests."""
    try:
        # Check cache first for each item
        uncached_batch = []
        uncached_indices = []
        
        if use_cache:
            for i, (prompt, cache_key, future) in enumerate(batch):
                if cache_key in content_cache:
                    # If cached, resolve the future immediately
                    future.set_result(content_cache[cache_key])
                else:
                    # Otherwise, add to uncached batch
                    uncached_batch.append((prompt, cache_key, future))
                    uncached_indices.append(i)
        else:
            uncached_batch = batch
            uncached_indices = list(range(len(batch)))
        
        # Process uncached items
        if uncached_batch:
            # For now, process items individually (in the future, if the LLM API supports batching, use it)
            for prompt, cache_key, future in uncached_batch:
                try:
                    response = llm.infer(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        response_model=None
                    )
                    
                    result = response.text
                    
                    # Cache the result if caching is enabled
                    if use_cache:
                        content_cache[cache_key] = result
                    
                    # Set the result for the future
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                
                # Small delay between API calls to avoid rate limits
                await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Set exceptions for all uncompleted futures
        for _, _, future in uncached_batch:
            if not future.done():
                future.set_exception(e)

async def extract_batch_content(
    urls_data: List[Tuple[str, dict, str]],
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    show_progress: bool = False,
    max_workers: int = None
) -> List[Tuple[str, dict, str, Optional[str]]]:
    """Extract content from multiple URLs in parallel
    
    Args:
        urls_data: List of tuples (concept, result, url)
        session: aiohttp session
        semaphore: Semaphore for rate limiting
        show_progress: Whether to show progress
        max_workers: Maximum workers for content extraction
        
    Returns:
        List of tuples (concept, result, url, content)
    """
    logger.info(f"Starting parallel content extraction for {len(urls_data)} URLs")
    extraction_tasks = []
    
    for concept, result, url in urls_data:
        task = extract_web_content(
            url,
            session,
            semaphore,
            show_progress,
            max_workers
        )
        extraction_tasks.append((concept, result, url, task))
    
    # Process content extraction in chunks to avoid memory issues
    results = []
    chunk_size = MAX_EXTRACTION_BATCH
    
    for i in range(0, len(extraction_tasks), chunk_size):
        chunk = extraction_tasks[i:i + chunk_size]
        chunk_start = time.time()
        logger.info(f"Processing URL extraction chunk {i//chunk_size + 1}/{(len(extraction_tasks) + chunk_size - 1)//chunk_size} ({len(chunk)} URLs)")
        
        # Create tasks and wait for all to complete
        tasks = [task for _, _, _, task in chunk]
        contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        success_count = 0
        failed_count = 0
        
        for (concept, result, url, _), content in zip(chunk, contents):
            if isinstance(content, Exception):
                logger.warning(f"Error extracting content from {url}: {content}")
                failed_count += 1
                content = None
            else:
                success_count += 1
                if content:
                    content_size = len(content) if content else 0
                    logger.debug(f"Extracted {content_size} chars from {url}")
                else:
                    logger.warning(f"No content extracted from {url}")
                    failed_count += 1
                    success_count -= 1
            
            results.append((concept, result, url, content))
            
        # Add a small delay between chunks to avoid resource exhaustion
        await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05
        
        chunk_time = time.time() - chunk_start
        logger.info(f"Extraction chunk completed in {chunk_time:.2f}s - Success: {success_count}, Failed: {failed_count}")
        
        # Free memory
        del chunk
        del tasks
        del contents
    
    # Calculate success rate
    success_count = sum(1 for _, _, _, content in results if content is not None)
    success_rate = (success_count / len(urls_data)) * 100 if urls_data else 0
    logger.info(f"Content extraction complete - Success rate: {success_rate:.1f}% ({success_count}/{len(urls_data)})")
    
    return results

async def process_content_batch_efficiently(
    batch_data: List[Tuple[str, dict, str, Optional[str]]],
    llm: BaseLLM,
    system_prompt: str,
    show_progress: bool = False,
    skip_verification: bool = False,
    use_cache: bool = True,
    use_batch_llm: bool = True,
    llm_queue: asyncio.Queue = None,
    skip_low_quality: bool = True,
    content_threshold: float = 1.5
) -> List[Tuple[str, Optional[WebContent]]]:
    """Process multiple content items efficiently
    
    Args:
        batch_data: List of tuples (concept, result, url, content)
        llm: LLM instance
        system_prompt: System prompt for LLM
        show_progress: Whether to show progress
        skip_verification: Whether to skip verification
        use_cache: Whether to use caching
        use_batch_llm: Whether to use batch LLM processing
        llm_queue: Queue for batch LLM processing
        skip_low_quality: Whether to skip low quality content early
        content_threshold: Threshold for raw content quality
        
    Returns:
        List of tuples (concept, web_content)
    """
    logger.info(f"Processing batch of {len(batch_data)} content items")
    # First filter out items with no content
    valid_items = [(concept, result, url, content) 
                   for concept, result, url, content in batch_data 
                   if content is not None]
    
    logger.info(f"Found {len(valid_items)}/{len(batch_data)} items with valid content")
    
    # Process educational score assessment in parallel if skipping low quality
    if skip_low_quality:
        logger.info(f"Assessing educational quality of {len(valid_items)} content items")
        quality_start = time.time()
        
        edu_score_tasks = []
        for concept, result, url, content in valid_items:
            task = get_educational_score_async(content)
            edu_score_tasks.append((concept, result, url, content, task))
            
        # Wait for all edu scores
        results = []
        high_quality = 0
        low_quality = 0
        
        for concept, result, url, content, task in edu_score_tasks:
            try:
                edu_score = await task
                # Only process content above threshold
                if edu_score >= content_threshold:
                    results.append((concept, result, url, content, True, edu_score))
                    high_quality += 1
                    logger.debug(f"High quality content from {url} (score: {edu_score:.2f})")
                else:
                    # Use snippet as processed content for low quality
                    results.append((concept, result, url, content, False, edu_score))
                    low_quality += 1
                    logger.info(f"Skipping low-quality content from {url} (score: {edu_score:.2f})")
            except Exception as e:
                logger.error(f"Error assessing content quality for {url}: {e}")
                # Include with default score to avoid losing content
                results.append((concept, result, url, content, True, 2.5))
        
        quality_time = time.time() - quality_start
        logger.info(f"Quality assessment complete in {quality_time:.2f}s - High: {high_quality}, Low: {low_quality}")
    else:
        # Include all content if not skipping low quality
        logger.info("Skipping quality assessment - processing all content")
        results = [(concept, result, url, content, True, 2.5) for concept, result, url, content in valid_items]
    
    # Process content with LLM in parallel
    logger.info(f"Processing {len(results)} content items with LLM")
    llm_start = time.time()
    
    processing_tasks = []
    for concept, result, url, content, process_with_llm, edu_score in results:
        task = asyncio.create_task(process_web_content(
            url=result['url'],
            title=result['title'],
            snippet=result['snippet'],
            raw_content=content,
            score=result['score'],
            llm=llm,
            system_prompt=system_prompt,
            show_progress=show_progress,
            skip_verification=skip_verification,
            use_cache=use_cache,
            use_batch_llm=use_batch_llm,
            llm_queue=llm_queue,
            process_with_llm=process_with_llm,
            edu_score=edu_score,
            term=concept,
            query=result['query']
        ))
        processing_tasks.append((concept, task))
    
    # Process in smaller chunks to manage memory
    final_results = []
    chunk_size = MAX_CONCURRENT_PROCESSING
    
    for i in range(0, len(processing_tasks), chunk_size):
        chunk = processing_tasks[i:i + chunk_size]
        chunk_start = time.time()
        logger.info(f"Processing LLM chunk {i//chunk_size + 1}/{(len(processing_tasks) + chunk_size - 1)//chunk_size} ({len(chunk)} items)")
        
        tasks = [task for _, task in chunk]
        contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        failed_count = 0
        
        for (concept, _), content in zip(chunk, contents):
            if isinstance(content, Exception):
                logger.error(f"Error processing content: {content}")
                content = None
                failed_count += 1
            else:
                success_count += 1
            final_results.append((concept, content))
        
        # Small delay between chunks
        await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05
        
        chunk_time = time.time() - chunk_start
        logger.info(f"LLM chunk processed in {chunk_time:.2f}s - Success: {success_count}, Failed: {failed_count}")
        
        # Free memory after each chunk
        del chunk
        del tasks
        del contents
    
    # Calculate stats
    success_count = sum(1 for _, content in final_results if content is not None)
    success_rate = (success_count / len(results)) * 100 if results else 0
    llm_time = time.time() - llm_start
    
    logger.info(f"LLM processing complete in {llm_time:.2f}s - Success rate: {success_rate:.1f}% ({success_count}/{len(results)})")
    
    return final_results

async def process_web_content(
    url: str,
    title: str,
    snippet: str,
    raw_content: str,
    score: float,
    llm: BaseLLM,
    system_prompt: Optional[str] = None,
    show_progress: bool = False,
    skip_verification: bool = False,
    use_cache: bool = True,
    use_batch_llm: bool = True,
    llm_queue: asyncio.Queue = None,
    process_with_llm: bool = True,
    edu_score: float = None,
    term: str = "",
    query: Optional[str] = None,
    **kwargs
) -> Optional[WebContent]:
    """Process web content using LLM"""
    try:
        # Fast path for very small content - probably not useful
        if raw_content and len(raw_content.strip()) < 100:
            logger.debug(f"Content too small from {url}, using snippet")
            # Create minimal WebContent with snippet
            return WebContent(
                url=url,
                title=title,
                snippet=snippet,
                raw_content=raw_content,
                processed_content=snippet,
                score=score * 0.5,  # Penalize small content
                is_verified=False,
                verification_reason="Content too short to be useful",
                query=query
            )
            
        # Use default system prompt if none provided
        if not system_prompt:
            system_prompt = """You are an expert in extractive summarization for academic and technical concepts.
Your task is to identify and extract ONLY the most relevant sentences from the source text that directly explain the concept.
DO NOT paraphrase, rewrite, or generate new content.
DO NOT add your own explanations or interpretations.
ONLY extract 1-3 of the most relevant sentences from the original text that best define or explain the concept.
If no relevant sentences exist in the text, respond with "No relevant information found."
"""

        # First check the educational score of the raw content if not already provided
        raw_edu_score = edu_score if edu_score is not None else await get_educational_score_async(raw_content)
        logger.debug(f"Educational score for raw content from {url}: {raw_edu_score:.2f}/5.0")
        
        # Only process with LLM if the score is above threshold or we're processing all content
        processed_content = ""
        if process_with_llm:
            # Create a cache key based on content hash
            cache_key = hash(raw_content)
            
            # Check cache if enabled
            if use_cache and cache_key in content_cache:
                processed_content = content_cache[cache_key]
                logger.debug(f"Using cached processed content for {url}")
            else:
                # Create the LLM prompt
                prompt = f"""Create an ABSTRACTIVE summary of the following content that focuses specifically on explaining the concept: "{term}".

RULES:
1. Create a coherent, flowing explanation that synthesizes the information about "{term}"
2. Use your own words to explain the concept clearly and concisely
3. Keep the summary focused and relevant to "{term}" - aim for 2-4 sentences
4. Maintain academic/technical accuracy while making the concept accessible
5. If "{term}" is not discussed in the text or no relevant information exists, respond with "No relevant information found."

Text:
{raw_content}

Return ONLY the abstractive summary about "{term}", with no additional text or commentary."""

                # Use batch processing if enabled
                if use_batch_llm and llm_queue:
                    # Create a future for the result
                    future = asyncio.Future()
                    
                    # Add to the processing queue
                    await llm_queue.put((prompt, cache_key, future))
                    
                    # Wait for the result
                    try:
                        processed_content = await future
                    except Exception as e:
                        logger.error(f"Error in LLM processing for {url}: {e}")
                        # Fall back to direct processing if batch processing fails
                        response = llm.infer(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            response_model=None
                        )
                        processed_content = response.text
                else:
                    # Direct processing
                    response = llm.infer(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        response_model=None
                    )
                    processed_content = response.text
                    
                # Cache the result for future use if caching is enabled
                if use_cache:
                    content_cache[cache_key] = processed_content
        else:
            # If not processing with LLM, use the snippet as processed content
            processed_content = snippet
            logger.debug(f"Using snippet as processed content for {url} (raw edu score: {raw_edu_score:.2f}/5.0)")
        
        # Create WebContent object first without verification
        web_content = WebContent(
            url=url,
            title=title,
            snippet=snippet,
            raw_content=raw_content,
            processed_content=processed_content,
            score=score,  # Initial score from domain priority
            query=query
        )
        
        # Skip verification if requested or if raw score was below threshold
        if skip_verification or (edu_score is not None and edu_score < 1.5):
            # For below-threshold content, always mark as unverified
            if edu_score is not None and edu_score < 1.5:
                is_verified = False
                reason = f"Content skipped due to low educational quality score: {edu_score:.2f}/5.0"
                edu_score_final = edu_score
                base_edu_score = edu_score_final # Set base score for consistency
            else:
                # For websites with high domain priority, consider them verified automatically when skipping
                is_verified = True
                reason = "Verification skipped, content accepted"
                edu_score_final = 3.0  # Neutral score
                base_edu_score = edu_score_final # Set base score for consistency
        else:
            # Verify content quality based on processed content
            # Unpack the four values returned by verify_content_async
            is_verified, reason, edu_score_final, base_edu_score = await verify_content_async(
                url, processed_content, min_score=kwargs.get('min_score', 1.5)
            )
        
        # Log verification failures but continue processing
        if not is_verified and not skip_verification:
            logger.warning(f"Content verification failed for {url}: {reason}")
        
        # Combine domain priority with educational score
        final_score = (score * 0.4) + (edu_score_final / 5.0 * 0.6)  # Weight educational score more
        
        # Update WebContent object with verification status and final score
        web_content.score = final_score
        web_content.is_verified = is_verified
        web_content.verification_reason = reason
        web_content.educational_score = edu_score_final # Store the final boosted score
        
        return web_content
    except Exception as e:
        logger.error(f"Error processing content from {url}: {e}")
        return None

async def search_web_content(
    concepts: List[str],
    settings: SearchSettings,
    continue_from: Optional[str] = None,
    save_interval: int = 1  # Save after every batch by default
) -> Dict[str, List[WebContent]]:
    """
    Search for web content related to concepts.
    
    Args:
        concepts: List of concepts to search for
        settings: Search settings
        continue_from: Optional path to continue from previous results
        save_interval: How often to save intermediate results (in batches)
        
    Returns:
        Dictionary mapping concepts to lists of WebContent objects
    """
    # Skip processing if no concepts provided
    if not concepts:
        logger.warning("No concepts provided for search")
        return {}
        
    # Load existing results if continue_from is specified
    results_by_concept = {}
    if continue_from:
        logger.info(f"Loading existing results from {continue_from}")
        results_by_concept = load_results(continue_from)
        # Only search for concepts that don't exist in the file or have no verified content
        concepts_to_search = []
        for concept in concepts:
            if (concept not in results_by_concept or 
                not results_by_concept[concept] or
                not any(content.is_verified for content in results_by_concept[concept])):
                concepts_to_search.append(concept)
                # Initialize empty list if concept doesn't exist
                if concept not in results_by_concept:
                    results_by_concept[concept] = []
        
        # If all concepts already have verified content, return the existing results
        if not concepts_to_search:
            logger.info("All concepts already have verified content in the continue-from file.")
            return results_by_concept
            
        # Otherwise, update the concepts list to only search for missing/incomplete concepts
        logger.info(f"Continuing search for {len(concepts_to_search)} concepts not in existing file.")
        concepts = concepts_to_search
    
    # Initialize seen URLs tracking for new search
    seen_urls_by_concept = {concept: set() for concept in concepts}
    # If continuing from existing results, add existing URLs to seen set
    if continue_from:
        for concept in concepts:
            if concept in results_by_concept:
                seen_urls_by_concept[concept] = {content.url for content in results_by_concept[concept]}
    
    # Initialize new concepts that weren't in the continue-from file
    for concept in concepts:
        if concept not in results_by_concept:
            results_by_concept[concept] = []
    
    # Create semaphore for rate limiting - use a consistent limit
    semaphore = Semaphore(min(settings.max_concurrent_requests, MAX_CONCURRENT_REQUESTS))  # Cap at defined constant
    logger.info(f"Using maximum {semaphore._value} concurrent requests")
    
    # Initialize process pool for trafilatura with conservative worker count
    max_workers = min(settings.max_workers, MAX_WORKERS)  # Cap at MAX_WORKERS constant
    if settings.safe_mode:
        # Even more conservative in safe mode
        max_workers = min(4, max_workers)  # Use 4 as a reasonable maximum for safe mode
    initialize_trafilatura_executor(max_workers)
    logger.info(f"Initialized content extraction with {max_workers} workers")
    
    # Initialize LLM for content processing
    provider = settings.provider or Provider.GEMINI
    model = GEMINI_MODELS['default'] if provider == Provider.GEMINI else OPENAI_MODELS['default']
    logger.info(f"Initializing LLM with provider: {provider}, model: {model}")
    
    llm = LLMFactory.create_llm(
        provider=provider,
        temperature=0.1,  # Low temperature for consistent responses
        model=model
    )
    
    # Start the batch processor task for LLM if batch processing is enabled
    # Create a queue in the current event loop
    batch_processor_task = None
    llm_queue = None
    if settings.use_batch_llm:
        # Create queue in current event loop
        logger.info("Initializing batch LLM processor")
        llm_queue = asyncio.Queue()
        batch_processor_task = asyncio.create_task(
            batch_llm_processor(llm, settings.system_prompt, settings.use_cache, llm_queue)
        )
    
    # Configure connection pool with consistent limits
    conn = aiohttp.TCPConnector(
        limit=min(settings.max_concurrent_requests, MAX_CONCURRENT_REQUESTS),  # Use constant
        ttl_dns_cache=300
    )
    
    try:
        async with aiohttp.ClientSession(connector=conn) as session:
            # Use RapidAPI search if enabled
            if settings.use_rapidapi:
                try:
                    # Use smaller batch size for safety
                    batch_size = min(BATCH_SIZE, settings.batch_size)  # Use BATCH_SIZE constant
                    logger.info(f"Starting search with batch size: {batch_size}")
                    
                    # Process terms in batches
                    batch_count = 0
                    for i in range(0, len(concepts), batch_size):
                        batch_count += 1
                        batch = concepts[i:i + batch_size]
                        
                        # Skip empty batches
                        if not batch:
                            continue
                        
                        # Start timing for this batch
                        batch_start_time = time.time()
                        logger.info(f"Processing batch {batch_count} of {(len(concepts) + batch_size - 1)//batch_size} ({len(batch)} terms)")
                        
                        # Create search queries: one regular and one Wikipedia-focused for each term
                        search_queries = []
                        query_to_concept_map = {}  # Map to track which concept each query belongs to
                        
                        for concept in batch:
                            # Query 1: Regular term query
                            regular_query = concept
                            search_queries.append(regular_query)
                            query_to_concept_map[regular_query] = concept
                            
                            # Query 2: Wikipedia-focused query
                            wiki_query = f"{concept} wikipedia"
                            search_queries.append(wiki_query)
                            query_to_concept_map[wiki_query] = concept
                        
                        if settings.show_progress:
                            logger.info(f"Generated {len(search_queries)} queries for {len(batch)} terms")
                        
                        try:
                            # Search and process results in parallel
                            logger.info(f"Sending batch search request to RapidAPI")
                            search_start_time = time.time()
                            
                            search_results = await search_rapidapi_batch(
                                search_queries,
                                session,
                                semaphore,
                                settings.show_progress
                            )
                            
                            search_time = time.time() - search_start_time
                            logger.info(f"RapidAPI search completed in {search_time:.2f}s")
                            
                            # Skip processing if no search results
                            if not search_results or not search_results.get("data"):
                                logger.warning("No search results returned, skipping batch")
                                continue
                            
                            process_start_time = time.time()
                            results = await process_rapidapi_results(
                                search_results,
                                settings.show_progress
                            )
                            
                            # Skip further processing if no results
                            if not results:
                                logger.warning("No processed results found, skipping batch")
                                continue
                                
                            process_time = time.time() - process_start_time
                            logger.info(f"Search results processed in {process_time:.2f}s - Found {len(results)} results")
                            
                            # Collect all URLs to process
                            urls_to_process = []
                            skipped_urls = 0
                            
                            for result in results:
                                query = result['query']
                                # Map back to the original concept
                                concept = query_to_concept_map.get(query, query)
                                
                                # Skip if concept not in our list (shouldn't happen)
                                if concept not in results_by_concept:
                                    continue
                                
                                url = standardize_url(result['url'])
                                
                                # Skip if we've already seen this URL for this concept
                                if url in seen_urls_by_concept[concept] or should_skip_domain(url):
                                    skipped_urls += 1
                                    continue
                                
                                seen_urls_by_concept[concept].add(url)
                                
                                # Add to batch processing list
                                urls_to_process.append((concept, result, url))
                            
                            # Skip if no URLs to process
                            if not urls_to_process:
                                logger.warning("No URLs to process after filtering, skipping batch")
                                continue
                                
                            logger.info(f"Prepared {len(urls_to_process)} URLs for extraction (skipped {skipped_urls} duplicates)")
                            
                            # Process content extraction more efficiently
                            if settings.parallel_extraction:
                                # New parallel extraction approach
                                extract_start_time = time.time()
                                logger.info(f"Starting parallel content extraction for {len(urls_to_process)} URLs")
                                
                                extracted_contents = await extract_batch_content(
                                    urls_to_process,
                                    session,
                                    semaphore,
                                    settings.show_progress,
                                    max_workers
                                )
                                
                                # Skip processing if no content extracted
                                if not any(content for _, _, _, content in extracted_contents):
                                    logger.warning("No content extracted from any URL, skipping batch")
                                    continue
                                
                                extract_time = time.time() - extract_start_time
                                extract_success = sum(1 for _, _, _, content in extracted_contents if content is not None)
                                logger.info(f"Content extraction completed in {extract_time:.2f}s - Success: {extract_success}/{len(urls_to_process)}")
                                
                                # Process extracted content in parallel
                                process_start_time = time.time()
                                logger.info(f"Starting content processing with LLM")
                                
                                processed_contents = await process_content_batch_efficiently(
                                    extracted_contents,
                                    llm,
                                    settings.system_prompt,
                                    settings.show_progress,
                                    settings.skip_verification,
                                    settings.use_cache,
                                    settings.use_batch_llm,
                                    llm_queue,
                                    settings.skip_low_quality,
                                    settings.content_threshold
                                )
                                
                                process_time = time.time() - process_start_time
                                process_success = sum(1 for _, content in processed_contents if content is not None)
                                logger.info(f"Content processing completed in {process_time:.2f}s - Success: {process_success}/{len(extracted_contents)}")
                                
                                # Add results to the output dictionary
                                for concept, web_content in processed_contents:
                                    if web_content:
                                        results_by_concept[concept].append(web_content)
                            else:
                                # Original approach with sequential content extraction
                                logger.info("Using sequential content extraction (slower)")
                                content_extraction_tasks = []
                                
                                for concept, result, url in urls_to_process:
                                    # Create task for content extraction
                                    task = extract_web_content(
                                        url,
                                        session,
                                        semaphore,
                                        settings.show_progress,
                                        max_workers  # Use the capped max_workers value
                                    )
                                    content_extraction_tasks.append((concept, result, task))
                                
                                # Skip if no tasks
                                if not content_extraction_tasks:
                                    continue
                                
                                # Process content extraction in parallel, but limit batch size to avoid memory issues
                                for j in range(0, len(content_extraction_tasks), MAX_EXTRACTION_BATCH):
                                    batch_tasks = content_extraction_tasks[j:j + MAX_EXTRACTION_BATCH]
                                    logger.info(f"Processing extraction batch {j//MAX_EXTRACTION_BATCH + 1}/{(len(content_extraction_tasks) + MAX_EXTRACTION_BATCH - 1)//MAX_EXTRACTION_BATCH}")
                                    
                                    # Process content extraction tasks in current batch
                                    content_processing_tasks = []
                                    
                                    for concept, result, task in batch_tasks:
                                        try:
                                            raw_content = await task
                                            
                                            if raw_content:
                                                # Create processing task but don't await yet
                                                processing_task = process_web_content(
                                                    url=result['url'],
                                                    title=result['title'],
                                                    snippet=result['snippet'],
                                                    raw_content=raw_content,
                                                    score=result['score'],
                                                    llm=llm,
                                                    system_prompt=settings.system_prompt,
                                                    show_progress=settings.show_progress,
                                                    skip_verification=settings.skip_verification,
                                                    use_cache=settings.use_cache,
                                                    use_batch_llm=settings.use_batch_llm,
                                                    llm_queue=llm_queue,  # Pass the queue created in this event loop
                                                    term=concept,
                                                    query=result['query']
                                                )
                                                content_processing_tasks.append((concept, processing_task))
                                        except Exception as e:
                                            logger.error(f"Error extracting content for URL {result['url']}: {str(e)}")
                                            # Continue with other tasks
                                    
                                    # Skip if no processing tasks
                                    if not content_processing_tasks:
                                        continue
                                    
                                    logger.info(f"Processing {len(content_processing_tasks)} content items with LLM")
                                    # Now process the LLM tasks in parallel - use the constant
                                    for i in range(0, len(content_processing_tasks), MAX_CONCURRENT_PROCESSING):
                                        current_batch = content_processing_tasks[i:i + MAX_CONCURRENT_PROCESSING]
                                        
                                        # Process this smaller batch concurrently
                                        batch_results = await asyncio.gather(
                                            *[task for _, task in current_batch],
                                            return_exceptions=True
                                        )
                                        
                                        # Add the results to the output dictionary
                                        for (concept, _), result in zip(current_batch, batch_results):
                                            if isinstance(result, Exception):
                                                logger.error(f"Error processing content: {str(result)}")
                                            elif result:
                                                results_by_concept[concept].append(result)
                                        
                                        # Add a small delay between processing batches
                                        await asyncio.sleep(0.05)
                                    
                                    # Free memory after each batch
                                    del batch_tasks
                                    del content_processing_tasks
                            
                            # Save intermediate results if continue_from is specified and we've processed enough batches
                            if continue_from and batch_count % save_interval == 0:
                                # Skip saving if no new results since last save
                                if batch_count > 1:
                                    logger.info(f"Saving intermediate results after batch {batch_count}...")
                                    save_results(results_by_concept, continue_from, merge_with_existing=True)
                            
                            # Calculate and log batch processing time
                            batch_end_time = time.time()
                            batch_processing_time = batch_end_time - batch_start_time
                            logger.info(f"Batch {batch_count} processing time: {batch_processing_time:.2f} seconds for {len(batch)} terms")
                            # Calculate time per term
                            time_per_term = batch_processing_time / len(batch)
                            logger.info(f"Average time per term: {time_per_term:.2f} seconds")
                            
                            # Log stats for current batch
                            batch_stats = {}
                            for concept in batch:
                                batch_stats[concept] = len(results_by_concept.get(concept, []))
                            
                            logger.info(f"Results count by term: {batch_stats}")
                            
                            # Add longer delay between batches to avoid resource exhaustion
                            await asyncio.sleep(RATE_LIMIT_DELAY)
                            
                        except Exception as e:
                            # Still log processing time even if there was an error
                            batch_end_time = time.time()
                            batch_processing_time = batch_end_time - batch_start_time
                            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                            logger.error(f"Failed batch processing time: {batch_processing_time:.2f} seconds")
                            # Add a longer delay after an error - use multiple of the constant
                            await asyncio.sleep(RATE_LIMIT_DELAY * 5)
                            # Continue with next batch
            
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error during RapidAPI search: {str(e)}\n{error_details}")
                    
                # Save final results if continue_from is specified
                if continue_from:
                    logger.info("Saving final results...")
                    save_results(results_by_concept, continue_from, merge_with_existing=True)
    finally:
        # Cancel the batch processor task when done
        if batch_processor_task:
            logger.info("Shutting down batch LLM processor")
            batch_processor_task.cancel()
            try:
                await batch_processor_task
            except asyncio.CancelledError:
                pass
    
    return results_by_concept

def load_results(input_file: str) -> Dict[str, List[WebContent]]:
    """Load existing mining results from a JSON file
    
    Args:
        input_file: Path to load results from
        
    Returns:
        Dictionary mapping concepts to their WebContent objects
    """
    try:
        if not os.path.exists(input_file):
            logger.warning(f"Continue-from file {input_file} does not exist. Will create a new file.")
            return {}
            
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Convert the loaded data back to WebContent objects
        results = {}
        for concept, contents in data.items():
            results[concept] = [WebContent(**content) for content in contents]
            
        logger.info(f"Loaded existing results from {input_file} with {len(results)} concepts")
        return results
    except Exception as e:
        logger.error(f"Error loading results from file: {e}")
        return {}

def save_results(results: Dict[str, List[WebContent]], output_file: str, merge_with_existing: bool = False) -> None:
    """
    Save results to a JSON file, optionally merging with existing results.
    
    Args:
        results: Dictionary mapping terms to lists of WebContent objects
        output_file: Path to the output file
        merge_with_existing: Whether to merge with existing results
    """
    # Convert WebContent objects to dictionaries
    converted_results = {}
    for term, contents in results.items():
        # Handle a list of WebContent objects or dictionaries
        converted_contents = []
        for content in contents:
            if hasattr(content, 'model_dump'):
                # Handle Pydantic models (WebContent)
                converted_contents.append(content.model_dump())
            else:
                # Already a dictionary or similar
                converted_contents.append(content)
        converted_results[term] = converted_contents
    
    try:
        # Merge with existing results if requested
        if merge_with_existing and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                
                # Merge the results
                for term, contents in converted_results.items():
                    if term in existing_results:
                        existing_results[term].extend(contents)
                    else:
                        existing_results[term] = contents
                
                # Use the merged results
                results_to_save = existing_results
                
            except Exception as e:
                logger.error(f"Error loading existing results from {output_file}: {e}")
                logger.warning("Proceeding with saving new results only")
                results_to_save = converted_results
        else:
            results_to_save = converted_results
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save the results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, cls=NumpyJSONEncoder)  # Changed indent from 2 to 4 for better readability
        
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results to file: {e}")
        print(f"Error saving results to file: {str(e)}")
        raise

async def process_all_content_for_terms(
    content_by_term: Dict[str, List[WebContent]],
    min_score: float,
    batch_size: int = BATCH_SIZE // 3,  # Use 1/3 of BATCH_SIZE as a reasonable default
    skip_verification: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """Process all content for multiple terms with verification.
    
    Args:
        content_by_term: Dictionary mapping terms to their WebContent objects
        min_score: Minimum score threshold for verification
        batch_size: Number of contents to process at once
        skip_verification: Whether to skip verification step
        
    Returns:
        Dictionary mapping terms to their processed content
    """
    results = {}
    
    # Short-circuit if empty
    if not content_by_term:
        return results
        
    # Skip processing and just convert to dictionaries if verification is skipped
    if skip_verification:
        # Fast path - just convert objects to dictionaries
        logger.info("Skipping verification - fast path conversion")
        for term, contents in content_by_term.items():
            results[term] = [content.model_dump() for content in contents]
        return results
    
    # Process terms in parallel instead of sequentially
    async def process_term(term: str, contents: List[WebContent]) -> Tuple[str, List[Dict[str, Any]]]:
        term_results = []
        
        # Skip processing if no contents
        if not contents:
            return term, []
        
        # Process content in smaller batches
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            processed = await process_content_batch(batch, term, min_score, skip_verification)
            
            # Include all processed content, regardless of verification status
            term_results.extend([r for r in processed if r is not None])
            
            # Free memory after each batch
            del batch
            del processed
            
        return term, term_results
    
    # Create tasks for all terms
    tasks = [process_term(term, contents) for term, contents in content_by_term.items()]
    
    # Process all terms in parallel
    term_results = await asyncio.gather(*tasks)
    
    # Combine results
    for term, term_content in term_results:
        results[term] = term_content
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Mine web content for academic concepts"
    )
    parser.add_argument("--concept", required=True, help="Concept to search for")
    parser.add_argument("--output", "-o", help="Path to save results as JSON")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum educational score threshold",
    )
    parser.add_argument("--provider", "-p", help="LLM provider (default: OpenAI)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--no-rapidapi", action="store_true", help="Disable RapidAPI search"
    )
    parser.add_argument(
        "--continue-from", 
        help="Continue from existing results file and update it in-place"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save interval in batches (default: after every batch)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for processing (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT_REQUESTS,
        help=f"Maximum concurrent requests (default: {MAX_CONCURRENT_REQUESTS})"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = SearchSettings(
        min_score=args.min_score,
        provider=args.provider,
        use_rapidapi=not args.no_rapidapi,
        batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrent,
    )

    # If both output and continue-from are specified, warn user that continue-from takes precedence
    if args.output and args.continue_from:
        logger.warning("Both --output and --continue-from specified. Results will be updated in --continue-from file.")
        output_file = args.continue_from
    else:
        output_file = args.continue_from or args.output

    results = asyncio.run(
        search_web_content(
            [args.concept], 
            settings, 
            continue_from=args.continue_from,
            save_interval=args.save_interval
        )
    )
    concept_results = results.get(args.concept, [])

    # Save results if output file is specified and we're not using continue-from
    if args.output and not args.continue_from:
        save_results(results, args.output)

    # Display results
    logger.info(f"\nFound {len(concept_results)} relevant results:")
    for i, content in enumerate(concept_results, 1):
        logger.info(f"\n=== Result {i} ===")
        logger.info(f"Source: {content.url}")
        logger.info(f"Title: {content.title}")
        logger.info("\nSnippet:")
        logger.info(content.snippet)
        logger.info("\nProcessed Content:")
        logger.info(
            content.processed_content[:MAX_CONTENT_LENGTH // 16] + "..."  # Use MAX_CONTENT_LENGTH to determine preview length
            if len(content.processed_content) > MAX_CONTENT_LENGTH // 16
            else content.processed_content
        )
        logger.info("-" * 80)

if __name__ == "__main__":
    main()
