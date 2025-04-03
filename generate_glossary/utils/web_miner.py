"""Unified web content mining module with support for general web and Wikipedia content."""

import os
import json
import asyncio
import aiohttp
import argparse
import subprocess
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
from tools.web_scraper import fetch_page, parse_html
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_CONCURRENT_REQUESTS = 10  # Increased from 5
RETRY_ATTEMPTS = 3
RATE_LIMIT_DELAY = 1
MAX_CONTENT_LENGTH = 8000
BATCH_SIZE = 32
CONNECT_TIMEOUT = 10  # Connection timeout in seconds
READ_TIMEOUT = 30    # Socket read timeout in seconds
TOTAL_TIMEOUT = 60   # Total operation timeout in seconds
MAX_RESULTS_PER_QUERY = 5  # Maximum results per query
MAX_WORKERS = max(4, cpu_count() - 1)  # Use most cores but leave one for system

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
    title: str
    snippet: str
    raw_content: str
    processed_content: str
    score: float = Field(default=0.5)  # Default score for unknown domains
    is_verified: bool = False
    verification_reason: str = ""

class SearchSettings(BaseModel):
    """Settings for web content search and verification"""
    min_score: float = Field(default=0.7)
    max_concurrent_requests: int = Field(default=MAX_CONCURRENT_REQUESTS)  # Increased default
    batch_size: int = Field(default=50)  # Default to 50 terms per batch (each term generates 2 queries)
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
    max_workers: int = Field(default=max(4, cpu_count() - 1))  # CPU workers for parallel processing
    use_cache: bool = Field(default=True)  # Whether to cache LLM results
    use_batch_llm: bool = Field(default=True)  # Whether to use batch processing for LLM requests

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
            workers = max_workers or min(4, max(1, cpu_count() // 4))
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

                html_content = await response.text()
                
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

                await asyncio.sleep(RATE_LIMIT_DELAY * 0.5)  # Reduced rate limit delay
                return content

        except asyncio.TimeoutError:
            logger.warning(f"Timeout error while fetching {url} (connect={CONNECT_TIMEOUT}s, read={READ_TIMEOUT}s)")
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
                    html_content = await response.text()
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
    """Write web mining results to multiple files.
    
    Args:
        results: Dictionary mapping terms to their web content
        output_path: Base path for output files (without extension)
    """
    # Convert NumPy types to Python native types
    converted_results = {
        term: [convert_numpy_types(content) for content in contents]
        for term, contents in results.items()
    }
    
    # Write the full mining results to JSON, including all content
    json_path = f"{output_path}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)

    # Write a summary file with statistics
    summary_path = f"{output_path}_summary.json"
    summary = {
        term: {
            "total_sources": len(contents),
            "verified_sources": len([c for c in contents if bool(c.get("is_verified", False))]),
            "wikipedia_sources": len([c for c in contents if "wikipedia.org" in c.get("url", "")]),
            "average_score": float(sum(c.get("score", 0) for c in contents)) / len(contents) if contents else 0
        }
        for term, contents in converted_results.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    # Write a simple text file with term-URL mappings (for easy viewing)
    text_path = f"{output_path}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for term, contents in converted_results.items():
            verified_contents = [c for c in contents if bool(c.get("is_verified", False))]
            if verified_contents:
                f.write(f"{term}:\n")
                for content in verified_contents:
                    f.write(f"  - {content.get('url')}\n")
                f.write("\n")

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

# Define a shared queue for LLM batch processing
llm_processing_queue = asyncio.Queue()
MAX_LLM_QUEUE_SIZE = 10  # Maximum batch size for LLM processing

async def batch_llm_processor(llm: BaseLLM, system_prompt: str, use_cache: bool = True):
    """Process LLM requests in batches to reduce API overhead.
    
    This function should be run as a background task that continuously processes
    the queue of LLM requests.
    """
    batch = []
    batch_futures = []
    
    while True:
        try:
            # Get the next item from the queue with a timeout
            try:
                item = await asyncio.wait_for(llm_processing_queue.get(), timeout=0.5)
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
            if len(batch) >= MAX_LLM_QUEUE_SIZE or llm_processing_queue.empty():
                await process_batch(llm, system_prompt, batch, use_cache)
                batch = []
                batch_futures = []
            
            # Mark task as done
            llm_processing_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in batch LLM processor: {e}")
            # For any items in the batch that weren't processed, set an error result
            for _, _, future in batch:
                if not future.done():
                    future.set_exception(e)
            batch = []
            batch_futures = []
            # Continue processing the queue despite errors
            await asyncio.sleep(1)

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
                await asyncio.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Set exceptions for all uncompleted futures
        for _, _, future in uncached_batch:
            if not future.done():
                future.set_exception(e)

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
    **kwargs
) -> Optional[WebContent]:
    """Process web content using LLM"""
    try:
        # Use default system prompt if none provided
        if not system_prompt:
            system_prompt = """You are an expert in extractive summarization for academic and technical concepts.
Your task is to identify and extract ONLY the most relevant sentences from the source text that directly explain the concept.
DO NOT paraphrase, rewrite, or generate new content.
DO NOT add your own explanations or interpretations.
ONLY extract 1-3 of the most relevant sentences from the original text that best define or explain the concept.
If no relevant sentences exist in the text, respond with "No relevant information found."
"""

        # Create a cache key based on content hash
        cache_key = hash(raw_content)
        
        # Check cache if enabled
        if use_cache and cache_key in content_cache:
            processed_content = content_cache[cache_key]
        else:
            # Create the LLM prompt
            prompt = f"""Extract ONLY the most relevant sentences from this text that directly explain the concept.

RULES:
1. ONLY extract 1-3 sentences from the original text that best define or explain the concept
2. DO NOT paraphrase or rewrite - use the EXACT sentences from the source text
3. DO NOT add any of your own explanations or interpretations
4. Focus ONLY on sentences that provide:
   - Clear definitions
   - Core meanings
   - Fundamental characteristics
5. If no relevant sentences exist, respond with "No relevant information found."

Text:
{raw_content}

Return ONLY the extracted sentences, with no additional text or commentary."""

            # Use batch processing if enabled
            if use_batch_llm:
                # Create a future for the result
                future = asyncio.Future()
                
                # Add to the processing queue
                await llm_processing_queue.put((prompt, cache_key, future))
                
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
        
        # Create WebContent object first without verification
        web_content = WebContent(
            url=url,
            title=title,
            snippet=snippet,
            raw_content=raw_content,
            processed_content=processed_content,
            score=score  # Initial score from domain priority
        )
        
        # Skip verification if requested
        if skip_verification:
            # For websites with high domain priority, consider them verified automatically
            is_verified = True
            reason = "Verification skipped, content accepted"
            edu_score = 3.0  # Neutral score
        else:
            # Verify content quality based on processed content
            is_verified, reason, edu_score = await verify_content_async(url, processed_content, **kwargs)
        
        # Log verification failures but continue processing
        if not is_verified and not skip_verification:
            logger.warning(f"Content verification failed for {url}: {reason}")
        
        # Combine domain priority with educational score
        final_score = (score * 0.4) + (edu_score / 5.0 * 0.6)  # Weight educational score more
        
        # Update WebContent object with verification status and final score
        web_content.score = final_score
        web_content.is_verified = is_verified
        web_content.verification_reason = reason
        
        return web_content
    except Exception as e:
        logger.error(f"Error processing content from {url}: {e}")
        return None

async def search_web_content(
    concepts: List[str],
    settings: SearchSettings
) -> Dict[str, List[WebContent]]:
    """
    Search for web content related to concepts.
    
    Args:
        concepts: List of concepts to search for
        settings: Search settings
        
    Returns:
        Dictionary mapping concepts to lists of WebContent objects
    """
    # Initialize results dictionary
    results_by_concept = {concept: [] for concept in concepts}
    seen_urls_by_concept = {concept: set() for concept in concepts}
    
    # Create semaphore for rate limiting - use a more conservative limit
    semaphore = Semaphore(min(settings.max_concurrent_requests, 10))  # Cap at 10 for safety
    
    # Initialize process pool for trafilatura with conservative worker count
    max_workers = min(settings.max_workers, 4)  # Cap at 4 for safety
    initialize_trafilatura_executor(max_workers)
    
    # Initialize LLM for content processing
    llm = LLMFactory.create_llm(
        provider=settings.provider or Provider.GEMINI,
        temperature=0.1  # Low temperature for consistent responses
    )
    
    # Start the batch processor task for LLM if batch processing is enabled
    batch_processor_task = None
    if settings.use_batch_llm:
        batch_processor_task = asyncio.create_task(
            batch_llm_processor(llm, settings.system_prompt, settings.use_cache)
        )
    
    # Configure connection pool with more conservative limits
    conn = aiohttp.TCPConnector(
        limit=min(settings.max_concurrent_requests, 10),  # Cap at 10
        ttl_dns_cache=300
    )
    
    try:
        async with aiohttp.ClientSession(connector=conn) as session:
            # Use RapidAPI search if enabled
            if settings.use_rapidapi:
                try:
                    # Use smaller batch size for safety
                    batch_size = min(30, settings.batch_size)  # Cap at 30 per batch
                    
                    # Process terms in batches
                    for i in range(0, len(concepts), batch_size):
                        batch = concepts[i:i + batch_size]
                        
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
                            logger.info(f"Processing batch {i//batch_size + 1} of {(len(concepts) + batch_size - 1)//batch_size}")
                            logger.info(f"Generated {len(search_queries)} queries for {len(batch)} terms")
                        
                        try:
                            # Search and process results in parallel
                            search_results = await search_rapidapi_batch(
                                search_queries,
                                session,
                                semaphore,
                                settings.show_progress
                            )
                            
                            results = await process_rapidapi_results(
                                search_results,
                                settings.show_progress
                            )
                            
                            # Group results by concept using the mapping
                            content_extraction_tasks = []
                            
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
                                    continue
                                
                                seen_urls_by_concept[concept].add(url)
                                
                                # Create task for content extraction
                                task = extract_web_content(
                                    url,
                                    session,
                                    semaphore,
                                    settings.show_progress,
                                    max_workers  # Use the capped max_workers value
                                )
                                content_extraction_tasks.append((concept, result, task))
                            
                            # Process content extraction in parallel, but limit batch size to avoid memory issues
                            MAX_EXTRACTION_BATCH = 10  # Reduced from 20 to 10
                            
                            for j in range(0, len(content_extraction_tasks), MAX_EXTRACTION_BATCH):
                                batch_tasks = content_extraction_tasks[j:j + MAX_EXTRACTION_BATCH]
                                
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
                                                use_batch_llm=settings.use_batch_llm
                                            )
                                            content_processing_tasks.append((concept, processing_task))
                                    except Exception as e:
                                        logger.error(f"Error extracting content for URL {result['url']}: {str(e)}")
                                        # Continue with other tasks
                                
                                # Now process the LLM tasks in parallel - limit concurrency further
                                MAX_CONCURRENT_PROCESSING = 5  # Limit concurrent LLM processing
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
                                    await asyncio.sleep(0.5)
                                
                                # Free memory after each batch
                                del batch_tasks
                                del content_processing_tasks
                            
                            # Add longer delay between batches to avoid resource exhaustion
                            await asyncio.sleep(RATE_LIMIT_DELAY * 2)
                            
                        except Exception as e:
                            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                            # Add a longer delay after an error
                            await asyncio.sleep(5)
                            # Continue with next batch
            
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.error(f"Error during RapidAPI search: {str(e)}\n{error_details}")
    finally:
        # Cancel the batch processor task when done
        if batch_processor_task:
            batch_processor_task.cancel()
            try:
                await batch_processor_task
            except asyncio.CancelledError:
                pass
    
    return results_by_concept

def save_results(results: Dict[str, List[WebContent]], output_file: str) -> None:
    """Save mining results to a JSON file
    
    Args:
        results: Dictionary mapping concepts to their WebContent objects
        output_file: Path to save results
    """
    try:
        # Convert to serializable format
        serializable_results = {
            concept: [content.model_dump() for content in contents]
            for concept, contents in results.items()
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to file: {e}")

async def process_all_content_for_terms(
    content_by_term: Dict[str, List[WebContent]],
    min_score: float,
    batch_size: int = 10,  # Increased from 5
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
    
    # Process terms in parallel instead of sequentially
    async def process_term(term: str, contents: List[WebContent]) -> Tuple[str, List[Dict[str, Any]]]:
        term_results = []
        
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
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = SearchSettings(
        min_score=args.min_score,
        provider=args.provider,
        use_rapidapi=not args.no_rapidapi,
    )

    results = asyncio.run(search_web_content([args.concept], settings))
    concept_results = results.get(args.concept, [])

    # Save results if output file is specified
    if args.output:
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
            content.processed_content[:500] + "..."
            if len(content.processed_content) > 500
            else content.processed_content
        )
        logger.info("-" * 80)

if __name__ == "__main__":
    main()
