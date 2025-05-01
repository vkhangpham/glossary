"""
Tavily-based web content mining for technical glossary generation.

This module provides functionality to search and process web content
for technical concepts using the Tavily API, with an interface compatible
with the original web_miner module.
"""

import os
import asyncio
import logging
import re
import time
import json
import csv
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urlunparse
import hashlib
from functools import lru_cache
from pydantic import BaseModel, Field
import aiohttp
from asyncio import Semaphore

# Import necessary modules from existing code
from .web_miner import (
    WebContent,
    SearchSettings,
    get_domain_priority,
    truncate_content,
    is_wikipedia_url,
    standardize_url,
    should_skip_domain,
    is_likely_binary_file,
)
from .verification_utils import get_educational_score_async, verify_content_async

# Setup logging
logger = logging.getLogger(__name__)

# Cache for LLM processed content
content_cache = {}

# Constants
MAX_RESULTS_PER_QUERY = 5 
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
RETRY_ATTEMPTS = 3
TAVILY_TIMEOUT = 60  # seconds - increased from 30 to 60
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Function to get Tavily API key with error checking
def get_tavily_api_key() -> str:
    """Get Tavily API key from environment variable with error checking."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Please set it before using Tavily API."
        )
    return api_key

def load_context_from_csv(csv_file_path: str, concept_column: str, context_column: str) -> Dict[str, List[str]]:
    """
    Load context information from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        concept_column: Name of the column containing concepts
        context_column: Name of the column containing context
        
    Returns:
        Dictionary mapping concepts to lists of unique context strings
    """
    if not csv_file_path or not os.path.exists(csv_file_path):
        logger.warning(f"CSV file not found or not specified: {csv_file_path}")
        return {}
        
    context_by_concept_sets = {} # Use sets for deduplication
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Validate columns exist
            if concept_column not in reader.fieldnames or context_column not in reader.fieldnames:
                available_columns = ", ".join(reader.fieldnames) if reader.fieldnames else "none"
                logger.error(f"Required columns '{concept_column}' and/or '{context_column}' not found in CSV. Available columns: {available_columns}")
                return {}
            
            # Process each row
            for row in reader:
                concept = row.get(concept_column, '').strip()
                context = row.get(context_column, '').strip()
                # Refined regex using word boundaries (\b)
                context = re.sub(r'\bcollege\s+of\b', '', context, flags=re.IGNORECASE)
                context = re.sub(r'\bdepartment\s+of\b', '', context, flags=re.IGNORECASE)
                # Clean up extra spaces left by substitution and strip ends
                context = re.sub(r'\s{2,}', ' ', context).strip()
                
                if concept and context:
                    if concept not in context_by_concept_sets:
                        context_by_concept_sets[concept] = set() # Initialize set
                    context_by_concept_sets[concept].add(context) # Add to set
    
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file_path}: {e}")
        return {}
        
    # Convert sets to lists before returning
    context_by_concept = {
        concept: list(contexts)
        for concept, contexts in context_by_concept_sets.items()
    }
        
    logger.info(f"Loaded unique contexts for {len(context_by_concept)} concepts from CSV")
    return context_by_concept

async def search_tavily(
    term: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    search_depth: str = "basic",
    max_results: int = MAX_RESULTS_PER_QUERY,
    include_domains: List[str] = None,
    exclude_domains: List[str] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Search using Tavily API with rate limiting and retries.
    
    Args:
        term: The term to search for
        session: aiohttp session for making requests
        semaphore: Semaphore for controlling concurrent requests
        search_depth: Either "basic" or "advanced" search depth
        max_results: Maximum number of results to return
        include_domains: List of domains to include in search results
        exclude_domains: List of domains to exclude from search results
        show_progress: Whether to show progress
        
    Returns:
        Dictionary with search results
    """
    api_key = get_tavily_api_key()
    
    # Configure search parameters
    payload = {
        "query": term,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": True,
        "max_results": max_results,
        "time_range": "year",
    }
    
    # Default exclusions for better quality
    default_exclude_domains = [
        "pinterest.com",
        "quora.com",
        "reddit.com",
        "twitter.com",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
    ]
    
    # Combine user exclusions with defaults
    if exclude_domains:
        exclude_domains = exclude_domains + default_exclude_domains
    else:
        exclude_domains = default_exclude_domains
        
    # Add optional parameters if provided
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    
    # Create headers with API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Track number of retries
    retry_count = 0
    while retry_count < RETRY_ATTEMPTS:
        try:
            async with semaphore:  # Control concurrent requests
                # Log request details
                logger.info(f"Sending request to Tavily API: {term} (depth: {search_depth}, max_results: {max_results})")
                print(f"Sending request to Tavily API: {term} (depth: {search_depth}, max_results: {max_results})")
                
                # Prepare the request data
                json_payload = json.dumps(payload)
                
                # Make the request to Tavily API using a simpler approach
                async with session.post(
                    "https://api.tavily.com/search",
                    data=json_payload,  # Use data instead of json for more direct control
                    headers=headers,
                    timeout=TAVILY_TIMEOUT,
                    ssl=False  # Disable SSL verification if needed
                ) as response:
                    # Handle response
                    status = response.status
                    if status != 200:
                        error_text = await response.text()
                        error_msg = f"Tavily API error: HTTP {status} - {error_text}"
                        logger.error(error_msg)
                        print(f"ERROR: {error_msg}")
                        response.raise_for_status()
                    
                    # Parse and return results
                    await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                    try:
                        data = await response.json()
                        logger.info(f"Received successful response from Tavily ({len(data.get('results', []))} results) for query: {term}")
                        print(f"✓ Received {len(data.get('results', []))} results from Tavily for: {term}")
                        return data
                    except Exception as json_error:
                        # If JSON parsing fails, try to get the raw text
                        raw_text = await response.text()
                        logger.error(f"Failed to parse JSON response: {json_error}. Raw response: {raw_text[:500]}...")
                        print(f"ERROR: Failed to parse Tavily response: {json_error}")
                        raise
                    
        except aiohttp.ClientError as e:
            retry_count += 1
            error_msg = f"Network error during Tavily search (attempt {retry_count}/{RETRY_ATTEMPTS}): {str(e.__class__.__name__)}: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            
            if retry_count >= RETRY_ATTEMPTS:
                logger.error(f"Max retries reached for Tavily search: {term}")
                return {"results": []}  # Return empty results on max retries
            
            # Exponential backoff
            wait_time = 2 ** retry_count
            logger.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
        
        except asyncio.TimeoutError as e:
            retry_count += 1
            error_msg = f"Timeout during Tavily search (attempt {retry_count}/{RETRY_ATTEMPTS}): Request timed out after {TAVILY_TIMEOUT} seconds"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            
            if retry_count >= RETRY_ATTEMPTS:
                logger.error(f"Max retries reached for Tavily search: {term}")
                return {"results": []}  # Return empty results on max retries
            
            # Exponential backoff
            wait_time = 2 ** retry_count
            logger.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            retry_count += 1
            error_msg = f"Error during Tavily search (attempt {retry_count}/{RETRY_ATTEMPTS}): {str(e.__class__.__name__)}: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            
            if retry_count >= RETRY_ATTEMPTS:
                logger.error(f"Max retries reached for Tavily search: {term}")
                return {"results": []}  # Return empty results on max retries
            
            # Exponential backoff
            wait_time = 2 ** retry_count
            logger.info(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
    
    # Should not reach here, but return empty results just in case
    return {"results": []}

async def process_tavily_results(search_results: Dict[str, Any], query: str) -> List[WebContent]:
    """
    Process Tavily search results into WebContent objects.
    
    Args:
        search_results: Dictionary with search results from Tavily
        query: The query used for the search
        
    Returns:
        List of WebContent objects
    """
    results = []
    
    if not search_results:
        logger.warning("Empty search results received from Tavily")
        return results
    
    # Extract results from the response
    items = search_results.get("results", [])
    if not items:
        logger.warning("No results found in Tavily response")
        return results
        
    # Get Tavily's generated answer if available
    tavily_answer = search_results.get("answer", "")
    if tavily_answer:
        logger.info(f"Tavily provided an answer for '{query}': {tavily_answer[:100]}...")
    else:
        logger.warning(f"No answer provided by Tavily for '{query}'")
    
    # Process each result
    for item in items:
        try:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("content", "")
            raw_content = item.get("raw_content", "")
            score = item.get("score", 0.5)  # Tavily provides a relevance score
            
            # Ensure raw_content is never None to prevent Pydantic validation errors
            if raw_content is None:
                raw_content = ""
                logger.warning(f"Received None for raw_content from Tavily for URL: {url}. Using empty string instead.")
            
            # Skip if URL is empty or should be skipped
            if not url or should_skip_domain(url) or is_likely_binary_file(url):
                continue
            
            # Standardize URL
            url = standardize_url(url)
            
            # Blend Tavily's score with our domain priority
            domain_score = get_domain_priority(url)
            combined_score = (domain_score * 0.3) + (score * 0.7)  # Weigh Tavily's score more heavily
            
            # If this is the top result and Tavily provided an answer, use it as processed content
            processed_content = snippet
            if tavily_answer and item == items[0]:  # First/top result
                processed_content = tavily_answer
            
            # Create WebContent object
            web_content = WebContent(
                url=url,
                title=title or "No title",  # Ensure title is never an empty string
                snippet=snippet or "No snippet available",  # Ensure snippet is never an empty string
                raw_content=raw_content,  # Now guaranteed to be a string, never None
                processed_content=processed_content,  # Use Tavily's answer for top result
                score=combined_score,  # Blended score
                query=query  # Add query here
            )
            
            results.append(web_content)
            
        except Exception as e:
            logger.error(f"Error processing Tavily result: {e}")
            continue
    
    return results

async def process_tavily_content(
    content: WebContent,
    term: str,
    llm,
    system_prompt: Optional[str] = None,
    skip_verification: bool = False,
    use_cache: bool = True,
    min_score: float = 1.3,
    skip_summarization: bool = False,
) -> Dict[str, Any]:
    """
    Process a single WebContent item using LLM and verification.
    
    Args:
        content: WebContent object to process
        term: The term this content belongs to
        llm: LLM instance for processing
        system_prompt: System prompt for LLM
        skip_verification: Whether to skip verification
        use_cache: Whether to use content cache
        min_score: Minimum score threshold for verification
        skip_summarization: Whether to skip LLM summarization
        
    Returns:
        Dictionary with processed content and verification info
    """
    try:
        # Ensure raw_content and other required fields are never None
        if content.raw_content is None:
            logger.warning(f"raw_content is None for URL: {content.url}. Using empty string.")
            content.raw_content = ""
            
        if content.title is None:
            content.title = "No title"
            
        if content.snippet is None:
            content.snippet = "No snippet available"
            
        if content.processed_content is None:
            content.processed_content = "No content available"

        # Fast path for very small content - probably not useful
        if content.raw_content and len(content.raw_content.strip()) < 100:
            logger.debug(f"Content too small from {content.url}, using snippet")
            # Convert to dict with minimal processing
            content_dict = content.model_dump()
            content_dict["processed_content"] = content.snippet if not skip_summarization else ""
            content_dict["is_verified"] = False
            content_dict["verification_reason"] = "Content too short to be useful"
            content_dict["score"] = float(content.score * 0.5)  # Penalize small content
            content_dict["term"] = term
            return content_dict
            
        # Use default system prompt if none provided
        if not system_prompt:
            system_prompt = """You are an expert in extractive summarization for academic and technical concepts.
Your task is to identify and extract ONLY the most relevant sentences from the source text that directly explain the concept.
DO NOT paraphrase, rewrite, or generate new content.
DO NOT add your own explanations or interpretations.
ONLY extract 1-3 of the most relevant sentences from the original text that best define or explain the concept.
If no relevant sentences exist in the text, respond with "No relevant information found."
"""

        # First check the educational score of the raw content
        raw_edu_score = await get_educational_score_async(content.raw_content)
        logger.debug(f"Educational score for raw content from {content.url}: {raw_edu_score:.2f}/5.0")
        
        # Define processed_content and tavily_answer_available variables
        processed_content = ""
        tavily_answer_available = content.processed_content != content.snippet
        
        # If summarization should be skipped, use empty processed content
        if skip_summarization:
            logger.debug(f"Skipping summarization for {content.url} as requested")
            processed_content = ""
        else:
            # Determine if we should use LLM processing
            skip_llm = tavily_answer_available and raw_edu_score >= 2.5
            
            if skip_llm:
                # Use the Tavily-provided answer directly, avoiding an LLM call
                logger.debug(f"Using Tavily's answer for {content.url}, skipping LLM processing")
                processed_content = content.processed_content
            elif raw_edu_score >= 1.3:
                # Process with LLM if educational score is good enough
                # Create a cache key based on content hash
                cache_key = hashlib.md5(content.raw_content.encode()).hexdigest()
                
                # Check cache if enabled
                if use_cache and cache_key in content_cache:
                    processed_content = content_cache[cache_key]
                    logger.debug(f"Using cached processed content for {content.url}")
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
{content.raw_content}

Return ONLY the abstractive summary about "{term}", with no additional text or commentary."""

                    # Direct processing with LLM
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
                processed_content = content.snippet
                logger.debug(f"Using snippet as processed content for {content.url} (raw edu score: {raw_edu_score:.2f}/5.0)")
        
        # Skip verification if requested or if raw score was below threshold
        if skip_verification or raw_edu_score < 1.3:
            # For below-threshold content, always mark as unverified
            if raw_edu_score < 1.3:
                is_verified = False
                reason = f"Content skipped due to low educational quality score: {raw_edu_score:.2f}/5.0"
                edu_score_final = raw_edu_score
                base_edu_score = edu_score_final # Set base score for consistency
            else:
                # For websites with high domain priority, consider them verified automatically when skipping
                is_verified = True
                reason = "Verification skipped, content accepted"
                edu_score_final = 3.0  # Neutral score
                base_edu_score = edu_score_final # Set base score for consistency
        else:
            # Verify content quality based on processed content or snippet if summarization is skipped
            content_to_verify = processed_content if not skip_summarization else content.snippet
            is_verified, reason, edu_score_final, base_edu_score = await verify_content_async(
                content.url, content_to_verify, min_score=min_score
            )
        
        # Combine domain priority with educational score - boost score if we have a Tavily answer
        tavily_boost = 0.1 if tavily_answer_available else 0
        final_score = (content.score * 0.4) + (edu_score_final / 5.0 * 0.6) + tavily_boost
        
        # Prepare result
        content_dict = content.model_dump()
        # Truncate raw and processed content
        content_dict["raw_content"] = truncate_content(content_dict.get("raw_content", ""))
        content_dict["processed_content"] = truncate_content(processed_content) if not skip_summarization else ""
        content_dict["is_verified"] = bool(is_verified)
        content_dict["verification_reason"] = reason
        content_dict["score"] = float(final_score)
        content_dict["term"] = term
        content_dict["used_tavily_answer"] = skip_llm if not skip_summarization else False  # Track if we used Tavily's answer
        content_dict["educational_score"] = float(edu_score_final) # Store the final boosted score
        
        return content_dict
        
    except Exception as e:
        logger.error(f"Error processing content from {content.url}: {e}")
        # Return basic content with error information
        content_dict = content.model_dump()
        content_dict["processed_content"] = content.snippet if not skip_summarization else ""
        content_dict["is_verified"] = False
        content_dict["verification_reason"] = f"Error during processing: {str(e)}"
        content_dict["score"] = float(content.score * 0.5)  # Penalize due to error
        content_dict["term"] = term
        return content_dict

async def process_tavily_content_batch(
    contents: List[WebContent],
    term: str,
    llm,
    system_prompt: Optional[str] = None,
    skip_verification: bool = False,
    use_cache: bool = True,
    min_score: float = 1.3,
    skip_summarization: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process a batch of WebContent items.
    
    Args:
        contents: List of WebContent objects to process
        term: The term these contents belong to
        llm: LLM instance for processing
        system_prompt: System prompt for LLM
        skip_verification: Whether to skip verification
        use_cache: Whether to use content cache
        min_score: Minimum score threshold for verification
        skip_summarization: Whether to skip LLM summarization
        
    Returns:
        List of dictionaries with processed content and verification info
    """
    results = []
    tavily_answer_count = 0
    llm_call_count = 0
    
    # Process each content item
    for content in contents:
        try:
            # Validate content fields before processing
            if not isinstance(content, WebContent):
                logger.warning(f"Skipping non-WebContent item for term '{term}'")
                continue
                
            # Fix any None values in required string fields
            for field in ["raw_content", "title", "snippet", "processed_content"]:
                if getattr(content, field, None) is None:
                    logger.warning(f"Field '{field}' is None for URL: {content.url}. Setting to empty string.")
                    setattr(content, field, "")
            
            # Process the content
            processed_content = await process_tavily_content(
                content,
                term,
                llm,
                system_prompt,
                skip_verification,
                use_cache,
                min_score,
                skip_summarization,
            )
            
            if processed_content:
                # Track if Tavily's answer was used
                if processed_content.get("used_tavily_answer", False):
                    tavily_answer_count += 1
                else:
                    llm_call_count += 1
                    
                results.append(processed_content)
        except Exception as e:
            logger.error(f"Error processing Tavily content: {e}")
            # Continue with the next content item
            continue
    
    # Log statistics about LLM usage
    if not skip_summarization and tavily_answer_count > 0:
        logger.info(f"Used Tavily's answer for {tavily_answer_count} out of {tavily_answer_count + llm_call_count} content items ({tavily_answer_count/(tavily_answer_count + llm_call_count)*100:.1f}%)")
        logger.info(f"Saved {tavily_answer_count} LLM calls for term '{term}'")
    
    return results

async def search_tavily_for_term(
    term: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    settings: SearchSettings,
    context_by_concept: Optional[Dict[str, List[str]]] = None,
) -> List[WebContent]:
    """
    Search for content for a single term using Tavily.
    
    Args:
        term: The term to search for
        session: aiohttp session for making requests
        semaphore: Semaphore for controlling concurrent requests
        settings: Search settings
        context_by_concept: Dictionary mapping concepts to context strings
        
    Returns:
        List of WebContent objects
    """
    # Generate better semantic search queries
    search_queries = [
        f"{term} wikipedia",
    ]
    
    # Add context-based queries if available
    if context_by_concept:
        # Look for exact match
        if term in context_by_concept:
            for context in context_by_concept[term]:
                search_queries.append(f"what is {term} in {context}")
        else:
            # Try to find partial matches
            for concept, contexts in context_by_concept.items():
                if term.lower() in concept.lower() or concept.lower() in term.lower():
                    logger.info(f"Found partial match between '{term}' and '{concept}'")
                    for context in contexts:
                        search_queries.append(f"what is {term} in {context}")
    
    all_contents = []
    seen_urls = set()
    
    # Process each query in parallel
    tasks = []
    for query in search_queries:
        logger.info(f"Searching Tavily for: '{query}'")
        task = asyncio.create_task(search_tavily(
            query,
            session,
            semaphore,
            search_depth="advanced" if "what is" in query else "basic",
            max_results=8,  # Slightly reduced from 10 to focus on quality
            show_progress=settings.show_progress,
        ))
        tasks.append((query, task))
    
    # Wait for all search tasks to complete
    for query, task in tasks:
        try:
            result = await task
            
            # Process this query's search results
            contents = await process_tavily_results(result, query)
            
            # Add non-duplicate results to the overall collection
            for content in contents:
                try:
                    # Add additional validation to prevent None values
                    for field in ["raw_content", "title", "snippet", "processed_content"]:
                        if getattr(content, field, None) is None:
                            logger.warning(f"Field '{field}' is None for URL: {content.url}. Setting to empty string.")
                            setattr(content, field, "")
                            
                    # Add only if URL not seen before
                    if content.url not in seen_urls:
                        seen_urls.add(content.url)
                        all_contents.append(content)
                except Exception as e:
                    logger.error(f"Error validating content: {e}")
                    continue
                    
            logger.info(f"Query '{query}' returned {len(contents)} results")
            
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {e}")
    
    # Prioritize results - first by score, then by whether they have Tavily answers
    all_contents.sort(key=lambda x: x.score, reverse=True)
    
    logger.info(f"Found {len(all_contents)} unique results for term '{term}'")
    
    return all_contents

async def tavily_search_web_content(
    concepts: List[str],
    settings: SearchSettings,
) -> Dict[str, List[WebContent]]:
    """
    Search web content for a list of concepts using Tavily API.
    
    Args:
        concepts: List of concepts to search for
        settings: Search settings
        
    Returns:
        Dictionary mapping concepts to lists of WebContent objects
    """
    # Initialize results
    content_by_term = {}
    
    # Load context from CSV if provided
    context_by_concept = {}
    if settings.context_csv_file and settings.context_column:
        logger.info(f"Loading context from CSV: {settings.context_csv_file}")
        context_by_concept = load_context_from_csv(
            settings.context_csv_file,
            settings.concept_column,
            settings.context_column
        )
    
    # Configure aiohttp session with custom timeout and connection settings
    timeout = aiohttp.ClientTimeout(total=TAVILY_TIMEOUT, connect=20, sock_connect=20, sock_read=TAVILY_TIMEOUT)
    tcp_connector = aiohttp.TCPConnector(
        limit=settings.max_concurrent_requests,
        ssl=False,  # Disable SSL verification for troubleshooting
        force_close=True,  # Close connections after use to prevent hanging
        enable_cleanup_closed=True  # Clean up closed connections
    )
    
    session_kwargs = {
        "timeout": timeout,
        "connector": tcp_connector,
        "raise_for_status": False,  # Handle status manually for better error messages
        "headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    }
    
    logger.info(f"Initializing aiohttp session with timeout={TAVILY_TIMEOUT}s, max_connections={settings.max_concurrent_requests}")
    print(f"✓ Initializing network connection with timeout={TAVILY_TIMEOUT}s")
    
    # Use this session configuration instead of default
    async with aiohttp.ClientSession(**session_kwargs) as session:
        # Create semaphore for rate limiting
        semaphore = Semaphore(settings.max_concurrent_requests)
        
        # Process terms in batches
        for batch_start in range(0, len(concepts), settings.batch_size):
            batch_concepts = concepts[batch_start:batch_start + settings.batch_size]
            
            logger.info(f"Processing batch of {len(batch_concepts)} terms")
            batch_start_time = time.time()
            
            # Create tasks for searching each term
            tasks = []
            for term in batch_concepts:
                task = asyncio.create_task(search_tavily_for_term(
                    term,
                    session,
                    semaphore,
                    settings,
                    context_by_concept,
                ))
                tasks.append((term, task))
            
            # Wait for all tasks to complete
            for term, task in tasks:
                try:
                    results = await task
                    content_by_term[term] = results
                    logger.info(f"Found {len(results)} results for term '{term}'")
                except Exception as e:
                    logger.error(f"Error searching for term '{term}': {e}")
                    content_by_term[term] = []
            
            # Log batch processing time
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch processing completed in {batch_time:.2f}s")
    
    return content_by_term

async def tavily_process_all_content(
    content_by_term: Dict[str, List[WebContent]],
    settings: SearchSettings,
    llm,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all content for all terms using LLM and verification.
    
    Args:
        content_by_term: Dictionary mapping terms to lists of WebContent objects
        settings: Search settings
        llm: LLM instance for processing
        
    Returns:
        Dictionary mapping terms to lists of processed content dictionaries
    """
    results = {}
    total_content_count = 0
    total_tavily_answers = 0
    
    # Process each term
    for term, contents in content_by_term.items():
        if not contents:
            logger.info(f"No content to process for term '{term}'")
            results[term] = []
            continue
            
        logger.info(f"Processing {len(contents)} content items for term '{term}'")
        term_tavily_count = 0
        
        try:
            # Process content in batch
            processed_contents = await process_tavily_content_batch(
                contents,
                term,
                llm,
                settings.system_prompt,
                settings.skip_verification,
                settings.use_cache,
                settings.min_score,
                settings.skip_summarization,
            )
            
            # Count how many used Tavily's answer
            term_tavily_count = sum(1 for c in processed_contents if c.get("used_tavily_answer", False))
            total_tavily_answers += term_tavily_count
            total_content_count += len(processed_contents)
            
            # Store results
            results[term] = processed_contents
            logger.info(f"Processed {len(processed_contents)} content items for term '{term}'")
            
            # Clean up used_tavily_answer field before returning results
            for content in processed_contents:
                if "used_tavily_answer" in content:
                    del content["used_tavily_answer"]
                    
        except Exception as e:
            logger.error(f"Error processing content for term '{term}': {e}")
            # Add an empty list as a fallback
            results[term] = []
    
    # Log overall statistics
    if total_content_count > 0 and not settings.skip_summarization:
        savings_percent = (total_tavily_answers / total_content_count) * 100
        logger.info(f"OVERALL STATS: Used Tavily's answers for {total_tavily_answers} out of {total_content_count} content items ({savings_percent:.1f}%)")
        logger.info(f"Saved a total of {total_tavily_answers} LLM calls across all terms")
    
    return results 