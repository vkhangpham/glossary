"""
Firecrawl SDK-based web mining module for academic glossary extraction.
Replaces the complex HTML parsing pipeline with Firecrawl's AI-powered extraction.
"""

# Import run_async_safely for handling event loop conflicts
try:
    from generate_glossary.utils.llm import run_async_safely
except ImportError:
    # Fallback for standalone execution
    def run_async_safely(async_func, *args, **kwargs):
        """Fallback implementation for standalone execution."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, run in a thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                return future.result()
        except RuntimeError:
            # No running loop, we can use asyncio.run directly
            return asyncio.run(async_func(*args, **kwargs))

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from generate_glossary.utils.error_handler import (
    ExternalServiceError, handle_error, processing_context
)
from generate_glossary.utils.logger import get_logger, log_processing_step, set_correlation_id, log_with_context

# Load environment variables
load_dotenv()

# Get enhanced logger
logger = get_logger(__name__)

# Import failure tracker
try:
    from generate_glossary.utils.failure_tracker import save_failure
except ImportError:
    # Fallback for standalone execution
    def save_failure(module, function, error_type, error_message, context=None, failure_dir=None):
        """Fallback implementation that just logs."""
        logger.warning(f"Failure in {module}.{function}: {error_type}: {error_message}")

# Initialize Firecrawl client
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    logger.warning("FIRECRAWL_API_KEY not found in environment. Set it to use Firecrawl.")
    
# Configuration is now centralized - these imports provide access to all constants
from generate_glossary.config import get_mining_config

class ConceptDefinition(BaseModel):
    """Schema for extracted academic concept definitions."""
    concept: str = Field(description="The academic term or concept")
    definition: str = Field(description="Clear, comprehensive definition")
    context: str = Field(description="Academic field or domain")
    key_points: List[str] = Field(default=[], description="Key characteristics")
    related_concepts: List[str] = Field(default=[], description="Related terms")
    source_quality: str = Field(default="general", description="Quality: authoritative/reliable/general")

class WebResource(BaseModel):
    """Schema for web resources containing definitions."""
    url: str
    title: str = ""
    definitions: List[ConceptDefinition] = Field(default_factory=list)
    domain: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        from urllib.parse import urlparse
        self.domain = urlparse(self.url).netloc

def initialize_firecrawl() -> Optional[FirecrawlApp]:
    """Initialize Firecrawl client with API key."""
    with processing_context("initialize_firecrawl") as correlation_id:
        if not FIRECRAWL_API_KEY:
            error_msg = "Firecrawl API key not configured"
            logger.error(error_msg)
            handle_error(
                ExternalServiceError(error_msg, service="firecrawl"),
                context={},
                operation="firecrawl_initialization"
            )
            return None
        
        try:
            app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
            logger.info("Firecrawl client initialized successfully")
            return app
        except Exception as e:
            handle_error(
                ExternalServiceError(f"Failed to initialize Firecrawl: {e}", service="firecrawl"),
                context={},
                operation="firecrawl_initialization"
            )
            logger.error(f"Failed to initialize Firecrawl: {e}")
            return None

def scrape_urls_with_cache(
    app: FirecrawlApp,
    urls: List[str],
    use_summary: bool = False,
    max_age: int = 172800000  # 2 days in milliseconds
) -> Dict[str, Any]:
    """
    Scrape URLs using Firecrawl's v2 scrape endpoint with caching.
    
    Args:
        app: Firecrawl client instance
        urls: List of URLs to scrape
        use_summary: Whether to use summary format for concise content
        max_age: Maximum age for cached content in milliseconds (default: 2 days)
        
    Returns:
        Dictionary mapping URLs to scraped content
    """
    with processing_context("scrape_urls_batch") as correlation_id:
        log_processing_step(
            logger,
            "scrape_urls_batch",
            "started", 
            {
                "urls_count": len(urls),
                "use_summary": use_summary,
                "max_age": max_age
            },
            correlation_id=correlation_id
        )
        
        results = {}
        
        try:
            for url in urls:
                try:
                    # Determine formats to extract
                    formats = ["summary"] if use_summary else ["markdown", "links"]
                    
                    # Scrape with v2 features
                    scrape_params = {
                        "url": url,
                        "formats": formats
                    }
                    
                    # Try v2 scrape with caching and optimizations
                    try:
                        result = app.scrape(
                            url=url,
                            formats=formats,
                            maxAge=max_age,  # Use cached content if available
                            blockAds=True,  # Block ads by default
                            skipTlsVerification=True,  # Skip TLS for faster scraping
                            removeBase64Images=True,  # Remove base64 images for smaller payload
                            onlyMainContent=True  # Focus on main content
                        )
                    except (TypeError, AttributeError):
                        # Fallback to standard scrape
                        result = app.scrape(**scrape_params)
                    
                    # Store result
                    if result:
                        results[url] = result
                        log_with_context(logger, logging.DEBUG, f"Successfully scraped {url} (cached: {result.get('fromCache', False)})", correlation_id=correlation_id)
                    
                except Exception as e:
                    handle_error(
                        ExternalServiceError(f"Failed to scrape {url}: {e}", service="firecrawl"),
                        context={
                            "url": url,
                            "use_summary": use_summary,
                            "max_age": max_age
                        },
                        operation="firecrawl_scrape_single_url"
                    )
                    log_with_context(logger, logging.WARNING, f"Failed to scrape {url}: {e}", correlation_id=correlation_id)
                    save_failure(
                        module="generate_glossary.mining.firecrawl",
                        function="scrape_urls_with_cache",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        context={"url": url, "use_summary": use_summary}
                    )
                    # Continue processing other URLs
                    results[url] = {"error": str(e)}
            
            log_processing_step(
                logger,
                "scrape_urls_batch",
                "completed",
                {
                    "urls_processed": len(urls),
                    "successful_scrapes": len([r for r in results.values() if "error" not in r]),
                    "failed_scrapes": len([r for r in results.values() if "error" in r])
                },
                correlation_id=correlation_id
            )
            
            return results
            
        except Exception as e:
            handle_error(
                e,
                context={
                    "urls_count": len(urls),
                    "use_summary": use_summary,
                    "max_age": max_age
                },
                operation="scrape_urls_batch",
                reraise=True
            )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True
)
def search_concept_firecrawl(app: FirecrawlApp, concept: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for a concept using Firecrawl's search endpoint with v2 features.
    
    Args:
        app: Firecrawl client instance
        concept: The concept to search for
        limit: Maximum number of results
        
    Returns:
        List of search results
    """
    with processing_context(f"search_concept_{concept}") as correlation_id:
        log_processing_step(
            logger,
            "search_concept_firecrawl",
            "started",
            {
                "concept": concept,
                "limit": limit
            },
            correlation_id=correlation_id
        )
        
        # Build academic-focused query before try block to ensure it's always available
        query = f'"{concept}" definition explanation academic OR wikipedia OR edu OR arxiv'
        
        try:
            
            log_with_context(logger, logging.INFO, f"Searching for: {concept}", correlation_id=correlation_id)
            
            # Use Firecrawl search with v2 features
            # Try to use research category if available
            search_params = {
                "query": query,
                "limit": limit
            }
            
            # Try v2 search with categories (if supported by SDK version)
            try:
                # Attempt to use research category for better academic results
                results = app.search(
                    query=query,
                    limit=limit,
                    sources=[{"type": "web"}],  # v2 format
                    categories=["research"]  # Filter for research/academic content
                )
            except (TypeError, AttributeError):
                # Fallback to standard search if v2 features not available
                results = app.search(**search_params)
            
            # Extract the results
            search_results = []
            if isinstance(results, dict):
                if 'web' in results:
                    # v2 format with categorized results
                    search_results = results['web']
                elif 'data' in results:
                    search_results = results['data']
                else:
                    search_results = []
            elif isinstance(results, list):
                search_results = results
            else:
                log_with_context(logger, logging.WARNING, f"Unexpected search response format: {type(results)}", correlation_id=correlation_id)
                search_results = []
            
            log_processing_step(
                logger,
                "search_concept_firecrawl",
                "completed",
                {
                    "concept": concept,
                    "results_count": len(search_results)
                },
                correlation_id=correlation_id
            )
            
            return search_results
                
        except (ConnectionError, TimeoutError) as e:
            handle_error(
                ExternalServiceError(f"Search failed for '{concept}': {e}", service="firecrawl"),
                context={
                    "concept": concept,
                    "limit": limit,
                    "query": query
                },
                operation="search_concept_firecrawl"
            )
            log_with_context(logger, logging.ERROR, f"Search failed for '{concept}': {e}", correlation_id=correlation_id)
            save_failure(
                module="generate_glossary.mining.firecrawl",
                function="search_concept_firecrawl",
                error_type=type(e).__name__,
                error_message=str(e),
                context={"concept": concept, "limit": limit}
            )
            raise
        except Exception as e:
            handle_error(
                e,
                context={
                    "concept": concept,
                    "limit": limit,
                    "query": query
                },
                operation="search_concept_firecrawl"
            )
            log_with_context(logger, logging.ERROR, f"Search failed for '{concept}': {e}", correlation_id=correlation_id)
            save_failure(
                module="generate_glossary.mining.firecrawl",
                function="search_concept_firecrawl",
                error_type=type(e).__name__,
                error_message=str(e),
                context={"concept": concept, "limit": limit}
            )
            return []

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True
)
def extract_definitions_firecrawl(
    app: FirecrawlApp, 
    urls: List[str], 
    concept: str
) -> List[WebResource]:
    """
    Extract structured definitions from URLs using Firecrawl's extract endpoint with v2 features.
    
    Args:
        app: Firecrawl client instance
        urls: List of URLs to extract from
        concept: The concept to extract definitions for
        
    Returns:
        List of WebResource objects with extracted definitions
    """
    with processing_context(f"extract_definitions_{concept}") as correlation_id:
        log_processing_step(
            logger,
            "extract_definitions_firecrawl",
            "started",
            {
                "concept": concept,
                "urls_count": len(urls)
            },
            correlation_id=correlation_id
        )
        
        if not urls:
            return []
        
        try:
            # Build extraction prompt - enhanced for multi-entity extraction
            prompt = f"""
            Extract comprehensive information about the academic concept "{concept}".
            
            For each occurrence, extract:
            1. A clear, authoritative definition
            2. The academic or technical context
            3. Key characteristics or properties (as a list)
            4. Related concepts mentioned
            5. Assess the source quality (authoritative/reliable/general)
            
            Focus on academic and technical definitions only.
            Consider multiple perspectives if available.
            """
            
            # Define schema for extraction - enhanced for better validation
            schema = {
                "type": "object",
                "properties": {
                    "definitions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "concept": {"type": "string"},
                                "definition": {"type": "string"},
                                "context": {"type": "string"},
                                "key_points": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "related_concepts": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "source_quality": {
                                    "type": "string",
                                    "enum": ["authoritative", "reliable", "general"]
                                }
                            },
                            "required": ["concept", "definition", "context", "source_quality"]
                        }
                    }
                },
                "required": ["definitions"]
            }
            
            log_with_context(logger, logging.INFO, f"Extracting definitions from {len(urls)} URLs for '{concept}'", correlation_id=correlation_id)
            
            # Try to use v2 extract features
            extract_params = {
                "urls": urls,
                "prompt": prompt,
                "schema": schema
            }
            
            # Add v2 features if supported
            try:
                # Try with enhanced v2 parameters
                result = app.extract(
                    urls=urls,
                    prompt=prompt,
                    schema=schema,
                    enableWebSearch=True,  # Enable web search for additional context
                    allowExternalLinks=False,  # Stay focused on provided URLs
                    includeSubdomains=False  # Don't crawl subdomains
                )
            except (TypeError, AttributeError):
                # Fallback to standard extract if v2 features not available
                result = app.extract(**extract_params)
            
            # Process results
            resources = []
            
            # Handle different response formats
            extracted_data = {}
            if isinstance(result, dict):
                if 'data' in result:
                    # New format: {'data': [...]}
                    for item in result['data']:
                        if 'url' in item and 'extracted' in item:
                            extracted_data[item['url']] = item['extracted']
                elif 'results' in result:
                    # Alternative format
                    for item in result['results']:
                        if 'url' in item and 'extracted' in item:
                            extracted_data[item['url']] = item['extracted']
                else:
                    # Direct URL mapping
                    extracted_data = result
            
            # Convert to WebResource objects
            for url, data in extracted_data.items():
                if data and 'definitions' in data:
                    definitions = []
                    for def_data in data['definitions']:
                        definitions.append(ConceptDefinition(**def_data))
                    
                    resources.append(WebResource(
                        url=url,
                        title=f"Content from {url}",
                        definitions=definitions
                    ))
            
            log_processing_step(
                logger,
                "extract_definitions_firecrawl",
                "completed",
                {
                    "concept": concept,
                    "resources_extracted": len(resources)
                },
                correlation_id=correlation_id
            )
            
            log_with_context(logger, logging.INFO, f"Extracted {len(resources)} resources with definitions", correlation_id=correlation_id)
            return resources
            
        except (ConnectionError, TimeoutError) as e:
            handle_error(
                ExternalServiceError(f"Extraction failed for concept '{concept}': {e}", service="firecrawl"),
                context={
                    "concept": concept,
                    "urls_count": len(urls)
                },
                operation="extract_definitions_firecrawl"
            )
            log_with_context(logger, logging.ERROR, f"Extraction failed for concept '{concept}': {e}", correlation_id=correlation_id)
            save_failure(
                module="generate_glossary.mining.firecrawl",
                function="extract_definitions_firecrawl",
                error_type=type(e).__name__,
                error_message=str(e),
                context={"concept": concept, "urls_count": len(urls)}
            )
            raise
        except Exception as e:
            handle_error(
                e,
                context={
                    "concept": concept,
                    "urls_count": len(urls)
                },
                operation="extract_definitions_firecrawl"
            )
            log_with_context(logger, logging.ERROR, f"Extraction failed for concept '{concept}': {e}", correlation_id=correlation_id)
            save_failure(
                module="generate_glossary.mining.firecrawl",
                function="extract_definitions_firecrawl",
                error_type=type(e).__name__,
                error_message=str(e),
                context={"concept": concept, "urls_count": len(urls)}
            )
            return []

async def mine_concept_async(app: FirecrawlApp, concept: str) -> Dict[str, Any]:
    """
    Mine web content for a single concept asynchronously.
    
    Args:
        app: Firecrawl client instance
        concept: The concept to mine
        
    Returns:
        Dictionary with concept definition and sources
    """
    with processing_context(f"mine_concept_async:{concept}") as correlation_id:
        log_processing_step(
            logger,
            "mine_concept_async",
            "started",
            {
                "concept": concept
            },
            correlation_id=correlation_id
        )
        
        try:
            log_with_context(logger, logging.INFO, f"Mining content for: {concept}", correlation_id=correlation_id)
            
            # Step 1: Search for relevant URLs
            search_results = await asyncio.to_thread(
                search_concept_firecrawl, app, concept, limit=5
            )
            
            if not search_results:
                log_with_context(logger, logging.WARNING, f"No search results for: {concept}", correlation_id=correlation_id)
                return {
                    "concept": concept,
                    "resources": [],
                    "summary": None,
                    "error": "No search results found"
                }
            
            # Extract URLs from search results
            mining_config = get_mining_config()
            urls = []
            for result in search_results[:mining_config.max_urls_per_concept]:
                if isinstance(result, dict) and 'url' in result:
                    urls.append(result['url'])
                elif isinstance(result, str):
                    urls.append(result)
            
            if not urls:
                return {
                    "concept": concept,
                    "resources": [],
                    "summary": None,
                    "error": "No valid URLs found"
                }
            
            log_with_context(logger, logging.INFO, f"Found {len(urls)} URLs for {concept}", correlation_id=correlation_id)
            
            # Step 2: Extract definitions from URLs
            resources = await asyncio.to_thread(
                extract_definitions_firecrawl, app, urls, concept
            )
            
            # Step 3: Aggregate and score results
            result = {
                "concept": concept,
                "resources": [],
                "summary": None
            }
            
            # Process resources and create aggregated summary
            all_definitions = []
            for resource in resources:
                # Filter for quality
                quality_definitions = [
                    d for d in resource.definitions 
                    if d.source_quality in ["authoritative", "reliable"]
                ]
                
                if quality_definitions:
                    result["resources"].append({
                        "url": resource.url,
                        "domain": resource.domain,
                        "definitions": [d.model_dump() for d in quality_definitions]
                    })
                    all_definitions.extend(quality_definitions)
            
            # Create best summary from authoritative sources
            if all_definitions:
                # Prioritize authoritative sources
                auth_defs = [d for d in all_definitions if d.source_quality == "authoritative"]
                best_def = auth_defs[0] if auth_defs else all_definitions[0]
            
                # Aggregate related concepts
                all_related = set()
                for d in all_definitions:
                    all_related.update(d.related_concepts)
                
                result["summary"] = {
                    "definition": best_def.definition,
                    "context": best_def.context,
                    "key_points": best_def.key_points,
                    "related_concepts": list(all_related)[:5],  # Top 5 related
                    "source_count": len(result["resources"])
                }
            
            log_processing_step(
                logger,
                "mine_concept_async",
                "completed",
                {
                    "concept": concept,
                    "resources_extracted": len(result['resources']),
                    "definitions_found": len(all_definitions)
                },
                correlation_id=correlation_id
            )
            
            log_with_context(logger, logging.INFO, f"Extracted {len(result['resources'])} quality resources for {concept}", correlation_id=correlation_id)
            return result
            
        except Exception as e:
            handle_error(
                e,
                context={
                    "concept": concept
                },
                operation="mine_concept_async",
                reraise=True
            )

def batch_scrape_urls(
    app: FirecrawlApp,
    urls: List[str],
    max_concurrent: int = 10
) -> Dict[str, Any]:
    """
    Use Firecrawl's batch scrape endpoint for efficient parallel scraping.
    
    Args:
        app: Firecrawl client instance
        urls: List of URLs to scrape
        max_concurrent: Maximum concurrent scrapes
        
    Returns:
        Dictionary with scraping results
    """
    with processing_context("batch_scrape_urls") as correlation_id:
        log_processing_step(
            logger,
            "batch_scrape_urls", 
            "started",
            {
                "urls_count": len(urls),
                "max_concurrent": max_concurrent
            },
            correlation_id=correlation_id
        )
        
        if not urls:
            return {}
        
        try:
            log_with_context(logger, logging.INFO, f"Batch scraping {len(urls)} URLs", correlation_id=correlation_id)
            
            # Try to use batch_scrape if available
            try:
                # Use v2 batch scrape with optimizations
                result = app.batch_scrape(
                    urls=urls,
                    formats=["markdown", "links"],
                    maxConcurrency=max_concurrent,
                    maxAge=172800000,  # 2 days cache
                    blockAds=True,
                    skipTlsVerification=True,
                    removeBase64Images=True,
                    onlyMainContent=True
                )
                
                # Wait for completion and get results
                if hasattr(result, 'wait_until_done'):
                    final_result = result.wait_until_done()
                    
                    log_processing_step(
                        logger,
                        "batch_scrape_urls",
                        "completed",
                        {
                            "urls_processed": len(urls),
                            "successful_scrapes": len([r for r in final_result.get('data', []) if 'error' not in r])
                        },
                        correlation_id=correlation_id
                    )
                    
                    return final_result
                else:
                    log_processing_step(
                        logger,
                        "batch_scrape_urls",
                        "completed",
                        {
                            "urls_processed": len(urls),
                            "result_type": "direct_return"
                        },
                        correlation_id=correlation_id
                    )
                    return result
                    
            except (AttributeError, TypeError) as e:
                handle_error(
                    e,
                    context={
                        "urls_count": len(urls),
                        "max_concurrent": max_concurrent,
                        "fallback": "sequential_scraping"
                    },
                    operation="batch_scrape_firecrawl"
                )
                log_with_context(logger, logging.WARNING, f"Batch scrape not available, falling back to sequential: {e}", correlation_id=correlation_id)
                # Fallback to sequential scraping
                return scrape_urls_with_cache(app, urls)
                
        except Exception as e:
            handle_error(
                e,
                context={
                    "urls_count": len(urls),
                    "max_concurrent": max_concurrent
                },
                operation="batch_scrape_urls",
                reraise=True
            )

async def mine_concepts_batch_async(
    app: FirecrawlApp,
    concepts: List[str],
    max_concurrent: int = None,
    use_batch_scrape: bool = True
) -> Dict[str, Any]:
    """
    Mine web content for multiple concepts in parallel with v2 optimizations.
    
    Args:
        app: Firecrawl client instance
        concepts: List of concepts to mine
        max_concurrent: Maximum concurrent operations
        use_batch_scrape: Whether to use batch scraping for URLs
        
    Returns:
        Dictionary with all results
    """
    mining_config = get_mining_config()
    if max_concurrent is None:
        max_concurrent = mining_config.max_concurrent_operations
    
    results = {}
    
    # Process in batches
    for i in range(0, len(concepts), mining_config.batch_size):
        batch = concepts[i:i + mining_config.batch_size]
        batch_num = i // mining_config.batch_size + 1
        total_batches = (len(concepts) + mining_config.batch_size - 1) // mining_config.batch_size
        
        log_with_context(logger, logging.INFO, f"Processing batch {batch_num}/{total_batches} ({len(batch)} concepts)")
        
        if use_batch_scrape:
            # Collect all URLs for batch scraping
            all_urls = []
            concept_url_map = {}
            
            # First, search for all concepts to get URLs
            for concept in batch:
                search_results = await asyncio.to_thread(
                    search_concept_firecrawl, app, concept, limit=3
                )
                
                urls = []
                for result in search_results[:mining_config.max_urls_per_concept]:
                    if isinstance(result, dict) and 'url' in result:
                        url = result['url']
                        urls.append(url)
                        all_urls.append(url)
                
                concept_url_map[concept] = urls
            
            # Batch scrape all URLs at once
            if all_urls:
                scraped_data = await asyncio.to_thread(
                    batch_scrape_urls, app, all_urls, max_concurrent
                )
                
                # Process results for each concept
                for concept, urls in concept_url_map.items():
                    concept_data = {
                        "concept": concept,
                        "resources": [],
                        "summary": None
                    }
                    
                    # Extract relevant scraped data for this concept
                    for url in urls:
                        if url in scraped_data and not scraped_data[url].get("error"):
                            concept_data["resources"].append({
                                "url": url,
                                "content": scraped_data[url]
                            })
                    
                    results[concept] = concept_data
            
        else:
            # Use original parallel processing
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def mine_with_limit(concept):
                async with semaphore:
                    return await mine_concept_async(app, concept)
            
            tasks = [mine_with_limit(concept) for concept in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for concept, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    log_with_context(logger, logging.ERROR, f"Failed to mine {concept}: {result}")
                    save_failure(
                        module="generate_glossary.mining.firecrawl",
                        function="mine_concepts_batch_async",
                        error_type=type(result).__name__,
                        error_message=str(result),
                        context={
                            "concept": concept,
                            "batch_num": batch_num,
                            "total_batches": total_batches
                        }
                    )
                    # Continue processing - save error but don't stop
                    results[concept] = {
                        "concept": concept,
                        "error": str(result),
                        "resources": [],
                        "summary": None
                    }
                else:
                    results[concept] = result
        
        # Small delay between batches
        if i + mining_config.batch_size < len(concepts):
            await asyncio.sleep(1)
    
    return results

async def mine_concepts_with_firecrawl_async(
    concepts: List[str],
    output_path: Optional[str] = None,
    use_batch_scrape: bool = True,
    use_cache: bool = True,
    max_age: int = 172800000  # 2 days default
) -> Dict[str, Any]:
    """
    Async version of mining concepts using Firecrawl SDK.
    """
    # Initialize Firecrawl
    app = initialize_firecrawl()
    if not app:
        logger.error("Cannot proceed without Firecrawl client")
        return {
            "error": "Firecrawl not configured",
            "results": {},
            "statistics": {"total": len(concepts), "successful": 0, "failed": len(concepts)}
        }
    
    logger.info(f"Starting web mining for {len(concepts)} concepts using Firecrawl SDK v2")
    logger.info(f"Features enabled: batch_scrape={use_batch_scrape}, cache={use_cache}, max_age={max_age/1000/60:.0f}min")
    
    start_time = time.time()
    
    # Run async mining with v2 optimizations
    results = await mine_concepts_batch_async(
        app, 
        concepts,
        use_batch_scrape=use_batch_scrape
    )
    
    # Calculate statistics
    stats = {
        "total_concepts": len(concepts),
        "successful": sum(1 for r in results.values() if r.get("summary")),
        "failed": sum(1 for r in results.values() if "error" in r),
        "total_resources": sum(len(r.get("resources", [])) for r in results.values()),
        "concepts_with_content": sum(1 for r in results.values() if r.get("resources")),
        "processing_time": time.time() - start_time,
        "features_used": {
            "batch_scrape": use_batch_scrape,
            "caching": use_cache,
            "cache_max_age_hours": max_age / 1000 / 60 / 60
        }
    }
    
    logger.info(f"Mining complete in {stats['processing_time']:.1f}s")
    logger.info(f"Success rate: {stats['successful']}/{stats['total_concepts']} ({stats['successful']/stats['total_concepts']*100:.1f}%)")
    
    # Prepare output
    output_data = {
        "results": results,
        "statistics": stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "firecrawl_version": "v2"
    }
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    
    return output_data


def mine_concepts_with_firecrawl(
    concepts: List[str],
    output_path: Optional[str] = None,
    use_batch_scrape: bool = True,
    use_cache: bool = True,
    max_age: int = 172800000  # 2 days default
) -> Dict[str, Any]:
    """
    Main entry point for mining concepts using Firecrawl SDK with v2 features.
    
    Args:
        concepts: List of concepts to mine
        output_path: Optional path to save results
        use_batch_scrape: Whether to use batch scraping (500% faster with v2)
        use_cache: Whether to use cached results (maxAge feature)
        max_age: Maximum age for cached content in milliseconds
        
    Returns:
        Dictionary with all results and statistics
    """
    # Use run_async_safely to handle event loop conflicts
    return run_async_safely(
        mine_concepts_with_firecrawl_async,
        concepts,
        output_path,
        use_batch_scrape,
        use_cache,
        max_age
    )

# Backwards compatibility wrapper
def search_and_extract_batch(
    concepts: List[str],
    settings: Optional[Any] = None
) -> Dict[str, Any]:
    """Backwards compatible wrapper for batch mining."""
    return mine_concepts_with_firecrawl(concepts)

if __name__ == "__main__":
    # Test with sample concepts
    test_concepts = [
        "machine learning",
        "deep learning", 
        "neural networks",
        "natural language processing",
        "computer vision"
    ]
    
    print("\n" + "="*60)
    print("Testing Firecrawl SDK Web Mining")
    print("="*60)
    
    if not FIRECRAWL_API_KEY:
        print("\n⚠️  Warning: FIRECRAWL_API_KEY not set in environment")
        print("Set it with: export FIRECRAWL_API_KEY='your-api-key'")
    else:
        print(f"✅ Firecrawl API key configured")
    
    print(f"\nProcessing {len(test_concepts)} test concepts...")
    
    # Run the mining
    results = mine_concepts_with_firecrawl(
        test_concepts,
        output_path="firecrawl_test_results.json"
    )
    
    # Display results
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    stats = results.get("statistics", {})
    print(f"Total concepts: {stats.get('total_concepts', 0)}")
    print(f"Successful: {stats.get('successful', 0)}")
    print(f"Failed: {stats.get('failed', 0)}")
    print(f"Processing time: {stats.get('processing_time', 0):.1f}s")
    print(f"Total resources: {stats.get('total_resources', 0)}")
    
    # Show sample results
    print("\n" + "-"*60)
    print("Sample Results:")
    print("-"*60)
    
    for concept, data in list(results.get("results", {}).items())[:3]:
        print(f"\n📚 {concept}:")
        if data.get("summary"):
            definition = data["summary"]["definition"]
            print(f"  Definition: {definition[:150]}...")
            print(f"  Context: {data['summary']['context']}")
            print(f"  Sources: {data['summary']['source_count']}")
            print(f"  Key points: {len(data['summary'].get('key_points', []))}")
        else:
            print(f"  Status: {data.get('error', 'No content found')}")