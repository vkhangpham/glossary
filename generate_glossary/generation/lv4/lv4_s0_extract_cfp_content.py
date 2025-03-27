import os
import sys
import time
import json
import asyncio
import aiohttp
import trafilatura
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydash import chunk
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from aiohttp import ClientTimeout
from functools import lru_cache
import hashlib
import chardet
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

# Load environment variables and setup logging
load_dotenv()
logger = setup_logger("lv4.s0")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# API Configuration
HEADERS = {
    "content-type": "application/json",
    "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY", ""),
    "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com",
}

WEB_SEARCH_URL = "https://real-time-web-search.p.rapidapi.com/search"
MAX_CONTENT_LENGTH = 25000  # Maximum words per content
RETRY_ATTEMPTS = 5  # Number of retry attempts for failed requests
REQUEST_WINDOW = 2.0  # Time window for rate limiting (2 seconds to be safe)
MAX_REQUESTS_PER_WINDOW = 10  # Maximum requests per window
MIN_RETRY_WAIT = 4  # Minimum wait time between retries in seconds
MAX_RETRY_WAIT = 10  # Maximum wait time between retries in seconds
MAX_PARALLEL_EXTRACTIONS = 20  # Maximum number of parallel content extractions
MAX_WORKERS = 4  # Number of process pool workers for CPU-intensive tasks

# Common encoding fallbacks
ENCODING_FALLBACKS = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']

# Caching
URL_CONTENT_CACHE = {}
SEARCH_RESULTS_CACHE = {}

class Config:
    """Configuration for CFP content extraction"""
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv3/lv3_s0_venue_names.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv4/lv4_s0_cfp_content.json")
    CACHE_DIR = os.path.join(BASE_DIR, "data/lv4/cache")
    BATCH_SIZE = 10  # Since each venue needs 2 queries (max 20 queries per batch)
    
    # Timeouts
    CONNECT_TIMEOUT = 10  # Connection timeout in seconds
    READ_TIMEOUT = 30    # Socket read timeout in seconds
    TOTAL_TIMEOUT = 60   # Total operation timeout in seconds
    
    # Create cache file path
    @staticmethod
    def get_cache_file_path(key: str) -> str:
        """Get cache file path for a key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(Config.CACHE_DIR, f"{key_hash}.json")

class RateLimiter:
    """Rate limiter to ensure we don't exceed API limits"""
    def __init__(self, max_requests: int, window: float):
        self.max_requests = max_requests
        self.window = window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            # Remove old requests outside the window
            self.requests = [t for t in self.requests if now - t < self.window]
            
            # If we've hit the limit, wait until we can make another request
            if len(self.requests) >= self.max_requests:
                wait_time = self.requests[0] + self.window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.requests = self.requests[1:]  # Remove oldest request
            
            # Add current request
            self.requests.append(now)
            
            # Add small delay between requests even within limit
            await asyncio.sleep(0.2)  # 200ms between requests

def check_cache(key: str) -> Optional[Dict[str, Any]]:
    """Check if content for key is cached"""
    # Check memory cache first
    if key in URL_CONTENT_CACHE:
        return URL_CONTENT_CACHE[key]
        
    # Then check disk cache
    cache_path = Config.get_cache_file_path(key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Update memory cache
                URL_CONTENT_CACHE[key] = data
                return data
        except Exception as e:
            logger.warning(f"Error reading cache for {key}: {str(e)}")
    return None

def save_to_cache(key: str, data: Dict[str, Any]) -> None:
    """Save content to cache"""
    # Update memory cache
    URL_CONTENT_CACHE[key] = data
    
    # Save to disk cache
    cache_path = Config.get_cache_file_path(key)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Error saving to cache for {key}: {str(e)}")

@lru_cache(maxsize=1000)
def should_skip_domain(url: str) -> bool:
    """Check if URL should be skipped (cached for performance)"""
    skip_domains = [
        'latimes.com', 'nytimes.com', 'wsj.com', 'bloomberg.com',
        'facebook.com', 'twitter.com', 'linkedin.com', 'youtube.com',
        'instagram.com', 'google.com', 'pinterest.com', 'reddit.com'
    ]
    return any(domain in url.lower() for domain in skip_domains)

@lru_cache(maxsize=1000)
def normalize_url(url: str) -> str:
    """Normalize URL to avoid duplicates"""
    # Remove trailing slashes
    url = url.rstrip('/')
    
    # Remove common tracking parameters
    for param in ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid']:
        if f'{param}=' in url:
            # Simple parameter removal - for more complex cases use urllib.parse
            parts = url.split('?')
            if len(parts) == 2:
                base, query = parts
                query_params = query.split('&')
                filtered_params = [p for p in query_params if not p.startswith(f'{param}=')]
                if filtered_params:
                    url = f"{base}?{'&'.join(filtered_params)}"
                else:
                    url = base
    
    return url

async def detect_and_decode(response: aiohttp.ClientResponse) -> Tuple[bytes, str]:
    """Detect encoding and get raw bytes from response"""
    # First try to get the encoding from the headers
    content_type = response.headers.get('Content-Type', '')
    encoding = None
    
    # Check if encoding is specified in headers
    if 'charset=' in content_type:
        encoding = content_type.split('charset=')[-1].strip()
        try:
            raw_bytes = await response.read()
            return raw_bytes, encoding
        except:
            encoding = None  # Reset if this fails
    
    # If encoding not in headers or failed, use chardet to detect
    raw_bytes = await response.read()
    
    if not encoding:
        # Detect encoding
        detected = chardet.detect(raw_bytes)
        encoding = detected.get('encoding', 'utf-8')
        
    # Make sure we have a valid encoding
    if not encoding or encoding.lower() == 'none':
        encoding = 'utf-8'  # Default to utf-8
        
    return raw_bytes, encoding

async def extract_content_from_url(
    url: str, 
    session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Extract content from a single URL using trafilatura with fallbacks"""
    # Check if URL should be skipped
    if should_skip_domain(url):
        return None
        
    # Normalize URL
    url = normalize_url(url)
    
    # Check cache
    cache_key = f"url_content_{url}"
    cached_content = check_cache(cache_key)
    if cached_content:
        logger.debug(f"Cache hit for {url}")
        return cached_content.get("content")
    
    # Use semaphore to limit parallel extractions
    async with semaphore:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
            
        try:
            # Rate limiting
            await rate_limiter.acquire()
            
            # Configure timeouts
            timeout = ClientTimeout(
                connect=Config.CONNECT_TIMEOUT,
                sock_read=Config.READ_TIMEOUT,
                total=Config.TOTAL_TIMEOUT
            )
            
            async with session.get(url, headers=headers, timeout=timeout, ssl=False) as response:
                if response.status != 200:
                    return None
                
                # Detect encoding and get raw bytes
                raw_bytes, encoding = await detect_and_decode(response)
                
                # Try to decode using detected encoding
                html_content = None
                successful_encoding = None
                
                # Try detected encoding first, then fallbacks
                encodings_to_try = [encoding] + [e for e in ENCODING_FALLBACKS if e.lower() != encoding.lower()]
                
                for enc in encodings_to_try:
                    try:
                        html_content = raw_bytes.decode(enc)
                        successful_encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error decoding with {enc}: {str(e)}")
                        continue
                
                if not html_content:
                    logger.warning(f"Failed to decode content from {url} with any encoding")
                    return None
                
                logger.debug(f"Successfully decoded {url} using {successful_encoding} encoding")
                
                # Extract content with trafilatura
                content = trafilatura.extract(
                    html_content,
                    include_comments=False,
                    favor_precision=True,
                    include_tables=False,
                    no_fallback=True,
                    include_links=False,
                    include_images=False
                )
                
                # If trafilatura fails, try web_scraper fallback
                if not content:
                    logger.info(f"Trafilatura failed for {url}, trying fallback")
                    # Implement fallback parsing logic
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style tags
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    content = '\n'.join(chunk for chunk in chunks if chunk)
                
                if not content:
                    return None
                    
                # Truncate if too long
                words = content.split()
                if len(words) > MAX_CONTENT_LENGTH:
                    content = ' '.join(words[:MAX_CONTENT_LENGTH])
                
                # Save to cache
                save_to_cache(cache_key, {"content": content, "url": url})
                
                return content
                
        except aiohttp.ClientError as e:
            logger.warning(f"Client error for {url}: {str(e)}")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout error for {url}")
            return None
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

@retry(
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
    before_sleep=lambda retry_state: logger.warning(
        f"Attempt {retry_state.attempt_number} failed. Retrying in {retry_state.next_action.sleep} seconds..."
    )
)
async def web_search_batch(
    queries: List[str], 
    session: aiohttp.ClientSession,
    rate_limiter: RateLimiter
) -> Dict:
    """Perform bulk web search for multiple queries"""
    assert len(queries) <= 20, "Maximum 20 queries per batch allowed"
    
    # Check cache first
    cache_key = f"search_batch_{','.join(sorted(queries))}"
    cached_results = check_cache(cache_key)
    if cached_results:
        logger.debug(f"Cache hit for search batch with {len(queries)} queries")
        return cached_results
    
    payload = {
        "queries": queries,
        "limit": "3",
    }
    
    await rate_limiter.acquire()
    try:
        async with session.post(WEB_SEARCH_URL, json=payload, headers=HEADERS) as response:
            if response.status == 429:  # Too Many Requests
                logger.warning("Rate limit exceeded, waiting before retry")
                # Wait longer than usual for rate limit
                await asyncio.sleep(10)
                raise aiohttp.ClientError("Rate limit exceeded")
                
            if response.status != 200:
                logger.error(f"Search API error: HTTP {response.status}")
                raise aiohttp.ClientError(f"HTTP error {response.status}")
                
            data = await response.json()
            
            if not data.get("data"):
                logger.error("No data in search response")
                raise aiohttp.ClientError("Invalid response format")
                
            # Save to cache
            save_to_cache(cache_key, data)
            return data
            
    except aiohttp.ClientError as e:
        logger.error(f"Error in web search: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in web search: {str(e)}")
        raise aiohttp.ClientError(f"Unexpected error: {str(e)}")

def build_venue_queries(venues: List[str]) -> List[str]:
    """Build search queries for venue CFPs and aims/scope"""
    queries = []
    for venue in venues:
        queries.append(f"{venue} call for papers")
        queries.append(f"{venue} aims and scope")
    return queries

async def process_search_results(
    search_results: Dict, 
    session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore
) -> Dict[str, List[Dict[str, Any]]]:
    """Process search results and extract content"""
    venue_results = {}
    processed_results = {}
    
    # Map results to venues
    for item in search_results.get("data", []):
        query = item.get("query", "").lower()
        results = item.get("results", [])
        
        if not results:
            continue
            
        # Determine venue name from query
        venue = None
        if "call for papers" in query:
            venue = query.replace("call for papers", "").strip()
        elif "aims and scope" in query:
            venue = query.replace("aims and scope", "").strip()
            
        if not venue:
            continue
            
        # Initialize venue data structure
        if venue not in venue_results:
            venue_results[venue] = {}
            
        # Determine search type
        search_type = "aims and scope" if "aims and scope" in query else "call for papers"
        # Take only top 2 results per search type
        venue_results[venue][search_type] = results[:2]
    
    # Create tasks for all venues
    all_tasks = []
    
    for venue, results in venue_results.items():
        for search_type, type_results in results.items():
            for result in type_results:
                url = result.get("url", "")
                if not url:
                    continue
                
                # Create task for content extraction
                task = extract_content_from_url(url, session, rate_limiter, semaphore)
                all_tasks.append((venue, search_type, result, task))
    
    # Process tasks in batches for better progress tracking
    batch_size = 20
    task_batches = [all_tasks[i:i + batch_size] for i in range(0, len(all_tasks), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(task_batches, desc="Extracting content")):
        tasks = [t[3] for t in batch]  # Get the task object
        logger.info(f"Processing batch {batch_idx+1}/{len(task_batches)} with {len(tasks)} URLs")
        
        if tasks:
            contents = await asyncio.gather(*tasks, return_exceptions=True)
            
            for (venue, search_type, result, _), content in zip(batch, contents):
                if isinstance(content, Exception) or not content:
                    continue
                    
                # Create entry
                entry = {
                    "content": content,
                    "url": result.get("url", ""),
                    "domain": result.get("domain", ""),
                    "title_ref": result.get("title", ""),
                    "url_ref": result.get("url", ""),
                    "snippet_ref": result.get("snippet", ""),
                    "search_type": search_type,
                    "topic": None  # Will be filled by step 1
                }
                
                # Add to processed results
                if venue not in processed_results:
                    processed_results[venue] = []
                processed_results[venue].append(entry)
    
    return processed_results

async def process_venues_batch(
    venues: List[str],
    session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore
) -> Dict[str, List[Dict[str, Any]]]:
    """Process a batch of venues"""
    start_time = time.time()
    try:
        # Build queries for the batch
        queries = build_venue_queries(venues)
        logger.info(f"Processing batch with {len(venues)} venues ({len(queries)} queries)")
        
        # Perform search
        search_results = await web_search_batch(queries, session, rate_limiter)
        
        # Process results
        processed_results = await process_search_results(search_results, session, rate_limiter, semaphore)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch processed in {elapsed_time:.2f} seconds with {len(processed_results)} successful venues")
        
        return processed_results
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error processing batch after {elapsed_time:.2f} seconds: {str(e)}")
        return {}

async def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        logger.info("Starting CFP content extraction")
        
        # Create cache directory if needed
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Read venue names
        with open(Config.INPUT_FILE, "r", encoding="utf-8") as f:
            venues = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Read {len(venues)} venues")
        
        # Create rate limiter
        rate_limiter = RateLimiter(
            max_requests=MAX_REQUESTS_PER_WINDOW,
            window=REQUEST_WINDOW
        )
        
        # Create semaphore for limiting parallel content extractions
        extraction_semaphore = asyncio.Semaphore(MAX_PARALLEL_EXTRACTIONS)
        
        # Process in batches
        all_results = {}
        venue_batches = chunk(venues, Config.BATCH_SIZE)
        logger.info(f"Processing venues in {len(venue_batches)} batches of size {Config.BATCH_SIZE}")
        
        # Configure timeout
        timeout = ClientTimeout(
            connect=Config.CONNECT_TIMEOUT,
            sock_read=Config.READ_TIMEOUT,
            total=Config.TOTAL_TIMEOUT
        )
        
        # Create progress bar
        batch_pbar = tqdm(total=len(venue_batches), desc="Processing venue batches")
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for batch_idx, batch in enumerate(venue_batches):
                try:
                    batch_start = time.time()
                    batch_results = await process_venues_batch(batch, session, rate_limiter, extraction_semaphore)
                    all_results.update(batch_results)
                    
                    batch_elapsed = time.time() - batch_start
                    logger.info(f"Batch {batch_idx+1}/{len(venue_batches)} processed in {batch_elapsed:.2f} seconds")
                    batch_pbar.update(1)
                    
                    # Add small delay between batches
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx+1}: {str(e)}")
                    continue
                    
        batch_pbar.close()
        
        # Create output directory if needed
        output_path = Path(Config.OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Log statistics
        total_venues = len(all_results)
        total_documents = sum(len(docs) for docs in all_results.values())
        success_rate = round(total_venues / len(venues) * 100, 2) if venues else 0
        
        elapsed_time = time.time() - start_time
        logger.info(f"CFP content extraction completed in {elapsed_time:.2f} seconds")
        logger.info(f"Extracted content for {total_venues}/{len(venues)} venues ({success_rate}% success rate)")
        logger.info(f"Total documents extracted: {total_documents}")
        logger.info(f"Cache hits: {len(URL_CONTENT_CACHE)}")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"An error occurred after {elapsed_time:.2f} seconds: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
