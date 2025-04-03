import os
import asyncio
import aiohttp
import ssl
import time
import random
import socket
import json
import re
import chardet
from urllib.parse import urlparse
from aiohttp import ClientTimeout, TCPConnector
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import certifi

# Constants
MAX_CONCURRENT_REQUESTS = 16
MAX_RETRIES = 4
RATE_LIMIT_DELAY = 1  # seconds
ENCODING_FALLBACKS = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
MAX_CONTENT_LENGTH = 500000  # Characters
CONNECT_TIMEOUT = 15  # seconds
READ_TIMEOUT = 40  # seconds
TOTAL_TIMEOUT = 80  # seconds
MAX_DELAY_BETWEEN_ATTEMPTS = 8  # Maximum delay between retry attempts

# Cloud Scraper import (optional)
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# User Agent rotation (optional)
try:
    from fake_useragent import UserAgent
    FAKE_UA_AVAILABLE = True
except ImportError:
    FAKE_UA_AVAILABLE = False


# Predefined user agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

# Common referrers for educational sites
REFERRERS = [
    "https://www.google.com/",
    "https://www.google.com/search?q=university+departments",
    "https://www.google.com/search?q=university+research",
    "https://scholar.google.com/",
    "https://www.bing.com/search?q=university+departments",
    "https://www.educationusa.state.gov/"
]

# List of problem domains known to have SSL issues
SSL_PROBLEM_DOMAINS = [
    "utk.edu",
    "northeastern.edu",
    "ucdavis.edu",
    "baruch.cuny.edu",
    "westpoint.edu"
]

# List of domains known to block scrapers (403)
STRICT_DOMAINS = [
    "gsu.edu",
    "acu.edu",
    "gcu.edu",
    "wakehealth.edu"
]

class HTMLFetchConfig:
    """Configuration for HTML fetching"""
    def __init__(self,
                 cache_dir: Optional[str] = None,
                 custom_user_agents: Optional[List[str]] = None,
                 custom_referrers: Optional[List[str]] = None,
                 custom_ssl_problem_domains: Optional[List[str]] = None,
                 custom_strict_domains: Optional[List[str]] = None):
        self.cache_dir = cache_dir
        self.user_agents = custom_user_agents or USER_AGENTS
        self.referrers = custom_referrers or REFERRERS
        self.ssl_problem_domains = custom_ssl_problem_domains or SSL_PROBLEM_DOMAINS
        self.strict_domains = custom_strict_domains or STRICT_DOMAINS
        
        # Add fake user agents if available
        if FAKE_UA_AVAILABLE:
            try:
                ua = UserAgent()
                self.user_agents.extend([ua.chrome, ua.firefox, ua.edge, ua.random])
            except:
                pass
    
    def get_cache_file_path(self, url: str) -> str:
        """Get cache file path for a URL"""
        if not self.cache_dir:
            raise ValueError("cache_dir not set in HTMLFetchConfig")
            
        # Create a hash of the URL for the filename
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.txt")


def should_skip_domain(url: str) -> bool:
    """
    Check if domain should be skipped
    
    Args:
        url: URL to check
        
    Returns:
        Boolean indicating if domain should be skipped
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Skip common irrelevant domains
        skip_domains = [
            'wikipedia.org',
            'youtube.com',
            'facebook.com',
            'twitter.com',
            'instagram.com',
            'amazon.com',
            'googleusercontent.com',
            'github.com',
            'linkedin.com'
        ]
        
        return any(skip in domain for skip in skip_domains)
    except:
        return True


def standardize_url(url: str) -> str:
    """
    Standardize URL format
    
    Args:
        url: URL to standardize
        
    Returns:
        Standardized URL
    """
    try:
        # Remove fragments
        url = url.split('#')[0]
        # Remove trailing slash
        if url.endswith('/'):
            url = url[:-1]
        return url
    except:
        return url


def get_random_headers(url: Optional[str] = None, config: Optional[HTMLFetchConfig] = None) -> Dict[str, str]:
    """
    Get a random set of headers to make requests look more organic
    
    Args:
        url: Optional URL to customize headers for problematic domains
        config: Optional HTMLFetchConfig with custom headers
        
    Returns:
        Dictionary of headers
    """
    if config is None:
        config = HTMLFetchConfig()
        
    user_agent = random.choice(config.user_agents)
    referrer = random.choice(config.referrers)
    
    # Standard headers that mimic a real browser
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "DNT": "1",
    }
    
    # Add Chrome-specific headers if using a Chrome user agent
    if "Chrome" in user_agent:
        headers["sec-ch-ua"] = '"Google Chrome";v="113", "Chromium";v="113"'
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = '"Windows"'
    
    # Add referrer only sometimes to vary request patterns
    if random.random() > 0.2:
        headers["Referer"] = referrer
    
    # Custom headers for problematic domains
    if url and config:
        domain = urlparse(url).netloc
        
        # Extra headers for domains that tend to block scrapers
        for strict_domain in config.strict_domains:
            if strict_domain in domain:
                # Add more headers that make the request look more like a real browser
                headers["Authority"] = domain
                headers["Method"] = "GET"
                headers["Path"] = urlparse(url).path
                headers["Scheme"] = "https"
                headers["Accept-Encoding"] = "gzip, deflate, br"
                headers["Priority"] = "u=0, i"
                # Add a cookie that many sites check
                headers["Cookie"] = f"_ga=GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}; _gid=GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}"
                break
    
    return headers


def create_custom_ssl_context(verify: bool = True) -> ssl.SSLContext:
    """
    Create a custom SSL context for handling various SSL issues
    
    Args:
        verify: Whether to verify SSL certificates
        
    Returns:
        SSL context
    """
    context = ssl.create_default_context(cafile=certifi.where())
    
    if not verify:
        # Disable verification for problem sites
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    
    # Add options to handle legacy servers and other SSL issues
    try:
        # These SSL options might not be available in all Python versions
        if hasattr(ssl, 'OP_LEGACY_SERVER_CONNECT'):
            context.options |= ssl.OP_LEGACY_SERVER_CONNECT  # For legacy renegotiation
        else:
            # Use the raw value (0x4) as a fallback
            context.options |= 0x4  # SSL_OP_LEGACY_SERVER_CONNECT
            
        # Allow insecure renegotiation for problematic servers
        if hasattr(ssl, 'OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION'):
            context.options |= ssl.OP_ALLOW_UNSAFE_LEGACY_RENEGOTIATION
    except (AttributeError, ValueError):
        pass
        
    return context


def save_to_cache(url: str, content: str, config: HTMLFetchConfig, term: Optional[str] = None) -> None:
    """
    Save content to cache
    
    Args:
        url: URL being cached
        content: HTML content to cache
        config: HTMLFetchConfig with cache settings
        term: Optional term that generated this URL
    """
    if not config.cache_dir:
        return
        
    cache_path = config.get_cache_file_path(url)
    try:
        # If it's HTML content, wrap it with metadata
        if content and content.strip().startswith(('<html', '<!DOCTYPE', '<HTML')):
            # Save with metadata
            metadata = {
                "url": url,
                "term": term,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content_length": len(content)
            }
            
            # Save as JSON with metadata and content fields
            with open(cache_path, "w", encoding="utf-8") as f:
                cache_data = {
                    "metadata": metadata,
                    "content": content
                }
                json.dump(cache_data, f, ensure_ascii=False)
        else:
            # For non-HTML content, just save as is
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
    except Exception:
        pass


def check_cache(url: str, config: HTMLFetchConfig) -> Optional[str]:
    """
    Check if content for URL is cached
    
    Args:
        url: URL to check
        config: HTMLFetchConfig with cache settings
        
    Returns:
        Cached content if available, None otherwise
    """
    if not config.cache_dir:
        return None
        
    cache_path = config.get_cache_file_path(url)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Check if it's JSON format with metadata
                if content.strip().startswith('{') and '"metadata":' in content:
                    try:
                        cache_data = json.loads(content)
                        return cache_data.get("content", "")
                    except json.JSONDecodeError:
                        # If JSON parsing fails, return content as is
                        return content
                else:
                    # It's not JSON, return as is
                    return content
        except Exception:
            pass
    return None


async def detect_and_decode(response: aiohttp.ClientResponse) -> Tuple[bytes, str]:
    """
    Detect encoding and decode response content
    
    Args:
        response: aiohttp ClientResponse object
        
    Returns:
        Tuple of (raw_bytes, encoding)
    """
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


async def fetch_with_cloudscraper(url: str) -> Optional[str]:
    """
    Use CloudScraper to bypass Cloudflare and other protection measures
    
    Args:
        url: URL to fetch
        
    Returns:
        HTML content if successful, None otherwise
    """
    if not CLOUDSCRAPER_AVAILABLE:
        return None
        
    try:
        # First check if domain is resolvable to avoid NameResolutionError
        domain = urlparse(url).netloc
        try:
            # Try to resolve the domain first to check if it's valid
            socket.gethostbyname(domain)
        except socket.gaierror:
            return None  # Return early if domain can't be resolved
            
        # Create a cloudscraper session with various browser emulation settings
        browser_options = [
            {'browser': 'chrome', 'platform': 'windows', 'desktop': True},
            {'browser': 'firefox', 'platform': 'darwin', 'desktop': True},
        ]
        
        # Use a random browser configuration
        browser_config = random.choice(browser_options)
        
        # Create the scraper
        scraper = cloudscraper.create_scraper(
            browser=browser_config,
            delay=random.uniform(2, 5),  # Random delay to avoid patterns
            doubleDown=True,  # Try harder to bypass protections
            interpreter='js2py'  # Use js2py for JavaScript challenge solving
        )
        
        # Add headers to make the request appear more legitimate
        headers = get_random_headers(url)
        
        # Add cookies to help with session management
        scraper.cookies.set('session_visit', 'true', domain=domain)
        scraper.cookies.set('_ga', f'GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}', domain=domain)
        scraper.cookies.set('_gid', f'GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}', domain=domain)
        
        # Add more specific cookies for sites known to use specific cookie checks
        if any(strict in domain for strict in STRICT_DOMAINS):
            scraper.cookies.set('visitor_id', f'{random.randint(10000, 99999)}', domain=domain)
            scraper.cookies.set('has_js', '1', domain=domain)
            scraper.cookies.set('PHPSESSID', ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=26)), domain=domain)
        
        # Perform the request with timeout and retries
        for attempt in range(3):  # Increased retries for CloudScraper
            try:
                response = scraper.get(
                    url, 
                    headers=headers, 
                    timeout=40,  # Increased timeout
                    allow_redirects=True
                )
                
                if response.status_code != 200:
                    if attempt < 2:  # Try more times with different settings
                        # Rotate browser config
                        browser_config = random.choice([b for b in browser_options if b != browser_config])
                        scraper = cloudscraper.create_scraper(
                            browser=browser_config,
                            delay=random.uniform(3, 8)  # Longer delay for retry
                        )
                        # New headers
                        headers = get_random_headers(url)
                        time.sleep(random.uniform(3, 6))  # Add a significant delay
                        continue
                    return None
                
                # Check if we got an actual HTML response (not a CAPTCHA or error page)
                content = response.text
                if len(content) < 500 or "captcha" in content.lower() or "denied" in content.lower() or "access is blocked" in content.lower():
                    if attempt < 2:
                        time.sleep(random.uniform(3, 6))
                        continue
                    return None
                    
                return content
            except Exception:
                if attempt < 2:
                    time.sleep(random.uniform(4, 8))
                    continue
                return None
                
        return None
    except Exception:
        return None


async def fetch_webpage(url: str, 
                      session: aiohttp.ClientSession, 
                      semaphore: asyncio.Semaphore,
                      config: HTMLFetchConfig,
                      term: Optional[str] = None,
                      logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Fetch webpage content with robust encoding handling
    
    Args:
        url: URL to fetch
        session: aiohttp ClientSession
        semaphore: asyncio Semaphore for concurrency control
        config: HTMLFetchConfig for settings
        term: Optional term for caching
        logger: Optional logger
        
    Returns:
        HTML content if successful, None otherwise
    """
    async with semaphore:  # Control concurrent requests
        if should_skip_domain(url):
            if logger:
                logger.debug(f"Skipping domain: {url}")
            return None
            
        # Standardize URL
        url = standardize_url(url)
        
        # Check cache first
        cached_content = check_cache(url, config)
        if cached_content:
            if logger:
                logger.debug(f"Using cached content for {url}")
            return cached_content
        
        # Get domain for special handling
        domain = urlparse(url).netloc
        
        # Check if this is a known SSL problem domain
        has_ssl_issues = any(problem_domain in domain for problem_domain in config.ssl_problem_domains)
        
        # Check if this is a known strict domain (403 errors)
        is_strict_domain = any(strict_domain in domain for strict_domain in config.strict_domains)
        
        # If we know this domain has SSL issues, try with a permissive context immediately
        if has_ssl_issues:
            if logger:
                logger.debug(f"Using permissive SSL context for known problematic domain: {domain}")
            try:
                # Create a session with SSL verification disabled
                ssl_context = create_custom_ssl_context(verify=False)
                timeout = ClientTimeout(connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT, total=TOTAL_TIMEOUT)
                
                async with aiohttp.ClientSession(connector=TCPConnector(ssl=ssl_context), timeout=timeout) as ssl_session:
                    headers = get_random_headers(url, config)
                    async with ssl_session.get(url, headers=headers, allow_redirects=True) as response:
                        if response.status == 200:
                            raw_bytes, encoding = await detect_and_decode(response)
                            
                            # Try to decode using detected encoding
                            for enc in [encoding] + ENCODING_FALLBACKS:
                                try:
                                    html_content = raw_bytes.decode(enc)
                                    # Save to cache
                                    save_to_cache(url, html_content, config, term)
                                    return html_content
                                except UnicodeDecodeError:
                                    continue
                                except Exception as e:
                                    if logger:
                                        logger.warning(f"Error decoding with {enc} (permissive SSL): {str(e)}")
            except Exception as e:
                if logger:
                    logger.warning(f"Error with permissive SSL context for {url}: {str(e)}")
        
        # If this is a known strict domain, try CloudScraper immediately
        if is_strict_domain and CLOUDSCRAPER_AVAILABLE:
            if logger:
                logger.debug(f"Using CloudScraper for known strict domain: {domain}")
            cloudscraper_content = await fetch_with_cloudscraper(url)
            if cloudscraper_content:
                # Save to cache
                save_to_cache(url, cloudscraper_content, config, term)
                return cloudscraper_content
                
        # Get random headers for each request to avoid detection
        headers = get_random_headers(url, config)
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Configure timeouts
                timeout = ClientTimeout(
                    connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,
                    total=TOTAL_TIMEOUT
                )
                
                # For subsequent attempts, rotate user agent and add a delay
                if attempt > 0:
                    headers = get_random_headers(url, config)  # Get a fresh set of headers
                    # Add progressively longer delays between retries
                    delay = min(RATE_LIMIT_DELAY * (1.5 ** attempt), MAX_DELAY_BETWEEN_ATTEMPTS)
                    await asyncio.sleep(delay)
                    if logger:
                        logger.debug(f"Retry #{attempt} for {url} with new headers (delay: {delay:.2f}s)")
                
                async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as response:
                    # Check for non-200 responses
                    if response.status != 200:
                        # Log intermediate HTTP errors as DEBUG, final as DEBUG too
                        log_level = logging.DEBUG # Always DEBUG for these errors now
                        if response.status == 403:
                            if logger:
                                logger.log(log_level, f"HTTP error 403 (Forbidden) for URL: {url} (attempt {attempt+1}/{MAX_RETRIES+1})")
                            # If this is the last attempt, try CloudScraper as a fallback
                            if attempt == MAX_RETRIES and CLOUDSCRAPER_AVAILABLE:
                                cloudscraper_content = await fetch_with_cloudscraper(url)
                                if cloudscraper_content:
                                    # Save to cache
                                    save_to_cache(url, cloudscraper_content, config, term)
                                    return cloudscraper_content
                            
                            # If we still have retries left, continue to the next attempt
                            if attempt < MAX_RETRIES:
                                continue
                        else:
                            if logger:
                                logger.log(log_level, f"HTTP error {response.status} for URL: {url} (attempt {attempt+1}/{MAX_RETRIES+1})")
                        
                        # If it failed on the last attempt or was not a 403, return None after logging
                        if attempt == MAX_RETRIES:
                             return None
                        else:
                             continue # Try next attempt for non-403 errors too if retries remain
                    
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
                            if logger:
                                logger.warning(f"Error decoding with {enc}: {str(e)}")
                            continue
                    
                    if not html_content:
                        if logger:
                            logger.warning(f"Failed to decode content from {url} with any encoding")
                        return None
                    
                    if logger:
                        logger.debug(f"Successfully decoded {url} using {successful_encoding} encoding")
                    
                    # Save to cache
                    save_to_cache(url, html_content, config, term)
                    
                    await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                    return html_content
                    
            except ssl.SSLError as e:
                # Log intermediate and final SSL errors as DEBUG
                log_level = logging.DEBUG # Always DEBUG
                if logger:
                    logger.log(log_level, f"SSL error for {url}: {str(e)} (attempt {attempt+1}/{MAX_RETRIES+1})")
                
                # If this is the last attempt, try with SSL verification disabled
                if attempt == MAX_RETRIES:
                    try:
                        # Create a new connector with SSL verification disabled
                        ssl_context = create_custom_ssl_context(verify=False)
                        
                        # Create a new session with SSL verification disabled
                        async with aiohttp.ClientSession(connector=TCPConnector(ssl=ssl_context)) as insecure_session:
                            async with insecure_session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as response:
                                if response.status != 200:
                                    if logger:
                                        logger.warning(f"HTTP error {response.status} for URL (with SSL disabled): {url}")
                                    return None
                                
                                raw_bytes, encoding = await detect_and_decode(response)
                                
                                # Try to decode using detected encoding
                                for enc in [encoding] + ENCODING_FALLBACKS:
                                    try:
                                        html_content = raw_bytes.decode(enc)
                                        # Save to cache
                                        save_to_cache(url, html_content, config, term)
                                        return html_content
                                    except UnicodeDecodeError:
                                        continue
                                    except Exception as e:
                                        if logger:
                                            logger.warning(f"Error decoding with {enc} (SSL disabled): {str(e)}")
                                        continue
                    except Exception as e:
                        if logger:
                            logger.warning(f"Error fetching with SSL verification disabled from {url}: {str(e)}")
                
                # If we still have retries left, continue to the next attempt
                if attempt < MAX_RETRIES:
                    continue
                return None
            except asyncio.TimeoutError:
                # Log intermediate and final Timeout errors as DEBUG
                log_level = logging.DEBUG # Always DEBUG
                if logger:
                    logger.log(log_level, f"Timeout error while fetching {url} (attempt {attempt+1}/{MAX_RETRIES+1})")
                if attempt < MAX_RETRIES:
                    continue
                return None
            except aiohttp.ClientConnectorError as e:
                # Log intermediate and final connection errors as DEBUG
                log_level = logging.DEBUG # Always DEBUG
                if logger:
                    logger.log(log_level, f"Connection error for {url}: {str(e)} (attempt {attempt+1}/{MAX_RETRIES+1})")
                
                if attempt < MAX_RETRIES:
                    continue
                return None
            except aiohttp.ClientError as e:
                # Log intermediate and final network errors as DEBUG
                log_level = logging.DEBUG # Always DEBUG
                if logger:
                    logger.log(log_level, f"Network error for {url}: {str(e)} (attempt {attempt+1}/{MAX_RETRIES+1})")
                if attempt < MAX_RETRIES:
                    continue
                return None
            except Exception as e:
                # Log unexpected errors as ERROR
                if logger:
                    logger.error(f"Unexpected error fetching content from {url}: {str(e)}", exc_info=True)
                return None
        
        # If we reach here, all attempts failed
        # Log final overall failure as DEBUG now as well
        if logger:
             logger.debug(f"All {MAX_RETRIES+1} attempts failed for URL: {url}") # Changed from WARNING to DEBUG
        return None 