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
from aiohttp import ClientTimeout, TCPConnector, ClientResponseError, ClientConnectorError, ServerDisconnectedError, ClientPayloadError
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import certifi

# Constants
MAX_CONCURRENT_REQUESTS = 16
MAX_CONCURRENT_BROWSERS = 4 # Limit concurrent headless browser instances
MAX_RETRIES = 4
RATE_LIMIT_DELAY = 1  # seconds
ENCODING_FALLBACKS = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
MAX_CONTENT_LENGTH = 500000  # Characters of CLEANED TEXT (not raw HTML)
CONNECT_TIMEOUT = 15  # seconds
READ_TIMEOUT = 40  # seconds
TOTAL_TIMEOUT = 80  # seconds
MAX_DELAY_BETWEEN_ATTEMPTS = 8  # Maximum delay between retry attempts
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504} # Status codes to retry on

# Add HTML cleaning libraries if available
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

# Cloud Scraper import (optional)
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# User Agent rotation (optional)
try:
    import fake_useragent
    FAKE_UA_AVAILABLE = True
    # Create a single global instance to avoid repeated file operations
    try:
        GLOBAL_USER_AGENT = fake_useragent.UserAgent()
    except Exception as ua_error:
        print(f"Warning: Could not initialize fake_useragent: {ua_error}")
        GLOBAL_USER_AGENT = None
        FAKE_UA_AVAILABLE = False
except ImportError:
    FAKE_UA_AVAILABLE = False
    GLOBAL_USER_AGENT = None

# Try to import nest_asyncio for fixing asyncio nested loop issues
try:
    import nest_asyncio
    NEST_ASYNCIO_AVAILABLE = True
    # Apply nest_asyncio patch to allow nested event loops
    nest_asyncio.apply()
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False

# Try to import optional headless browser libraries
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    import seleniumwire.undetected_chromedriver as uc
    UNDETECTED_CHROME_AVAILABLE = True
except ImportError:
    UNDETECTED_CHROME_AVAILABLE = False

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
    "westpoint.edu",
    "unc.edu" # Added based on observed errors
]

# List of domains known to block scrapers (403)
STRICT_DOMAINS = [
    "gsu.edu",
    "acu.edu",
    "gcu.edu",
    "wakehealth.edu",
    # Add more edu domains known to have strict protections
    "harvard.edu",
    "stanford.edu",
    "mit.edu",
    "columbia.edu",
    "yale.edu",
    "princeton.edu",
    "cornell.edu",
    "upenn.edu",
    "berkeley.edu",
    "umich.edu"
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
        
        # Add fake user agents if available and global instance exists
        if FAKE_UA_AVAILABLE and GLOBAL_USER_AGENT:
            try:
                # Use the global instance
                self.user_agents.extend([
                    GLOBAL_USER_AGENT.chrome, 
                    GLOBAL_USER_AGENT.firefox, 
                    GLOBAL_USER_AGENT.edge, 
                    GLOBAL_USER_AGENT.random
                ])
            except Exception as e:
                print(f"Warning: Error using pre-initialized fake_useragent: {e}")
    
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
    
    # Latest Chrome, Firefox, and Edge user agents (updated 2025)
    MODERN_USER_AGENTS = [
        # Chrome
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        # Firefox
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
    ]
    
    # Prioritize modern user agents, but fall back to config-provided ones if available
    user_agents_to_use = MODERN_USER_AGENTS + config.user_agents if config.user_agents else MODERN_USER_AGENTS
    
    # Use the global fake_useragent directly if needed (avoid creating new instances)
    if FAKE_UA_AVAILABLE and GLOBAL_USER_AGENT:
        try:
            # Use the global instance, but only add if we can successfully get values
            random_ua = GLOBAL_USER_AGENT.random
            if random_ua:
                user_agents_to_use.append(random_ua)
        except Exception:
            # Silently continue if fake_useragent fails
            pass
    
    # Select a random user agent
    user_agent = random.choice(user_agents_to_use)
    
    # Modern educational referrers
    edu_referrers = [
        "https://scholar.google.com/",
        "https://www.researchgate.net/",
        "https://www.academia.edu/",
        "https://www.jstor.org/",
        "https://www.sciencedirect.com/",
        "https://link.springer.com/",
        "https://eric.ed.gov/",
        "https://www.proquest.com/",
        "https://www.ebsco.com/",
        "https://www.refseek.com/",
        "https://www.educationcorner.com/",
        "https://www.googlescholar.com/"
    ]
    
    # Combine with standard referrers
    referrers_to_use = config.referrers + edu_referrers if config.referrers else edu_referrers
    referrer = random.choice(referrers_to_use)
    
    # Base headers common to all browsers
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }
    
    # Browser-specific headers
    if "Chrome" in user_agent and "Edg" not in user_agent:
        # Chrome-specific headers
        chrome_version = re.search(r'Chrome/(\d+)', user_agent)
        chrome_ver = chrome_version.group(1) if chrome_version else "123"
        
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "sec-ch-ua": f'"Google Chrome";v="{chrome_ver}", "Not:A-Brand";v="8", "Chromium";v="{chrome_ver}"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"' if "Windows" in user_agent else ('"macOS"' if "Macintosh" in user_agent else '"Linux"'),
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        })
    elif "Firefox" in user_agent:
        # Firefox-specific headers
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "TE": "trailers",
            "Pragma": "no-cache"
        })
    elif "Edg" in user_agent:
        # Edge-specific headers
        edge_version = re.search(r'Edg/(\d+)', user_agent)
        edge_ver = edge_version.group(1) if edge_version else "123"
        
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "sec-ch-ua": f'"Microsoft Edge";v="{edge_ver}", "Not:A-Brand";v="8", "Chromium";v="{edge_ver}"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"' if "Windows" in user_agent else ('"macOS"' if "Macintosh" in user_agent else '"Linux"'),
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        })
        
    # Add referrer with 80% probability
    if random.random() > 0.2:
        headers["Referer"] = referrer
    
    # Add random viewport dimensions to further randomize requests
    viewports = [
        {"width": 1920, "height": 1080},
        {"width": 1440, "height": 900},
        {"width": 1366, "height": 768},
        {"width": 2560, "height": 1440},
        {"width": 1536, "height": 864}
    ]
    viewport = random.choice(viewports)
    
    # Add viewport as a custom header (some sites check this)
    headers["X-Viewport-Width"] = str(viewport["width"])
    headers["X-Viewport-Height"] = str(viewport["height"])
    
    # Custom headers for problematic domains
    if url and config:
        domain = urlparse(url).netloc.lower()
        
        # Handle .edu domains specially
        if domain.endswith('.edu'):
            # Create realistic cookie values for academic sites
            session_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))
            current_ts = str(int(time.time()))
            ga_id = str(random.randint(1000000, 9999999))
            
            # Set of cookies commonly found on .edu sites
            headers["Cookie"] = f"_ga=GA1.2.{ga_id}.{current_ts}; _gid=GA1.2.{random.randint(1000000, 9999999)}.{current_ts}; _gat=1; _fbp=fb.1.{current_ts}.{random.randint(1000000, 9999999)}; PHPSESSID={session_id}; has_js=1; visitor_id={random.randint(100000, 999999)}; EduAccepted=1; cookieconsent_status=dismiss"
            
            # Set academic referrer for .edu sites
            headers["Referer"] = random.choice(edu_referrers)
        
        # Extra headers for domains that tend to block scrapers
        for strict_domain in config.strict_domains:
            if strict_domain in domain:
                # Add more headers that make the request look more like a real browser
                headers["Authority"] = domain
                headers["Method"] = "GET"
                headers["Path"] = urlparse(url).path
                headers["Scheme"] = "https"
                headers["Priority"] = "u=0, i"
                
                # Even more sophisticated cookies for strict domains
                headers["Cookie"] = f"_ga=GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}; _gid=GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}; _gat=1; _fbp=fb.1.{int(time.time())}.{random.randint(1000000, 9999999)}; PHPSESSID={''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))}; has_js=1; visitor_id={random.randint(100000, 999999)}; EduAccepted=1; cookieconsent_status=dismiss; session_id={random.randint(1000000000, 9999999999)}"
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
    except (AttributeError, ValueError) as e:
        print(f"Warning: Could not set SSL option OP_LEGACY_SERVER_CONNECT: {e}")
        pass
        
    # Enable options to improve compatibility
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1
    context.options |= ssl.OP_SINGLE_DH_USE
    context.options |= ssl.OP_SINGLE_ECDH_USE
    context.set_ciphers('HIGH:!DH') # Use modern ciphers

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


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Cleaned text content
    """
    # Use BeautifulSoup if available
    if BEAUTIFULSOUP_AVAILABLE:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(["script", "style", "meta", "noscript", "iframe", "svg"]):
                element.decompose()
                
            # Get the text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception:
            # Fall back to regex if BeautifulSoup parsing fails
            pass
    
    # Fallback: Use simple regex to clean HTML
    try:
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Remove multiple whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception:
        # If all cleaning fails, return original length to be safe
        return html_content


async def fetch_with_advanced_techniques(url: str,
                                     session: aiohttp.ClientSession,
                                     headers: Dict[str, str],
                                     ssl_context: ssl.SSLContext,
                                     config: HTMLFetchConfig,
                                     logger: logging.Logger,
                                     browser_semaphore: asyncio.Semaphore) -> Optional[str]:
    """
    Advanced techniques to bypass edu site protections

    Args:
        url: URL to fetch
        session: aiohttp session
        headers: Request headers
        ssl_context: SSL context
        config: HTMLFetchConfig with custom settings
        logger: Logger instance
        browser_semaphore: Semaphore for headless browsers

    Returns:
        HTML content if successful, None otherwise
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    domain_is_edu = domain.endswith('.edu')
    
    # Try CloudScraper first for edu domains if available
    if domain_is_edu and CLOUDSCRAPER_AVAILABLE:
        logger.debug(f"Attempting .edu site with CloudScraper: {url}")
        content = await fetch_with_cloudscraper(url)
        if content:
            logger.debug(f"CloudScraper succeeded for {url}")
            return content
            
    # Enhanced cookie and session management for edu sites
    if domain_is_edu:
        # Create a more sophisticated browser fingerprint
        advanced_headers = headers.copy()
        
        # Add more realistic browser fingerprinting
        advanced_headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
        advanced_headers["Accept-Encoding"] = "gzip, deflate, br"
        advanced_headers["Accept-Language"] = "en-US,en;q=0.9"
        advanced_headers["Sec-Ch-Ua"] = '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"'
        advanced_headers["Sec-Ch-Ua-Mobile"] = "?0"
        advanced_headers["Sec-Ch-Ua-Platform"] = '"Windows"'
        advanced_headers["Sec-Fetch-Dest"] = "document"
        advanced_headers["Sec-Fetch-Mode"] = "navigate"
        advanced_headers["Sec-Fetch-Site"] = "none"
        advanced_headers["Sec-Fetch-User"] = "?1"
        advanced_headers["Upgrade-Insecure-Requests"] = "1"
        
        # Add cookie consent and session cookies many EDU sites check for
        advanced_headers["Cookie"] = f"_ga=GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}; _gid=GA1.2.{random.randint(1000000, 9999999)}.{int(time.time())}; cookieconsent_status=dismiss; session_id={random.randint(1000000000, 9999999999)}; has_js=1; visitor_id={random.randint(100000, 999999)}; JSESSIONID={''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))}"
        
        # Try a "softer" approach first - appear to be coming from Google search
        search_referrers = [
            "https://www.google.com/search?q=site%3A" + domain + "+department+information",
            "https://www.google.com/search?q=site%3A" + domain + "+academic+programs",
            "https://www.google.com/search?q=site%3A" + domain + "+faculty+research",
            "https://scholar.google.com/scholar?q=site%3A" + domain
        ]
        advanced_headers["Referer"] = random.choice(search_referrers)
        
        try:
            logger.debug(f"Attempting .edu site with enhanced headers: {url}")
            async with session.get(
                url,
                headers=advanced_headers,
                ssl=ssl_context,
                timeout=ClientTimeout(total=TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT),
                allow_redirects=True,
                max_redirects=5
            ) as response:
                if response.status == 200:
                    content_bytes, encoding = await detect_and_decode(response)
                    content = content_bytes.decode(encoding, errors='replace')
                    logger.debug(f"Enhanced technique worked for {url}")
                    return content
                    
        except Exception as e:
            logger.debug(f"Enhanced technique failed for {url}: {str(e)}")
            
        # If .edu domain and regular techniques fail, try with headless browser directly
        if PLAYWRIGHT_AVAILABLE or UNDETECTED_CHROME_AVAILABLE:
            try:
                logger.debug(f"Attempting direct headless browser for .edu domain: {url}")
                browser_content = await fetch_with_headless_browser(url, logger, browser_semaphore)
                if browser_content:
                    logger.debug(f"Direct headless browser succeeded for {url}")
                    return browser_content
            except Exception as e:
                # Check specifically for the 'cannot access local variable' error
                if "cannot access local variable" in str(e):
                    logger.warning(f"Caught 'cannot access local variable' error for {url}. Trying alternative approach.")
                    # Try simplified version with selenium as fallback
                    try:
                        if UNDETECTED_CHROME_AVAILABLE:
                            # Implement direct synchronous call to avoid asyncio issues
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                browser_content = await asyncio.get_event_loop().run_in_executor(
                                    executor,
                                    lambda: fetch_with_undetected_chrome_sync(url, logger)
                                )
                                if browser_content:
                                    logger.debug(f"Direct headless browser (sync fallback) succeeded for {url}")
                                    return browser_content
                    except Exception as fallback_e:
                        logger.error(f"Fallback headless browser approach failed for {url}: {str(fallback_e)}")
                else:
                    logger.error(f"Headless browser failed for {url}: {str(e)}")
    
    # If all special techniques fail, return None and let the caller try standard methods
    return None

def fetch_with_undetected_chrome_sync(url: str, logger: logging.Logger) -> Optional[str]:
    """
    Synchronous function to fetch a URL using undetected_chromedriver.
    This avoids asyncio issues when running in threads.
    
    Args:
        url: URL to fetch
        logger: Logger instance
        
    Returns:
        HTML content if successful, None otherwise
    """
    if not UNDETECTED_CHROME_AVAILABLE:
        return None
        
    driver = None
    try:
        # Initialize undetected Chrome
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-extensions")
        
        driver = uc.Chrome(options=chrome_options)
        
        # Navigate to URL
        driver.get(url)
        
        # Wait for page to load
        time.sleep(3)
        
        # Execute some scrolling and human-like behavior
        driver.execute_script("""
            window.scrollTo(0, 300);
            setTimeout(() => window.scrollTo(0, 600), 300);
            setTimeout(() => window.scrollTo(0, 900), 600);
        """)
        
        time.sleep(2)
        
        # Get page content
        content = driver.page_source
        
        # Close driver
        driver.quit()
        
        return content
    except Exception as inner_e:
        logger.error(f"Undetected Chrome sync error: {str(inner_e)}")
        if driver:
            try:
                driver.quit()
            except:
                pass
        return None

async def fetch_webpage(url: str,
                      session: aiohttp.ClientSession,
                      semaphore: asyncio.Semaphore,
                      browser_semaphore: asyncio.Semaphore,
                      config: HTMLFetchConfig,
                      term: Optional[str] = None,
                      logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Fetch HTML content from a webpage with automatic retry and fallback strategies.

    Args:
        url: URL to fetch
        session: aiohttp ClientSession
        semaphore: Semaphore for concurrency control (general fetches)
        browser_semaphore: Semaphore for concurrency control (headless browser fetches)
        config: Fetch configuration
        term: Optional context term for logging
        logger: Optional logger

    Returns:
        HTML content as string or None if all fetch attempts fail
    """
    if not logger:
        logger = logging.getLogger("html_fetch")

    # Log start of fetch process with term context if available
    context_info = f" for term '{term}'" if term else ""
    logger.info(f"⏳ Starting fetch{context_info}: {url}")

    # Skip URLs pointing directly to common file types
    file_extensions_to_skip = (
        '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
        '.zip', '.rar', '.tar', '.gz', '.7z', '.csv', '.xml', '.json',
        '.ics', '.txt', '.rtf', '.mp3', '.mp4', '.avi', '.mov', '.wmv',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tif', '.tiff'
    )
    parsed_url_path = urlparse(url).path.lower()
    if parsed_url_path.endswith(file_extensions_to_skip):
        logger.warning(f"⚠️ Skipping URL pointing to a file: {url}")
        return None

    # Skip known problematic domains
    if should_skip_domain(url):
        logger.warning(f"⚠️ Skipping known problematic domain: {url}")
        return None

    # Standardize URL (fixes common issues)
    url = standardize_url(url)

    # Check cache first
    if config.cache_dir:
        cached_content = check_cache(url, config)
        if cached_content:
            logger.info(f"✅ Using cached content{context_info}: {url}")
            return cached_content
        logger.debug(f"Cache miss{context_info}: {url}")
    
    # Prepare headers
    headers = get_random_headers(url, config)
    
    # Create a custom SSL context
    ssl_context = create_custom_ssl_context()
    
    # Process with concurrency control using semaphore
    fetch_start_time = time.time()
    try:
        async with semaphore:
            # Log acquiring semaphore
            logger.debug(f"🔒 Acquired semaphore for{context_info}: {url}")
            
            # Try standard aiohttp fetch first
            try:
                logger.debug(f"Attempting standard fetch{context_info}: {url}")
                async with session.get(url, headers=headers, ssl=ssl_context, timeout=30, allow_redirects=True) as response:
                    if response.status == 200:
                        content_bytes, encoding = await detect_and_decode(response)
                        html_content = content_bytes.decode(encoding, errors='replace')
                        
                        # Cache successful result
                        if config.cache_dir:
                            save_to_cache(url, html_content, config, term)
                            
                        fetch_time = time.time() - fetch_start_time
                        logger.info(f"✅ Successfully fetched{context_info} in {fetch_time:.2f}s: {url}")
                        return html_content
                    else:
                        logger.warning(f"⚠️ HTTP {response.status}{context_info}: {url}")
            except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
                logger.warning(f"⚠️ Standard fetch failed{context_info}: {url} - {type(e).__name__}: {str(e)}")
            
            # Try advanced fetch techniques if standard fetch fails
            logger.debug(f"Attempting advanced fetch techniques{context_info}: {url}")
            advanced_content = await fetch_with_advanced_techniques(url, session, headers, ssl_context, config, logger, browser_semaphore)
            if advanced_content:
                # Cache successful result
                if config.cache_dir:
                    save_to_cache(url, advanced_content, config, term)
                    
                fetch_time = time.time() - fetch_start_time
                logger.info(f"✅ Successfully fetched with advanced techniques{context_info} in {fetch_time:.2f}s: {url}")
                return advanced_content
            
            # Try cached version or Wayback Machine if advanced techniques fail
            logger.debug(f"Attempting cached version fetch{context_info}: {url}")
            cached_version = await fetch_cached_version(url, session, ssl_context, headers, logger)
            if cached_version:
                if config.cache_dir:
                    save_to_cache(url, cached_version, config, term)
                    
                fetch_time = time.time() - fetch_start_time
                logger.info(f"✅ Successfully fetched from cached version{context_info} in {fetch_time:.2f}s: {url}")
                return cached_version
            
            # Final attempt with headless browser
            logger.debug(f"Attempting headless browser fetch{context_info}: {url}")
            browser_content = await fetch_with_headless_browser(url, logger, browser_semaphore)
            if browser_content:
                if config.cache_dir:
                    save_to_cache(url, browser_content, config, term)
                
                fetch_time = time.time() - fetch_start_time
                logger.info(f"✅ Successfully fetched with headless browser{context_info} in {fetch_time:.2f}s: {url}")
                return browser_content
            
            # All attempts failed
            logger.error(f"❌ Failed to fetch{context_info} after all attempts: {url}")
            return None
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching{context_info}: {url} - {type(e).__name__}: {str(e)}")
        return None
    finally:
        # Log releasing semaphore
        logger.debug(f"🔓 Released semaphore for{context_info}: {url}")
        
        # Close any resources explicitly
        # (No additional resources to close in this function, but good practice to have the finally block)

async def fetch_cached_version(url: str, session: aiohttp.ClientSession, 
                          ssl_context: ssl.SSLContext, headers: Dict[str, str],
                          logger: logging.Logger) -> Optional[str]:
    """
    Attempt to fetch a cached version of the webpage from Google Cache or Wayback Machine
    when direct access fails, especially useful for .edu sites.
    
    Args:
        url: URL to fetch cached version for
        session: aiohttp ClientSession
        ssl_context: SSL context for requests
        headers: Headers to use for requests
        logger: Logger instance
        
    Returns:
        Cached HTML content if successful, None otherwise
    """
    original_url = url
    domain = urlparse(url).netloc
    
    # Try Google Cache first
    try:
        # Format Google Cache URL
        google_cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{original_url}"
        logger.debug(f"Attempting to fetch from Google Cache: {google_cache_url}")
        
        cache_headers = headers.copy()
        # Use a different referrer for Google Cache
        cache_headers["Referer"] = "https://www.google.com/search"
        
        async with session.get(
            google_cache_url,
            headers=cache_headers,
            ssl=ssl_context,
            timeout=ClientTimeout(total=60, connect=20, sock_read=40),
            allow_redirects=True
        ) as response:
            if response.status == 200:
                content_bytes, encoding = await detect_and_decode(response)
                content = content_bytes.decode(encoding, errors='replace')
                
                # Check if it's actually a cached page and not an error page
                if "This is Google's cache of" in content or "Esta es la versión en caché de Google de" in content:
                    logger.debug(f"Successfully fetched from Google Cache: {original_url}")
                    return content
                logger.debug(f"Google Cache returned a page but not a cached version: {original_url}")
    except Exception as e:
        logger.debug(f"Error fetching from Google Cache for {original_url}: {str(e)}")
    
    # If Google Cache fails, try Wayback Machine
    try:
        # Get the latest snapshot from Wayback Machine API
        wayback_api_url = f"https://archive.org/wayback/available?url={original_url}"
        logger.debug(f"Checking Wayback Machine availability: {wayback_api_url}")
        
        async with session.get(
            wayback_api_url,
            ssl=ssl_context,
            timeout=ClientTimeout(total=30, connect=10, sock_read=20)
        ) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    if data.get('archived_snapshots', {}).get('closest', {}).get('available', False):
                        wayback_url = data['archived_snapshots']['closest']['url']
                        logger.debug(f"Found Wayback Machine snapshot: {wayback_url}")
                        
                        # Fetch the actual content from Wayback Machine
                        async with session.get(
                            wayback_url,
                            ssl=ssl_context,
                            timeout=ClientTimeout(total=60, connect=20, sock_read=40),
                            allow_redirects=True
                        ) as wb_response:
                            if wb_response.status == 200:
                                wb_content_bytes, wb_encoding = await detect_and_decode(wb_response)
                                wb_content = wb_content_bytes.decode(wb_encoding, errors='replace')
                                logger.debug(f"Successfully fetched from Wayback Machine: {original_url}")
                                return wb_content
                except Exception as inner_e:
                    logger.debug(f"Error processing Wayback Machine content for {original_url}: {str(inner_e)}")
    except Exception as e:
        logger.debug(f"Error fetching from Wayback Machine for {original_url}: {str(e)}")
    
    # If all attempts fail, return None
    return None 

async def fetch_with_headless_browser(url: str, logger: logging.Logger, browser_semaphore: asyncio.Semaphore) -> Optional[str]:
    """
    Use headless browser (Playwright or Undetected Chrome) for difficult websites
    that require JavaScript execution and more human-like behavior

    Args:
        url: URL to fetch
        logger: Logger instance
        browser_semaphore: Semaphore to limit concurrent browser launches

    Returns:
        HTML content if successful, None otherwise
    """
    # Acquire browser semaphore before launching
    async with browser_semaphore:
        logger.debug(f"🔒 Acquired BROWSER semaphore for: {url}")
        # First try with Playwright if available
        if PLAYWRIGHT_AVAILABLE:
            try:
                logger.debug(f"Trying Playwright for {url} (under semaphore)")
                # Import here to ensure we're working with the module, not a variable
                from playwright.async_api import async_playwright

                # Launch playwright using async_playwright context manager
                async with async_playwright() as playwright_instance:
                    # Launch browser with stealth mode
                    browser = await playwright_instance.chromium.launch(
                        headless=True,
                        args=[
                            '--disable-blink-features=AutomationControlled',
                            '--disable-extensions',
                            '--disable-component-extensions-with-background-pages',
                            '--disable-default-apps',
                            '--disable-features=site-per-process,TranslateUI,BlinkGenPropertyTrees',
                            '--disable-hang-monitor',
                            '--disable-ipc-flooding-protection',
                            '--disable-popup-blocking',
                            '--disable-prompt-on-repost',
                            '--disable-renderer-backgrounding',
                            '--disable-sync',
                            '--disable-web-security',
                            '--disable-client-side-phishing-detection',
                            '--disable-features=IsolateOrigins,site-per-process',
                            '--disable-site-isolation-trials',
                            '--no-default-browser-check',
                            '--no-experiments',
                            '--no-pings',
                            '--window-size=1920,1080',
                            '--no-sandbox',
                            '--disable-infobars',
                            '--disable-dev-shm-usage',
                            '--ignore-certificate-errors',
                            '--enable-features=NetworkService,NetworkServiceInProcess'
                        ]
                    )
                    
                    # Prepare context with realistic viewport and user agent
                    random_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                    context = await browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        user_agent=random_ua,
                        java_script_enabled=True,
                        is_mobile=False,
                        has_touch=False,
                        locale="en-US",
                        timezone_id="America/New_York"
                    )
                    
                    # Add stealth script
                    await context.add_init_script("""
                        // Overwrite the navigator properties to remove webdriver flags
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => false,
                        });
                        
                        // Prevent detection via permissions
                        const originalQuery = window.navigator.permissions.query;
                        window.navigator.permissions.query = (parameters) => (
                            parameters.name === 'notifications' ?
                                Promise.resolve({state: Notification.permission}) :
                                originalQuery(parameters)
                        );
                        
                        // Add plugins to appear more like a real browser
                        Object.defineProperty(navigator, 'plugins', {
                            get: () => [1, 2, 3, 4, 5].map(() => ({
                                0: {
                                    type: 'application/x-google-chrome-pdf',
                                    suffixes: 'pdf',
                                    description: 'Portable Document Format'
                                },
                                name: 'Chrome PDF Plugin',
                                filename: 'internal-pdf-viewer',
                                description: 'Portable Document Format',
                                length: 1
                            }))
                        });
                    """)
                    
                    # Create new page
                    page = await context.new_page()
                    
                    # Add stealth behaviors
                    await page.add_init_script("""
                        // Add chrome global
                        window.chrome = {
                            runtime: {},
                            loadTimes: function() {},
                            csi: function() {},
                            app: {}
                        };
                        
                        // Modify navigator
                        window.navigator.connection = {
                            effectiveType: '4g',
                            rtt: 50,
                            downlink: 10,
                            saveData: false
                        };
                    """)
                    
                    try:
                        # Add random delays and mouse movements to appear more human
                        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                        
                        # Add random mouse movements and scrolling
                        for i in range(2):
                            x = 100 + int(random.random() * 800)
                            y = 100 + int(random.random() * 600)
                            await page.mouse.move(x, y)
                            await asyncio.sleep(random.uniform(0.5, 1.5))
                        
                        # Simulate scrolling
                        await page.evaluate("""
                            () => {
                                window.scrollTo(0, 300);
                                setTimeout(() => window.scrollTo(0, 600), 200);
                                setTimeout(() => window.scrollTo(0, 900), 400);
                                setTimeout(() => window.scrollTo(0, 1200), 600);
                            }
                        """)
                        
                        await asyncio.sleep(random.uniform(3, 5))
                        
                        # Wait for network to be idle
                        await page.wait_for_load_state("networkidle")
                        
                        # Get the content
                        content = await page.content()
                        
                        if content:
                            logger.debug(f"Successfully fetched with Playwright: {url}")
                            return content
                            
                    except Exception as inner_e:
                        logger.error(f"Playwright page navigation error for {url}: {str(inner_e)}")
                    
                    # Ensure browser is closed even if an error occurs
                    await browser.close()
            
            except Exception as e:
                logger.error(f"Playwright error for {url}: {str(e)}")
            
        # Try with Undetected Chrome if Playwright failed
        if UNDETECTED_CHROME_AVAILABLE:
            try:
                logger.debug(f"Trying Undetected Chrome for {url} (under semaphore)")

                # Use asyncio to run Undetected Chrome in a non-blocking way
                loop = asyncio.get_event_loop()

                def _fetch_with_undetected_chrome():
                    driver = None
                    try:
                        # Initialize undetected Chrome
                        chrome_options = uc.ChromeOptions()
                        chrome_options.add_argument("--headless")
                        chrome_options.add_argument("--no-sandbox")
                        chrome_options.add_argument("--disable-gpu")
                        chrome_options.add_argument("--window-size=1920,1080")
                        chrome_options.add_argument("--disable-dev-shm-usage")
                        chrome_options.add_argument("--disable-extensions")

                        driver = uc.Chrome(options=chrome_options)

                        # Navigate to URL
                        driver.get(url)

                        # Wait for page to load
                        time.sleep(3)

                        # Execute some scrolling and human-like behavior
                        driver.execute_script("""
                            window.scrollTo(0, 300);
                            setTimeout(() => window.scrollTo(0, 600), 300);
                            setTimeout(() => window.scrollTo(0, 900), 600);
                        """)

                        time.sleep(2)

                        # Get page content
                        content = driver.page_source

                        # Close driver
                        driver.quit()

                        return content
                    except Exception as inner_e:
                        logger.error(f"Undetected Chrome inner error: {str(inner_e)}")
                        if driver:
                            try: # Add try-except for quit
                                driver.quit()
                            except Exception as quit_e:
                                logger.error(f"Error quitting undetected chrome driver: {quit_e}")
                        return None

                # Run in executor to avoid blocking
                content = await loop.run_in_executor(None, _fetch_with_undetected_chrome)

                if content:
                    logger.debug(f"Successfully fetched with Undetected Chrome: {url}")
                    return content

            except Exception as e:
                logger.error(f"Undetected Chrome error for {url}: {str(e)}")

        # If all headless browser attempts fail, return None
        logger.error(f"All headless browser attempts failed for {url} (under semaphore)")
        # Implicitly release semaphore here as `async with` block ends
        return None
    # Implicitly release semaphore if we exit the `async with` block
    # Add explicit logging for release
    logger.debug(f"🔓 Released BROWSER semaphore for: {url}")
    return None # Should not be reached if successful, but safety return

    # If all special techniques fail, return None and let the caller try standard methods
    return None

    # If all special techniques fail, return None and let the caller try standard methods
    return None 