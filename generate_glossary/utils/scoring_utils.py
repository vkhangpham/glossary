"""Pure functions for source prioritization scoring."""

from functools import lru_cache
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .verification_utils import get_educational_scores_async

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 4  # Maximum number of worker processes/threads

# Constants for domain trust scoring
TRUSTED_TLDS = {
    'edu': 1.0,  # Educational institutions
    'gov': 0.9,  # Government institutions
    'org': 0.7,  # Non-profit organizations
    'ac.uk': 0.9,  # UK academic institutions
    'edu.au': 0.9,  # Australian educational institutions
}

TRUSTED_DOMAINS = {
    'wikipedia.org': 0.8,
    'arxiv.org': 0.8,
    'scholar.google.com': 0.8,
    'researchgate.net': 0.7,
    'sciencedirect.com': 0.8,
    'ieee.org': 0.8,
    'acm.org': 0.8,
    'springer.com': 0.8,
    'nature.com': 0.8,
    'science.org': 0.8,
}

class ScoringError(Exception):
    """Base exception for scoring-related errors."""
    pass

class InvalidWeightsError(ScoringError):
    """Exception raised for invalid weight values."""
    pass

class InvalidURLError(ScoringError):
    """Exception raised for invalid or malformed URLs."""
    pass

@lru_cache(maxsize=1024)
def get_domain_trust_score(url: str) -> float:
    """
    Calculate domain trust score based on TLD and domain.
    Returns a score between 0.0 and 1.0.
    
    Args:
        url: Source URL
        
    Returns:
        Domain trust score between 0.0 and 1.0
        
    Raises:
        InvalidURLError: If the URL is malformed or invalid
    """
    try:
        domain = urlparse(url).netloc.lower()
        if not domain:
            raise InvalidURLError(f"Invalid URL: {url}")
        
        logger.debug(f"Calculating domain trust score for {domain}")
        
        # Check for trusted domains first
        for trusted_domain, score in TRUSTED_DOMAINS.items():
            if trusted_domain in domain:
                logger.debug(f"Found trusted domain {trusted_domain} with score {score}")
                return score
        
        # Check for trusted TLDs
        for tld, score in TRUSTED_TLDS.items():
            if domain.endswith(f'.{tld}'):
                logger.debug(f"Found trusted TLD {tld} with score {score}")
                return score
        
        # Default score for unknown domains
        logger.debug(f"Using default score 0.5 for unknown domain {domain}")
        return 0.5
        
    except Exception as e:
        logger.error(f"Error calculating domain trust score for {url}: {e}")
        raise InvalidURLError(f"Error processing URL {url}: {str(e)}")

def normalize_educational_score(raw_score: float) -> float:
    """
    Normalize educational score from 0-5 range to 0-1 range.
    
    Args:
        raw_score: Raw educational score (0-5)
        
    Returns:
        Normalized score between 0.0 and 1.0
    """
    logger.debug(f"Normalizing educational score {raw_score}")
    return min(1.0, max(0.0, raw_score / 5.0))

def validate_weights(weights: Tuple[float, float]) -> None:
    """
    Validate weight values.
    
    Args:
        weights: Tuple of (domain_weight, educational_weight)
        
    Raises:
        InvalidWeightsError: If weights are invalid
    """
    domain_weight, edu_weight = weights
    
    if not (isinstance(domain_weight, (int, float)) and isinstance(edu_weight, (int, float))):
        raise InvalidWeightsError("Weights must be numeric values")
    
    if not (0.0 <= domain_weight <= 1.0 and 0.0 <= edu_weight <= 1.0):
        raise InvalidWeightsError(
            f"Weights must be between 0.0 and 1.0, got: domain={domain_weight}, edu={edu_weight}"
        )
    
    if abs(domain_weight + edu_weight - 1.0) > 1e-6:
        raise InvalidWeightsError(
            f"Weights must sum to 1.0, got: domain={domain_weight}, edu={edu_weight}, sum={domain_weight + edu_weight}"
        )

async def calculate_source_score_async(
    url: str,
    content: str,
    weights: Tuple[float, float] = (0.5, 0.5)
) -> float:
    """
    Calculate composite source score based on domain trust and educational quality.
    
    Args:
        url: Source URL
        content: Source content
        weights: Tuple of (domain_weight, educational_weight), must sum to 1.0
        
    Returns:
        Composite score between 0.0 and 1.0
        
    Raises:
        InvalidURLError: If the URL is invalid
        InvalidWeightsError: If weights are invalid
    """
    try:
        logger.debug(f"Calculating source score for {url}")
        logger.debug(f"Using weights: domain={weights[0]}, edu={weights[1]}")
        
        # Validate weights
        validate_weights(weights)
        domain_weight, edu_weight = weights
        
        # Get individual scores in parallel
        domain_score = get_domain_trust_score(url)
        edu_scores = await get_educational_scores_async([content])
        edu_score = normalize_educational_score(edu_scores[0])
        
        logger.debug(f"Individual scores: domain={domain_score}, edu={edu_score}")
        
        # Calculate weighted average
        composite_score = domain_score * domain_weight + edu_score * edu_weight
        logger.debug(f"Composite score: {composite_score}")
        
        return composite_score
        
    except (InvalidURLError, InvalidWeightsError):
        raise
    except Exception as e:
        logger.error(f"Error calculating source score: {e}")
        raise ScoringError(f"Error calculating source score: {str(e)}")

def calculate_source_score(
    url: str,
    content: str,
    weights: Tuple[float, float] = (0.5, 0.5)
) -> float:
    """Calculate composite source score."""
    return asyncio.run(calculate_source_score_async(url, content, weights))

async def prioritize_sources_async(
    sources: List[Tuple[str, str]],
    weights: Tuple[float, float] = (0.5, 0.5),
    show_progress: bool = False
) -> List[Tuple[str, str, float]]:
    """
    Score and sort sources by priority asynchronously.
    
    Args:
        sources: List of (url, content) tuples
        weights: Tuple of (domain_weight, educational_weight)
        show_progress: Whether to show progress bar
        
    Returns:
        List of (url, content, score) tuples, sorted by score in descending order
        
    Raises:
        InvalidWeightsError: If weights are invalid
        ScoringError: If there's an error scoring sources
    """
    try:
        logger.debug(f"Prioritizing {len(sources)} sources")
        logger.debug(f"Using weights: domain={weights[0]}, edu={weights[1]}")
        
        # Validate weights
        validate_weights(weights)
        
        # Score sources in parallel using ThreadPoolExecutor
        scored_sources = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create tasks for each source
            tasks = []
            for url, content in sources:
                task = asyncio.create_task(calculate_source_score_async(url, content, weights))
                tasks.append((url, content, task))
            
            # Wait for tasks to complete
            if show_progress:
                for url, content, task in tqdm(tasks, desc="Scoring sources"):
                    try:
                        score = await task
                        scored_sources.append((url, content, score))
                        logger.debug(f"Scored {url}: {score}")
                    except (InvalidURLError, ScoringError) as e:
                        logger.warning(f"Error scoring source {url}: {e}")
                        continue
            else:
                for url, content, task in tasks:
                    try:
                        score = await task
                        scored_sources.append((url, content, score))
                        logger.debug(f"Scored {url}: {score}")
                    except (InvalidURLError, ScoringError) as e:
                        logger.warning(f"Error scoring source {url}: {e}")
                        continue
        
        # Sort by score
        sorted_sources = sorted(scored_sources, key=lambda x: x[2], reverse=True)
        logger.debug("Sources prioritized successfully")
        
        return sorted_sources
        
    except InvalidWeightsError:
        raise
    except Exception as e:
        logger.error(f"Error prioritizing sources: {e}")
        raise ScoringError(f"Error prioritizing sources: {str(e)}")

def prioritize_sources(
    sources: List[Tuple[str, str]],
    weights: Tuple[float, float] = (0.5, 0.5),
    show_progress: bool = False
) -> List[Tuple[str, str, float]]:
    """Score and sort sources by priority."""
    return asyncio.run(prioritize_sources_async(sources, weights, show_progress)) 