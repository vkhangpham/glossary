"""Functional utilities for web content verification and quality scoring."""

from functools import lru_cache
from typing import Callable, List, Tuple, TypeVar, Dict, Any, Optional, Union
from urllib.parse import urlparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from tqdm import tqdm
import os
import json
import torch
import numpy as np
from unicodedata import normalize
import re

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
ContentType = TypeVar("ContentType")
RuleType = Callable[[ContentType], bool]
ScoreType = Callable[[ContentType], float]
VerificationResult = Tuple[bool, str, float]  # (is_verified, reason, score)

# Constants for FineWeb-Edu classifier
FINEWEB_MODEL_NAME = "HuggingFaceFW/fineweb-edu-classifier"
FINEWEB_MAX_LENGTH = 512  # Maximum sequence length for the model
BATCH_SIZE = 32  # Batch size for model inference
MAX_WORKERS = min(4, os.cpu_count() or 1)  # Maximum number of worker processes/threads

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

class InvalidURLError(Exception):
    """Exception raised for invalid or malformed URLs."""
    pass

# Global variables for model and tokenizer
_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_wikipedia_url(url: str) -> bool:
    """Check if a URL is from Wikipedia"""
    return "wikipedia.org" in url.lower()


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


def _get_fineweb_resources():
    """Lazy loading of FineWeb-Edu classifier resources"""
    global _model, _tokenizer
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(FINEWEB_MODEL_NAME).to(_device)
        _model.eval()  # Set to evaluation mode
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINEWEB_MODEL_NAME)
    return _tokenizer, _model


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to Python native types.
    
    Args:
        obj: Any object that might contain NumPy types
        
    Returns:
        Object with all NumPy types converted to Python native types
    """
    # Handle NumPy scalars
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    # Handle lists recursively
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    # Handle tuples recursively
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    # Return other types unchanged
    return obj


def _process_batch(batch: List[str]) -> List[float]:
    """Process a batch of content using FineWeb-Edu classifier."""
    tokenizer, model = _get_fineweb_resources()

    # Tokenize batch
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=FINEWEB_MAX_LENGTH,
    )
    # Move inputs to device
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1).float().cpu().numpy()

    # Process each score in the batch
    scores = []
    for score in logits:
        # Get raw score (0-5)
        raw_score = max(0, min(score, 5))
        # Convert NumPy float32 to Python float
        scores.append(float(raw_score))

    return scores


async def get_educational_scores_async(contents: List[str], show_progress: bool = False) -> List[float]:
    """Score multiple contents using FineWeb-Edu classifier asynchronously."""
    # Process in batches using ThreadPoolExecutor
    scores = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create batches
        batches = [
            contents[i:i + BATCH_SIZE]
            for i in range(0, len(contents), BATCH_SIZE)
        ]
        
        # Process batches
        futures = []
        for batch in batches:
            future = asyncio.get_event_loop().run_in_executor(
                executor, _process_batch, batch
            )
            futures.append(future)
        
        # Wait for all batches to complete
        if show_progress:
            for future in tqdm(
                asyncio.as_completed(futures),
                total=len(futures),
                desc="Scoring content"
            ):
                batch_scores = await future
                scores.extend(batch_scores)
        else:
            batch_results = await asyncio.gather(*futures)
            for batch_scores in batch_results:
                scores.extend(batch_scores)

    return scores


async def get_educational_score_async(content: str) -> float:
    """Score single content using FineWeb-Edu classifier."""
    scores = await get_educational_scores_async([content])
    return scores[0]


async def verify_content_async(
    url: str,
    content: str,
    min_score: float = 2.6,
    **kwargs
) -> VerificationResult:
    """Verify content based on educational quality score asynchronously.
    
    Args:
        url: URL of the content
        content: Processed content to verify
        min_score: Minimum score threshold for verification
        
    Returns:
        Tuple of (is_verified, reason, score)
    """
    # Get educational quality score for processed content
    edu_score = await get_educational_score_async(content)
    
    # Apply domain-based boosting
    boost = 0.0
    boost_reason = ""
    
    # Special case for Wikipedia (highest boost)
    if is_wikipedia_url(url):
        boost = 1.0
        boost_reason = "Wikipedia content"
    else:
        # Get domain trust score (0.5-1.0) and calculate boost (0.0-0.5)
        try:
            domain_trust = get_domain_trust_score(url)
            # Only boost if domain trust is above the baseline (0.5)
            if domain_trust > 0.5:
                boost = (domain_trust - 0.5) * 2  # Scale 0.5-1.0 to 0.0-1.0
                domain = urlparse(url).netloc
                boost_reason = f"trusted domain ({domain})"
        except Exception as e:
            logger.warning(f"Error calculating domain trust for {url}: {e}")
    
    # Apply the boost
    boosted_score = min(5.0, edu_score + boost)
    
    # Determine verification result
    is_verified = boosted_score >= min_score
    
    # Create detailed reason
    if boost > 0:
        reason = (
            f"Content verified with educational score {edu_score:.2f}/5.0 + {boost:.2f} boost for {boost_reason} = {boosted_score:.2f}/5.0"
            if is_verified
            else f"Content failed verification with educational score {edu_score:.2f}/5.0 + {boost:.2f} boost for {boost_reason} = {boosted_score:.2f}/5.0"
        )
    else:
        reason = (
            f"Content verified with educational score {boosted_score:.2f}/5.0"
            if is_verified
            else f"Content failed verification with educational score {boosted_score:.2f}/5.0"
        )

    # Convert NumPy types to Python native types
    return bool(is_verified), reason, float(boosted_score)


async def verify_batch_content_async(
    urls: List[str],
    contents: List[str],
    min_score: float = 2.6,
    show_progress: bool = False
) -> List[VerificationResult]:
    """Verify a batch of content asynchronously.
    
    Args:
        urls: List of URLs corresponding to the contents
        contents: List of processed content to verify
        min_score: Minimum score threshold for verification
        show_progress: Whether to show progress bars
        
    Returns:
        List of verification results (is_verified, reason, score)
    """
    # Get educational scores in batches for processed content
    edu_scores = await get_educational_scores_async(contents, show_progress)

    # Process results in parallel
    async def process_result(url: str, edu_score: float) -> VerificationResult:
        # Apply domain-based boosting
        boost = 0.0
        boost_reason = ""
        
        # Special case for Wikipedia (highest boost)
        if is_wikipedia_url(url):
            boost = 1.0
            boost_reason = "Wikipedia content"
        else:
            # Get domain trust score (0.5-1.0) and calculate boost (0.0-0.5)
            try:
                domain_trust = get_domain_trust_score(url)
                # Only boost if domain trust is above the baseline (0.5)
                if domain_trust > 0.5:
                    boost = (domain_trust - 0.5) * 2  # Scale 0.5-1.0 to 0.0-1.0
                    domain = urlparse(url).netloc
                    boost_reason = f"trusted domain ({domain})"
            except Exception as e:
                logger.warning(f"Error calculating domain trust for {url}: {e}")
        
        # Apply the boost
        boosted_score = min(5.0, edu_score + boost)
        
        # Determine verification result
        is_verified = boosted_score >= min_score
        
        # Create detailed reason
        if boost > 0:
            reason = (
                f"Content verified with educational score {edu_score:.2f}/5.0 + {boost:.2f} boost for {boost_reason} = {boosted_score:.2f}/5.0"
                if is_verified
                else f"Content failed verification with educational score {edu_score:.2f}/5.0 + {boost:.2f} boost for {boost_reason} = {boosted_score:.2f}/5.0"
            )
        else:
            reason = (
                f"Content verified with educational score {boosted_score:.2f}/5.0"
                if is_verified
                else f"Content failed verification with educational score {boosted_score:.2f}/5.0"
            )

        # Convert NumPy types to Python native types
        return bool(is_verified), reason, float(boosted_score)

    # Create tasks for parallel processing
    tasks = [process_result(url, score) for url, score in zip(urls, edu_scores)]
    
    # Wait for all tasks to complete
    if show_progress:
        results = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Verifying content"
        ):
            result = await task
            results.append(result)
    else:
        results = await asyncio.gather(*tasks)

    return results
