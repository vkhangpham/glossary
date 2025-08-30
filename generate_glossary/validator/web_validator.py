"""
Web content-based validation for terms.

This module validates terms by analyzing their associated web content
for relevance and quality.
"""

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .utils import calculate_relevance_score, extract_web_content_fields

# Default thresholds
DEFAULT_MIN_SCORE = 0.5
DEFAULT_MIN_RELEVANCE_SCORE = 0.5
MIN_VERIFIED_SOURCES = 1


def validate_with_web_content(
    terms: List[str],
    web_content: Dict[str, List[Dict[str, Any]]],
    min_score: float = DEFAULT_MIN_SCORE,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    show_progress: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Validate terms using their web content.
    
    Args:
        terms: List of terms to validate
        web_content: Dictionary mapping terms to their web content
        min_score: Minimum content score threshold
        min_relevance_score: Minimum relevance score threshold
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping terms to validation results
    """
    # Create validation function with fixed parameters
    def validate_term(term):
        return _validate_single_term_web(
            term, 
            web_content.get(term, []),
            min_score,
            min_relevance_score
        )
    
    # Use thread pool for parallel validation
    with ThreadPoolExecutor() as executor:
        if show_progress:
            results = list(tqdm(
                executor.map(validate_term, terms),
                total=len(terms),
                desc="Web validation"
            ))
        else:
            results = list(executor.map(validate_term, terms))
    
    # Convert to dictionary
    return {r["term"]: r for r in results}


def _validate_single_term_web(
    term: str,
    contents: List[Dict[str, Any]],
    min_score: float,
    min_relevance_score: float
) -> Dict[str, Any]:
    """
    Validate a single term using its web content.
    
    Args:
        term: Term to validate
        contents: List of web content for the term
        min_score: Minimum content score
        min_relevance_score: Minimum relevance score
        
    Returns:
        Validation result dictionary
    """
    if not contents:
        return {
            "term": term,
            "is_valid": False,
            "confidence": 0.0,
            "mode": "web",
            "details": {
                "error": "No web content available",
                "num_sources": 0,
                "verified_sources": [],
                "relevant_sources": [],
                "high_quality_sources": []
            }
        }
    
    # Analyze each content source
    verified_sources = []
    relevant_sources = []
    high_quality_sources = []
    all_scores = []
    
    for content in contents:
        # Extract content fields
        url, title, score, is_verified = extract_web_content_fields(content)
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(term, content)
        
        # Create source info
        source_info = {
            "url": url,
            "title": title,
            "content_score": round(score, 3),
            "relevance_score": round(relevance_score, 3),
            "is_verified": is_verified
        }
        
        # Categorize source
        if is_verified and score >= min_score:
            verified_sources.append(source_info)
            
            if relevance_score >= min_relevance_score:
                relevant_sources.append(source_info)
                
                # High quality if both scores are high
                if score >= 0.7 and relevance_score >= 0.7:
                    high_quality_sources.append(source_info)
        
        # Track all scores for statistics
        all_scores.append({
            "content": score,
            "relevance": relevance_score
        })
    
    # Sort sources by quality (combined score)
    for source_list in [verified_sources, relevant_sources, high_quality_sources]:
        source_list.sort(
            key=lambda x: x["content_score"] * x["relevance_score"],
            reverse=True
        )
    
    # Calculate validation confidence
    confidence = _calculate_web_confidence(
        len(verified_sources),
        len(relevant_sources),
        len(high_quality_sources),
        len(contents)
    )
    
    # Determine validity
    is_valid = len(relevant_sources) >= MIN_VERIFIED_SOURCES
    
    # Calculate average scores
    avg_content_score = sum(s["content"] for s in all_scores) / len(all_scores) if all_scores else 0
    avg_relevance_score = sum(s["relevance"] for s in all_scores) / len(all_scores) if all_scores else 0
    
    return {
        "term": term,
        "is_valid": is_valid,
        "confidence": round(confidence, 3),
        "mode": "web",
        "details": {
            "num_sources": len(contents),
            "verified_sources": verified_sources,
            "relevant_sources": relevant_sources,
            "high_quality_sources": high_quality_sources,
            "avg_content_score": round(avg_content_score, 3),
            "avg_relevance_score": round(avg_relevance_score, 3),
            "has_high_quality": bool(high_quality_sources)
        }
    }


def _calculate_web_confidence(
    verified_count: int,
    relevant_count: int,
    high_quality_count: int,
    total_count: int
) -> float:
    """
    Calculate confidence score based on web content analysis.
    
    Args:
        verified_count: Number of verified sources
        relevant_count: Number of relevant sources
        high_quality_count: Number of high quality sources
        total_count: Total number of sources
        
    Returns:
        Confidence score between 0 and 1
    """
    if total_count == 0:
        return 0.0
    
    # Weight different factors
    verified_ratio = verified_count / total_count
    relevant_ratio = relevant_count / max(1, verified_count)
    quality_ratio = high_quality_count / total_count
    
    # Weighted combination
    confidence = (
        0.3 * verified_ratio +
        0.4 * relevant_ratio +
        0.3 * quality_ratio
    )
    
    # Bonus for having multiple high-quality sources
    if high_quality_count >= 3:
        confidence = min(1.0, confidence + 0.1)
    
    return min(1.0, max(0.0, confidence))


def analyze_web_coverage(
    terms: List[str],
    web_content: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Analyze web content coverage for terms.
    
    Useful for understanding data quality before validation.
    
    Args:
        terms: List of terms to analyze
        web_content: Web content dictionary
        
    Returns:
        Coverage analysis statistics
    """
    coverage_stats = {
        "total_terms": len(terms),
        "terms_with_content": 0,
        "terms_without_content": 0,
        "total_sources": 0,
        "avg_sources_per_term": 0,
        "coverage_rate": 0,
        "missing_terms": []
    }
    
    sources_per_term = []
    
    for term in terms:
        contents = web_content.get(term, [])
        if contents:
            coverage_stats["terms_with_content"] += 1
            sources_per_term.append(len(contents))
            coverage_stats["total_sources"] += len(contents)
        else:
            coverage_stats["terms_without_content"] += 1
            coverage_stats["missing_terms"].append(term)
    
    if sources_per_term:
        coverage_stats["avg_sources_per_term"] = round(
            sum(sources_per_term) / len(sources_per_term), 2
        )
    
    if terms:
        coverage_stats["coverage_rate"] = round(
            coverage_stats["terms_with_content"] / len(terms), 3
        )
    
    return coverage_stats