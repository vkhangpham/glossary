"""
Web-based deduplication for adding edges to the graph.

Uses web resources like URLs and content overlap to determine similarity.
"""

import logging
from typing import Dict, List, Any, Optional
import networkx as nx


def add_web_based_edges(
    graph: nx.Graph,
    web_content: Dict[str, Any],
    terms: Optional[List[str]] = None,
    min_url_overlap: int = 2,
    min_relevance_score: float = 0.3
) -> nx.Graph:
    """
    Add edges based on web resource overlap.
    
    Args:
        graph: NetworkX graph to add edges to
        web_content: Dictionary mapping terms to their web resources
        terms: Optional list of terms to process (if None, process all)
        min_url_overlap: Minimum number of overlapping URLs
        min_relevance_score: Minimum average relevance score
        
    Returns:
        Updated graph with web-based edges
    """
    if terms is None:
        terms = [term for term in graph.nodes() if term in web_content]
    else:
        terms = [term for term in terms if term in graph and term in web_content]
    
    edges_added = 0
    
    for i, term1 in enumerate(terms):
        content1 = web_content.get(term1, [])
        if not content1:
            continue
            
        urls1 = extract_urls(content1)
        
        for term2 in terms[i+1:]:
            if graph.has_edge(term1, term2):
                continue
                
            content2 = web_content.get(term2, [])
            if not content2:
                continue
                
            urls2 = extract_urls(content2)
            
            # Calculate URL overlap
            overlap = calculate_url_overlap(urls1, urls2)
            
            if overlap >= min_url_overlap:
                # Calculate relevance-weighted similarity
                relevance_score = calculate_relevance_score(content1, content2)
                
                if relevance_score >= min_relevance_score:
                    weight = min(1.0, relevance_score + 0.1 * overlap)
                    graph.add_edge(
                        term1, term2,
                        weight=weight,
                        edge_type="web_overlap",
                        method="web_based",
                        url_overlap=overlap,
                        relevance_score=relevance_score
                    )
                    edges_added += 1
                    logging.debug(
                        f"Added web edge: {term1} <-> {term2} "
                        f"(overlap={overlap}, relevance={relevance_score:.3f})"
                    )
    
    logging.info(f"Added {edges_added} web-based edges")
    return graph


def extract_urls(web_content: Any) -> set:
    """
    Extract URLs from web content.
    
    Args:
        web_content: Web content (list or dict format)
        
    Returns:
        Set of URLs
    """
    urls = set()
    
    if isinstance(web_content, list):
        for item in web_content:
            if isinstance(item, dict):
                if "url" in item:
                    urls.add(normalize_url(item["url"]))
                elif "link" in item:
                    urls.add(normalize_url(item["link"]))
    elif isinstance(web_content, dict):
        if "urls" in web_content:
            for url in web_content["urls"]:
                urls.add(normalize_url(url))
        if "sources" in web_content:
            for source in web_content["sources"]:
                if isinstance(source, dict) and "url" in source:
                    urls.add(normalize_url(source["url"]))
    
    return urls


def normalize_url(url: str) -> str:
    """
    Normalize URL for comparison.
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL
    """
    # Remove protocol
    url = url.replace("https://", "").replace("http://", "")
    
    # Remove www
    if url.startswith("www."):
        url = url[4:]
    
    # Remove trailing slash
    url = url.rstrip("/")
    
    # Remove fragment
    if "#" in url:
        url = url.split("#")[0]
    
    # Remove common tracking parameters
    if "?" in url:
        base, params = url.split("?", 1)
        # Keep only essential parameters
        essential_params = []
        for param in params.split("&"):
            if not any(track in param.lower() for track in ["utm_", "ref=", "source="]):
                essential_params.append(param)
        if essential_params:
            url = base + "?" + "&".join(essential_params)
        else:
            url = base
    
    return url.lower()


def calculate_url_overlap(urls1: set, urls2: set) -> int:
    """
    Calculate the number of overlapping URLs.
    
    Args:
        urls1: First set of URLs
        urls2: Second set of URLs
        
    Returns:
        Number of overlapping URLs
    """
    return len(urls1.intersection(urls2))


def calculate_relevance_score(content1: Any, content2: Any) -> float:
    """
    Calculate relevance score based on web content.
    
    Args:
        content1: First term's web content
        content2: Second term's web content
        
    Returns:
        Relevance score between 0 and 1
    """
    scores1 = extract_relevance_scores(content1)
    scores2 = extract_relevance_scores(content2)
    
    if not scores1 or not scores2:
        return 0.0
    
    # Average of both terms' average scores
    avg1 = sum(scores1) / len(scores1)
    avg2 = sum(scores2) / len(scores2)
    
    return (avg1 + avg2) / 2


def extract_relevance_scores(web_content: Any) -> List[float]:
    """
    Extract relevance scores from web content.
    
    Args:
        web_content: Web content
        
    Returns:
        List of relevance scores
    """
    scores = []
    
    if isinstance(web_content, list):
        for item in web_content:
            if isinstance(item, dict):
                if "relevance" in item:
                    scores.append(float(item["relevance"]))
                elif "score" in item:
                    scores.append(float(item["score"]))
                elif "confidence" in item:
                    scores.append(float(item["confidence"]))
    elif isinstance(web_content, dict):
        if "relevance_scores" in web_content:
            scores.extend(web_content["relevance_scores"])
    
    return scores


def add_domain_specific_edges(
    graph: nx.Graph,
    web_content: Dict[str, Any],
    terms: Optional[List[str]] = None,
    domain_patterns: Optional[Dict[str, float]] = None
) -> nx.Graph:
    """
    Add edges based on domain-specific patterns in URLs.
    
    Args:
        graph: NetworkX graph
        web_content: Web content for terms
        terms: Optional list of terms to process
        domain_patterns: Domain patterns and their weights
        
    Returns:
        Updated graph with domain-specific edges
    """
    if domain_patterns is None:
        domain_patterns = {
            "arxiv.org": 0.9,  # Academic papers
            "acm.org": 0.85,  # ACM Digital Library
            "ieee.org": 0.85,  # IEEE
            "springer.com": 0.8,  # Springer
            "sciencedirect.com": 0.8,  # ScienceDirect
            ".edu": 0.75,  # Educational institutions
            "wikipedia.org": 0.7,  # Wikipedia
            "github.com": 0.65  # GitHub
        }
    
    if terms is None:
        terms = [term for term in graph.nodes() if term in web_content]
    else:
        terms = [term for term in terms if term in graph and term in web_content]
    
    edges_added = 0
    
    for i, term1 in enumerate(terms):
        domains1 = extract_domains(web_content.get(term1, []))
        
        for term2 in terms[i+1:]:
            if graph.has_edge(term1, term2):
                continue
            
            domains2 = extract_domains(web_content.get(term2, []))
            
            # Calculate domain-weighted similarity
            similarity = calculate_domain_similarity(domains1, domains2, domain_patterns)
            
            if similarity > 0.5:  # Threshold for domain similarity
                graph.add_edge(
                    term1, term2,
                    weight=similarity,
                    edge_type="domain_specific",
                    method="web_based"
                )
                edges_added += 1
                logging.debug(f"Added domain edge: {term1} <-> {term2} (sim={similarity:.3f})")
    
    if edges_added > 0:
        logging.info(f"Added {edges_added} domain-specific edges")
    
    return graph


def extract_domains(web_content: Any) -> List[str]:
    """
    Extract domains from web content URLs.
    
    Args:
        web_content: Web content
        
    Returns:
        List of domains
    """
    urls = extract_urls(web_content)
    domains = []
    
    for url in urls:
        # Extract domain from URL
        if "/" in url:
            domain = url.split("/")[0]
        else:
            domain = url
        domains.append(domain)
    
    return domains


def calculate_domain_similarity(
    domains1: List[str],
    domains2: List[str],
    domain_patterns: Dict[str, float]
) -> float:
    """
    Calculate similarity based on domain patterns.
    
    Args:
        domains1: First list of domains
        domains2: Second list of domains
        domain_patterns: Domain patterns and weights
        
    Returns:
        Domain similarity score
    """
    if not domains1 or not domains2:
        return 0.0
    
    score = 0.0
    matches = 0
    
    for domain1 in domains1:
        for domain2 in domains2:
            for pattern, weight in domain_patterns.items():
                if pattern in domain1 and pattern in domain2:
                    score += weight
                    matches += 1
                    break
    
    if matches > 0:
        # Normalize by the number of possible comparisons
        max_comparisons = min(len(domains1), len(domains2))
        return score / max_comparisons
    
    return 0.0


def add_content_similarity_edges(
    graph: nx.Graph,
    web_content: Dict[str, Any],
    terms: Optional[List[str]] = None,
    min_content_similarity: float = 0.6
) -> nx.Graph:
    """
    Add edges based on web content text similarity.
    
    Args:
        graph: NetworkX graph
        web_content: Web content for terms
        terms: Optional list of terms to process
        min_content_similarity: Minimum content similarity threshold
        
    Returns:
        Updated graph with content similarity edges
    """
    if terms is None:
        terms = [term for term in graph.nodes() if term in web_content]
    else:
        terms = [term for term in terms if term in graph and term in web_content]
    
    edges_added = 0
    
    for i, term1 in enumerate(terms):
        text1 = extract_text_content(web_content.get(term1, []))
        if not text1:
            continue
        
        for term2 in terms[i+1:]:
            if graph.has_edge(term1, term2):
                continue
            
            text2 = extract_text_content(web_content.get(term2, []))
            if not text2:
                continue
            
            # Calculate content similarity
            similarity = calculate_text_overlap(text1, text2)
            
            if similarity >= min_content_similarity:
                graph.add_edge(
                    term1, term2,
                    weight=similarity,
                    edge_type="content_similarity",
                    method="web_based"
                )
                edges_added += 1
                logging.debug(f"Added content edge: {term1} <-> {term2} (sim={similarity:.3f})")
    
    if edges_added > 0:
        logging.info(f"Added {edges_added} content similarity edges")
    
    return graph


def extract_text_content(web_content: Any) -> str:
    """
    Extract text content from web resources.
    
    Args:
        web_content: Web content
        
    Returns:
        Combined text content
    """
    texts = []
    
    if isinstance(web_content, list):
        for item in web_content:
            if isinstance(item, dict):
                if "title" in item:
                    texts.append(item["title"])
                if "description" in item:
                    texts.append(item["description"])
                if "snippet" in item:
                    texts.append(item["snippet"])
                if "content" in item:
                    texts.append(item["content"])
    elif isinstance(web_content, dict):
        if "text" in web_content:
            texts.append(web_content["text"])
        if "summary" in web_content:
            texts.append(web_content["summary"])
    
    return " ".join(texts).lower()


def calculate_text_overlap(text1: str, text2: str) -> float:
    """
    Calculate text overlap using word-level Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score
    """
    # Simple word-level Jaccard similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0