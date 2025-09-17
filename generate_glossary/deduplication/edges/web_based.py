"""
Web-based deduplication for adding edges to the graph.

Uses web resources like URLs and content overlap to determine similarity.
"""

import logging
from typing import Dict, List, Any, Optional
import networkx as nx

from ..types import Edge, WebConfig, create_default_domain_patterns


def create_web_edges(
    terms: List[str], 
    web_content: Dict[str, Dict[str, Any]], 
    config: Optional[WebConfig] = None
) -> List[Edge]:
    """
    Create web-based edges using pure functional approach.
    
    Args:
        terms: List of terms to process
        web_content: Dictionary mapping terms to their web resources.
                    Expected structure for each term:
                    {
                        "term_name": {
                            "results": [
                                {
                                    "url": str,
                                    "title": Optional[str],
                                    "content": Optional[str], 
                                    "relevance_score": Optional[float],
                                    "domain": Optional[str]
                                },
                                ...
                            ],
                            "metadata": Optional[Dict[str, Any]]
                        }
                    }
        config: Optional WebConfig object (uses defaults if None)
        
    Returns:
        List of Edge objects representing web-based relationships
        
    Raises:
        ValueError: If web_content structure is invalid or missing required keys
    """
    if config is None:
        config = WebConfig()
    
    # Validate web_content structure and filter valid terms
    valid_terms = []
    for term in terms:
        if term not in web_content:
            continue
            
        term_data = web_content[term]
        if not isinstance(term_data, dict):
            logging.warning(f"Invalid web_content structure for term '{term}': expected dict, got {type(term_data)}")
            continue
            
        if "results" not in term_data:
            logging.warning(f"Missing 'results' key in web_content for term '{term}'")
            continue
            
        if not isinstance(term_data["results"], list):
            logging.warning(f"Invalid 'results' structure for term '{term}': expected list, got {type(term_data['results'])}")
            continue
            
        valid_terms.append(term)
    
    if not valid_terms:
        logging.warning("No terms with valid web_content structure found")
        return []
    
    edges = []
    
    # Add web overlap edges
    edges.extend(_find_web_overlap_edges(valid_terms, web_content, config))
    
    # Add domain-specific edges (if enabled)
    if config.enable_domain_specific:
        edges.extend(_find_domain_specific_edges(valid_terms, web_content, config))
    
    # Add content similarity edges (if enabled)
    if config.enable_content_similarity:
        edges.extend(_find_content_similarity_edges(valid_terms, web_content, config))
    
    return edges


def _find_web_overlap_edges(
    terms: List[str], 
    web_content: Dict[str, Any], 
    config: WebConfig
) -> List[Edge]:
    """Find web overlap edges between terms."""
    edges = []
    
    for i, term1 in enumerate(terms):
        content1 = web_content.get(term1, {"results": []})
        if not content1:
            continue

        urls1 = extract_urls(content1.get("results", []))

        for term2 in terms[i+1:]:
            content2 = web_content.get(term2, {"results": []})
            if not content2:
                continue

            urls2 = extract_urls(content2.get("results", []))
            
            # Calculate URL overlap
            overlap = calculate_url_overlap(urls1, urls2)
            
            if overlap >= config.min_url_overlap:
                # Calculate relevance-weighted similarity
                relevance_score = calculate_relevance_score(content1.get("results", []), content2.get("results", []))
                
                if relevance_score >= config.min_relevance_score:
                    weight = min(1.0, relevance_score + 0.1 * overlap)
                    edges.append(create_web_overlap_edge(
                        term1, term2, weight, overlap, relevance_score
                    ))
    
    return edges


def _find_domain_specific_edges(
    terms: List[str], 
    web_content: Dict[str, Any], 
    config: WebConfig
) -> List[Edge]:
    """Find domain-specific edges between terms."""
    edges = []
    
    for i, term1 in enumerate(terms):
        domains1 = extract_domains(web_content.get(term1, {"results": []}).get("results", []))

        for term2 in terms[i+1:]:
            domains2 = extract_domains(web_content.get(term2, {"results": []}).get("results", []))
            
            # Calculate domain-weighted similarity
            similarity = calculate_domain_similarity(domains1, domains2, config.domain_patterns)
            
            if similarity > 0.5:  # Threshold for domain similarity
                edges.append(create_domain_specific_edge(term1, term2, similarity))
    
    return edges


def _find_content_similarity_edges(
    terms: List[str], 
    web_content: Dict[str, Any], 
    config: WebConfig
) -> List[Edge]:
    """Find content similarity edges between terms."""
    edges = []
    
    for i, term1 in enumerate(terms):
        text1 = extract_text_content(web_content.get(term1, {"results": []}).get("results", []))
        if not text1:
            continue

        for term2 in terms[i+1:]:
            text2 = extract_text_content(web_content.get(term2, {"results": []}).get("results", []))
            if not text2:
                continue
            
            # Calculate content similarity
            similarity = calculate_text_overlap(text1, text2)
            
            if similarity >= config.min_content_similarity:
                edges.append(create_content_similarity_edge(term1, term2, similarity))
    
    return edges

def add_web_based_edges(
    graph: nx.Graph,
    web_content: Dict[str, Any],
    terms: Optional[List[str]] = None,
    min_url_overlap: int = 2,
    min_relevance_score: float = 0.3,
    *,
    config: Optional[WebConfig] = None
) -> nx.Graph:
    """
    Add edges based on web resource overlap.

    DEPRECATED: Use create_web_edges() for pure functional approach.
    This function is kept for backward compatibility.

    Args:
        graph: NetworkX graph to add edges to
        web_content: Dictionary mapping terms to their web resources
        terms: Optional list of terms to process (if None, process all)
        min_url_overlap: Minimum number of overlapping URLs (fallback for backward compatibility)
        min_relevance_score: Minimum average relevance score (fallback for backward compatibility)
        config: Optional WebConfig object

    Returns:
        Updated graph with web-based edges
    """
    import warnings
    warnings.warn(
        "add_web_based_edges is deprecated. Use create_web_edges() for pure functional approach.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if terms is None:
        terms = [term for term in graph.nodes() if term in web_content]
    else:
        terms = [term for term in terms if term in graph and term in web_content]

    # Use config values if provided, otherwise create fallback config
    if config is None:
        config = WebConfig(
            min_url_overlap=min_url_overlap,
            min_relevance_score=min_relevance_score
        )

    # Get edges from pure function
    edges = create_web_edges(terms, web_content, config)
    
    # Add edges to graph
    edges_added = 0
    for edge in edges:
        # Only add if both terms exist in graph and edge doesn't exist
        if (edge.source in graph and edge.target in graph and 
            not graph.has_edge(edge.source, edge.target)):
            # Add edge with all metadata
            edge_data = {
                "weight": edge.weight,
                "edge_type": edge.edge_type,
                "method": edge.method
            }
            # Add any metadata from the edge
            if edge.metadata:
                edge_data.update(edge.metadata)
                
            graph.add_edge(edge.source, edge.target, **edge_data)
            edges_added += 1
            
            # Log with appropriate detail based on edge type
            if edge.edge_type == "web_overlap":
                ol = edge.metadata.get('url_overlap') if edge.metadata else None
                rs = edge.metadata.get('relevance_score') if edge.metadata else None
                logging.debug(
                    f"Added web edge: {edge.source} <-> {edge.target} "
                    f"(overlap={ol if ol is not None else 'N/A'}, "
                    f"relevance={rs:.3f if isinstance(rs, (int, float)) else 'N/A'})"
                )
            else:
                logging.debug(f"Added {edge.edge_type} edge: {edge.source} <-> {edge.target}")
    
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
        # Check for nested results structure
        if "results" in web_content:
            for item in web_content["results"]:
                if isinstance(item, dict):
                    if "relevance" in item:
                        scores.append(float(item["relevance"]))
                    elif "score" in item:
                        scores.append(float(item["score"]))
                    elif "confidence" in item:
                        scores.append(float(item["confidence"]))
        # Fallback to existing behavior for flat dicts
        elif "relevance_scores" in web_content:
            scores.extend(web_content["relevance_scores"])
    
    return scores


def add_domain_specific_edges(
    graph: nx.Graph,
    web_content: Dict[str, Any],
    terms: Optional[List[str]] = None,
    domain_patterns: Optional[Dict[str, float]] = None,
    *,
    config: Optional[WebConfig] = None
) -> nx.Graph:
    """
    Add edges based on domain-specific patterns in URLs.

    DEPRECATED: Use create_web_edges() for pure functional approach.
    This function is kept for backward compatibility.

    Args:
        graph: NetworkX graph
        web_content: Web content for terms
        terms: Optional list of terms to process
        domain_patterns: Domain patterns and their weights (fallback for backward compatibility)
        config: Optional WebConfig object

    Returns:
        Updated graph with domain-specific edges
    """
    import warnings
    warnings.warn(
        "add_domain_specific_edges is deprecated. Use create_web_edges() for pure functional approach.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if terms is None:
        terms = [term for term in graph.nodes() if term in web_content]
    else:
        terms = [term for term in terms if term in graph and term in web_content]

    # Check if domain-specific edges are enabled
    if config is not None and not config.enable_domain_specific:
        logging.debug("Domain-specific edges disabled in config, skipping")
        return graph

    # Use config patterns if provided, otherwise use individual parameter or defaults
    if config is None:
        patterns = domain_patterns if domain_patterns is not None else create_default_domain_patterns()
        config = WebConfig(domain_patterns=patterns, enable_domain_specific=True)

    # Get domain-specific edges from pure function
    edges = _find_domain_specific_edges(terms, web_content, config)
    
    # Add edges to graph
    edges_added = 0
    for edge in edges:
        # Only add if both terms exist in graph and edge doesn't exist
        if (edge.source in graph and edge.target in graph and 
            not graph.has_edge(edge.source, edge.target)):
            graph.add_edge(
                edge.source, edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type,
                method=edge.method
            )
            edges_added += 1
            logging.debug(f"Added domain edge: {edge.source} <-> {edge.target} (sim={edge.weight:.3f})")

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
    min_content_similarity: float = 0.6,
    *,
    config: Optional[WebConfig] = None
) -> nx.Graph:
    """
    Add edges based on web content text similarity.

    DEPRECATED: Use create_web_edges() for pure functional approach.
    This function is kept for backward compatibility.

    Args:
        graph: NetworkX graph
        web_content: Web content for terms
        terms: Optional list of terms to process
        min_content_similarity: Minimum content similarity threshold (fallback for backward compatibility)
        config: Optional WebConfig object

    Returns:
        Updated graph with content similarity edges
    """
    import warnings
    warnings.warn(
        "add_content_similarity_edges is deprecated. Use create_web_edges() for pure functional approach.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if terms is None:
        terms = [term for term in graph.nodes() if term in web_content]
    else:
        terms = [term for term in terms if term in graph and term in web_content]

    # Check if content similarity edges are enabled
    if config is not None and not config.enable_content_similarity:
        logging.debug("Content similarity edges disabled in config, skipping")
        return graph

    # Use config values if provided, otherwise create fallback config
    if config is None:
        config = WebConfig(
            min_content_similarity=min_content_similarity,
            enable_content_similarity=True
        )

    # Get content similarity edges from pure function
    edges = _find_content_similarity_edges(terms, web_content, config)
    
    # Add edges to graph
    edges_added = 0
    for edge in edges:
        # Only add if both terms exist in graph and edge doesn't exist
        if (edge.source in graph and edge.target in graph and 
            not graph.has_edge(edge.source, edge.target)):
            graph.add_edge(
                edge.source, edge.target,
                weight=edge.weight,
                edge_type=edge.edge_type,
                method=edge.method
            )
            edges_added += 1
            logging.debug(f"Added content edge: {edge.source} <-> {edge.target} (sim={edge.weight:.3f})")

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
        # Check for nested results structure
        if "results" in web_content:
            for item in web_content["results"]:
                if isinstance(item, dict):
                    if "title" in item:
                        texts.append(item["title"])
                    if "description" in item:
                        texts.append(item["description"])
                    if "snippet" in item:
                        texts.append(item["snippet"])
                    if "content" in item:
                        texts.append(item["content"])
        # Fallback to existing behavior for flat dicts
        else:
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


# Edge creation helper functions (for future functional conversion)

def create_web_overlap_edge(term1: str, term2: str, weight: float, url_overlap: int, relevance_score: float) -> Edge:
    """Create a web overlap edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=weight,
        edge_type="web_overlap",
        method="web_based",
        metadata={
            "url_overlap": url_overlap,
            "relevance_score": relevance_score
        }
    )


def create_domain_specific_edge(term1: str, term2: str, similarity: float) -> Edge:
    """Create a domain-specific edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=similarity,
        edge_type="domain_specific",
        method="web_based"
    )


def create_content_similarity_edge(term1: str, term2: str, similarity: float) -> Edge:
    """Create a content similarity edge object."""
    return Edge(
        source=term1,
        target=term2,
        weight=similarity,
        edge_type="content_similarity",
        method="web_based"
    )