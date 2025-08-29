"""
API functions for querying the deduplication graph.

These functions provide different views of the graph data
without modifying the graph itself.
"""

import logging
from typing import Dict, List, Any, Set, Optional
import networkx as nx

from .canonical_selector import (
    select_canonical_terms as _select_canonical,
    get_canonical_groups,
    filter_canonical_by_level
)
from .graph_io import load_graph


def get_canonical_terms(
    graph: nx.Graph,
    prefer_higher_level: bool = False,
    level: Optional[int] = None
) -> List[str]:
    """
    Get the list of canonical terms from the graph.
    
    Args:
        graph: Deduplication graph
        prefer_higher_level: Whether to prefer higher level terms
        level: Optional - only return canonicals for a specific level
        
    Returns:
        List of canonical terms
    """
    # Get canonical mapping
    canonical_mapping = _select_canonical(graph, prefer_higher_level)
    
    # Filter by level if specified
    if level is not None:
        return filter_canonical_by_level(canonical_mapping, graph, level)
    
    # Return all unique canonical terms
    return sorted(set(canonical_mapping.values()))


def get_terms_with_variations(
    graph: nx.Graph,
    prefer_higher_level: bool = False
) -> Dict[str, List[str]]:
    """
    Get canonical terms with their variations.
    
    Args:
        graph: Deduplication graph
        prefer_higher_level: Whether to prefer higher level terms
        
    Returns:
        Dictionary mapping canonical terms to their variations
    """
    # Get canonical mapping
    canonical_mapping = _select_canonical(graph, prefer_higher_level)
    
    # Group by canonical
    groups = get_canonical_groups(canonical_mapping)
    
    # Format result - exclude canonical from its own variations
    result = {}
    for canonical, terms in groups.items():
        variations = [t for t in terms if t != canonical]
        result[canonical] = variations
    
    return result


def get_variations_for_term(
    graph: nx.Graph,
    term: str
) -> List[str]:
    """
    Get all variations of a specific term (its connected component).
    
    Args:
        graph: Deduplication graph
        term: Term to get variations for
        
    Returns:
        List of variations (including the term itself)
    """
    if term not in graph:
        return []
    
    # Get connected component
    component = nx.node_connected_component(graph, term)
    return sorted(component)


def get_duplicate_pairs(graph: nx.Graph) -> List[tuple]:
    """
    Get all pairs of terms that are duplicates (have edges between them).
    
    Args:
        graph: Deduplication graph
        
    Returns:
        List of (term1, term2) tuples
    """
    return list(graph.edges())


def get_duplicate_clusters(graph: nx.Graph) -> List[Set[str]]:
    """
    Get all duplicate clusters (connected components).
    
    Args:
        graph: Deduplication graph
        
    Returns:
        List of sets, each containing terms in a cluster
    """
    return list(nx.connected_components(graph))


def is_duplicate_pair(
    graph: nx.Graph,
    term1: str,
    term2: str
) -> bool:
    """
    Check if two terms are duplicates (connected in the graph).
    
    Args:
        graph: Deduplication graph
        term1: First term
        term2: Second term
        
    Returns:
        True if terms are in the same connected component
    """
    if term1 not in graph or term2 not in graph:
        return False
    
    # Check if they're in the same component
    return nx.has_path(graph, term1, term2)


def get_edge_info(
    graph: nx.Graph,
    term1: str,
    term2: str
) -> Dict[str, Any]:
    """
    Get information about the edge between two terms.
    
    Args:
        graph: Deduplication graph
        term1: First term
        term2: Second term
        
    Returns:
        Edge data dictionary or empty dict if no edge
    """
    if graph.has_edge(term1, term2):
        return graph.edges[term1, term2]
    return {}


def get_term_info(
    graph: nx.Graph,
    term: str
) -> Dict[str, Any]:
    """
    Get all information about a term from the graph.
    
    Args:
        graph: Deduplication graph
        term: Term to get info for
        
    Returns:
        Dictionary with term info including neighbors, edges, etc.
    """
    if term not in graph:
        return {}
    
    # Get node data
    info = dict(graph.nodes[term])
    
    # Add connectivity info
    info["degree"] = graph.degree(term)
    info["neighbors"] = list(graph.neighbors(term))
    
    # Add component info
    component = nx.node_connected_component(graph, term)
    info["component_size"] = len(component)
    info["component_members"] = sorted(component)
    
    # Add edge info
    edges = []
    for neighbor in graph.neighbors(term):
        edge_data = graph.edges[term, neighbor].copy()
        edge_data["neighbor"] = neighbor
        edges.append(edge_data)
    info["edges"] = edges
    
    return info


def get_graph_summary(graph: nx.Graph) -> Dict[str, Any]:
    """
    Get a summary of the graph structure.
    
    Args:
        graph: Deduplication graph
        
    Returns:
        Summary statistics
    """
    components = list(nx.connected_components(graph))
    component_sizes = [len(c) for c in components]
    
    summary = {
        "total_terms": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "num_components": len(components),
        "num_singletons": sum(1 for size in component_sizes if size == 1),
        "num_duplicates": sum(1 for size in component_sizes if size > 1),
        "largest_cluster_size": max(component_sizes) if component_sizes else 0,
        "avg_cluster_size": sum(component_sizes) / len(component_sizes) if component_sizes else 0
    }
    
    # Add level distribution
    level_counts = {}
    for node in graph.nodes():
        level = graph.nodes[node].get("level", -1)
        level_counts[level] = level_counts.get(level, 0) + 1
    summary["terms_by_level"] = level_counts
    
    # Add edge type distribution
    edge_types = {}
    for _, _, data in graph.edges(data=True):
        edge_type = data.get("edge_type", "unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    summary["edges_by_type"] = edge_types
    
    return summary


def query_graph(
    graph_path: str,
    query_type: str,
    **kwargs
) -> Any:
    """
    Load a graph and query it.
    
    Args:
        graph_path: Path to saved graph
        query_type: Type of query to run
        **kwargs: Additional arguments for the query
        
    Returns:
        Query result
    """
    # Load graph
    graph = load_graph(graph_path)
    
    # Run query based on type
    if query_type == "canonical":
        return get_canonical_terms(graph, **kwargs)
    elif query_type == "variations":
        return get_terms_with_variations(graph, **kwargs)
    elif query_type == "clusters":
        return get_duplicate_clusters(graph)
    elif query_type == "summary":
        return get_graph_summary(graph)
    elif query_type == "term_info":
        return get_term_info(graph, kwargs.get("term"))
    else:
        raise ValueError(f"Unknown query type: {query_type}")


# Convenience functions for common queries
def get_deduplicated_terms(
    graph: nx.Graph,
    level: Optional[int] = None
) -> List[str]:
    """
    Get deduplicated terms (canonical terms only).
    
    This is what you use when you want a clean list without duplicates.
    
    Args:
        graph: Deduplication graph
        level: Optional level filter
        
    Returns:
        List of canonical terms
    """
    return get_canonical_terms(graph, level=level)


def get_all_variations(graph: nx.Graph) -> Dict[str, List[str]]:
    """
    Get all terms grouped by their canonical form.
    
    This is what you use when you need to know all variations.
    
    Args:
        graph: Deduplication graph
        
    Returns:
        Dictionary of canonical -> variations
    """
    return get_terms_with_variations(graph)