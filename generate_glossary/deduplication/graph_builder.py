"""
Core graph builder for deduplication.

The graph is the first-class citizen - all deduplication revolves around building
and managing this graph progressively.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx


def create_deduplication_graph() -> nx.Graph:
    """
    Create an empty deduplication graph.
    
    Returns:
        Empty NetworkX graph with metadata
    """
    graph = nx.Graph()
    graph.graph["type"] = "deduplication"
    graph.graph["levels_processed"] = []
    graph.graph["total_nodes"] = 0
    graph.graph["total_edges"] = 0
    graph.graph["edge_types"] = set()
    return graph


def add_terms_as_nodes(
    graph: nx.Graph,
    terms: List[str],
    level: int,
    metadata: Optional[Dict[str, Any]] = None
) -> nx.Graph:
    """
    Add terms as nodes to the graph.
    
    Args:
        graph: Deduplication graph
        terms: List of terms to add
        level: Hierarchy level of these terms
        metadata: Optional metadata for terms
        
    Returns:
        Updated graph with new nodes
    """
    nodes_added = 0
    
    for term in terms:
        if term not in graph:
            node_attrs = {
                "level": level,
                "original_term": term,
                "canonical": None,  # Will be set during canonical selection
            }
            
            # Add metadata if provided
            if metadata and term in metadata:
                node_attrs.update(metadata[term])
            
            graph.add_node(term, **node_attrs)
            nodes_added += 1
    
    # Update graph metadata
    graph.graph["total_nodes"] = graph.number_of_nodes()
    if level not in graph.graph["levels_processed"]:
        graph.graph["levels_processed"].append(level)
    
    logging.info(f"Added {nodes_added} nodes at level {level} (total: {graph.number_of_nodes()})")
    return graph


def get_graph_stats(graph: nx.Graph) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the graph.
    
    Args:
        graph: Deduplication graph
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_components": nx.number_connected_components(graph),
        "levels_processed": sorted(graph.graph.get("levels_processed", [])),
        "edge_types": list(graph.graph.get("edge_types", set())),
        "nodes_by_level": {},
        "edges_by_type": {},
        "edges_by_method": {},
        "component_sizes": []
    }
    
    # Count nodes by level
    for node in graph.nodes():
        level = graph.nodes[node].get("level", -1)
        stats["nodes_by_level"][level] = stats["nodes_by_level"].get(level, 0) + 1
    
    # Count edges by type and method
    for u, v, data in graph.edges(data=True):
        edge_type = data.get("edge_type", "unknown")
        method = data.get("method", "unknown")
        
        stats["edges_by_type"][edge_type] = stats["edges_by_type"].get(edge_type, 0) + 1
        stats["edges_by_method"][method] = stats["edges_by_method"].get(method, 0) + 1
    
    # Get component sizes
    for component in nx.connected_components(graph):
        stats["component_sizes"].append(len(component))
    stats["component_sizes"].sort(reverse=True)
    
    # Add summary statistics
    if stats["component_sizes"]:
        stats["largest_component"] = stats["component_sizes"][0]
        stats["num_singletons"] = sum(1 for size in stats["component_sizes"] if size == 1)
    
    return stats


def find_connected_components(graph: nx.Graph) -> List[set]:
    """
    Find all connected components in the graph.
    
    Args:
        graph: Deduplication graph
        
    Returns:
        List of sets, each containing terms in a component
    """
    return list(nx.connected_components(graph))


def get_component_for_term(graph: nx.Graph, term: str) -> set:
    """
    Get the connected component containing a specific term.
    
    Args:
        graph: Deduplication graph
        term: Term to find component for
        
    Returns:
        Set of terms in the same component
    """
    if term not in graph:
        return set()
    
    return nx.node_connected_component(graph, term)










def remove_weak_edges(
    graph: nx.Graph,
    threshold: float = 0.3
) -> nx.Graph:
    """
    Remove edges with weight below threshold.
    
    Args:
        graph: Deduplication graph
        threshold: Minimum weight to keep edge
        
    Returns:
        Graph with weak edges removed
    """
    edges_to_remove = []
    
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 0)
        if weight < threshold:
            edges_to_remove.append((u, v))
    
    graph.remove_edges_from(edges_to_remove)
    
    if edges_to_remove:
        logging.info(f"Removed {len(edges_to_remove)} weak edges (threshold={threshold})")
    
    # Update graph metadata
    graph.graph["total_edges"] = graph.number_of_edges()
    
    return graph


def merge_components(
    graph: nx.Graph,
    component1: set,
    component2: set,
    edge_data: Optional[Dict[str, Any]] = None
) -> nx.Graph:
    """
    Merge two components by adding an edge between them.
    
    Args:
        graph: Deduplication graph
        component1: First component
        component2: Second component
        edge_data: Optional edge attributes
        
    Returns:
        Graph with components merged
    """
    # Find best representatives from each component
    rep1 = select_canonical_from_component(graph, component1)
    rep2 = select_canonical_from_component(graph, component2)
    
    # Add edge with high weight to ensure components stay merged
    if edge_data is None:
        edge_data = {"weight": 0.95, "edge_type": "manual_merge", "method": "manual"}
    
    graph.add_edge(rep1, rep2, **edge_data)
    
    logging.info(f"Merged components: {rep1} <-> {rep2}")
    
    return graph


def validate_graph(graph: nx.Graph) -> Tuple[bool, List[str]]:
    """
    Validate the graph structure and data.
    
    Args:
        graph: Deduplication graph
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for isolated nodes
    isolated = list(nx.isolates(graph))
    if isolated:
        issues.append(f"Found {len(isolated)} isolated nodes")
    
    # Check for missing node attributes
    for node in graph.nodes():
        if "level" not in graph.nodes[node]:
            issues.append(f"Node '{node}' missing 'level' attribute")
            break  # Only report first to avoid spam
    
    # Check for missing edge attributes
    for u, v in graph.edges():
        data = graph.edges[u, v]
        if "weight" not in data:
            issues.append(f"Edge {u}-{v} missing 'weight' attribute")
            break  # Only report first
        if "edge_type" not in data:
            issues.append(f"Edge {u}-{v} missing 'edge_type' attribute")
            break
    
    # Check for self-loops
    if list(nx.selfloop_edges(graph)):
        issues.append("Graph contains self-loops")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logging.warning(f"Graph validation found {len(issues)} issue(s)")
    
    return is_valid, issues