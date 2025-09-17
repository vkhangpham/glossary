"""
Graph I/O functions for explicit saving and loading of deduplication graphs.

The graph is a primary output artifact, not a cache.
"""

import pickle
import json
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging


def save_graph(
    graph: nx.Graph,
    output_path: str,
    format: str = "pickle"
) -> Path:
    """
    Save graph to file as a primary output.
    
    Args:
        graph: NetworkX graph to save
        output_path: Path to save graph (without extension)
        format: Format to save in ('pickle', 'graphml', 'json')
        
    Returns:
        Path where graph was saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        # Most efficient, preserves all attributes
        file_path = output_path.with_suffix('.graph.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    elif format == "graphml":
        # Standard graph format, readable by other tools
        file_path = output_path.with_suffix('.graphml')
        nx.write_graphml(graph, file_path)
        
    elif format == "json":
        # Human-readable but may lose some attributes
        file_path = output_path.with_suffix('.graph.json')
        data = nx.node_link_data(graph)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logging.info(
        f"Saved graph to {file_path} "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    
    return file_path


def load_graph(
    input_path: str,
    format: Optional[str] = None
) -> nx.Graph:
    """
    Load graph from file.
    
    Args:
        input_path: Path to graph file
        format: Format to load ('pickle', 'graphml', 'json') or auto-detect
        
    Returns:
        Loaded NetworkX graph
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Graph file not found: {input_path}")
    
    # Auto-detect format from extension if not specified
    if format is None:
        if input_path.suffix == '.pkl':
            format = "pickle"
        elif input_path.suffix == '.graphml':
            format = "graphml"
        elif input_path.suffix == '.json':
            format = "json"
        else:
            # Try to detect from content
            format = "pickle"  # Default guess
    
    if format == "pickle":
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
            
    elif format == "graphml":
        graph = nx.read_graphml(input_path)
        
    elif format == "json":
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        graph = nx.node_link_graph(data)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logging.info(
        f"Loaded graph from {input_path} "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    
    return graph


def save_graph_with_metadata(
    graph: nx.Graph,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Path, Path]:
    """
    Save graph and its metadata as separate files.
    
    Args:
        graph: NetworkX graph to save
        output_path: Base path for output files (without extension)
        metadata: Additional metadata to save
        
    Returns:
        Tuple of (graph_path, metadata_path)
    """
    output_path = Path(output_path)
    
    # Save graph
    graph_path = save_graph(graph, str(output_path), format="pickle")
    
    # Prepare metadata
    meta = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_components": nx.number_connected_components(graph),
        "nodes_by_level": {},
        "edges_by_type": {}
    }
    
    # Count nodes by level
    for node in graph.nodes():
        level = graph.nodes[node].get("level", -1)
        meta["nodes_by_level"][str(level)] = meta["nodes_by_level"].get(str(level), 0) + 1
    
    # Count edges by type
    for _, _, edge_data in graph.edges(data=True):
        edge_type = edge_data.get("type", "unknown")
        meta["edges_by_type"][edge_type] = meta["edges_by_type"].get(edge_type, 0) + 1
    
    # Add custom metadata
    if metadata:
        meta.update(metadata)
    
    # Save metadata
    metadata_path = output_path.with_suffix('.graph_meta.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved graph metadata to {metadata_path}")
    
    return graph_path, metadata_path


def load_graph_with_metadata(
    input_path: str
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Load graph and its metadata.
    
    Args:
        input_path: Base path for input files (without extension)
        
    Returns:
        Tuple of (graph, metadata)
    """
    input_path = Path(input_path)
    
    # Try different extensions for graph file
    graph_file = None
    for ext in ['.graph.pkl', '.pkl', '.graphml', '.graph.json', '.json']:
        candidate = input_path.with_suffix(ext)
        if candidate.exists():
            graph_file = candidate
            break
    
    if not graph_file:
        raise FileNotFoundError(f"No graph file found with base path: {input_path}")
    
    # Load graph
    graph = load_graph(str(graph_file))
    
    # Load metadata if exists
    metadata = {}
    metadata_file = input_path.with_suffix('.graph_meta.json')
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logging.info(f"Loaded metadata from {metadata_file}")
    
    return graph, metadata


def export_graph_summary(
    graph: nx.Graph,
    output_path: str
) -> Path:
    """
    Export a human-readable summary of the graph.
    
    Args:
        graph: NetworkX graph to summarize
        output_path: Path for summary file
        
    Returns:
        Path where summary was saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("# Deduplication Graph Summary\n")
    summary_lines.append(f"Total Nodes: {graph.number_of_nodes()}")
    summary_lines.append(f"Total Edges: {graph.number_of_edges()}")
    summary_lines.append(f"Connected Components: {nx.number_connected_components(graph)}\n")
    
    # Nodes by level
    nodes_by_level = {}
    for node in graph.nodes():
        level = graph.nodes[node].get("level", -1)
        nodes_by_level[level] = nodes_by_level.get(level, [])
        nodes_by_level[level].append(node)
    
    summary_lines.append("## Nodes by Level")
    for level in sorted(nodes_by_level.keys()):
        summary_lines.append(f"Level {level}: {len(nodes_by_level[level])} nodes")
    
    # Edges by type
    edges_by_type = {}
    for _, _, edge_data in graph.edges(data=True):
        edge_type = edge_data.get("type", "unknown")
        edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1
    
    summary_lines.append("\n## Edges by Type")
    for edge_type in sorted(edges_by_type.keys()):
        summary_lines.append(f"{edge_type}: {edges_by_type[edge_type]} edges")
    
    # Component sizes
    components = list(nx.connected_components(graph))
    component_sizes = sorted([len(c) for c in components], reverse=True)
    
    summary_lines.append("\n## Component Sizes")
    summary_lines.append(f"Largest component: {component_sizes[0] if component_sizes else 0} nodes")
    summary_lines.append(f"Singleton components: {sum(1 for s in component_sizes if s == 1)}")
    
    # Sample large components
    if components:
        summary_lines.append("\n## Sample Large Components")
        for i, component in enumerate(sorted(components, key=len, reverse=True)[:5]):
            sample = sorted(list(component))[:5]
            summary_lines.append(f"Component {i+1} ({len(component)} nodes): {', '.join(sample)}...")
    
    # Write summary
    summary_path = output_path.with_suffix('.graph_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    logging.info(f"Exported graph summary to {summary_path}")
    
    return summary_path