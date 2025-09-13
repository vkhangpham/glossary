#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = 'data'
FINAL_DIR = os.path.join(DATA_DIR, 'final')


def load_metadata(level: int, verbose: bool = False) -> Dict[str, Dict[str, List[str]]]:
    """Load metadata from the specified level.
    
    Prioritizes data from the final directory (data/final) created by the metadata collector.
    Falls back to the original location only if final directory data doesn't exist.
    """
    # First try to load from the final directory
    final_metadata_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_metadata.json')
    
    # Fall back to original location if final doesn't exist
    original_metadata_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_metadata.json')
    
    # Choose the appropriate file to load
    using_final = os.path.exists(final_metadata_file)
    metadata_file = final_metadata_file if using_final else original_metadata_file
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return {}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if verbose:
        source = "final" if using_final else "original"
        print(f"Loaded metadata for level {level} from {source} directory: {metadata_file}")
        
    return data


def load_resources(level: int, verbose: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Load resources from the specified level.
    
    Prioritizes data from the final directory (data/final) created by the metadata collector.
    Falls back to other locations only if final directory data doesn't exist.
    """
    # First try final directory resources
    final_resources_file = os.path.join(FINAL_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
    
    # Then try filtered resources in original location
    filtered_resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_filtered_resources.json')
    
    # Then fall back to full resources in original location
    resources_file = os.path.join(DATA_DIR, f'lv{level}', f'lv{level}_resources.json')
    
    # Try each location in order
    if os.path.exists(final_resources_file):
        with open(final_resources_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if verbose:
                print(f"Loaded resources for level {level} from final directory: {final_resources_file}")
            return data
    elif os.path.exists(filtered_resources_file):
        with open(filtered_resources_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if verbose:
                print(f"Loaded resources for level {level} from filtered resources: {filtered_resources_file}")
            return data
    elif os.path.exists(resources_file):
        with open(resources_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if verbose:
                print(f"Loaded resources for level {level} from original resources: {resources_file}")
            return data
    
    return {}


def build_hierarchy(output_file: str = None, 
                    transfer_sources: bool = True, 
                    transfer_resources: bool = True,
                    rank_parents: bool = False,
                    min_parent_score: float = 2.0,
                    max_parents: int = 3,
                    verbose: bool = False) -> Dict[str, Any]:
    """Build a hierarchical data structure from metadata across all levels."""
    # Check if final directory exists
    final_dir_exists = os.path.exists(FINAL_DIR) and any(os.path.exists(os.path.join(FINAL_DIR, f'lv{level}')) for level in range(4))
    
    if not final_dir_exists:
        print("\nWARNING: The data/final directory was not found or is empty.")
        print("         This usually means that the metadata collector has not been run with promotion enabled.")
        print("         The hierarchy will be built using the original data files, which may not have")
        print("         the proper term promotion for consistent parent-child relationships.\n")
        print("         Consider running: python -m generate_glossary.metadata_collector_cli 3 -v\n")
    else:
        print(f"Building hierarchy using data from data/final directory")
    
    # Load metadata from all levels
    metadata = {}
    resources = {}
    
    for level in range(4):  # Levels 0, 1, 2, 3
        level_metadata = load_metadata(level, verbose)
        if verbose:
            print(f"Loaded {len(level_metadata)} terms from level {level}")
        metadata[level] = level_metadata
        
        if transfer_resources:
            level_resources = load_resources(level, verbose)
            if verbose:
                print(f"Loaded resources for {len(level_resources)} terms from level {level}")
            resources[level] = level_resources
    
    # Build the hierarchy
    hierarchy = {
        "levels": {0: [], 1: [], 2: [], 3: []},  # Added level 3
        "relationships": {
            "parent_child": [],  # (parent, child, level) tuples
            "variations": []     # (term, variation) tuples
        },
        "terms": {},  # All terms with their metadata
    }
    
    # Process each level and populate the hierarchy
    for level, level_metadata in metadata.items():
        for term, term_data in level_metadata.items():
            # Add term to the level list
            hierarchy["levels"][level].append(term)
            
            # Add term to the terms dictionary
            if term not in hierarchy["terms"]:
                hierarchy["terms"][term] = {
                    "level": level,
                    "sources": term_data["sources"],
                    "parents": term_data["parents"],
                    "variations": term_data["variations"],
                    "children": [],
                    "resources": resources.get(level, {}).get(term, []),
                    "related_concepts": term_data.get("related_concepts", {}),
                    "definition": term_data.get("definition", "")
                }
            
            # Process parent-child relationships
            for parent in term_data["parents"]:
                # Only add valid parent-child relationship if parent exists in our hierarchy
                for parent_level in range(level):
                    if parent in metadata[parent_level]:
                        hierarchy["relationships"]["parent_child"].append((parent, term, level))
                        
                        # Add child to parent's children list
                        if parent in hierarchy["terms"]:
                            if term not in hierarchy["terms"][parent]["children"]:
                                hierarchy["terms"][parent]["children"].append(term)
            
            # Process variations
            for variation in term_data["variations"]:
                hierarchy["relationships"]["variations"].append((term, variation))
    
    # Transfer sources and resources from variations to canonical terms
    if transfer_sources or transfer_resources:
        # Create a lookup dictionary for variations to canonical terms
        variation_to_canonical = {}
        for canonical, variation in hierarchy["relationships"]["variations"]:
            variation_to_canonical[variation] = canonical
            
        variation_count = 0
        
        # First collect all terms including variations from all metadata
        all_terms = set()
        for level, level_metadata in metadata.items():
            all_terms.update(level_metadata.keys())
        
        # For any term that is a variation, transfer its metadata to the canonical term
        for term in all_terms:
            if term in variation_to_canonical:
                canonical = variation_to_canonical[term]
                variation_count += 1
                
                # Find the term's metadata in all levels
                for level, level_metadata in metadata.items():
                    if term in level_metadata:
                        var_data = level_metadata[term]
                        
                        # Transfer sources if enabled
                        if transfer_sources and canonical in hierarchy["terms"]:
                            for source in var_data["sources"]:
                                if source not in hierarchy["terms"][canonical]["sources"]:
                                    hierarchy["terms"][canonical]["sources"].append(source)
                
                # Transfer resources if enabled
                if transfer_resources and canonical in hierarchy["terms"]:
                    for level, level_resources in resources.items():
                        if term in level_resources:
                            # Avoid duplicates by checking URLs
                            existing_urls = {r["url"] for r in hierarchy["terms"][canonical]["resources"] 
                                           if "url" in r}
                            
                            for resource in level_resources[term]:
                                if "url" in resource and resource["url"] not in existing_urls:
                                    hierarchy["terms"][canonical]["resources"].append(resource)
                                    existing_urls.add(resource["url"])
        
        if verbose:
            print(f"Transferred metadata from {variation_count} variations to their canonical terms")
    
    # Compute additional statistics
    hierarchy["stats"] = {
        "total_terms": sum(len(terms) for terms in hierarchy["levels"].values()),
        "terms_by_level": {level: len(terms) for level, terms in hierarchy["levels"].items()},
        "total_relationships": len(hierarchy["relationships"]["parent_child"]),
        "total_variations": len(hierarchy["relationships"]["variations"])
    }
    
    # Save hierarchy to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"Hierarchy saved to {output_file}")
    
    return hierarchy


def export_graph(hierarchy: Dict[str, Any], output_file: str, format: str = 'gml', verbose: bool = False) -> None:
    """Export the hierarchy as a graph file in the specified format."""
    G = nx.DiGraph()
    
    # Add nodes for all terms
    for level, terms in hierarchy["levels"].items():
        for term in terms:
            G.add_node(term, level=level)
    
    # Add edges for parent-child relationships
    for parent, child, _ in hierarchy["relationships"]["parent_child"]:
        G.add_edge(parent, child, relationship_type="parent_child")
    
    # Add edges for variations
    for term, variation in hierarchy["relationships"]["variations"]:
        if variation not in G:
            G.add_node(variation, level=-1, is_variation=True)
        G.add_edge(term, variation, relationship_type="variation")
    
    # Export the graph
    if format == 'gml':
        nx.write_gml(G, output_file)
    elif format == 'graphml':
        nx.write_graphml(G, output_file)
    elif format == 'json':
        data = nx.node_link_data(G)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if verbose:
        print(f"Graph exported to {output_file} in {format} format")


def visualize_sample(hierarchy: Dict[str, Any], output_file: str = None, 
                     max_nodes: int = 100, verbose: bool = False) -> None:
    """Create a visualization of a sample of the hierarchy."""
    # Create a graph
    G = nx.DiGraph()
    
    # Sample some terms from each level
    sampled_terms = set()
    for level in range(4):  # Sample from all 4 levels
        level_terms = hierarchy["levels"][level]
        sample_size = min(max_nodes // 4, len(level_terms))  # Adjusted for 4 levels
        
        # Take sample_size random terms
        import random
        sampled_level_terms = random.sample(level_terms, sample_size)
        sampled_terms.update(sampled_level_terms)
    
    # Add nodes for sampled terms
    for term in sampled_terms:
        term_data = hierarchy["terms"][term]
        G.add_node(term, level=term_data["level"])
    
    # Add edges for parent-child relationships
    for parent, child, _ in hierarchy["relationships"]["parent_child"]:
        if parent in sampled_terms and child in sampled_terms:
            G.add_edge(parent, child)
    
    # Create positions for nodes
    pos = {}
    level_counts = defaultdict(int)
    
    for node in G.nodes():
        level = G.nodes[node].get("level", 0)
        pos[node] = (level_counts[level], -level)
        level_counts[level] += 1
    
    # Adjust positions to spread nodes evenly
    for level in level_counts:
        count = level_counts[level]
        if count > 1:
            for node in G.nodes():
                if G.nodes[node].get("level", 0) == level:
                    x, y = pos[node]
                    pos[node] = (x / (count - 1) * 10, y)
    
    # Draw the graph
    plt.figure(figsize=(15, 10))
    
    # Draw nodes by level with different colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Added color for level 3
    for level in range(4):  # Draw all 4 levels
        level_nodes = [node for node in G.nodes() if G.nodes[node].get("level", -1) == level]
        nx.draw_networkx_nodes(G, pos, nodelist=level_nodes, node_color=colors[level], 
                              node_size=100, alpha=0.8, label=f"Level {level}")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, arrows=True, arrowsize=10)
    
    # Draw labels for nodes
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.title("Academic Hierarchy Sample Visualization")
    plt.legend()
    plt.axis('off')
    
    # Save or show the visualization
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Build and visualize hierarchy from term metadata')
    parser.add_argument('-o', '--output', type=str, default='data/hierarchy.json',
                        help='Output file path for the hierarchy (default: data/hierarchy.json)')
    parser.add_argument('-g', '--graph', type=str, default=None,
                        help='Export graph to file (specified by path)')
    parser.add_argument('-f', '--format', type=str, default='gml', choices=['gml', 'graphml', 'json'],
                        help='Format for graph export (default: gml)')
    parser.add_argument('-v', '--visualize', type=str, default=None,
                        help='Create a visualization and save to the specified path')
    parser.add_argument('-m', '--max-nodes', type=int, default=100,
                        help='Maximum number of nodes to include in visualization (default: 100)')
    parser.add_argument('--no-transfer-sources', action='store_true',
                        help='Disable transferring sources from variations to canonical terms')
    parser.add_argument('--no-transfer-resources', action='store_true',
                        help='Disable transferring resources from variations to canonical terms')
    parser.add_argument('--rank-parents', action='store_true', default=False,
                        help='Apply parent ranking to filter and prioritize parent relationships')
    parser.add_argument('--min-parent-score', type=float, default=2.0,
                        help='Minimum score for a parent to be included when ranking (default: 2.0)')
    parser.add_argument('--max-parents', type=int, default=3,
                        help='Maximum number of parents to keep per term when ranking (default: 3)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Build the hierarchy
    hierarchy = build_hierarchy(
        args.output, 
        not args.no_transfer_sources, 
        not args.no_transfer_resources,
        args.rank_parents,
        args.min_parent_score,
        args.max_parents,
        args.verbose
    )
    
    # Export graph if requested
    if args.graph:
        export_graph(hierarchy, args.graph, args.format, args.verbose)
    
    # Visualize if requested
    if args.visualize:
        visualize_sample(hierarchy, args.visualize, args.max_nodes, args.verbose)
    
    # Print summary
    if args.verbose:
        print("\nHierarchy Summary:")
        print(f"Total terms: {hierarchy['stats']['total_terms']}")
        print(f"Terms by level: {hierarchy['stats']['terms_by_level']}")
        print(f"Total parent-child relationships: {hierarchy['stats']['total_relationships']}")
        print(f"Total variations: {hierarchy['stats']['total_variations']}")


if __name__ == '__main__':
    main() 