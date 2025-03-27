#!/usr/bin/env python
"""
Test script for academic suffix detection in graph_dedup.py
"""

import logging
import sys
import os
import networkx as nx
from pprint import pprint

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    print(f"Adding {project_root} to sys.path")
    sys.path.append(project_root)

try:
    from generate_glossary.deduplicator.graph_dedup import build_term_graph, add_rule_based_edges
    from generate_glossary.deduplicator.dedup_utils import (
        get_term_variations, get_plural_variations, get_spelling_variations, get_dash_space_variations
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

# Set up logging to show debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    """Test academic suffix detection with specific examples."""
    print("Starting test script")
    
    # Create a test set of terms
    test_terms = [
        "Asian studies",
        "Asian law",
        "Asian languages",
        "art education",
        "art history",
        "art",
        "art theory",
        "communication",
        "communication disorders", 
        "communication sciences",
        "communication studies",
        "family medicine",
        "family studies",
        "science education",
        "science"
    ]
    
    print(f"Test terms: {test_terms}")
    
    # Create a mock terms_by_level dictionary (all at level 1)
    terms_by_level = {1: test_terms}
    
    # Debug: Show term variations for key terms
    interesting_terms = [
        "Asian studies", "Asian law", "art education", "art history", 
        "communication", "communication disorders"
    ]
    
    print("Checking term variations:")
    for term in interesting_terms:
        print(f"\n### Variations for '{term}':")
        print(f"- Term variations: {get_term_variations(term)}")
        print(f"- Plural variations: {get_plural_variations(term)}")
        print(f"- Spelling variations: {get_spelling_variations(term)}")
        print(f"- Dash variations: {get_dash_space_variations(term)}")
    
    # Build the graph
    print("\nBuilding graph...")
    G = build_term_graph(terms_by_level)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Add rule-based edges (this is where academic suffixes are processed)
    print("\nAdding rule-based edges...")
    G = add_rule_based_edges(G, terms_by_level)
    print(f"After adding rule-based edges: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Print the graph structure
    print("\n### Graph Structure:")
    for u, v, data in G.edges(data=True):
        print(f"Edge: {u} - {v}")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Find and print paths between specific problem terms
    print("\n### Paths between problem term pairs:")
    problem_pairs = [
        ("Asian studies", "Asian law"),
        ("art education", "art history"),
        ("communication", "communication disorders")
    ]
    
    for source, target in problem_pairs:
        print(f"\nPaths from '{source}' to '{target}':")
        if nx.has_path(G, source, target):
            for path in nx.all_simple_paths(G, source, target, cutoff=3):
                print(f"  Path: {' -> '.join(path)}")
                # Print edge details for each step in the path
                for i in range(len(path) - 1):
                    edge_data = G.edges[path[i], path[i+1]]
                    print(f"    Edge {path[i]} -> {path[i+1]}: {edge_data}")
        else:
            # If no direct path, check if there's an indirect path through common neighbors
            source_neighbors = set(G.neighbors(source))
            target_neighbors = set(G.neighbors(target))
            common = source_neighbors & target_neighbors
            
            if common:
                print(f"  No direct path, but common neighbors: {common}")
                for neighbor in common:
                    print(f"    {source} -> {neighbor}: {G.edges[source, neighbor]}")
                    print(f"    {neighbor} -> {target}: {G.edges[neighbor, target]}")
            else:
                print("  No path or common neighbors found")

if __name__ == "__main__":
    main() 