#!/usr/bin/env python
"""
Test script that writes to a file
"""

import os
import sys
import logging
import networkx as nx

# Write to a file
with open('test_output.txt', 'w') as f:
    f.write("Starting test script\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Current working directory: {os.getcwd()}\n")
    
    try:
        f.write("Attempting to import modules...\n")
        from generate_glossary.deduplicator.graph_dedup import build_term_graph, add_rule_based_edges
        from generate_glossary.deduplicator.dedup_utils import (
            get_term_variations, get_plural_variations, get_spelling_variations, get_dash_space_variations
        )
        f.write("Successfully imported modules\n\n")
        
        # Create a test set of terms
        test_terms = [
            "Asian studies",
            "Asian law",
            "art education",
            "art history",
            "communication",
            "communication disorders", 
            "communication sciences",
            "communication studies"
        ]
        
        f.write(f"Test terms: {test_terms}\n\n")
        
        # Create a mock terms_by_level dictionary (all at level 1)
        terms_by_level = {1: test_terms}
        
        # Test term variations
        for term in test_terms:
            f.write(f"Variations for '{term}':\n")
            f.write(f"- Term variations: {get_term_variations(term)}\n")
            f.write(f"- Plural variations: {get_plural_variations(term)}\n")
            f.write(f"- Spelling variations: {get_spelling_variations(term)}\n")
            f.write(f"- Dash variations: {get_dash_space_variations(term)}\n\n")
        
        # Build the graph
        f.write("Building graph...\n")
        G = build_term_graph(terms_by_level)
        f.write(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges\n")
        
        # Add rule-based edges (this is where academic suffixes are processed)
        f.write("\nAdding rule-based edges...\n")
        G = add_rule_based_edges(G, terms_by_level)
        f.write(f"After adding rule-based edges: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges\n\n")
        
        # Print the graph structure
        f.write("Graph Structure:\n")
        for u, v, data in G.edges(data=True):
            f.write(f"Edge: {u} - {v}\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Check for paths between problem pairs
        problem_pairs = [
            ("Asian studies", "Asian law"),
            ("art education", "art history"),
            ("communication", "communication disorders")
        ]
        
        f.write("\nPaths between problem term pairs:\n")
        
        for source, target in problem_pairs:
            f.write(f"\nPaths from '{source}' to '{target}':\n")
            
            if source not in G.nodes or target not in G.nodes:
                f.write(f"Error: One or both nodes not in graph ({source} or {target})\n")
                continue
                
            if nx.has_path(G, source, target):
                for path in nx.all_simple_paths(G, source, target, cutoff=3):
                    f.write(f"  Path: {' -> '.join(path)}\n")
                    # Print edge details for each step in the path
                    for i in range(len(path) - 1):
                        edge_data = G.edges[path[i], path[i+1]]
                        f.write(f"    Edge {path[i]} -> {path[i+1]}: {edge_data}\n")
            else:
                # If no direct path, check if there's an indirect path through common neighbors
                source_neighbors = set(G.neighbors(source))
                target_neighbors = set(G.neighbors(target))
                common = source_neighbors & target_neighbors
                
                if common:
                    f.write(f"  No direct path, but common neighbors: {common}\n")
                    for neighbor in common:
                        f.write(f"    {source} -> {neighbor}: {G.edges[source, neighbor]}\n")
                        f.write(f"    {neighbor} -> {target}: {G.edges[neighbor, target]}\n")
                else:
                    f.write("  No path or common neighbors found\n")
                    
    except ImportError as e:
        f.write(f"Import error: {e}\n")
    except Exception as e:
        f.write(f"Unexpected error: {e}\n")
        
print("Test script completed. Check test_output.txt for results.") 