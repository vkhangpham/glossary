#!/usr/bin/env python
"""
Simplified test script for debugging import issues
"""

import os
import sys
import logging

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    print("Attempting to import from generate_glossary...")
    import generate_glossary
    print(f"generate_glossary package is located at: {generate_glossary.__file__}")
    
    print("Attempting to import deduplicator module...")
    from generate_glossary import deduplicator
    print(f"deduplicator module is located at: {deduplicator.__file__}")
    
    print("Attempting to import graph_dedup...")
    from generate_glossary.deduplicator import graph_dedup
    print(f"graph_dedup module is located at: {graph_dedup.__file__}")
    
    print("Attempting to import specific functions...")
    from generate_glossary.deduplicator.graph_dedup import build_term_graph, add_rule_based_edges
    print("Successfully imported build_term_graph and add_rule_based_edges")
    
    print("Attempting to import from dedup_utils...")
    from generate_glossary.deduplicator.dedup_utils import get_term_variations
    print("Successfully imported get_term_variations")
    
    # Basic test of functionality
    print("\nTesting a simple function call:")
    variations = get_term_variations("test")
    print(f"Variations of 'test': {variations}")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}") 