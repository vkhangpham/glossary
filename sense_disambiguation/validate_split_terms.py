#!/usr/bin/env python3
"""
Validate which terms from split proposals actually exist in the hierarchy.
"""

import json
import os
from typing import Dict, List, Set

def load_hierarchy(hierarchy_file: str) -> Set[str]:
    """Load hierarchy and return set of term names."""
    with open(hierarchy_file, 'r') as f:
        data = json.load(f)
    return set(data.get('terms', {}).keys())

def load_split_proposals(split_results_dir: str) -> List[str]:
    """Load all accepted split proposal terms."""
    all_terms = []
    
    for level in range(4):
        filepath = os.path.join(split_results_dir, f"split_proposals_level{level}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for proposal in data.get('accepted_proposals', []):
                all_terms.append(proposal['original_term'])
    
    return all_terms

def main():
    hierarchy_file = "data/hierarchy.json"
    split_results_dir = "sense_disambiguation/data/sense_disambiguation_results/20250522_082713"
    
    print("Loading hierarchy terms...")
    hierarchy_terms = load_hierarchy(hierarchy_file)
    print(f"Found {len(hierarchy_terms)} terms in hierarchy")
    
    print("\nLoading split proposals...")
    split_terms = load_split_proposals(split_results_dir)
    print(f"Found {len(split_terms)} terms in split proposals")
    
    print("\nValidating which split terms exist in hierarchy:")
    found_terms = []
    missing_terms = []
    
    for term in split_terms:
        if term in hierarchy_terms:
            found_terms.append(term)
            print(f"✓ {term}")
        else:
            missing_terms.append(term)
            print(f"✗ {term}")
    
    print(f"\nSummary:")
    print(f"Terms found in hierarchy: {len(found_terms)}")
    print(f"Terms missing from hierarchy: {len(missing_terms)}")
    
    if missing_terms:
        print(f"\nMissing terms:")
        for term in missing_terms[:10]:  # Show first 10
            print(f"  - {term}")
        if len(missing_terms) > 10:
            print(f"  ... and {len(missing_terms) - 10} more")

if __name__ == "__main__":
    main() 