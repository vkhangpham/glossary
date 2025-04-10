#!/usr/bin/env python

import json
import os

def check_level_sources(level, pattern):
    """Check if the sources in the given level metadata contain the pattern."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'))
    metadata_file = os.path.join(data_dir, f'lv{level}', f'lv{level}_metadata.json')
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pattern_count = 0
    terms_with_pattern = []
    
    for term, term_data in data.items():
        for source in term_data.get('sources', []):
            if pattern.lower() in source.lower():
                pattern_count += 1
                if term not in terms_with_pattern:
                    terms_with_pattern.append(term)
                print(f"Term '{term}' has source containing '{pattern}': {source}")
    
    print(f"Total sources with '{pattern}': {pattern_count}")
    print(f"Total terms with sources containing '{pattern}': {len(terms_with_pattern)}")

if __name__ == "__main__":
    print("Checking level 1 sources for 'college of':")
    check_level_sources(1, "college of")
    
    print("\nChecking level 2 sources for 'department of':")
    check_level_sources(2, "department of") 