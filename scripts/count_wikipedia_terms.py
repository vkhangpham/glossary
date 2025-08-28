#!/usr/bin/env python3
"""
Script to count how many terms have at least one Wikipedia resource.
Searches through data/final/lvX/lvX_filtered_resources.json files.
"""
import json
import os
import glob
import re
from pathlib import Path

def is_wikipedia_url(url):
    """Check if a URL is a Wikipedia URL from any language edition."""
    # Match Wikipedia URLs for any language edition and protocol
    # Examples: https://en.wikipedia.org/wiki/, https://fr.wikipedia.org/wiki/, 
    #           http://simple.wikipedia.org/wiki/, https://commons.wikipedia.org/wiki/
    pattern = r'https?://(www\.)?[a-z-]+\.wikipedia\.org/wiki/'
    return bool(re.match(pattern, url))

def count_wikipedia_terms_in_file(file_path):
    """Count terms with Wikipedia resources in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        wikipedia_terms = set()
        total_terms = len(data)
        
        for term, resources in data.items():
            has_wikipedia = any(is_wikipedia_url(resource['url']) for resource in resources)
            if has_wikipedia:
                wikipedia_terms.add(term)
        
        return len(wikipedia_terms), total_terms, wikipedia_terms
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, set()

def main():
    # Find all filtered_resources.json files
    base_path = Path("data/final")
    pattern = "lv*/lv*_filtered_resources.json"
    
    files = list(base_path.glob(pattern))
    files.sort()  # Sort for consistent ordering
    
    if not files:
        print("No filtered_resources.json files found!")
        return
    
    print("Wikipedia Resource Count Analysis")
    print("=" * 50)
    
    total_wikipedia_terms = set()
    total_terms_overall = 0
    level_stats = []
    
    for file_path in files:
        level = file_path.parent.name
        wikipedia_count, total_count, wikipedia_terms = count_wikipedia_terms_in_file(file_path)
        
        level_stats.append({
            'level': level,
            'wikipedia_count': wikipedia_count,
            'total_count': total_count,
            'percentage': (wikipedia_count / total_count * 100) if total_count > 0 else 0
        })
        
        total_wikipedia_terms.update(wikipedia_terms)
        total_terms_overall += total_count
        
        print(f"\n{level.upper()}:")
        print(f"  Terms with Wikipedia resources: {wikipedia_count:,}")
        print(f"  Total terms: {total_count:,}")
        print(f"  Percentage: {wikipedia_count / total_count * 100:.1f}%")
    
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY:")
    print(f"  Unique terms with Wikipedia resources: {len(total_wikipedia_terms):,}")
    print(f"  Total terms across all levels: {total_terms_overall:,}")
    print(f"  Overall percentage: {len(total_wikipedia_terms) / total_terms_overall * 100:.1f}%")
    
    # Show breakdown by level
    print("\n" + "=" * 50)
    print("LEVEL BREAKDOWN:")
    for stat in level_stats:
        print(f"  {stat['level']:3}: {stat['wikipedia_count']:4,} / {stat['total_count']:4,} ({stat['percentage']:5.1f}%)")

if __name__ == "__main__":
    main() 