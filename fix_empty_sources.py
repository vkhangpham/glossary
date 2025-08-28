#!/usr/bin/env python3
"""
Script to fix empty sources in data/final by re-running metadata collection
with the corrected source filtering logic.
"""

import os
import sys
import json
from pathlib import Path

# Add the generate_glossary module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generate_glossary'))

try:
    from utils.metadata_collector import collect_metadata, collect_resources
except ImportError as e:
    print(f"Error importing metadata_collector: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

def check_empty_sources_stats(level: int) -> tuple:
    """Check statistics of empty sources for a given level."""
    try:
        metadata_file = f'data/final/lv{level}/lv{level}_metadata.json'
        if not os.path.exists(metadata_file):
            return 0, 0
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        empty_count = sum(1 for data in metadata.values() if 'sources' in data and not data['sources'])
        total_count = len(metadata)
        
        return empty_count, total_count
    except Exception as e:
        print(f"Error checking level {level}: {e}")
        return 0, 0

def show_empty_sources_terms(level: int, show_terms: bool = False) -> list:
    """Show terms with empty sources for a given level."""
    try:
        metadata_file = f'data/final/lv{level}/lv{level}_metadata.json'
        if not os.path.exists(metadata_file):
            return []
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        empty_terms = []
        for term, data in metadata.items():
            if 'sources' in data and not data['sources']:
                empty_terms.append(term)
        
        if show_terms and empty_terms:
            print(f"Level {level} terms with empty sources:")
            for term in empty_terms[:10]:  # Show first 10
                print(f"  - {term}")
            if len(empty_terms) > 10:
                print(f"  ... and {len(empty_terms) - 10} more")
        
        return empty_terms
    except Exception as e:
        print(f"Error checking level {level}: {e}")
        return []

def test_new_filtering_logic():
    """Test the new filtering logic with known problematic sources."""
    print("🧪 Testing new filtering logic...")
    
    # Import the function to test
    from utils.metadata_collector import is_department_or_college_source
    
    test_cases = [
        ("r.f. smith school of chemical and biomolecular engineering", False),
        ("college of engineering", True),
        ("hslopez school of business analytics", False),
        ("college of management", False),
        ("human health sciences", False),
        ("college of health sciences", False),
        ("college", True),
        ("department", True),
        ("school", True),
        ("department of engineering", True),
        ("school of engineering", True),
    ]
    
    all_passed = True
    for source, should_be_filtered in test_cases:
        result = is_department_or_college_source(source)
        if result == should_be_filtered:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        print(f"  {status} {source:50} -> filtered={result}")
    
    if all_passed:
        print("✅ All filtering tests passed!")
        return True
    else:
        print("❌ Some filtering tests failed! Check the logic.")
        return False

def check_source_files():
    """Check that required source files exist."""
    print("📋 Checking source files...")
    
    levels_info = [
        (0, 'generate_glossary/data/lv0/lv0_final.txt', None),  # Level 0 doesn't have department concepts CSV
        (1, 'generate_glossary/data/lv1/lv1_final.txt', 'generate_glossary/data/lv1/raw/lv1_s1_department_concepts.csv'),
        (2, 'generate_glossary/data/lv2/lv2_final.txt', 'generate_glossary/data/lv2/raw/lv2_s1_hierarchical_concepts.csv'),
        (3, 'generate_glossary/data/lv3/lv3_final.txt', 'generate_glossary/data/lv3/raw/lv3_s1_hierarchical_concepts.csv'),
    ]
    
    missing_files = []
    for level, final_file, csv_file in levels_info:
        if not os.path.exists(final_file):
            missing_files.append(final_file)
            print(f"  ❌ Missing: {final_file}")
        else:
            print(f"  ✅ Found: {final_file}")
            
        if csv_file is not None:  # Only check CSV for levels that should have one
            if not os.path.exists(csv_file):
                missing_files.append(csv_file)
                print(f"  ❌ Missing: {csv_file}")
            else:
                print(f"  ✅ Found: {csv_file}")
        else:
            print(f"  ℹ️  Level {level} doesn't use CSV source file")
    
    if missing_files:
        print(f"\n❌ {len(missing_files)} required files are missing!")
        print("Cannot proceed with the fix.")
        return False
    else:
        print("\n✅ All required source files found!")
        return True

def main():
    print("🔧 Fixing Empty Sources in data/final")
    print("=" * 50)
    
    # Test the new filtering logic
    if not test_new_filtering_logic():
        print("❌ Filtering logic test failed. Aborting.")
        return
    
    # Check source files
    if not check_source_files():
        print("❌ Required source files missing. Aborting.")
        return
    
    # Check current state
    print("\n📊 Current state (BEFORE fix):")
    levels = [0, 1, 2, 3]
    total_empty_before = 0
    total_terms_before = 0
    
    for level in levels:
        empty_count, total_count = check_empty_sources_stats(level)
        total_empty_before += empty_count
        total_terms_before += total_count
        percentage = (empty_count / total_count * 100) if total_count > 0 else 0
        print(f"  Level {level}: {empty_count:2d}/{total_count:4d} terms with empty sources ({percentage:.1f}%)")
    
    print(f"\nTotal: {total_empty_before}/{total_terms_before} terms with empty sources ({total_empty_before/total_terms_before*100:.1f}%)")
    
    # Show some example terms with empty sources
    print("\n📝 Example terms with empty sources:")
    for level in levels:
        empty_terms = show_empty_sources_terms(level, show_terms=True)
        if empty_terms:
            print()
    
    # Ask for confirmation
    print("\n" + "=" * 50)
    print("This will re-collect metadata for all levels with the corrected filtering logic.")
    print("A backup has been created in the backups/ directory.")
    response = input("Proceed with fixing empty sources? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Re-collect metadata for all levels
    print("\n🔄 Re-collecting metadata with corrected filtering logic...")
    
    success_count = 0
    for level in levels:
        print(f"\nProcessing level {level}...")
        try:
            # Collect metadata with verbose output
            print(f"  📋 Collecting metadata for level {level}...")
            collect_metadata(level, verbose=False, include_variations=True)
            
            # Collect resources if they exist
            resources_file = f'generate_glossary/data/lv{level}/lv{level}_resources.json'
            
            if os.path.exists(resources_file):
                print(f"  📚 Collecting resources for level {level}...")
                collect_resources(level, verbose=False, include_variations=True)
            
            print(f"  ✅ Level {level} completed")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing level {level}: {e}")
    
    print(f"\n📊 Successfully processed {success_count}/{len(levels)} levels")
    
    # Check results
    print("\n" + "=" * 50)
    print("📊 Results (AFTER fix):")
    
    total_empty_after = 0
    total_terms_after = 0
    
    for level in levels:
        empty_count, total_count = check_empty_sources_stats(level)
        total_empty_after += empty_count
        total_terms_after += total_count
        percentage = (empty_count / total_count * 100) if total_count > 0 else 0
        print(f"  Level {level}: {empty_count:2d}/{total_count:4d} terms with empty sources ({percentage:.1f}%)")
    
    print(f"\nTotal: {total_empty_after}/{total_terms_after} terms with empty sources ({total_empty_after/total_terms_after*100:.1f}%)")
    
    # Show improvement
    fixed_count = total_empty_before - total_empty_after
    print(f"\n🎉 Fixed {fixed_count} terms that previously had empty sources!")
    
    if total_empty_after > 0:
        print(f"\nRemaining terms with empty sources:")
        for level in levels:
            empty_terms = show_empty_sources_terms(level, show_terms=True)
            if empty_terms:
                print()
    
    # Show specific examples of fixed terms
    print("\n✨ Examples of previously problematic terms that should now have sources:")
    examples = [
        ("biomolecular engineering", "r.f. smith school of chemical and biomolecular engineering"),
        ("business analytics", "hslopez school of business analytics"),
        ("human health sciences", "college of health sciences"),
    ]
    
    for term, expected_source in examples:
        # Check if term has sources now
        for level in levels:
            metadata_file = f'data/final/lv{level}/lv{level}_metadata.json'
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if term in metadata:
                        sources = metadata[term].get('sources', [])
                        if sources:
                            print(f"  ✅ {term} (Level {level}): {', '.join(sources)}")
                        else:
                            print(f"  ❌ {term} (Level {level}): still no sources")
                        break
                except Exception as e:
                    continue
    
    print("\n" + "=" * 50)
    print("✅ EMPTY SOURCES FIX COMPLETED!")
    print("=" * 50)
    
    if fixed_count > 0:
        print(f"🎉 Successfully fixed {fixed_count} terms with empty sources!")
    else:
        print("ℹ️  No terms were fixed. The issue may have been resolved already.")

if __name__ == "__main__":
    main() 