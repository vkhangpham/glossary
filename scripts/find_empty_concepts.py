#!/usr/bin/env python3
"""
Script to find concepts that do not have a definition or any resources.
Checks lvX_metadata.json for definitions and lvX_filtered_resources.json for resources.
"""
import json
from pathlib import Path

def get_terms_from_metadata(metadata_file_path):
    """Extracts all terms and terms with definitions from a metadata file."""
    terms_with_definitions = set()
    all_terms_in_metadata = set()
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for term, attributes in data.items():
            all_terms_in_metadata.add(term)
            if attributes.get('definition') and attributes['definition'].strip():
                terms_with_definitions.add(term)
    except FileNotFoundError:
        print(f"Warning: Metadata file not found: {metadata_file_path}")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from metadata file: {metadata_file_path}")
    except Exception as e:
        print(f"Error processing metadata file {metadata_file_path}: {e}")
    return all_terms_in_metadata, terms_with_definitions

def get_terms_with_resources(resources_file_path):
    """Extracts all terms that have at least one resource from a resources file."""
    terms_with_resources_set = set()
    try:
        with open(resources_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for term, resources_list in data.items():
            if resources_list: # Check if the list of resources is not empty
                terms_with_resources_set.add(term)
    except FileNotFoundError:
        print(f"Warning: Resources file not found: {resources_file_path}")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from resources file: {resources_file_path}")
    except Exception as e:
        print(f"Error processing resources file {resources_file_path}: {e}")
    return terms_with_resources_set

def main():
    base_path = Path("data/final")
    levels = [f"lv{i}" for i in range(4)] # lv0, lv1, lv2, lv3

    print("Analysis of Concepts Without Definitions or Resources")
    print("=" * 70)

    all_empty_concepts = set()

    for level_name in levels:
        level_path = base_path / level_name
        metadata_file = level_path / f"{level_name}_metadata.json"
        resources_file = level_path / f"{level_name}_filtered_resources.json"

        print(f"\nProcessing Level: {level_name.upper()}")

        all_metadata_terms, terms_with_defs = get_terms_from_metadata(metadata_file)
        terms_with_res = get_terms_with_resources(resources_file)

        if not all_metadata_terms:
            print(f"  No terms found in metadata for {level_name}. Skipping.")
            continue
            
        print(f"  Terms in metadata: {len(all_metadata_terms)}")
        print(f"  Terms with definitions: {len(terms_with_defs)}")
        print(f"  Terms with resources: {len(terms_with_res)}")

        # Concepts are terms present in metadata
        # Empty concepts are those in metadata without a definition AND without resources
        
        terms_without_definitions = all_metadata_terms - terms_with_defs
        terms_without_resources = all_metadata_terms - terms_with_res
        
        empty_concepts_in_level = terms_without_definitions.intersection(terms_without_resources)
        
        if empty_concepts_in_level:
            print(f"  Found {len(empty_concepts_in_level)} concepts without definition AND without resources:")
            for term in sorted(list(empty_concepts_in_level))[:10]: # Print first 10
                 print(f"    - {term}")
            if len(empty_concepts_in_level) > 10:
                print(f"    ... and {len(empty_concepts_in_level) - 10} more.")
            all_empty_concepts.update(empty_concepts_in_level)
        else:
            print("  No concepts found without both a definition and resources in this level.")

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY:")
    if all_empty_concepts:
        print(f"Total unique concepts without definition AND without resources across all levels: {len(all_empty_concepts)}")
        # If you need to see all of them:
        # print("\nList of all empty concepts:")
        # for term in sorted(list(all_empty_concepts)):
        #     print(f"  - {term}")
    else:
        print("No concepts found without definitions or resources across all levels.")

if __name__ == "__main__":
    main() 