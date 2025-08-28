#!/usr/bin/env python3
"""
Script to analyze concept completeness.
Identifies terms in metadata that are missing definitions and terms missing resources.
Saves these lists to files in data/analysis/.
"""
import json
from pathlib import Path
import os

# Ensure the script can correctly resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_PATH = PROJECT_ROOT / "data"
ANALYSIS_OUTPUT_DIR = BASE_DATA_PATH / "analysis"

def get_terms_from_metadata(metadata_file_path: Path) -> tuple[set[str], set[str]]:
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

def get_terms_with_resources(resources_file_path: Path) -> set[str]:
    """Extracts all terms that have at least one resource from a resources file."""
    terms_with_resources_set = set()
    try:
        with open(resources_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for term, resources_list in data.items():
            if resources_list:  # Check if the list of resources is not empty
                terms_with_resources_set.add(term)
    except FileNotFoundError:
        print(f"Warning: Resources file not found: {resources_file_path}")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from resources file: {resources_file_path}")
    except Exception as e:
        print(f"Error processing resources file {resources_file_path}: {e}")
    return terms_with_resources_set

def save_to_json(data: dict, file_path: Path):
    """Saves dictionary data to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

def save_to_txt(terms_set: set[str], file_path: Path):
    """Saves a set of terms to a TXT file, one term per line."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for term in sorted(list(terms_set)):
                f.write(f"{term}\n")
        print(f"Successfully saved term list to {file_path}")
    except Exception as e:
        print(f"Error saving term list to {file_path}: {e}")

def main():
    final_data_path = BASE_DATA_PATH / "final"
    levels = [f"lv{i}" for i in range(4)]  # lv0, lv1, lv2, lv3

    print("Analyzing Concept Completeness...")
    print("=" * 70)

    all_terms_needing_definitions = {}
    all_terms_missing_resources = {}

    overall_unique_terms_needing_definitions = set()
    overall_unique_terms_missing_resources = set()

    for level_name in levels:
        print(f"\nProcessing Level: {level_name.upper()}")
        level_path = final_data_path / level_name
        metadata_file = level_path / f"{level_name}_metadata.json"
        resources_file = level_path / f"{level_name}_filtered_resources.json"

        all_metadata_terms, terms_with_defs = get_terms_from_metadata(metadata_file)
        terms_with_res_entries = get_terms_with_resources(resources_file)

        if not all_metadata_terms:
            print(f"  No terms found in metadata for {level_name}. Skipping.")
            continue

        print(f"  Terms in metadata: {len(all_metadata_terms)}")
        print(f"  Terms with definitions: {len(terms_with_defs)}")
        print(f"  Terms with resource entries: {len(terms_with_res_entries)}")

        terms_needing_def_for_level = sorted(list(all_metadata_terms - terms_with_defs))
        terms_missing_res_for_level = sorted(list(all_metadata_terms - terms_with_res_entries))

        if terms_needing_def_for_level:
            all_terms_needing_definitions[level_name] = terms_needing_def_for_level
            overall_unique_terms_needing_definitions.update(terms_needing_def_for_level)
            print(f"  Found {len(terms_needing_def_for_level)} terms needing definitions.")
        else:
            print("  No terms needing definitions found in this level.")

        if terms_missing_res_for_level:
            all_terms_missing_resources[level_name] = terms_missing_res_for_level
            overall_unique_terms_missing_resources.update(terms_missing_res_for_level)
            print(f"  Found {len(terms_missing_res_for_level)} terms missing resources.")
        else:
            print("  No terms missing resources found in this level.")

    # Save the results
    output_file_needing_definitions = ANALYSIS_OUTPUT_DIR / "terms_needing_definitions.json"
    output_file_missing_resources_json = ANALYSIS_OUTPUT_DIR / "terms_missing_resources.json"
    output_file_missing_resources_txt = ANALYSIS_OUTPUT_DIR / "terms_missing_resources.txt"

    save_to_json(all_terms_needing_definitions, output_file_needing_definitions)
    save_to_json(all_terms_missing_resources, output_file_missing_resources_json)
    save_to_txt(overall_unique_terms_missing_resources, output_file_missing_resources_txt)

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY:")
    print(f"Total unique terms needing definitions across all levels: {len(overall_unique_terms_needing_definitions)}")
    print(f" (Saved to {output_file_needing_definitions})")
    print(f"Total unique terms missing resources across all levels: {len(overall_unique_terms_missing_resources)}")
    print(f" (JSON saved to {output_file_missing_resources_json})")
    print(f" (TXT saved to {output_file_missing_resources_txt})")
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 