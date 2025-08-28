#!/usr/bin/env python3
"""
Cleanup script for remaining Config.* references after centralized config migration.

This script handles the Config references that the main migration script missed,
particularly constants and validation file paths.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

def update_remaining_config_refs(file_path: Path) -> bool:
    """Update remaining Config references in a file"""
    print(f"Cleaning up remaining Config refs in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return False
    
    original_content = content
    
    # Replace remaining Config references
    replacements = [
        # Validation file paths
        (r'Config\.VALIDATION_META_FILE', 'level_config.get_validation_metadata_file(3)'),
        
        # Constants  
        (r'Config\.NON_ACADEMIC_TERMS', 'processing_config.non_academic_terms'),
        (r'Config\.MAX_EXAMPLES', 'processing_config.max_examples'),
        (r'Config\.COOLDOWN_PERIOD', 'processing_config.cooldown_period'),
        (r'Config\.COOLDOWN_FREQUENCY', 'processing_config.cooldown_frequency'),
        (r'Config\.MAX_RETRIES', 'processing_config.max_retries'),
        
        # Level-specific metadata files - these need special handling
        (r'Config\.METADATA_S0_FILE', 'level_config.get_step_metadata_file(0)'),
        (r'Config\.META_FILE_STEP0', 'level_config.get_step_metadata_file(0)'),
        
        # Cache and directory paths - these need level-specific handling
        (r'Config\.CACHE_DIR', 'level_config.data_dir / "cache"'),
        (r'Config\.RAW_SEARCH_DIR', 'level_config.data_dir / "raw_search"'),
        (r'Config\.DETAILED_META_DIR', 'level_config.data_dir / "detailed_meta"'),
        (r'Config\.RAW_RESULTS_DIR', 'level_config.data_dir / "raw_results"'),
        
        # Base directory references
        (r'Config\.BASE_DIR', 'str(level_config.data_dir.parent.parent)'),
        
        # Input file references
        (r'Config\.LV0_INPUT_FILE', 'str(get_level_config(0).get_final_file())'),
        (r'Config\.DEFAULT_LV1_INPUT_FILE', 'str(get_level_config(1).get_final_file())'),
        (r'Config\.DEFAULT_LV2_INPUT_FILE', 'str(get_level_config(2).get_final_file())'),
    ]
    
    for old_pattern, new_value in replacements:
        content = re.sub(old_pattern, new_value, content)
    
    # Handle function parameter defaults that use Config
    content = re.sub(
        r'cooldown: int = Config\.COOLDOWN_PERIOD',
        'cooldown: int = processing_config.cooldown_period',
        content
    )
    content = re.sub(
        r'cooldown_freq: int = Config\.COOLDOWN_FREQUENCY', 
        'cooldown_freq: int = processing_config.cooldown_frequency',
        content
    )
    
    # Write back if changes were made
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Successfully updated: {file_path.name}")
            return True
        except Exception as e:
            print(f"  Error writing {file_path}: {e}")
            return False
    else:
        print(f"  No changes needed: {file_path.name}")
        return True

def main():
    """Main cleanup function"""
    print("🧹 Starting cleanup of remaining Config references")
    
    # Find all Python files in generation directory
    generation_dir = Path("generate_glossary/generation")
    if not generation_dir.exists():
        print(f"❌ Generation directory not found: {generation_dir}")
        return
    
    python_files = []
    for py_file in generation_dir.rglob("*.py"):
        python_files.append(py_file)
    
    print(f"📁 Found {len(python_files)} Python files to check")
    
    # Check which files still have Config references
    files_with_config = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'Config.' in content:
                    files_with_config.append(py_file)
        except Exception:
            continue
    
    print(f"🔍 Found {len(files_with_config)} files with remaining Config references")
    
    # Update each file
    success_count = 0
    for py_file in files_with_config:
        if update_remaining_config_refs(py_file):
            success_count += 1
    
    print(f"\n✅ Cleanup complete: {success_count}/{len(files_with_config)} files updated")
    
    # Check if any Config references remain
    remaining_refs = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'Config.' in content:
                    # Count occurrences
                    config_refs = re.findall(r'Config\.\w+', content)
                    if config_refs:
                        remaining_refs.append((py_file, config_refs))
        except Exception:
            continue
    
    if remaining_refs:
        print(f"\n⚠️  {len(remaining_refs)} files still have Config references:")
        for py_file, refs in remaining_refs[:5]:  # Show first 5
            print(f"  {py_file.name}: {', '.join(set(refs))}")
        if len(remaining_refs) > 5:
            print(f"  ... and {len(remaining_refs) - 5} more")
    else:
        print(f"\n🎉 All Config references successfully migrated!")

if __name__ == "__main__":
    main()