"""
File discovery utilities for glossary data.

This module handles finding various files in the glossary data structure,
including step files, metadata files, and final output files.
"""

import os
from pathlib import Path
from typing import Optional, List


def find_step_file(level_dir, level, step, extension=None):
    """Find file for a specific step using regex."""
    pattern = f"lv{level}_s{step}_*"
    if extension:
        pattern += f".{extension}"
    
    # List all files matching the pattern
    matches = list(level_dir.glob(pattern))
    
    # Return the first match if any
    if matches:
        return matches[0]
    
    # If no matches with direct extension, try more flexible approach
    if extension:
        matches = list(level_dir.glob(f"lv{level}_s{step}_*"))
        for match in matches:
            if match.suffix.lstrip('.') == extension or match.name.endswith(f".{extension}"):
                return match
    
    return None


def find_step_metadata(raw_dir, level, step):
    """Find metadata.json file for a specific step."""
    # Try standard naming pattern
    metadata_file = raw_dir / f"lv{level}_s{step}_metadata.json"
    if metadata_file.exists():
        return metadata_file
    
    # Try flexible pattern matching
    metadata_files = list(raw_dir.glob(f"lv{level}_s{step}*metadata*.json"))
    if metadata_files:
        return metadata_files[0]
    
    return None


def find_final_file(level_dir, level):
    """Find the final output file for a level."""
    # First try the standard naming convention
    final_file = level_dir / f'lv{level}_final.txt'
    if final_file.exists():
        return final_file
    
    # Try pattern matching if standard file not found
    final_files = list(level_dir.glob(f'**/lv{level}_final.txt'))
    if final_files:
        return final_files[0]
    
    # Try even more flexible pattern matching
    final_files = list(level_dir.glob(f'**/*final*.txt'))
    for file in final_files:
        if f'lv{level}' in file.name:
            return file
    
    return None


def find_dedup_files(level: int) -> List[str]:
    """
    Find all deduplication files for a given level.
    
    Args:
        level: The level number (0, 1, 2, or 3)
        
    Returns:
        List of paths to deduplication files
    """
    # Base directory for the level
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'generate_glossary' / 'data'
    level_dir = base_dir / f"lv{level}"
    postprocessed_dir = level_dir / "postprocessed"
    
    dedup_files = []
    
    # Look for dedup files in postprocessed directory
    if postprocessed_dir.exists():
        # Common dedup file patterns
        patterns = [
            f"lv{level}_*dedup*.json",
            f"lv{level}_*dedup*.txt",
        ]
        
        for pattern in patterns:
            for file in postprocessed_dir.glob(pattern):
                if file.is_file():
                    dedup_files.append(str(file))
    
    return dedup_files


def ensure_final_dirs_exist():
    """Ensure all final directories exist."""
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'data' / 'final'
    
    # Create main final directory
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create level-specific directories
    for level in range(4):
        level_dir = base_dir / f'lv{level}'
        level_dir.mkdir(parents=True, exist_ok=True)