"""
Data extraction utilities for glossary metadata.

This module handles extracting information from various data sources,
including college names, parent relationships, and source metadata.
"""

import re
import json
from typing import Optional, Dict, Any, List, Set
from pathlib import Path


def extract_parent_from_college(college_name):
    """Extract parent term from college name."""
    # Extract the main subject from "college of X" pattern
    match = re.search(r'college of (\w+)', college_name.lower())
    if match:
        return match.group(1)
    return None


def clean_parent_term(parent: str) -> str:
    """
    Clean and normalize a parent term.
    
    Args:
        parent: Raw parent term string
        
    Returns:
        Cleaned parent term
    """
    if not parent:
        return ""
        
    # Remove extra whitespace
    parent = parent.strip()
    
    # Remove quotes if present
    if parent.startswith('"') and parent.endswith('"'):
        parent = parent[1:-1]
    elif parent.startswith("'") and parent.endswith("'"):
        parent = parent[1:-1]
    
    # Normalize case
    parent = parent.lower()
    
    # Remove special characters except spaces and hyphens
    parent = re.sub(r'[^\w\s-]', '', parent)
    
    # Replace multiple spaces with single space
    parent = re.sub(r'\s+', ' ', parent)
    
    return parent.strip()


def is_department_or_college_source(source: str) -> bool:
    """
    Check if a source string represents a department or college.
    
    Args:
        source: Source string to check
        
    Returns:
        True if source is a department or college
    """
    if not source:
        return False
        
    source_lower = source.lower()
    
    # Department patterns
    department_patterns = [
        r'department of',
        r'dept\.? of',
        r'school of',
        r'institute of',
        r'institute for',
        r'center for',
        r'centre for',
        r'division of',
        r'program in',
        r'programme in',
    ]
    
    # College patterns
    college_patterns = [
        r'college of',
        r'faculty of',
        r'school of',
    ]
    
    # Check department patterns
    for pattern in department_patterns:
        if re.search(pattern, source_lower):
            return True
    
    # Check if it ends with common department suffixes
    department_suffixes = [
        'department',
        'dept',
        'school',
        'institute',
        'center',
        'centre',
        'division',
        'program',
        'programme',
    ]
    
    for suffix in department_suffixes:
        if source_lower.endswith(suffix):
            return True
    
    # Check college patterns
    for pattern in college_patterns:
        if re.search(pattern, source_lower):
            # Additional check to distinguish from department
            if 'department' not in source_lower and 'dept' not in source_lower:
                return True
    
    return False


def extract_concept_from_source(source: str) -> Optional[str]:
    """
    Extract the main concept from a department/college source string.
    
    Args:
        source: Source string (e.g., "Department of Computer Science")
        
    Returns:
        Extracted concept (e.g., "computer science") or None
    """
    if not source:
        return None
        
    source_lower = source.lower()
    
    # Patterns to extract concept from
    patterns = [
        r'department of (.+?)(?:\s*[-–]\s*|$)',
        r'dept\.? of (.+?)(?:\s*[-–]\s*|$)',
        r'school of (.+?)(?:\s*[-–]\s*|$)',
        r'institute of (.+?)(?:\s*[-–]\s*|$)',
        r'institute for (.+?)(?:\s*[-–]\s*|$)',
        r'center for (.+?)(?:\s*[-–]\s*|$)',
        r'centre for (.+?)(?:\s*[-–]\s*|$)',
        r'division of (.+?)(?:\s*[-–]\s*|$)',
        r'program in (.+?)(?:\s*[-–]\s*|$)',
        r'programme in (.+?)(?:\s*[-–]\s*|$)',
        r'college of (.+?)(?:\s*[-–]\s*|$)',
        r'faculty of (.+?)(?:\s*[-–]\s*|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, source_lower)
        if match:
            concept = match.group(1).strip()
            # Clean up common suffixes
            concept = re.sub(r'\s*\([^)]*\)\s*$', '', concept)  # Remove parenthetical
            concept = re.sub(r'\s*department\s*$', '', concept)  # Remove trailing 'department'
            concept = re.sub(r'\s*school\s*$', '', concept)  # Remove trailing 'school'
            return clean_parent_term(concept)
    
    return None


def extract_metadata_from_json(json_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary containing extracted metadata
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {json_path}: {e}")
        return {}


def extract_variations_from_dedup(dedup_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract term variations from deduplication data.
    
    Args:
        dedup_data: Dictionary containing deduplication information
        
    Returns:
        Dictionary mapping primary terms to their variations
    """
    variations = {}
    
    # Handle different dedup data formats
    if 'groups' in dedup_data:
        # Graph dedup format
        for group in dedup_data['groups']:
            primary = group.get('primary', '')
            members = group.get('members', [])
            if primary:
                variations[primary] = [m for m in members if m != primary]
    
    elif 'duplicates' in dedup_data:
        # Rule-based dedup format
        for primary, dupes in dedup_data['duplicates'].items():
            variations[primary] = dupes
    
    return variations