"""
API for working with validation results.
"""

from typing import Dict, List, Any
import json
from pathlib import Path

# Type aliases
ValidationResults = Dict[str, Dict[str, Any]]


def get_valid_terms(results: ValidationResults) -> List[str]:
    """Get all valid terms from validation results."""
    return [
        term for term, result in results.items()
        if result.get("is_valid", False)
    ]


def get_invalid_terms(results: ValidationResults) -> List[str]:
    """Get all invalid terms from validation results."""
    return [
        term for term, result in results.items()
        if not result.get("is_valid", False)
    ]


def save_validation_results(
    results: ValidationResults,
    output_path: str,
    format: str = "both"
) -> None:
    """
    Save validation results to file(s).
    
    Args:
        results: Validation results dictionary
        output_path: Base path for output (without extension)
        format: Output format ('json', 'txt', 'both')
    """
    output_path = Path(output_path)
    
    # Save JSON if requested
    if format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save text file with valid terms if requested
    if format in ["txt", "both"]:
        txt_path = output_path.with_suffix(".txt")
        valid_terms = get_valid_terms(results)
        with open(txt_path, "w", encoding="utf-8") as f:
            for term in sorted(valid_terms):
                f.write(f"{term}\n")


def load_validation_results(filepath: str) -> ValidationResults:
    """
    Load validation results from a JSON file.
    
    Args:
        filepath: Path to validation results JSON file
        
    Returns:
        Validation results dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)