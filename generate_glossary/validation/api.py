"""
API for working with validation results.

This module provides utilities for working with both legacy dictionary-based
validation results and the new immutable ValidationResult structures.
"""

from typing import Dict, List, Any, Union, overload
import json
from pathlib import Path

from .core import ValidationResult, get_validation_summary

# Type aliases
ValidationResults = Dict[str, Dict[str, Any]]  # Legacy format
FunctionalResults = Dict[str, ValidationResult]  # New functional format
AnyValidationResults = Union[ValidationResults, FunctionalResults]


def _is_functional_results(results: AnyValidationResults) -> bool:
    """Detect if results are in functional ValidationResult format."""
    if not results:
        return False
    # Check first value to determine format
    first_value = next(iter(results.values()))
    return isinstance(first_value, ValidationResult)


@overload
def get_valid_terms(results: ValidationResults) -> List[str]: ...

@overload  
def get_valid_terms(results: FunctionalResults) -> List[str]: ...

def get_valid_terms(results: AnyValidationResults) -> List[str]:
    """Get all valid terms from validation results (supports both formats)."""
    if _is_functional_results(results):
        return [
            term for term, result in results.items()
            if result.is_valid
        ]
    else:
        return [
            term for term, result in results.items()
            if result.get("is_valid", False)
        ]


@overload
def get_invalid_terms(results: ValidationResults) -> List[str]: ...

@overload
def get_invalid_terms(results: FunctionalResults) -> List[str]: ...

def get_invalid_terms(results: AnyValidationResults) -> List[str]:
    """Get all invalid terms from validation results (supports both formats)."""
    if _is_functional_results(results):
        return [
            term for term, result in results.items()
            if not result.is_valid
        ]
    else:
        return [
            term for term, result in results.items()
            if not result.get("is_valid", False)
        ]


def _convert_functional_to_serializable(results: FunctionalResults) -> ValidationResults:
    """Convert ValidationResult objects to serializable dictionary format."""
    serializable = {}
    
    for term, result in results.items():
        # Convert ValidationResult to dictionary
        result_dict = {
            "term": result.term,
            "is_valid": result.is_valid,
            "confidence": result.confidence,
            "score": result.score,
        }
        
        # Add optional fields if they exist
        if result.relevance_score is not None:
            result_dict["relevance_score"] = result.relevance_score
        if result.errors:
            result_dict["errors"] = list(result.errors)
        
        # Add mode results if available
        mode_results = {}
        modes_used = []
        
        if result.rule_result:
            mode_results["rule"] = dict(result.rule_result)
            modes_used.append("rule")
        if result.web_result:
            mode_results["web"] = dict(result.web_result)
            modes_used.append("web")
        if result.llm_result:
            mode_results["llm"] = dict(result.llm_result)
            modes_used.append("llm")
        
        if mode_results:
            result_dict["mode_results"] = mode_results
            result_dict["modes_used"] = modes_used
        
        serializable[term] = result_dict
    
    return serializable


@overload
def save_validation_results(
    results: ValidationResults,
    output_path: str,
    format: str = "both"
) -> None: ...

@overload
def save_validation_results(
    results: FunctionalResults,
    output_path: str,
    format: str = "both"
) -> None: ...

def save_validation_results(
    results: AnyValidationResults,
    output_path: str,
    format: str = "both"
) -> None:
    """
    Save validation results to file(s) (supports both formats).
    
    Args:
        results: Validation results (dictionary or ValidationResult objects)
        output_path: Base path for output (without extension)
        format: Output format ('json', 'txt', 'both')
    """
    output_path = Path(output_path)
    
    # Convert to serializable format if needed
    if _is_functional_results(results):
        serializable_results = _convert_functional_to_serializable(results)
        # Also save summary for functional results
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        summary = get_validation_summary(results)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    else:
        serializable_results = results
    
    # Save JSON if requested
    if format in ["json", "both"]:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
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
        Validation results dictionary (legacy format)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_results_summary(results: AnyValidationResults) -> Dict[str, Any]:
    """Get summary statistics from validation results (supports both formats)."""
    if _is_functional_results(results):
        return get_validation_summary(results)
    else:
        # Calculate summary for legacy format
        total_terms = len(results)
        valid_terms = len(get_valid_terms(results))
        invalid_terms = total_terms - valid_terms
        
        if total_terms > 0:
            avg_confidence = sum(
                r.get("confidence", 0.0) for r in results.values()
            ) / total_terms
            validity_rate = valid_terms / total_terms
        else:
            avg_confidence = 0.0
            validity_rate = 0.0
        
        return {
            "total_terms": total_terms,
            "valid_terms": valid_terms,
            "invalid_terms": invalid_terms,
            "validity_rate": validity_rate,
            "average_confidence": avg_confidence
        }


def filter_results_by_confidence(
    results: AnyValidationResults,
    min_confidence: float
) -> AnyValidationResults:
    """Filter results to only include terms above confidence threshold."""
    if _is_functional_results(results):
        return {
            term: result for term, result in results.items()
            if result.confidence >= min_confidence
        }
    else:
        return {
            term: result for term, result in results.items()
            if result.get("confidence", 0.0) >= min_confidence
        }


def filter_results_by_score(
    results: AnyValidationResults,
    min_score: float
) -> AnyValidationResults:
    """Filter results to only include terms above score threshold."""
    if _is_functional_results(results):
        return {
            term: result for term, result in results.items()
            if result.score >= min_score
        }
    else:
        return {
            term: result for term, result in results.items()
            if result.get("score", result.get("confidence", 0.0)) >= min_score
        }