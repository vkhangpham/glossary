"""
Test script for web content relevance scoring.

This script demonstrates how the relevance scoring works for web content validation.
It takes a term and a web content file, and shows the relevance scores for each content.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

from generate_glossary.validator.validation_utils import calculate_relevance_score
from generate_glossary.validator.validation_modes import validate_web, DEFAULT_MIN_SCORE, DEFAULT_MIN_RELEVANCE_SCORE

def read_web_content(filepath: str, term: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read web content from JSON file.
    
    Args:
        filepath: Path to web content JSON file
        term: Optional term to filter content for
        
    Returns:
        Dict mapping terms to their web content
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)
            
            if term and term in content:
                # Return only content for the specified term
                return {term: content[term]}
            elif term:
                print(f"Warning: Term '{term}' not found in web content file", file=sys.stderr)
                return {}
            
            return content
    except Exception as e:
        print(f"Error loading web content: {e}", file=sys.stderr)
        return {}

def display_relevance_scores(term: str, web_contents: List[Dict[str, Any]]) -> None:
    """
    Display relevance scores for web content.
    
    Args:
        term: Term to calculate relevance for
        web_contents: List of web content objects
    """
    print(f"Relevance scores for term: '{term}'")
    print("-" * 80)
    
    for i, content in enumerate(web_contents, 1):
        # Extract basic info
        url = content.get("url", "")
        title = content.get("title", "")
        is_verified = content.get("is_verified", False)
        content_score = content.get("score", 0.0)
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(term, content)
        
        # Display information
        print(f"Content #{i}:")
        print(f"  URL: {url}")
        print(f"  Title: {title}")
        print(f"  Content Score: {content_score:.2f}")
        print(f"  Is Verified: {is_verified}")
        print(f"  Relevance Score: {relevance_score:.2f}")
        
        # Show relevance status
        if relevance_score >= DEFAULT_MIN_RELEVANCE_SCORE:
            print(f"  Status: RELEVANT (≥ {DEFAULT_MIN_RELEVANCE_SCORE})")
        else:
            print(f"  Status: NOT RELEVANT (< {DEFAULT_MIN_RELEVANCE_SCORE})")
        
        print()

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test relevance scoring for web content validation."
    )
    
    parser.add_argument(
        "term",
        help="Term to calculate relevance for"
    )
    
    parser.add_argument(
        "web_content",
        help="Path to web content JSON file"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run full web validation with relevance scoring"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum score for web content validation (default: {DEFAULT_MIN_SCORE})"
    )
    
    parser.add_argument(
        "--min-relevance-score",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help=f"Minimum relevance score (default: {DEFAULT_MIN_RELEVANCE_SCORE})"
    )
    
    args = parser.parse_args()
    
    try:
        # Read web content
        web_content = read_web_content(args.web_content, args.term)
        
        if not web_content or args.term not in web_content:
            print(f"No web content found for term: '{args.term}'")
            sys.exit(1)
        
        # Get content for the term
        term_content = web_content[args.term]
        
        # Display relevance scores
        display_relevance_scores(args.term, term_content)
        
        # Run full validation if requested
        if args.validate:
            print("\nRunning full web validation with relevance scoring:")
            print("-" * 80)
            
            result = validate_web(
                args.term, 
                term_content, 
                min_score=args.min_score,
                min_relevance_score=args.min_relevance_score
            )
            
            # Display validation result
            print(f"Validation Result: {'VALID' if result['is_valid'] else 'INVALID'}")
            print(f"Number of Sources: {result['details']['num_sources']}")
            print(f"Verified Sources: {len(result['details']['verified_sources'])}")
            print(f"Relevant Sources: {len(result['details']['relevant_sources'])}")
            
            if result['details']['relevant_sources']:
                print("\nTop Relevant Sources:")
                for i, source in enumerate(result['details']['relevant_sources'][:3], 1):
                    print(f"  {i}. {source['title']} (Relevance: {source['relevance_score']}, Score: {source['score']})")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 