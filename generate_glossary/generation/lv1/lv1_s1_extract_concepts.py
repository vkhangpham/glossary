#!/usr/bin/env python3
"""
Level 1 Step 1: Extract Academic Concepts from Department Names

This script uses LLM-based extraction to identify academic concepts
from department names collected in Step 0.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

from generate_glossary.utils.logger import setup_logger
from ..shared.concept_extraction import extract_concepts_llm
from ..shared.level_config import get_level_config, get_step_file_paths


def to_test_path(path: Path) -> Path:
    """Convert a path to its test mode equivalent."""
    path_str = str(path)
    # Handle both absolute and relative paths
    if '/data/' in path_str:
        return Path(path_str.replace('/data/', '/data/test/'))
    elif path_str.startswith('data/'):
        return Path(path_str.replace('data/', 'data/test/', 1))
    else:
        # If path doesn't contain data/, assume it's already a test path
        return path

# Constants
LEVEL = 1
STEP = "s1"

# Setup logger
logger = setup_logger("lv1.s1")


def main(test_mode: bool = False, provider: str = "openai") -> None:
    """
    Main function to extract academic concepts from department names.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        provider: LLM provider to use (openai, anthropic, gemini)
    """
    # Note: model parameter removed as it's not used by the underlying extract_concepts_llm function
    try:
        logger.info(f"Starting Level {LEVEL} Step 1: Concept Extraction")
        logger.info(f"Using LLM provider: {provider}" + (f" with model: {model}" if model else ""))
        
        # Get configuration and file paths
        config = get_level_config(LEVEL)
        input_file, output_file, metadata_file = get_step_file_paths(LEVEL, STEP)
        
        # Convert to Path objects
        input_file = Path(input_file)
        output_file = Path(output_file)
        metadata_file = Path(metadata_file)
        
        # Use test paths if in test mode
        if test_mode:
            logger.info("Running in TEST mode")
            # Modify paths for test mode
            input_file = to_test_path(input_file)
            output_file = to_test_path(output_file)
            metadata_file = to_test_path(metadata_file)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Reading department names from: {input_file}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract concepts using LLM
        logger.info("Extracting academic concepts from department names...")
        result = extract_concepts_llm(
            input_file=str(input_file),
            level=LEVEL,
            output_file=str(output_file),
            metadata_file=str(metadata_file),
            provider=provider
        )
        
        if result and result.get('success'):
            logger.info(f"Successfully extracted concepts from {result['input_items_count']} departments")
            logger.info(f"Total concepts extracted: {result['extracted_concepts_count']}")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
            
            # Report statistics
            logger.info("Extraction Statistics:")
            logger.info(f"  - Average concepts per department: {result.get('concepts_per_input', 0):.2f}")
            logger.info(f"  - Processing time: {result.get('processing_time_seconds', 0):.2f} seconds")
        else:
            logger.warning("No concepts extracted")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 1: {str(e)}")
        raise


def test() -> None:
    """
    Test function for Level 1 Step 1.
    Uses test directories and smaller datasets.
    """
    logger.info("=" * 60)
    logger.info("Running Level 1 Step 1 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv1/raw"),
        Path("data/test/lv1/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    test_input = Path("data/test/lv1/raw/lv1_s0_department_names.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample department names...")
        test_depts = [
            "Computer Science Department",
            "Department of Electrical Engineering",
            "Mathematics Department",
            "Physics Department",
            "Chemistry Department"
        ]
        test_input.write_text("\n".join(test_depts))
        logger.info(f"Created test input with {len(test_depts)} departments")
    
    # Run main in test mode with default provider
    main(test_mode=True, provider="openai")
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 1 Step 1")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Level 1 Step 1: Extract Academic Concepts from Department Names"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with smaller datasets"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default=os.getenv("GLOSSARY_LLM_PROVIDER", "openai"),
        help="LLM provider to use (default: openai or GLOSSARY_LLM_PROVIDER env var)"
    )
    # Note: --model parameter removed as it's not currently used by the underlying functions
    # The LLM tier system selects models automatically based on provider
    
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        main(provider=args.provider)