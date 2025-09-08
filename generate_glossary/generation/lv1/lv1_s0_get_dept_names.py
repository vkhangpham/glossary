#!/usr/bin/env python3
"""
Level 1 Step 0: Extract Department Names from College Web Pages

This script extracts department names from college websites using Firecrawl.
It processes the colleges from Level 0 and extracts their departments.
"""

import sys
from pathlib import Path
from typing import Optional

from generate_glossary.utils.logger import setup_logger
from ..shared.web_extraction_firecrawl import extract_web_content
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
STEP = "s0"

# Setup logger
logger = setup_logger("lv1.s0")


def main(test_mode: bool = False) -> None:
    """
    Main function to extract department names from college websites.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
    """
    try:
        logger.info(f"Starting Level {LEVEL} Step 0: Department Name Extraction")
        
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
        
        # Input file already set from get_step_file_paths
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logger.info(f"Reading colleges from: {input_file}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract web content for Level 1 (departments)
        logger.info("Extracting department names from college websites...")
        result = extract_web_content(
            input_file=str(input_file),
            level=LEVEL,
            output_file=str(output_file),
            metadata_file=str(metadata_file)
        )
        
        if result and not result.get('error'):
            logger.info(f"Successfully extracted {result['extracted_terms_count']} departments")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
        elif result and result.get('error'):
            logger.error(f"Extraction failed: {result['error']}")
        else:
            logger.warning("No departments extracted")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 0: {str(e)}")
        raise


def test() -> None:
    """
    Test function for Level 1 Step 0.
    Uses test directories and smaller datasets.
    """
    logger.info("=" * 60)
    logger.info("Running Level 1 Step 0 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv0"),
        Path("data/test/lv1/raw"),
        Path("data/test/lv1/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    test_input = Path("data/test/lv0/lv0_final.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample colleges...")
        test_colleges = [
            "College of Engineering",
            "School of Medicine",
            "College of Liberal Arts and Sciences"
        ]
        test_input.write_text("\n".join(test_colleges))
        logger.info(f"Created test input with {len(test_colleges)} colleges")
    
    # Run main in test mode
    main(test_mode=True)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 1 Step 0")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 1 Step 0: Extract Department Names from College Web Pages"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with smaller datasets"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        main()