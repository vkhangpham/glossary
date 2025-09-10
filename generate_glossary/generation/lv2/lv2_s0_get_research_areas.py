#!/usr/bin/env python3
"""
Level 2 Step 0: Extract Research Areas from Department Web Pages

This script extracts research areas from department websites using Firecrawl.
It processes the departments from Level 1 and extracts their research areas.
"""

from pathlib import Path
from typing import Optional

from generate_glossary.utils.logger import get_logger
from ..web_extraction_firecrawl import extract_web_content
from ...config import get_level_config


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
LEVEL = 2
STEP = "s0"

# Setup logger
logger = get_logger("lv2.s0")


def main(test_mode: bool = False, input_file: Optional[str] = None) -> None:
    """
    Main function to extract research areas from department websites.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        input_file: Optional custom input file path
    """
    try:
        logger.info(f"Starting Level {LEVEL} Step 0: Research Area Extraction")
        
        # Get configuration and file paths
        config = get_level_config(LEVEL)
        default_input = str(config.get_step_input_file(0))
        output_file = str(config.get_step_output_file(0))
        metadata_file = str(config.get_step_metadata_file(0))
        
        # Use custom input file if provided, otherwise use default
        if input_file:
            input_file = Path(input_file)
        else:
            input_file = Path(default_input)
        
        # Convert to Path objects
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
        
        logger.info(f"Reading departments from: {input_file}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract web content for Level 2 (research areas)
        logger.info("Extracting research areas from department websites...")
        result = extract_web_content(
            input_file=str(input_file),
            level=LEVEL,
            output_file=str(output_file),
            metadata_file=str(metadata_file)
        )
        
        if result and not result.get('error'):
            extracted_count = result.get('extracted_terms_count', 0)
            logger.info(f"Successfully extracted {extracted_count} research areas")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
        elif result and result.get('error'):
            logger.error(f"Extraction failed: {result.get('error', 'Unknown error')}")
        else:
            logger.warning("No research areas extracted")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 0: {str(e)}")
        raise


def test() -> None:
    """
    Test function for Level 2 Step 0.
    Uses test directories and smaller datasets.
    """
    logger.info("=" * 60)
    logger.info("Running Level 2 Step 0 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv1"),
        Path("data/test/lv2/raw"),
        Path("data/test/lv2/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    test_input = Path("data/test/lv1/lv1_final.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample department URLs for Firecrawl...")
        test_department_urls = [
            "https://cs.stanford.edu/",
            "https://www.eecs.berkeley.edu/",
            "https://me.mit.edu/"
        ]
        test_input.write_text("\n".join(test_department_urls))
        logger.info(f"Created test input with {len(test_department_urls)} department URLs")
    
    # Run main in test mode
    main(test_mode=True)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 2 Step 0")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 2 Step 0: Extract Research Areas from Department Web Pages"
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