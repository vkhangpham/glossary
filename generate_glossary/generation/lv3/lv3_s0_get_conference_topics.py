#!/usr/bin/env python3
"""
Level 3 Step 0: Extract Conference Topics from Research Area Web Pages

This script extracts conference topics from research area websites using Firecrawl.
It processes the research areas from Level 2 and extracts conference topics.
"""

import sys
import os
import json
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
LEVEL = 3
STEP = "s0"

# Setup logger
logger = get_logger("lv3.s0")


def main(test_mode: bool = False, input_file: Optional[str] = None) -> None:
    """
    Main function to extract conference topics from research area websites.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        input_file: Optional custom input file path
    """
    try:
        logger.info(f"Starting Level {LEVEL} Step 0: Conference Topic Extraction")
        
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
        
        logger.info(f"Reading research areas from: {input_file}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract web content for Level 3 (conference topics)
        logger.info("Extracting conference topics from research area websites...")
        result = extract_web_content(
            input_file=str(input_file),
            level=LEVEL,
            output_file=str(output_file),
            metadata_file=str(metadata_file)
        )
        
        if result and not result.get('error'):
            logger.info(f"Successfully extracted {result['extracted_terms_count']} conference topics")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
        elif result and result.get('error'):
            logger.error(f"Extraction failed: {result['error']}")
        else:
            logger.warning("No conference topics extracted")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 0: {str(e)}")
        raise


def test() -> None:
    """
    Test function for Level 3 Step 0.
    Uses test directories and smaller datasets.
    """
    logger.info("=" * 60)
    logger.info("Running Level 3 Step 0 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv2"),
        Path("data/test/lv3/raw"),
        Path("data/test/lv3/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    test_input = Path("data/test/lv2/lv2_final.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample research areas...")
        test_research_areas = [
            "Machine Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Robotics",
            "Database Systems"
        ]
        test_input.write_text("\n".join(test_research_areas))
        logger.info(f"Created test input with {len(test_research_areas)} research areas")
    
    # Check if Firecrawl API key is available
    if not os.getenv('FIRECRAWL_API_KEY'):
        logger.warning("FIRECRAWL_API_KEY not found - creating mock test data instead")
        
        # Create mock output files for test mode
        test_output = Path("data/test/lv3/raw/lv3_s0_conference_topics.txt")
        test_metadata = Path("data/test/lv3/raw/lv3_s0_conference_topics_metadata.json")
        
        # Create mock conference topics
        mock_topics = [
            "ICML - International Conference on Machine Learning",
            "NeurIPS - Neural Information Processing Systems",
            "CVPR - Computer Vision and Pattern Recognition",
            "ICCV - International Conference on Computer Vision",
            "ACL - Association for Computational Linguistics",
            "EMNLP - Empirical Methods in Natural Language Processing",
            "ICRA - International Conference on Robotics and Automation",
            "VLDB - Very Large Data Bases",
            "SIGMOD - Special Interest Group on Management of Data"
        ]
        
        test_output.write_text("\n".join(mock_topics))
        logger.info(f"Created mock output with {len(mock_topics)} conference topics")
        
        # Create minimal metadata
        metadata = {
            "test_mode": True,
            "firecrawl_skipped": True,
            "reason": "FIRECRAWL_API_KEY not available",
            "mock_data": True,
            "topics_count": len(mock_topics),
            "level": LEVEL,
            "step": STEP
        }
        
        with open(test_metadata, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Mock output saved to: {test_output}")
        logger.info(f"Mock metadata saved to: {test_metadata}")
        logger.info("Firecrawl was skipped in test mode - using mock data")
    else:
        # Run main in test mode with Firecrawl
        main(test_mode=True)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 3 Step 0")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 3 Step 0: Extract Conference Topics from Research Area Web Pages"
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