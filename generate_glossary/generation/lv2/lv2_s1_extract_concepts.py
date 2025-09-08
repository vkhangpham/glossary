#!/usr/bin/env python3
"""
Level 2 Step 1: Extract Academic Concepts from Research Areas

This script uses LLM-based extraction to identify academic concepts
from research areas collected in Step 0.
"""

from pathlib import Path

from generate_glossary.utils.logger import setup_logger
from ..concept_extraction import extract_concepts_llm
from ..level_config import get_level_config, get_step_file_paths


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
STEP = "s1"

# Setup logger
logger = setup_logger("lv2.s1")


def main(test_mode: bool = False, provider: str = "openai") -> None:
    """
    Main function to extract academic concepts from research areas.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        provider: LLM provider to use (openai, anthropic, gemini)
    """
    # Note: model parameter removed as it's not used by the underlying extract_concepts_llm function
    try:
        logger.info(f"Starting Level {LEVEL} Step 1: Concept Extraction")
        logger.info(f"Using LLM provider: {provider}")
        
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
        
        logger.info(f"Reading research areas from: {input_file}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract concepts using LLM
        logger.info("Extracting academic concepts from research areas...")
        result = extract_concepts_llm(
            input_file=str(input_file),
            level=LEVEL,
            output_file=str(output_file),
            metadata_file=str(metadata_file),
            provider=provider
        )
        
        if result and result.get('success'):
            input_items_count = result.get('input_items_count', 0)
            extracted_concepts_count = result.get('extracted_concepts_count', 0)
            logger.info(f"Successfully extracted concepts from {input_items_count} research areas")
            logger.info(f"Total concepts extracted: {extracted_concepts_count}")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
            
            # Report statistics
            logger.info("Extraction Statistics:")
            logger.info(f"  - Average concepts per research area: {result.get('concepts_per_input', 0):.2f}")
            logger.info(f"  - Processing time: {result.get('processing_time_seconds', 0):.2f} seconds")
        else:
            logger.warning("No concepts extracted")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 1: {str(e)}")
        raise


def test(provider: str = "openai") -> None:
    """
    Test function for Level 2 Step 1.
    Uses test directories and smaller datasets.
    
    Args:
        provider: LLM provider to use for testing
    """
    logger.info("=" * 60)
    logger.info("Running Level 2 Step 1 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv2/raw"),
        Path("data/test/lv2/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    test_input = Path("data/test/lv2/raw/lv2_s0_research_areas.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample research areas...")
        test_areas = [
            "Machine Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Robotics",
            "Cybersecurity"
        ]
        test_input.write_text("\n".join(test_areas))
        logger.info(f"Created test input with {len(test_areas)} research areas")
    
    # Run main in test mode with specified provider
    main(test_mode=True, provider=provider)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 2 Step 1")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Level 2 Step 1: Extract Academic Concepts from Research Areas"
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
        help="LLM provider to use (maps to specific models: openai->gpt-4o-mini, anthropic->claude-3-haiku, gemini->gemini-1.5-flash)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test(provider=args.provider)
    else:
        main(provider=args.provider)