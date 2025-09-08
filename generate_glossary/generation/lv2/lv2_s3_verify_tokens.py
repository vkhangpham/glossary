#!/usr/bin/env python3
"""
Level 2 Step 3: Verify Single-Token Academic Terms

This script uses LLM-based verification to validate single-word academic terms
while automatically passing multi-word terms that have already been validated.
"""

from pathlib import Path

from generate_glossary.utils.logger import setup_logger
from ..token_verification import verify_single_tokens
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
STEP = "s3"

# Setup logger
logger = setup_logger("lv2.s3")


def main(test_mode: bool = False, provider: str = "openai") -> None:
    """
    Main function to verify single-token academic terms.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        provider: LLM provider to use (openai, anthropic, gemini)
    """
    # Note: model parameter removed as it's not used by the underlying verify_single_tokens function
    try:
        logger.info(f"Starting Level {LEVEL} Step 3: Token Verification")
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
        
        logger.info(f"Reading filtered concepts from: {input_file}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Verify single tokens using LLM
        logger.info("Verifying single-token academic terms...")
        result = verify_single_tokens(
            input_file=str(input_file),
            level=LEVEL,
            output_file=str(output_file),
            metadata_file=str(metadata_file),
            provider=provider
        )
        
        if result and result.get('success'):
            total_input = result.get('total_input_terms', 0)
            single_verified = result.get('single_word_verified_count', 0)
            single_rejected = result.get('single_word_rejected_count', 0)
            multi_word = result.get('multi_word_terms_count', 0)
            verification_rate = result.get('verification_rate', 0)
            logger.info(f"Verification complete:")
            logger.info(f"  - Total input: {total_input}")
            logger.info(f"  - Single-word verified: {single_verified}")
            logger.info(f"  - Single-word rejected: {single_rejected}")
            logger.info(f"  - Multi-word (auto-passed): {multi_word}")
            logger.info(f"  - Acceptance rate: {verification_rate:.1%}")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
            
            # Report processing time
            logger.info(f"Processing time: {result.get('verification_time_seconds', 0):.2f} seconds")
        else:
            logger.warning("No concepts verified")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 3: {str(e)}")
        raise


def test(provider: str = "openai") -> None:
    """
    Test function for Level 2 Step 3.
    Uses test directories and smaller datasets.
    
    Args:
        provider: LLM provider to use for testing
    """
    logger.info("=" * 60)
    logger.info("Running Level 2 Step 3 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv2/raw"),
        Path("data/test/lv2/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    # Note: s3 expects input from s2, which should be in raw/ directory
    test_input = Path("data/test/lv2/raw/lv2_s2_filtered_concepts.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample concepts...")
        test_concepts = [
            # Single-word terms (need verification)
            "Algorithms",
            "Networks",
            "Systems",
            "Security",
            "Optimization",
            "Learning",  # Might be rejected as too generic
            "Processing",   # Might be rejected as too generic
            # Multi-word terms (auto-passed)
            "Deep Learning",
            "Neural Networks",
            "Computer Vision",
            "Natural Language Processing",
            "Reinforcement Learning",
        ]
        test_input.write_text("\n".join(test_concepts))
        logger.info(f"Created test input with {len(test_concepts)} concepts")
        logger.info(f"  - Single-word: {len([c for c in test_concepts if ' ' not in c])}")
        logger.info(f"  - Multi-word: {len([c for c in test_concepts if ' ' in c])}")
    
    # Run main in test mode with specified provider
    main(test_mode=True, provider=provider)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 2 Step 3")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Level 2 Step 3: Verify Single-Token Academic Terms"
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
        help="LLM provider to use (maps to specific models for verification)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test(provider=args.provider)
    else:
        main(provider=args.provider)