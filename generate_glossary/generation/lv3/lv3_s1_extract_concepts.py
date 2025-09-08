"""
Level 3 Step 1: Extract Academic Concepts from Conference Topics

This script uses LLM-based extraction to identify academic concepts
from conference topics collected in Step 0.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

from generate_glossary.utils.logger import setup_logger
from ..concept_extraction import extract_concepts_llm
from ..level_config import get_level_config, get_step_file_paths


def to_test_path(path: Path) -> Path:
    """Convert a path to its test mode equivalent."""
    path_str = str(path)
    if '/data/' in path_str:
        return Path(path_str.replace('/data/', '/data/test/'))
    elif path_str.startswith('data/'):
        return Path(path_str.replace('data/', 'data/test/', 1))
    else:
        return path

LEVEL = 3
STEP = "s1"

logger = setup_logger("lv3.s1")


def main(test_mode: bool = False, provider: str = "openai", dry_run: bool = False) -> None:
    """
    Main function to extract academic concepts from conference topics.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        provider: LLM provider to use (openai, anthropic, gemini)
        dry_run: If True, creates synthetic output without calling LLM
    """
    try:
        logger.info(f"Starting Level {LEVEL} Step 1: Concept Extraction")
        if dry_run:
            logger.info("Running in DRY-RUN mode - synthetic output will be generated")
        else:
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
        
        logger.info(f"Reading conference topics from: {input_file}")
        
        # Read input topics
        with open(input_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if dry_run:
            # Generate synthetic output without calling LLM
            logger.info("Generating synthetic concepts (dry-run mode)...")
            
            # Create synthetic concepts based on input
            synthetic_concepts = []
            for topic in topics[:5]:  # Process only first 5 for dry-run
                # Generate a few mock concepts per topic
                base_name = topic.split('-')[0].strip() if '-' in topic else topic
                synthetic_concepts.extend([
                    f"{base_name} - Deep Learning Applications",
                    f"{base_name} - Neural Network Architectures",
                    f"{base_name} - Optimization Methods"
                ])
            
            # Write synthetic output
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(synthetic_concepts))
            
            # Create synthetic metadata
            metadata = {
                "dry_run": True,
                "level": LEVEL,
                "step": STEP,
                "input_items_count": len(topics),
                "extracted_concepts_count": len(synthetic_concepts),
                "concepts_per_input": len(synthetic_concepts) / len(topics) if topics else 0,
                "processing_time_seconds": 0.1,
                "llm_provider": "none (dry-run)",
                "success": True,
                "message": "Dry-run mode - synthetic data generated without LLM calls"
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Dry-run complete:")
            logger.info(f"  - Input topics: {len(topics)}")
            logger.info(f"  - Synthetic concepts generated: {len(synthetic_concepts)}")
            logger.info(f"  - Output saved to: {output_file}")
            logger.info(f"  - Metadata saved to: {metadata_file}")
            logger.info("Dry-run mode was used - no LLM calls were made")
        else:
            # Extract concepts using LLM
            logger.info("Extracting academic concepts from conference topics...")
            result = extract_concepts_llm(
                input_file=str(input_file),
                level=LEVEL,
                output_file=str(output_file),
                metadata_file=str(metadata_file),
                provider=provider
            )
            
            if result and result.get('success'):
                logger.info(f"Successfully extracted concepts from {result['input_items_count']} conference topics")
                logger.info(f"Total concepts extracted: {result['extracted_concepts_count']}")
                logger.info(f"Output saved to: {output_file}")
                logger.info(f"Metadata saved to: {metadata_file}")
                
                # Report statistics
                logger.info("Extraction Statistics:")
                logger.info(f"  - Average concepts per conference topic: {result.get('concepts_per_input', 0):.2f}")
                logger.info(f"  - Processing time: {result.get('processing_time_seconds', 0):.2f} seconds")
            else:
                logger.warning("No concepts extracted")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 1: {str(e)}")
        raise


def test(provider: str = "openai", dry_run: bool = False) -> None:
    """
    Test function for Level 3 Step 1.
    Uses test directories and smaller datasets.
    
    Args:
        provider: LLM provider to use for testing
        dry_run: If True, use dry-run mode
    """
    logger.info("=" * 60)
    logger.info("Running Level 3 Step 1 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv3/raw"),
        Path("data/test/lv3/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    test_input = Path("data/test/lv3/raw/lv3_s0_conference_topics.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample conference topics...")
        test_topics = [
            "International Conference on Machine Learning",
            "Conference on Computer Vision and Pattern Recognition",
            "Annual Meeting of the Association for Computational Linguistics",
            "IEEE International Conference on Robotics and Automation",
            "International Conference on Very Large Data Bases"
        ]
        test_input.write_text("\n".join(test_topics))
        logger.info(f"Created test input with {len(test_topics)} conference topics")
    
    # Check for LLM credentials if not in dry-run mode
    if not dry_run:
        api_key = os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(f"No API key found for {provider} - switching to dry-run mode")
            dry_run = True
    
    # Run main in test mode with specified provider
    main(test_mode=True, provider=provider, dry_run=dry_run)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 3 Step 1")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 3 Step 1: Extract Academic Concepts from Conference Topics"
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic output without making LLM calls (useful for testing without API credentials)"
    )
    
    args = parser.parse_args()
    
    # Check for credentials and auto-enable dry-run if missing
    if not args.dry_run and args.test:
        api_key = os.getenv(f"{args.provider.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY") 
        if not api_key:
            logger.warning(f"No API key found for {args.provider} - auto-enabling dry-run mode")
            args.dry_run = True
    
    if args.test:
        test(provider=args.provider, dry_run=args.dry_run)
    else:
        main(provider=args.provider, dry_run=args.dry_run)