#!/usr/bin/env python3
"""
Level 3 Step 3: Verify Single-Token Conference Terms

This script uses LLM-based verification to validate single-word conference terms
while automatically passing multi-word terms that have already been validated.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

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
LEVEL = 3
STEP = "s3"

# Setup logger
logger = setup_logger("lv3.s3")


def main(test_mode: bool = False, provider: str = "openai", dry_run: bool = False) -> None:
    """
    Main function to verify single-token conference terms.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        provider: LLM provider to use (openai, anthropic, gemini)
        dry_run: If True, creates synthetic output without calling LLM
    """
    # Note: model parameter removed as it's not used by the underlying verify_single_tokens function
    try:
        logger.info(f"Starting Level {LEVEL} Step 3: Token Verification")
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
        
        logger.info(f"Reading filtered concepts from: {input_file}")
        
        # Read input concepts
        with open(input_file, 'r', encoding='utf-8') as f:
            concepts = [line.strip() for line in f if line.strip()]
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if dry_run:
            # Generate synthetic output without calling LLM
            logger.info("Generating synthetic verification results (dry-run mode)...")
            
            verified_concepts = []
            single_word_verified = 0
            single_word_rejected = 0
            multi_word_count = 0
            
            for concept in concepts:
                if ' ' in concept:
                    # Multi-word terms auto-pass
                    verified_concepts.append(concept)
                    multi_word_count += 1
                else:
                    # Simulate verification for single words
                    # Accept specific academic-sounding terms, reject generic ones
                    generic_terms = {'systems', 'science', 'data', 'information', 'technology', 'analysis'}
                    if concept.lower() in generic_terms:
                        single_word_rejected += 1
                        logger.debug(f"Rejected (simulated): {concept}")
                    else:
                        verified_concepts.append(concept)
                        single_word_verified += 1
                        logger.debug(f"Verified (simulated): {concept}")
            
            # Write synthetic output
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(verified_concepts))
            
            # Create synthetic metadata
            total_input = len(concepts)
            total_verified = len(verified_concepts)
            verification_rate = total_verified / total_input if total_input > 0 else 0
            
            metadata = {
                "dry_run": True,
                "level": LEVEL,
                "step": STEP,
                "total_input_terms": total_input,
                "single_word_verified_count": single_word_verified,
                "single_word_rejected_count": single_word_rejected,
                "multi_word_terms_count": multi_word_count,
                "total_verified_terms": total_verified,
                "verification_rate": verification_rate,
                "verification_time_seconds": 0.1,
                "llm_provider": "none (dry-run)",
                "success": True,
                "message": "Dry-run mode - synthetic verification performed without LLM calls"
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Dry-run verification complete:")
            logger.info(f"  - Total input: {total_input}")
            logger.info(f"  - Single-word verified: {single_word_verified}")
            logger.info(f"  - Single-word rejected: {single_word_rejected}")
            logger.info(f"  - Multi-word (auto-passed): {multi_word_count}")
            logger.info(f"  - Acceptance rate: {verification_rate:.1%}")
            logger.info(f"  - Output saved to: {output_file}")
            logger.info(f"  - Metadata saved to: {metadata_file}")
            logger.info("Dry-run mode was used - no LLM calls were made")
        else:
            # Verify single tokens using LLM
            logger.info("Verifying single-token conference terms...")
            result = verify_single_tokens(
                input_file=str(input_file),
                level=LEVEL,
                output_file=str(output_file),
                metadata_file=str(metadata_file),
                provider=provider
            )
            
            if result and result.get('success'):
                logger.info(f"Verification complete:")
                logger.info(f"  - Total input: {result['total_input_terms']}")
                logger.info(f"  - Single-word verified: {result['single_word_verified_count']}")
                logger.info(f"  - Single-word rejected: {result['single_word_rejected_count']}")
                logger.info(f"  - Multi-word (auto-passed): {result['multi_word_terms_count']}")
                logger.info(f"  - Acceptance rate: {result['verification_rate']:.1%}")
                logger.info(f"Output saved to: {output_file}")
                logger.info(f"Metadata saved to: {metadata_file}")
                
                # Report processing time
                logger.info(f"Processing time: {result.get('verification_time_seconds', 0):.2f} seconds")
            else:
                logger.warning("No concepts verified")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 3: {str(e)}")
        raise


def test(provider: str = "openai", dry_run: bool = False) -> None:
    """
    Test function for Level 3 Step 3.
    Uses test directories and smaller datasets.
    
    Args:
        provider: LLM provider to use for testing
        dry_run: If True, use dry-run mode
    """
    logger.info("=" * 60)
    logger.info("Running Level 3 Step 3 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv3/raw"),
        Path("data/test/lv3/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    # Note: s3 expects input from s2, which should be in raw/ directory
    test_input = Path("data/test/lv3/raw/lv3_s2_filtered_concepts.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample concepts...")
        test_concepts = [
            # Single-word terms (need verification)
            "Learning",
            "Vision",
            "Processing",
            "Networks",
            "Intelligence",
            "Systems",  # Might be rejected as too generic
            "Science",  # Might be rejected as too generic
            # Multi-word terms (auto-passed)
            "Deep Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Machine Learning",
            "Neural Networks",
            "Reinforcement Learning",
        ]
        test_input.write_text("\n".join(test_concepts))
        logger.info(f"Created test input with {len(test_concepts)} concepts")
        logger.info(f"  - Single-word: {len([c for c in test_concepts if ' ' not in c])}")
        logger.info(f"  - Multi-word: {len([c for c in test_concepts if ' ' in c])}")
    
    # Check for LLM credentials if not in dry-run mode
    if not dry_run:
        api_key = os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(f"No API key found for {provider} - switching to dry-run mode")
            dry_run = True
    
    # Run main in test mode with specified provider
    main(test_mode=True, provider=provider, dry_run=dry_run)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 3 Step 3")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 3 Step 3: Verify Single-Token Conference Terms"
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