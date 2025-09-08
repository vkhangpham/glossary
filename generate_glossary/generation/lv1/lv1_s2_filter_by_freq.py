#!/usr/bin/env python3
"""
Level 1 Step 2: Filter Concepts by Institutional Frequency

This script filters extracted concepts based on their frequency across
different institutions, keeping only those that appear in multiple sources.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from collections import Counter

from generate_glossary.utils.logger import setup_logger
from ..shared.frequency_filtering import filter_by_frequency
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
STEP = "s2"

# Setup logger
logger = setup_logger("lv1.s2")


def main(test_mode: bool = False, min_frequency: Optional[int] = None, threshold_percent: Optional[float] = None) -> None:
    """
    Main function to filter concepts by institutional frequency.
    
    Args:
        test_mode: If True, uses test directories and smaller datasets
        min_frequency: Minimum frequency count (overrides config if provided)
        threshold_percent: Minimum percentage threshold (overrides config if provided)
    """
    try:
        logger.info(f"Starting Level {LEVEL} Step 2: Frequency Filtering")
        
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
        
        logger.info(f"Reading extracted concepts from: {input_file}")
        
        # Apply CLI overrides if provided
        # Handle precedence: threshold_percent takes priority over min_frequency
        if threshold_percent is not None and min_frequency is not None:
            logger.warning(f"Both --threshold-percent and --min-frequency provided. Using --threshold-percent={threshold_percent}, ignoring --min-frequency={min_frequency}")
            min_frequency = None  # Clear min_frequency to prevent processing
        
        if threshold_percent is not None:
            logger.info(f"Overriding threshold percent from config ({config.frequency_threshold}) to {threshold_percent}")
            import generate_glossary.generation.shared.level_config as lc
            lc.LEVEL_CONFIGS[LEVEL].frequency_threshold = threshold_percent
        elif min_frequency is not None:
            # Compute threshold_percent from min_frequency
            # Count distinct sources from input file
            sources = set()
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Handle format: "source - concept" or just "concept"
                    if ' - ' in line:
                        source, _ = line.split(' - ', 1)
                        sources.add(source.strip())
                    else:
                        # No source info, treat as generic
                        sources.add('generic')
            
            sources_count = len(sources)
            if sources_count > 0:
                # Derive threshold percent and clamp to [0, 1]
                derived_threshold = min_frequency / sources_count
                derived_threshold = max(0.0, min(1.0, derived_threshold))
                
                logger.info(f"Computed threshold from --min-frequency={min_frequency}:")
                logger.info(f"  - Found {sources_count} distinct sources")
                logger.info(f"  - Derived threshold: {derived_threshold:.2%}")
                logger.info(f"  - Overriding config threshold ({config.frequency_threshold}) to {derived_threshold}")
                
                import generate_glossary.generation.shared.level_config as lc
                lc.LEVEL_CONFIGS[LEVEL].frequency_threshold = derived_threshold
            else:
                logger.warning(f"No sources found in input file, ignoring --min-frequency={min_frequency}")
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply frequency filtering
        logger.info("Applying frequency-based filtering...")
        result = filter_by_frequency(
            input_file=str(input_file),
            level=LEVEL,
            output_file=str(output_file),
            metadata_file=str(metadata_file)
        )
        
        if result and result.get('success'):
            input_count = result.get('input_concepts_count', 0)
            filtered_count = result.get('filtered_concepts_count', 0)
            filter_rate = 1 - (filtered_count / input_count) if input_count > 0 else 0
            
            logger.info(f"Filtering complete:")
            logger.info(f"  - Input concepts: {input_count}")
            logger.info(f"  - Kept concepts: {filtered_count}")
            logger.info(f"  - Filtered out: {input_count - filtered_count}")
            logger.info(f"  - Filter rate: {filter_rate:.1%}")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
            
            # Report frequency distribution from nested statistics
            if 'statistics' in result and 'frequency_distribution' in result['statistics']:
                dist = result['statistics']['frequency_distribution']
                logger.info("Frequency Distribution:")
                for freq, count in sorted(dist.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
                    logger.info(f"  - Frequency {freq}: {count} concepts")
        else:
            logger.warning("No concepts passed frequency filtering")
            
    except Exception as e:
        logger.error(f"Error in Level {LEVEL} Step 2: {str(e)}")
        raise


def test() -> None:
    """
    Test function for Level 1 Step 2.
    Uses test directories and smaller datasets.
    """
    logger.info("=" * 60)
    logger.info("Running Level 1 Step 2 in TEST mode")
    logger.info("=" * 60)
    
    # Create test directories if needed
    test_dirs = [
        Path("data/test/lv1/raw"),
        Path("data/test/lv1/processed"),
    ]
    for dir_path in test_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a small test input file if it doesn't exist
    test_input = Path("data/test/lv1/raw/lv1_s1_extracted_concepts.txt")
    if not test_input.exists():
        logger.info("Creating test input file with sample concepts...")
        # Create concepts with varying frequencies
        test_concepts = [
            "Machine Learning",  # High frequency
            "Machine Learning",
            "Machine Learning",
            "Artificial Intelligence",  # Medium frequency
            "Artificial Intelligence",
            "Data Science",  # Medium frequency
            "Data Science",
            "Computer Vision",  # Low frequency
            "Natural Language Processing",  # Low frequency
            "Robotics",  # Single occurrence
        ]
        test_input.write_text("\n".join(test_concepts))
        logger.info(f"Created test input with {len(test_concepts)} concept instances")
    
    # Run main in test mode with lower thresholds for testing
    main(test_mode=True, min_frequency=2, threshold_percent=0.2)
    
    logger.info("=" * 60)
    logger.info("Test completed for Level 1 Step 2")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Level 1 Step 2: Filter Concepts by Institutional Frequency"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with smaller datasets"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        help="Minimum frequency count (default: from config)"
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        help="Minimum percentage threshold (default: from config)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        main(
            min_frequency=args.min_frequency,
            threshold_percent=args.threshold_percent
        )