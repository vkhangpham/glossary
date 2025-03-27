#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from generate_glossary.metadata_collector import collect_metadata, find_step_file, find_final_file, find_step_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Collect metadata for terms in a specific level')
    parser.add_argument('-l', '--level', type=int, required=True,
                        help='Level number (0, 1, 2, etc.)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (default: data/lvX/metadata.json)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Set default output path if not provided
    if args.output is None:
        args.output = f'data/lv{args.level}/lv{args.level}_metadata.json'
    
    # Ensure level directory exists
    level_dir = Path(f'data/lv{args.level}')
    if not level_dir.exists():
        logger.error(f"Level directory {level_dir} does not exist")
        return 1
    
    # Check for final terms file
    final_file = find_final_file(level_dir, args.level)
    if not final_file:
        logger.error(f"Could not find final terms file for level {args.level}")
        logger.error("Have you completed the full pipeline including deduplication?")
        return 1
    
    logger.info(f"Found final terms file: {final_file}")
    
    # Check for raw directory
    raw_dir = level_dir / 'raw'
    if not raw_dir.exists():
        logger.warning(f"Raw directory {raw_dir} does not exist")
        logger.warning("Will rely on other metadata sources if available")
    else:
        # Check for source file and metadata files
        source_file = find_step_file(raw_dir, args.level, 1, 'csv')
        if source_file:
            logger.info(f"Found source file: {source_file}")
        
        # Log available metadata files
        metadata_files = []
        for step in range(4):  # 0, 1, 2, 3
            metadata_file = find_step_metadata(raw_dir, args.level, step)
            if metadata_file:
                metadata_files.append(metadata_file)
                logger.info(f"Found metadata file for step {step}: {metadata_file}")
        
        if not source_file and not metadata_files:
            logger.warning("No source files or metadata files found in raw directory")
    
    # Check for postprocessed directory
    postprocessed_dir = level_dir / 'postprocessed'
    if not postprocessed_dir.exists():
        logger.warning(f"Postprocessed directory {postprocessed_dir} does not exist")
        logger.warning("Variations data will be incomplete")
    elif not any(postprocessed_dir.glob('*.json')):
        logger.warning(f"No JSON files found in {postprocessed_dir}")
        logger.warning("Variations data will be incomplete")
    else:
        logger.debug(f"Found {len(list(postprocessed_dir.glob('*.json')))} JSON files in {postprocessed_dir}")
    
    # Collect metadata
    logger.info(f"Collecting metadata for level {args.level}")
    try:
        collect_metadata(args.level, args.output)
        logger.info(f"Metadata collection complete. Output saved to {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Error collecting metadata: {e}", exc_info=args.verbose)
        return 1

if __name__ == '__main__':
    exit(main()) 