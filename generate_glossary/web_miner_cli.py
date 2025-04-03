"""
Command-line interface for web content mining.

This module provides a CLI for mining web content for technical concepts,
with support for both general web and Wikipedia content.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import logging
import multiprocessing

from generate_glossary.utils.web_miner import (
    SearchSettings,
    search_web_content,
    process_all_content_for_terms,
    write_results,
)

from dotenv import load_dotenv

load_dotenv('.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get number of CPU cores for default settings - use a more conservative default
DEFAULT_CPU_PERCENT = 0.5  # Use 50% of available CPUs by default
CPU_COUNT = max(1, int(multiprocessing.cpu_count() * DEFAULT_CPU_PERCENT))

# Constants for memory management - with more conservative defaults
PROCESS_BATCH_SIZE = min(5, max(2, CPU_COUNT))  # More conservative batch size
DEFAULT_CONCURRENT = min(10, max(5, CPU_COUNT))  # Reduced concurrency

def read_terms(filepath: str) -> List[str]:
    """Read terms from file, one term per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_existing_results(filepath: str) -> Dict[str, Any]:
    """Load existing results from a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load existing results from {filepath}: {e}")
        return {}

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mine web content for technical concepts."
    )
    
    # Input arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Terms to mine web content for (file path)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Base path for output files (will create .txt, .json, and _summary.json)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--min-score",
        type=float,
        default=2.6,  # Default from verification_utils.py
        help="Minimum educational score threshold for verification (default: 2.6)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,  # Reduced from 50
        help="Batch size for processing (1-100, default: 30). RapidAPI supports up to 100 queries per batch."
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_CONCURRENT,
        help=f"Maximum concurrent requests (default: {DEFAULT_CONCURRENT}, based on CPU count)"
    )
    parser.add_argument(
        "--process-batch-size", 
        type=int,
        default=PROCESS_BATCH_SIZE,
        help=f"Number of contents to process at once (default: {PROCESS_BATCH_SIZE}, based on CPU count)"
    )
    parser.add_argument(
        "--provider",
        help="LLM provider for content verification",
        default="gemini"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bar"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        default=False,
        help="Skip content verification step to speed up processing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=CPU_COUNT,
        help=f"Maximum number of worker processes for CPU-bound tasks (default: {CPU_COUNT}, based on CPU count)"
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=CPU_COUNT,
        help=f"Maximum threads for OpenMP and other thread-based libraries (default: {CPU_COUNT})"
    )
    # Note: We now always generate both regular and Wikipedia-focused queries for each term
    parser.add_argument(
        "--system-prompt",
        help="Custom system prompt for the LLM that processes content. If not provided, a default extractive summarization prompt will be used."
    )
    parser.add_argument(
        "--cache-results",
        action="store_true",
        default=True,
        help="Cache processing results to avoid duplicate LLM calls (default: enabled)"
    )
    parser.add_argument(
        "--continue-from",
        help="Path to existing JSON results file to continue mining from. Will update this file in place."
    )
    parser.add_argument(
        "--no-batch-llm",
        action="store_true",
        default=False,
        help="Disable batch LLM processing (default: batch processing enabled)"
    )
    
    args = parser.parse_args()
    
    try:
        # Set OpenMP thread limit to avoid resource exhaustion
        os.environ["OMP_NUM_THREADS"] = str(args.max_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.max_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.max_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.max_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.max_threads)
        
        # Read terms
        all_terms = read_terms(args.input)
        
        # If continuing from existing results, load them and filter terms
        existing_results = {}
        terms_to_process = all_terms
        
        if args.continue_from:
            if os.path.exists(args.continue_from):
                logger.info(f"Loading existing results from {args.continue_from}")
                existing_results = load_existing_results(args.continue_from)
                
                # Filter out terms that are already in the existing results
                existing_terms = set(existing_results.keys())
                terms_to_process = [term for term in all_terms if term not in existing_terms]
                
                logger.info(f"Found {len(existing_terms)} existing terms")
                logger.info(f"Will process {len(terms_to_process)} new terms")
            else:
                logger.warning(f"Specified continue file {args.continue_from} does not exist, will create a new file")
        
        # Skip processing if there are no new terms
        if not terms_to_process:
            logger.info("No new terms to process, exiting")
            return
        
        # Configure settings
        settings = SearchSettings(
            min_score=args.min_score,
            max_concurrent_requests=args.max_concurrent,
            batch_size=args.batch_size,
            provider=args.provider,
            show_progress=not args.no_progress,
            system_prompt=args.system_prompt,  # Use custom prompt if provided
            skip_verification=args.skip_verification,  # Skip verification if requested
            max_workers=args.max_workers,  # Set max workers for multiprocessing
            use_cache=args.cache_results,  # Use caching if enabled
            use_batch_llm=not args.no_batch_llm,  # Use batch LLM processing unless disabled
        )
        
        # Set environment variable for multiprocessing
        os.environ["PYTHONUNBUFFERED"] = "1"  # Prevent output buffering
        
        # Run web mining
        logger.info(f"Mining web content for {len(terms_to_process)} terms...")
        logger.info(f"Using up to {args.max_concurrent} concurrent requests and {args.max_workers} worker processes")
        logger.info(f"Processing in batches of {args.batch_size} terms and {args.process_batch_size} contents")
        
        content_by_term = asyncio.run(search_web_content(terms_to_process, settings))
        
        # Process all content asynchronously
        new_results = asyncio.run(process_all_content_for_terms(
            content_by_term, 
            args.min_score,
            args.process_batch_size,
            args.skip_verification
        ))
        
        # If continuing, merge the new results with existing ones
        if args.continue_from and existing_results:
            # Update existing results with new results
            merged_results = {**existing_results, **new_results}
            
            # Write the merged results back to the original file
            output_base = os.path.splitext(args.continue_from)[0]
            write_results(merged_results, output_base)
            
            # Print summary of merged results
            total_terms = len(merged_results)
            total_content = sum(len(contents) for contents in merged_results.values())
            verified_content = sum(
                len([c for c in contents if c.get("is_verified", False)])
                for contents in merged_results.values()
            )
            logger.info(f"\nMining completed and results merged:")
            logger.info(f"- Total terms: {total_terms} ({len(existing_results)} existing + {len(new_results)} new)")
            logger.info(f"- Total content: {total_content}")
            logger.info(f"- Verified content: {verified_content}")
            logger.info(f"- Updated results saved to:")
            logger.info(f"  - {args.continue_from} (full results)")
            logger.info(f"  - {output_base}.txt (term-URL mappings)")
            logger.info(f"  - {output_base}_summary.json (statistics)")
        else:
            # Write results to the specified output path
            write_results(new_results, args.output)
            
            # Print summary of new results
            total_content = sum(len(contents) for contents in new_results.values())
            verified_content = sum(
                len([c for c in contents if c.get("is_verified", False)])
                for contents in new_results.values()
            )
            logger.info(f"\nMining completed:")
            logger.info(f"- Total terms processed: {len(terms_to_process)}")
            logger.info(f"- Total content found: {total_content}")
            logger.info(f"- Verified content: {verified_content}")
            logger.info(f"- Results saved to:")
            logger.info(f"  - {args.output}.json (full results)")
            logger.info(f"  - {args.output}.txt (term-URL mappings)")
            logger.info(f"  - {args.output}_summary.json (statistics)")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 