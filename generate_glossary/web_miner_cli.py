"""
Command-line interface for web content mining.

This module provides a CLI for mining web content for technical concepts,
with support for both general web and Wikipedia content.
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import logging
import multiprocessing

# Set OpenMP thread limits early to avoid resource exhaustion
# This must be set before importing any libraries that might use OpenMP
os.environ["OMP_NUM_THREADS"] = "4"  # Hard cap at 4 threads for OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
# Limit TensorFlow/PyTorch thread usage
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
# Required to fix the "Thread creation failed: Resource temporarily unavailable" error
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenization

from generate_glossary.utils.web_miner import (
    SearchSettings,
    search_web_content,
    process_all_content_for_terms,
    write_results,
)

from dotenv import load_dotenv

load_dotenv('.env')

# Constants for memory management - with more conservative defaults
DEFAULT_CPU_PERCENT = 0.3  # Use only 30% of available CPUs by default (reduced from 50%)
# Limit max CPU count more aggressively to avoid resource exhaustion
CPU_COUNT = max(1, min(4, int(multiprocessing.cpu_count() * DEFAULT_CPU_PERCENT)))

# Constants for memory management - with more conservative defaults
PROCESS_BATCH_SIZE = min(3, max(1, CPU_COUNT))  # More conservative batch size (reduced from 5)
DEFAULT_CONCURRENT = min(8, max(4, CPU_COUNT))  # Reduced concurrency (was 10)
CHECKPOINT_INTERVAL = 10  # Save checkpoints every N terms

def configure_logging(log_level: str) -> None:
    """Configure logging with the specified level."""
    # Map string log levels to their numeric values
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    # Set the numeric level (default to ERROR if invalid level specified)
    numeric_level = levels.get(log_level.upper(), logging.ERROR)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(message)s'  # Simplified format
    )
    
    # Also set levels for specific loggers that might be too verbose
    logging.getLogger("aiohttp").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("tokenizers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    
    # Get our module logger
    logger = logging.getLogger(__name__)
    return logger

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

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.json')]
    if not checkpoint_files:
        return None
        
    # Sort by timestamp in filename (checkpoint_TIMESTAMP.json)
    checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    return latest_checkpoint

def save_checkpoint(results: Dict[str, Any], checkpoint_dir: str, batch_index: int) -> str:
    """Save current results as a checkpoint file."""
    # Print directly to stdout to ensure visibility regardless of log level
    print(f"\n=== CHECKPOINT: Saving progress to checkpoint directory ===")
    
    try:
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ensure the directory was created successfully
        if not os.path.isdir(checkpoint_dir):
            print(f"ERROR: Failed to create checkpoint directory: {checkpoint_dir}")
            logger.error(f"Failed to create checkpoint directory: {checkpoint_dir}")
            return ""
            
        # Create a unique filename with timestamp
        timestamp = int(time.time())
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.json")
        
        # Count terms in results for informational purposes
        term_count = len(results)
        
        # Write the checkpoint file
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Verify file was created successfully
        if not os.path.exists(checkpoint_file):
            print(f"ERROR: Checkpoint file was not created: {checkpoint_file}")
            logger.error(f"Checkpoint file was not created: {checkpoint_file}")
            return ""
            
        # Get file size for confirmation
        file_size = os.path.getsize(checkpoint_file) / (1024 * 1024)  # Size in MB
        
        # Success message to both stdout and logs
        success_msg = f"✓ Checkpoint saved: {checkpoint_file} ({term_count} terms, {file_size:.2f} MB)"
        print(success_msg)
        logger.info(success_msg)
        
        # List all checkpoints in the directory
        all_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".json")]
        print(f"✓ Total checkpoints: {len(all_checkpoints)} files in {os.path.abspath(checkpoint_dir)}")
        
        return checkpoint_file
    except Exception as e:
        error_msg = f"ERROR saving checkpoint: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        return ""

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
        default=30,  # Reduced default batch size
        help="Batch size for processing (1-50, default: 30). RapidAPI supports up to 100 queries per batch."
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
        default=min(4, CPU_COUNT),  # Hard cap at 4 threads
        help=f"Maximum threads for OpenMP and other thread-based libraries (default: min(4, {CPU_COUNT}))"
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
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory to save checkpoint files. If specified, checkpoints will be saved periodically."
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f"Number of term batches to process before saving a checkpoint (default: {CHECKPOINT_INTERVAL})"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",
        help="Set the logging level (default: ERROR to suppress most messages)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress all output except critical errors (equivalent to --log-level CRITICAL)"
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=True,  # Enable safe mode by default
        help="Enable safe mode with conservative thread settings to prevent resource exhaustion (default: enabled)"
    )
    
    args = parser.parse_args()
    
    # Configure logging based on command line arguments
    if args.quiet:
        log_level = "CRITICAL"
    else:
        log_level = args.log_level
    
    global logger
    logger = configure_logging(log_level)
    
    # Validate checkpoint directory early if provided
    if args.checkpoint_dir:
        print(f"\n=== CHECKPOINT: Initializing checkpointing to directory: {args.checkpoint_dir} ===")
        print(f"Checkpoint interval: Every {args.checkpoint_interval} batches")
        # Create the directory if it doesn't exist
        try:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            print(f"✓ Checkpoint directory ready: {os.path.abspath(args.checkpoint_dir)}")
        except Exception as e:
            print(f"ERROR: Failed to create checkpoint directory: {e}")
            logger.error(f"Failed to create checkpoint directory: {e}")
    
    try:
        # Apply safe mode settings if enabled (which is the default)
        if args.safe_mode:
            # Hard-cap threads to prevent resource exhaustion
            args.max_threads = min(4, args.max_threads)
            args.max_workers = min(4, args.max_workers)
            args.max_concurrent = min(8, args.max_concurrent)
            args.process_batch_size = min(3, args.process_batch_size)
            # Ensure batch sizes are reasonable
            args.batch_size = min(30, args.batch_size)
            
            logger.info("Safe mode enabled: Thread and batch limits applied to prevent resource exhaustion")
    
        # Set thread limits for OpenMP and other libraries
        # These must be applied even if set at the beginning, as they might get overridden
        max_threads = str(args.max_threads)
        os.environ["OMP_NUM_THREADS"] = max_threads
        os.environ["OPENBLAS_NUM_THREADS"] = max_threads
        os.environ["MKL_NUM_THREADS"] = max_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads
        os.environ["NUMEXPR_NUM_THREADS"] = max_threads
        os.environ["TF_NUM_INTRAOP_THREADS"] = max_threads
        os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, min(2, args.max_threads // 2)))
        
        # Read terms
        all_terms = read_terms(args.input)
        
        # If continuing from existing results, load them and filter terms
        existing_results = {}
        terms_to_process = all_terms
        
        # First check if we should continue from checkpoint
        checkpoint_file = None
        if args.checkpoint_dir:
            checkpoint_file = find_latest_checkpoint(args.checkpoint_dir)
            if checkpoint_file:
                print(f"✓ Found checkpoint file: {checkpoint_file}")
                logger.info(f"Found checkpoint file: {checkpoint_file}")
                logger.info(f"Loading results from checkpoint")
                existing_results = load_existing_results(checkpoint_file)
                
                # Filter out terms that are already in the checkpoint results
                existing_terms = set(existing_results.keys())
                terms_to_process = [term for term in all_terms if term not in existing_terms]
                
                print(f"✓ Loaded {len(existing_terms)} terms from checkpoint")
                print(f"✓ Will process {len(terms_to_process)} remaining terms")
                logger.info(f"Found {len(existing_terms)} terms in checkpoint")
                logger.info(f"Will process {len(terms_to_process)} remaining terms")
                
        # If no checkpoint or specified continue file, use continue-from argument
        if not existing_results and args.continue_from:
            if os.path.exists(args.continue_from):
                print(f"✓ Loading existing results from {args.continue_from}")
                logger.info(f"Loading existing results from {args.continue_from}")
                existing_results = load_existing_results(args.continue_from)
                
                # Filter out terms that are already in the existing results
                existing_terms = set(existing_results.keys())
                terms_to_process = [term for term in all_terms if term not in existing_terms]
                
                print(f"✓ Found {len(existing_terms)} existing terms")
                print(f"✓ Will process {len(terms_to_process)} new terms")
                logger.info(f"Found {len(existing_terms)} existing terms")
                logger.info(f"Will process {len(terms_to_process)} new terms")
            else:
                warning_msg = f"Specified continue file {args.continue_from} does not exist, will create a new file"
                print(f"WARNING: {warning_msg}")
                logger.warning(warning_msg)
        
        # Skip processing if there are no new terms
        if not terms_to_process:
            logger.info("No new terms to process, exiting")
            print("✓ No new terms to process, exiting")
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
            log_level=log_level,  # Pass the log level to settings
            safe_mode=args.safe_mode,  # Pass safe mode setting
        )
        
        # Set environment variable for multiprocessing
        os.environ["PYTHONUNBUFFERED"] = "1"  # Prevent output buffering
        
        # Run web mining
        print(f"\n=== Starting web content mining for {len(terms_to_process)} terms ===")
        logger.info(f"Mining web content for {len(terms_to_process)} terms...")
        logger.info(f"Using up to {args.max_concurrent} concurrent requests and {args.max_workers} worker processes")
        logger.info(f"Thread limits: OpenMP={os.environ['OMP_NUM_THREADS']}, Process batch size={args.process_batch_size}")
        
        # Process terms in smaller batches to allow checkpointing
        batch_size = min(args.batch_size, 20)  # Cap batch size for safety (reduced from 30)
        new_results = {}
        
        for batch_index, i in enumerate(range(0, len(terms_to_process), batch_size)):
            batch_terms = terms_to_process[i:i + batch_size]
            batch_msg = f"Processing batch {batch_index + 1} of {(len(terms_to_process) + batch_size - 1) // batch_size}"
            print(f"\n=== BATCH {batch_index + 1}: {batch_msg} ({len(batch_terms)} terms) ===")
            logger.info(batch_msg)
            logger.info(f"Terms in this batch: {len(batch_terms)}")
            
            try:
                # Process this batch of terms
                batch_content_by_term = asyncio.run(search_web_content(batch_terms, settings))
                
                # Process all content asynchronously for this batch
                batch_results = asyncio.run(process_all_content_for_terms(
                    batch_content_by_term, 
                    args.min_score,
                    args.process_batch_size,
                    args.skip_verification
                ))
                
                # Add batch results to new results
                new_results.update(batch_results)
                
                # If checkpoint directory is specified, save checkpoint after the specified interval
                if args.checkpoint_dir and (batch_index + 1) % args.checkpoint_interval == 0:
                    # Combine existing and new results
                    checkpoint_results = {**existing_results, **new_results}
                    save_checkpoint(checkpoint_results, args.checkpoint_dir, batch_index + 1)
                
                print(f"✓ Batch {batch_index + 1} completed: Processed {len(batch_terms)} terms")
                
            except RuntimeError as e:
                # Check for specific thread creation error
                if "Thread creation failed" in str(e) or "Resource temporarily unavailable" in str(e):
                    error_msg = f"Thread resource error: {e}"
                    print(f"ERROR: {error_msg}")
                    logger.error(error_msg)
                    print("ERROR: The system ran out of thread resources. Try reducing --max-threads and --max-workers.")
                    logger.error("The system ran out of thread resources. Try reducing --max-threads and --max-workers.")
                    # Save checkpoint before exiting
                    if args.checkpoint_dir and new_results:
                        checkpoint_results = {**existing_results, **new_results}
                        checkpoint_file = save_checkpoint(checkpoint_results, args.checkpoint_dir, f"thread_error_{batch_index}")
                        print(f"✓ Saved progress to checkpoint: {checkpoint_file}")
                        print(f"✓ You can resume from this checkpoint with reduced thread settings:")
                        print(f"  --max-threads 2 --max-workers 2 --max-concurrent 4 --process-batch-size 2")
                        logger.info(f"Saved progress to checkpoint: {checkpoint_file}")
                        logger.info(f"You can resume from this checkpoint with reduced thread settings:")
                        logger.info(f"  --max-threads 2 --max-workers 2 --max-concurrent 4 --process-batch-size 2")
                    sys.exit(1)
                else:
                    error_msg = f"Error processing batch {batch_index + 1}: {e}"
                    print(f"ERROR: {error_msg}")
                    logger.error(error_msg)
                    # Save what we have so far as a checkpoint if checkpoint dir is specified
                    if args.checkpoint_dir and new_results:
                        checkpoint_results = {**existing_results, **new_results}
                        checkpoint_file = save_checkpoint(checkpoint_results, args.checkpoint_dir, batch_index + 1)
                        print(f"✓ Saved progress to checkpoint: {checkpoint_file}")
                        print(f"✓ You can resume from this checkpoint later")
                        logger.info(f"Saved progress to checkpoint: {checkpoint_file}")
                        logger.info(f"You can resume from this checkpoint later")
                    continue
            except Exception as e:
                error_msg = f"Error processing batch {batch_index + 1}: {e}"
                print(f"ERROR: {error_msg}")
                logger.error(error_msg)
                # Save what we have so far as a checkpoint if checkpoint dir is specified
                if args.checkpoint_dir and new_results:
                    checkpoint_results = {**existing_results, **new_results}
                    checkpoint_file = save_checkpoint(checkpoint_results, args.checkpoint_dir, batch_index + 1)
                    print(f"✓ Saved progress to checkpoint: {checkpoint_file}")
                    print(f"✓ You can resume from this checkpoint later")
                    logger.info(f"Saved progress to checkpoint: {checkpoint_file}")
                    logger.info(f"You can resume from this checkpoint later")
                continue
        
        # After all batches, if there are results to save
        if new_results:
            # Save final checkpoint if checkpoint dir is specified
            if args.checkpoint_dir:
                print("\n=== FINAL CHECKPOINT ===")
                checkpoint_results = {**existing_results, **new_results}
                save_checkpoint(checkpoint_results, args.checkpoint_dir, "final")
            
            # If continuing from existing results, merge with new results
            merged_results = {**existing_results, **new_results}
            
            # Determine output path
            if args.continue_from and os.path.exists(args.continue_from):
                output_base = os.path.splitext(args.continue_from)[0]
                output_json = args.continue_from
            else:
                output_base = args.output
                output_json = f"{args.output}.json"
            
            # Write the final results
            print("\n=== Saving final results ===")
            write_results(merged_results, output_base)
            
            # Print summary of merged results
            total_terms = len(merged_results)
            total_content = sum(len(contents) for contents in merged_results.values())
            verified_content = sum(
                len([c for c in contents if c.get("is_verified", False)])
                for contents in merged_results.values()
            )
            
            summary = [
                f"\n=== Mining completed ===",
                f"✓ Total terms: {total_terms} ({len(existing_results)} existing + {len(new_results)} new)",
                f"✓ Total content: {total_content}",
                f"✓ Verified content: {verified_content}",
                f"✓ Results saved to:",
                f"  - {output_json} (full results)",
                f"  - {output_base}.txt (term-URL mappings)",
                f"  - {output_base}_summary.json (statistics)"
            ]
            
            # Print summary to both stdout and log
            for line in summary:
                print(line)
                logger.info(line)
            
    except RuntimeError as e:
        # Special handling for thread creation errors
        if "Thread creation failed" in str(e) or "Resource temporarily unavailable" in str(e):
            error_msg = f"Thread resource error: {e}"
            print(f"ERROR: {error_msg}")
            logger.error(error_msg)
            print("ERROR: The system ran out of thread resources. Try reducing --max-threads and --max-workers.")
            logger.error("The system ran out of thread resources. Try reducing --max-threads and --max-workers.")
            # Save checkpoint before exiting if possible
            if 'args' in locals() and hasattr(args, 'checkpoint_dir') and args.checkpoint_dir and 'new_results' in locals() and 'existing_results' in locals():
                checkpoint_results = {**existing_results, **new_results}
                checkpoint_file = save_checkpoint(checkpoint_results, args.checkpoint_dir, "thread_error")
                print(f"✓ Saved progress to checkpoint before error: {checkpoint_file}")
                print(f"✓ You can resume from this checkpoint with reduced thread settings:")
                print(f"  --max-threads 2 --max-workers 2 --max-concurrent 4 --process-batch-size 2")
                logger.info(f"Saved progress to checkpoint before error: {checkpoint_file}")
                logger.info(f"You can resume from this checkpoint with reduced thread settings:")
                logger.info(f"  --max-threads 2 --max-workers 2 --max-concurrent 4 --process-batch-size 2")
        else:
            print(f"ERROR: {e}")
            logger.error(f"Error: {e}")
            # Try to save checkpoint if an error occurs and checkpoint dir is specified
            if 'args' in locals() and hasattr(args, 'checkpoint_dir') and args.checkpoint_dir and 'new_results' in locals() and 'existing_results' in locals():
                checkpoint_results = {**existing_results, **new_results}
                checkpoint_file = save_checkpoint(checkpoint_results, args.checkpoint_dir, "error")
                print(f"✓ Saved progress to checkpoint before error: {checkpoint_file}")
                logger.info(f"Saved progress to checkpoint before error: {checkpoint_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        logger.error(f"Error: {e}")
        # Try to save checkpoint if an error occurs and checkpoint dir is specified
        if 'args' in locals() and hasattr(args, 'checkpoint_dir') and args.checkpoint_dir and 'new_results' in locals() and 'existing_results' in locals():
            checkpoint_results = {**existing_results, **new_results}
            checkpoint_file = save_checkpoint(checkpoint_results, args.checkpoint_dir, "error")
            print(f"✓ Saved progress to checkpoint before error: {checkpoint_file}")
            logger.info(f"Saved progress to checkpoint before error: {checkpoint_file}")
        sys.exit(1)

if __name__ == "__main__":
    main() 