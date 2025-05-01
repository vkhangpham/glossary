"""
Implementation module for web content mining.

This module contains the actual implementation logic for the web content mining CLI,
keeping the CLI interface code clean and minimal.
"""

import os
import json
import time
import asyncio
import logging
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any, Optional

from generate_glossary.utils.web_miner import (
    SearchSettings,
    WebContent,
    search_web_content,
    process_all_content_for_terms,
    write_results,
    load_results,
    save_results,
)

from generate_glossary.utils.tavily_miner import (
    tavily_search_web_content,
    tavily_process_all_content,
)

# Constants for memory management - with more conservative defaults
DEFAULT_CPU_PERCENT = 0.5
CPU_COUNT = max(2, min(6, int(multiprocessing.cpu_count() * DEFAULT_CPU_PERCENT)))
PROCESS_BATCH_SIZE = min(5, max(2, CPU_COUNT))
DEFAULT_CONCURRENT = min(12, max(6, CPU_COUNT))
CHECKPOINT_INTERVAL = 10

# Available search providers
SEARCH_PROVIDERS = ["rapidapi", "tavily"]

# Setup logger
logger = logging.getLogger(__name__)

def configure_logging(log_level: str) -> logging.Logger:
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
        format='%(asctime)s | %(levelname)s | %(message)s',  # Include timestamp and level
        datefmt='%H:%M:%S'  # Short time format
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
    return logging.getLogger(__name__)

def read_terms(filepath: str) -> List[str]:
    """Read terms from file, one term per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

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
            json.dump(results, f, indent=4, ensure_ascii=False)
        
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

def set_thread_limits(max_threads: int):
    """Set thread limits for various libraries."""
    max_threads_str = str(max_threads)
    os.environ["OMP_NUM_THREADS"] = max_threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = max_threads_str
    os.environ["MKL_NUM_THREADS"] = max_threads_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = max_threads_str
    os.environ["TF_NUM_INTRAOP_THREADS"] = max_threads_str
    os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, min(2, max_threads // 2)))

def setup_file_logging(log_file: str) -> bool:
    """Setup logging to a file in addition to console."""
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
        print(f"✓ Logging to file: {log_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to set up file logging: {e}")
        return False

def apply_performance_settings(args):
    """Apply performance optimization settings based on command-line arguments."""
    if args.high_performance:
        logger.info("HIGH PERFORMANCE MODE: Optimizing for maximum speed")
        # Increase concurrency to maximum safe values
        args.max_concurrent = max(15, min(30, multiprocessing.cpu_count() * 2))
        args.max_workers = max(4, min(8, multiprocessing.cpu_count()))
        args.max_threads = min(6, args.max_workers)
        args.process_batch_size = min(8, max(4, args.max_workers))
        # Optimize batch size
        args.batch_size = 50  # Use maximum batch size
        # Skip verification for speed
        args.skip_verification = True
        # Use faster threshold
        args.content_threshold = 1.5
        # Performance optimizations
        args.no_parallel_extraction = False
        args.no_skip_low_quality = False
        args.performance_mode = True
        
        logger.info(f"High performance settings: max_concurrent={args.max_concurrent}, " +
                  f"max_workers={args.max_workers}, batch_size={args.batch_size}")
        if not args.minimal_output:
            print(f"🚀 HIGH PERFORMANCE MODE enabled with max_concurrent={args.max_concurrent}, " +
                  f"max_workers={args.max_workers}")
    
    # Apply performance mode settings if enabled
    elif args.performance_mode:
        logger.info("Performance mode enabled: Using optimized settings for faster processing")
        # Increase concurrency but still stay within safe limits
        args.max_concurrent = min(20, max(10, args.max_concurrent))
        # Ensure we're using parallel extraction
        args.no_parallel_extraction = False
        # Skip low quality content to avoid processing worthless data
        args.no_skip_low_quality = False
        # Use conservative content threshold to quickly filter out poor content
        args.content_threshold = 1.8
        # Raise batch size slightly for more efficient processing
        args.batch_size = min(40, max(30, args.batch_size))
        # Skip verification on most content to speed up processing
        if not args.skip_verification:
            logger.info("Performance mode: Enabling skip_verification to speed up processing")
            args.skip_verification = True
        
        logger.info(f"Performance mode settings: max_concurrent={args.max_concurrent}, " +
                  f"batch_size={args.batch_size}, content_threshold={args.content_threshold}")
        
    # Apply safe mode settings if enabled (which is the default)
    elif args.safe_mode:
        # Hard-cap threads to prevent resource exhaustion
        args.max_threads = min(4, args.max_threads)
        args.max_workers = min(4, args.max_workers)
        args.max_concurrent = min(10, args.max_concurrent)
        args.process_batch_size = min(3, args.process_batch_size)
        # Ensure batch sizes are reasonable
        args.batch_size = min(30, args.batch_size)
        
        logger.info("Safe mode enabled: Thread and batch limits applied to prevent resource exhaustion")
    
    return args

def load_existing_results(checkpoint_dir: Optional[str], continue_from: Optional[str], all_terms: List[str], minimal_output: bool = False):
    """Load existing results from checkpoint or continue-from file."""
    existing_results = {}
    terms_to_process = all_terms
    
    # First check if we should continue from checkpoint
    checkpoint_file = None
    if checkpoint_dir:
        checkpoint_file = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_file:
            logger.info(f"Found checkpoint file: {checkpoint_file}")
            logger.info(f"Loading results from checkpoint")
            existing_results = load_results(checkpoint_file)
            
            # Filter out terms that are already in the checkpoint results
            existing_terms = set(existing_results.keys())
            terms_to_process = [term for term in all_terms if term not in existing_terms]

            logger.info(f"Found {len(existing_terms)} terms in checkpoint")
            logger.info(f"Will process {len(terms_to_process)} remaining terms")
            
    # If no checkpoint or specified continue file, use continue-from argument
    if not existing_results and continue_from:
        if os.path.exists(continue_from):
            logger.info(f"Total original terms: {len(all_terms)}")
            logger.info(f"Loading existing results from {continue_from}")
            existing_results = load_results(continue_from)
            
            # Filter out terms that are already in the existing results
            existing_terms = set(existing_results.keys())
            terms_to_process = [term for term in all_terms if term not in existing_terms]
            
            logger.info(f"Found {len(existing_terms)} existing terms")
            logger.info(f"Will process {len(terms_to_process)} new terms")
        else:
            warning_msg = f"Specified continue file {continue_from} does not exist, will create a new file"
            if not minimal_output:
                print(f"WARNING: {warning_msg}")
            logger.warning(warning_msg)
    
    return existing_results, terms_to_process

async def process_term_batch(
    batch_terms: List[str], 
    settings: SearchSettings, 
    use_tavily: bool,
    batch_index: int,
    save_interval: int,
    continue_from: Optional[str],
    minimal_output: bool,
    skip_verification_step: bool,
    process_batch_size: int
) -> Dict[str, List[WebContent]]:
    """Process a batch of terms using either Tavily or RapidAPI."""
    batch_start_time = time.time()
    batch_results = {}
    
    try:
        # Process this batch of terms with the selected provider
        logger.info("Starting search for batch content...")
        
        if use_tavily:
            print(f"Using Tavily search provider for batch {batch_index + 1}")
            logger.info(f"Using Tavily search provider for batch {batch_index + 1}")
            
            try:
                # Try to import needed modules for Tavily
                try:
                    print("Attempting to import get_llm function...")
                    logger.info("Attempting to import get_llm function...")
                    from generate_glossary.utils.llm import get_llm
                    print(f"Using LLM provider: {settings.provider}")
                    logger.info(f"Using LLM provider: {settings.provider}")
                    llm = get_llm(settings.provider)
                    print(f"Successfully initialized LLM: {llm.__class__.__name__}")
                    logger.info(f"Successfully initialized LLM: {llm.__class__.__name__}")
                except (ImportError, AttributeError) as import_err:
                    # Handle the case where get_llm doesn't exist
                    error_msg = f"Could not import get_llm function: {import_err}"
                    logger.error(error_msg)
                    if not minimal_output:
                        print(f"ERROR: {error_msg}")
                    print("Falling back to RapidAPI search provider instead of Tavily")
                    logger.info("Falling back to RapidAPI search provider")
                    use_tavily = False
                
                if use_tavily:
                    # Use Tavily for search
                    print(f"Calling tavily_search_web_content with {len(batch_terms)} terms")
                    logger.info(f"Calling tavily_search_web_content with {len(batch_terms)} terms")
                    batch_content_by_term = await tavily_search_web_content(
                        batch_terms, 
                        settings
                    )
                    
                    # Process content with Tavily processor
                    print(f"Calling tavily_process_all_content for batch {batch_index + 1}")
                    logger.info(f"Calling tavily_process_all_content for batch {batch_index + 1}")
                    batch_results = await tavily_process_all_content(
                        batch_content_by_term,
                        settings,
                        llm
                    )
            except Exception as e:
                logger.error(f"Error using Tavily provider: {e}")
                if not minimal_output:
                    print(f"ERROR using Tavily: {e}")
                print("Falling back to RapidAPI search provider")
                logger.info("Falling back to RapidAPI search provider due to error")
                use_tavily = False
        
        # If Tavily failed or was not requested, use RapidAPI
        if not use_tavily:
            # Use RapidAPI for search
            batch_content_by_term = await search_web_content(
                batch_terms, 
                settings,
                continue_from=continue_from if batch_index == 0 else None,  # Only use continue_from for first batch
                save_interval=save_interval  # Use checkpoint interval as save interval
            )
            
            # Skip if no content was found
            if not batch_content_by_term or not any(len(c) > 0 for c in batch_content_by_term.values()):
                logger.warning("No content found for this batch, skipping processing step")
                # Calculate and log batch processing time
                batch_end_time = time.time()
                batch_processing_time = batch_end_time - batch_start_time
                if not minimal_output:
                    print(f"✓ Batch {batch_index + 1}: No content found for any terms.")
                logger.info(f"Batch processing time (no content): {batch_processing_time:.2f} seconds")
                return {}
            
            # Log content stats
            content_count = sum(len(contents) for contents in batch_content_by_term.values())
            logger.info(f"Search complete. Found {content_count} content items across {len(batch_content_by_term)} terms")
            
            # Process all content asynchronously for this batch
            logger.info("Starting content verification and processing...")
            batch_results = await process_all_content_for_terms(
                batch_content_by_term, 
                settings.min_score,
                process_batch_size,
                skip_verification_step
            )
        
        # Log verification stats
        if batch_results:
            verified_count = sum(
                len([c for c in contents if c.get("is_verified", False)])
                for contents in batch_results.values()
            )
            content_count = sum(len(contents) for contents in batch_results.values())
            logger.info(f"Processing complete. {verified_count} verified items out of {content_count} total")
        else:
            logger.warning("No results after processing")
        
        # Calculate and log batch processing time
        batch_end_time = time.time()
        batch_processing_time = batch_end_time - batch_start_time
        time_per_term = batch_processing_time/len(batch_terms) if batch_terms else 0
        batch_time_msg = f"Completed in {batch_processing_time:.2f} seconds ({time_per_term:.2f} sec/term)"
        if not minimal_output:
            print(f"✓ Batch {batch_index + 1} completed: Processed {len(batch_terms)} terms. {batch_time_msg}")
        logger.info(f"Batch processing time: {batch_processing_time:.2f} seconds")
        logger.info(f"Average time per term: {time_per_term:.2f} seconds")
        
        return batch_results
        
    except Exception as e:
        # Calculate processing time even for failed batches
        batch_end_time = time.time()
        batch_processing_time = batch_end_time - batch_start_time
        logger.error(f"Error processing batch {batch_index + 1}: {e}")
        logger.error(f"Failed batch processing time: {batch_processing_time:.2f} seconds")
        return {}

async def run_mining_pipeline(args) -> Dict[str, List[WebContent]]:
    """Run the web content mining pipeline based on command-line arguments."""
    # Configure logging
    if args.quiet or args.minimal_output:
        log_level = "CRITICAL"
    elif args.verbose_logging:
        # Force INFO level or lower when verbose logging is requested
        log_level = "INFO" if args.log_level in ["ERROR", "WARNING", "CRITICAL"] else args.log_level
    else:
        log_level = args.log_level
    
    global logger
    logger = configure_logging(log_level)
    
    # Configure file logging if requested
    if args.log_file:
        setup_file_logging(args.log_file)
    
    # Log start of execution with clear banner
    if not args.minimal_output:
        startup_banner = f"\n{'='*50}\nWEB CONTENT MINING STARTED\n{'='*50}"
        logger.info(startup_banner)
        if args.verbose_logging:
            print(f"✓ Verbose logging enabled (log level: {log_level})")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only if there's a directory part (not just a filename)
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Ensuring output directory exists: {output_dir}")
            if not os.path.isdir(output_dir):
                error_msg = f"Failed to create output directory: {output_dir}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                return {}
        except Exception as e:
            error_msg = f"Error creating output directory {output_dir}: {e}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            return {}
    
    # Validate checkpoint directory early if provided
    if args.checkpoint_dir:
        if not args.minimal_output:
            print(f"\n=== CHECKPOINT: Initializing checkpointing to directory: {args.checkpoint_dir} ===")
            print(f"Checkpoint interval: Every {args.checkpoint_interval} batches")
        # Create the directory if it doesn't exist
        try:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if not args.minimal_output:
                print(f"✓ Checkpoint directory ready: {os.path.abspath(args.checkpoint_dir)}")
        except Exception as e:
            print(f"ERROR: Failed to create checkpoint directory: {e}")
            logger.error(f"Failed to create checkpoint directory: {e}")
    
    try:
        # Read terms - do this early to fail fast if file doesn't exist
        try:
            all_terms = read_terms(args.input)
            if not all_terms:
                print(f"ERROR: No terms found in input file: {args.input}")
                logger.error(f"No terms found in input file: {args.input}")
                return {}
        except FileNotFoundError:
            print(f"ERROR: Input file not found: {args.input}")
            logger.error(f"Input file not found: {args.input}")
            return {}
            
        # Apply performance settings
        args = apply_performance_settings(args)
    
        # Set thread limits for OpenMP and other libraries
        set_thread_limits(args.max_threads)
        
        # If continuing from existing results, load them and filter terms
        existing_results, terms_to_process = load_existing_results(
            args.checkpoint_dir,
            args.continue_from,
            all_terms,
            args.minimal_output
        )
        
        # Skip processing if there are no new terms
        if not terms_to_process:
            logger.info("No new terms to process, exiting")
            if not args.minimal_output:
                print("✓ No new terms to process, exiting")
            return existing_results
        
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
            safe_mode=args.safe_mode and not (args.performance_mode or args.high_performance),  # Safe mode only if not performance mode
            parallel_extraction=not args.no_parallel_extraction,  # Use parallel extraction by default
            skip_low_quality=not args.no_skip_low_quality,  # Skip low quality content by default
            content_threshold=args.content_threshold,  # Content quality threshold
            # Pass context CSV arguments
            context_csv_file=args.context_csv_file,
            concept_column=args.concept_column,
            context_column=args.context_column,
            skip_summarization=args.skip_summarization,  # Add the skip_summarization flag
        )
        
        # Set environment variable for multiprocessing
        os.environ["PYTHONUNBUFFERED"] = "1"  # Prevent output buffering
        
        # Run web mining with the selected provider
        if not args.minimal_output:
            print(f"\n=== Starting web content mining for {len(terms_to_process)} terms using {args.search_provider} ===")
        logger.info(f"Mining web content for {len(terms_to_process)} terms using {args.search_provider}...")
        logger.info(f"Using up to {args.max_concurrent} concurrent requests and {args.max_workers} worker processes")
        logger.info(f"Thread limits: OpenMP={os.environ['OMP_NUM_THREADS']}, Process batch size={args.process_batch_size}")
        
        # Show performance settings
        if settings.parallel_extraction and not args.minimal_output:
            msg = "Using parallel content extraction (faster)"
            print(f"✓ {msg}")
            logger.info(msg)
        if settings.skip_low_quality and not args.minimal_output:
            msg = f"Skipping low-quality content below threshold {settings.content_threshold}"
            print(f"✓ {msg}")
            logger.info(msg)
        if args.high_performance and not args.minimal_output:
            msg = "HIGH PERFORMANCE MODE enabled (maximum speed)"
            print(f"🚀 {msg}")
            logger.info(msg)
        elif args.performance_mode and not args.minimal_output:
            msg = "Performance mode enabled (optimized for speed)"
            print(f"✓ {msg}")
            logger.info(msg)
        
        # Check if we should use Tavily
        use_tavily = args.search_provider == "tavily"
        if use_tavily:
            # Ensure TAVILY_API_KEY is set
            tavily_api_key = os.environ.get("TAVILY_API_KEY")
            print(f"Checking for TAVILY_API_KEY environment variable...")
            logger.info(f"Checking for TAVILY_API_KEY environment variable...")
            
            if not tavily_api_key:
                error_msg = "TAVILY_API_KEY environment variable not set. Please set it before using Tavily as a search provider."
                print(f"ERROR: {error_msg}")
                logger.error(error_msg)
                print("To set the key, run: export TAVILY_API_KEY=your_key_here")
                return existing_results
            else:
                # Don't print the actual key, just a masked version
                masked_key = tavily_api_key[:4] + "..." + tavily_api_key[-4:] if len(tavily_api_key) > 8 else "***"
                print(f"✓ Found TAVILY_API_KEY (starts with {masked_key[:6]})")
                logger.info(f"Found TAVILY_API_KEY (masked: {masked_key})")
                
                # Check if tavily_miner module is available
                try:
                    print("Checking if tavily_miner module is available...")
                    logger.info("Checking if tavily_miner module is available...")
                    import importlib
                    tavily_module = importlib.import_module("generate_glossary.utils.tavily_miner")
                    print(f"✓ Tavily module found: {tavily_module.__name__}")
                    logger.info(f"Tavily module found: {tavily_module.__name__}")
                except ImportError as e:
                    error_msg = f"Could not import tavily_miner module: {e}"
                    print(f"ERROR: {error_msg}")
                    logger.error(error_msg)
                    print("Falling back to RapidAPI search provider")
                    use_tavily = False
        
        # Process terms in smaller batches to allow checkpointing
        batch_size = min(50, args.batch_size)  # Increased from 40 to match BATCH_SIZE constant
        new_results = {}
        
        # Skip intermediate verification step if possible
        skip_verification_step = args.skip_verification
        
        # Track overall stats
        overall_start_time = time.time()
        total_processed_terms = 0
        
        for batch_index, i in enumerate(range(0, len(terms_to_process), batch_size)):
            batch_terms = terms_to_process[i:i + batch_size]
            if not batch_terms:  # Skip empty batches
                continue
                
            batch_msg = f"Processing batch {batch_index + 1} of {(len(terms_to_process) + batch_size - 1) // batch_size}"
            batch_banner = f"\n{'='*25} BATCH {batch_index + 1} {'='*25}\n{batch_msg} ({len(batch_terms)} terms)"
            if not args.minimal_output:
                print(f"\n=== BATCH {batch_index + 1}: {batch_msg} ({len(batch_terms)} terms) ===")
            logger.info(batch_banner)
            
            if args.verbose_logging:
                logger.info(f"Terms in this batch: {', '.join(batch_terms[:5])}{'...' if len(batch_terms) > 5 else ''}")
            
            # Process this batch of terms
            batch_results = await process_term_batch(
                batch_terms=batch_terms,
                settings=settings,
                use_tavily=use_tavily,
                batch_index=batch_index,
                save_interval=args.checkpoint_interval,
                continue_from=args.continue_from if batch_index == 0 else None,
                minimal_output=args.minimal_output,
                skip_verification_step=skip_verification_step,
                process_batch_size=args.process_batch_size
            )
            
            # Add batch results to new results
            new_results.update(batch_results)
            
            # Save intermediate results to a temporary file after each batch
            try:
                temp_output_json = f"{args.output}_TEMP.json"
                current_merged_results = {**existing_results, **new_results}
                if current_merged_results: # Only save if there's something to save
                    logger.info(f"Saving intermediate results ({len(current_merged_results)} terms) to temporary file: {temp_output_json}")
                    # Use save_results from web_miner to overwrite the temp file
                    save_results(current_merged_results, temp_output_json, merge_with_existing=False)
                    if not args.minimal_output:
                        print(f"✓ Intermediate results saved to {temp_output_json}")
            except Exception as save_err:
                logger.error(f"Error saving intermediate results to {temp_output_json}: {save_err}")
                if not args.minimal_output:
                    print(f"WARNING: Failed to save intermediate results: {save_err}")
            
            # If checkpoint directory is specified, save checkpoint after the specified interval
            if args.checkpoint_dir and (batch_index + 1) % args.checkpoint_interval == 0:
                # Combine existing and new results
                logger.info(f"Saving checkpoint after batch {batch_index + 1}...")
                checkpoint_results = {**existing_results, **new_results}
                checkpoint_file = save_checkpoint(checkpoint_results, args.checkpoint_dir, batch_index + 1)
                logger.info(f"Checkpoint saved: {checkpoint_file}")
            
            # Update total processed terms
            total_processed_terms += len(batch_terms)
            
            # Calculate and log overall progress and ETA
            elapsed_time = time.time() - overall_start_time
            avg_time_per_term = elapsed_time / total_processed_terms
            remaining_terms = len(terms_to_process) - total_processed_terms
            estimated_time_remaining = remaining_terms * avg_time_per_term
            
            # Format as hours:minutes:seconds
            eta_hours = int(estimated_time_remaining // 3600)
            eta_minutes = int((estimated_time_remaining % 3600) // 60)
            eta_seconds = int(estimated_time_remaining % 60)
            
            progress_pct = (total_processed_terms / len(terms_to_process)) * 100
            eta_msg = f"Progress: {progress_pct:.1f}% ({total_processed_terms}/{len(terms_to_process)} terms)"
            eta_msg += f" | ETA: {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}"
            if not args.minimal_output:
                print(f"→ {eta_msg}")
            logger.info(eta_msg)
        
        # After all batches, if there are results to save
        if new_results:
            # Save final checkpoint if checkpoint dir is specified
            if args.checkpoint_dir:
                if not args.minimal_output:
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
            if not args.minimal_output:
                print("\n=== Saving final results ===")
            write_results(merged_results, output_base)
            
            # Print summary of merged results
            total_terms = len(merged_results)
            total_content = sum(len(contents) for contents in merged_results.values())
            verified_content = sum(
                len([c for c in contents if c.get("is_verified", False)])
                for contents in merged_results.values()
            )
            
            # Calculate overall stats
            total_time = time.time() - overall_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            summary = [
                f"\n{'='*50}",
                f"MINING COMPLETED - Total time: {time_formatted}",
                f"{'='*50}",
                f"✓ Total terms: {total_terms} ({len(existing_results)} existing + {len(new_results)} new)",
                f"✓ Total content: {total_content}",
                f"✓ Verified content: {verified_content}",
                f"✓ Average time per term: {total_time/len(terms_to_process):.2f} seconds",
                f"✓ Results saved to:",
                f"  - {output_json} (full results)",
                f"  - {output_base}.txt (term-URL mappings)",
                f"  - {output_base}_summary.json (statistics)"
            ]
            
            # Clean up temporary file upon successful completion
            temp_output_json_final = f"{output_base}_TEMP.json"
            try:
                if os.path.exists(temp_output_json_final):
                    os.remove(temp_output_json_final)
                    logger.info(f"Successfully removed temporary file: {temp_output_json_final}")
            except Exception as remove_err:
                logger.warning(f"Could not remove temporary file {temp_output_json_final}: {remove_err}")
            
            # Print summary to both stdout and log
            for line in summary:
                if not args.minimal_output:
                    print(line)
                logger.info(line)
            
            return merged_results
        
        return existing_results
            
    except Exception as e:
        logger.error(f"Error in mining pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return existing_results 