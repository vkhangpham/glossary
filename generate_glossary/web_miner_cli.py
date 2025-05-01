"""
Command-line interface for web content mining.

This module provides a clean CLI wrapper for mining web content for technical concepts,
delegating all implementation details to the web_miner_runner module.
"""

import argparse
import sys
import os
import asyncio
from dotenv import load_dotenv

# Import runner functionality
from generate_glossary.utils.web_miner_runner import (
    run_mining_pipeline,
    SEARCH_PROVIDERS,
    DEFAULT_CONCURRENT,
    PROCESS_BATCH_SIZE,
    CPU_COUNT,
    CHECKPOINT_INTERVAL
)

# Required to fix the "Thread creation failed: Resource temporarily unavailable" error
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenization

# Load environment variables
load_dotenv('.env')

def setup_argument_parser():
    """Create and configure the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Mine web content for technical concepts with support for multiple search providers."
    )
    
    # Create argument groups for better organization
    input_group = parser.add_argument_group('Input/Output Options')
    search_group = parser.add_argument_group('Search Provider Options')
    context_group = parser.add_argument_group('Context CSV Options')
    perf_group = parser.add_argument_group('Performance Options')
    llm_group = parser.add_argument_group('LLM and Content Processing Options')
    checkpoint_group = parser.add_argument_group('Checkpointing Options')
    logging_group = parser.add_argument_group('Logging Options')
    
    # Input arguments
    input_group.add_argument(
        "-i", "--input",
        required=True,
        help="Terms to mine web content for (file path)"
    )
    input_group.add_argument(
        "-o", "--output",
        required=True,
        help="Base path for output files (will create .txt, .json, and _summary.json)"
    )
    input_group.add_argument(
        "--continue-from",
        help="Path to existing JSON results file to continue mining from. Will update this file in place."
    )
    
    # Context CSV options
    context_group.add_argument(
        "--context-csv",
        dest="context_csv_file",
        help="Path to CSV file containing context information for search terms"
    )
    context_group.add_argument(
        "--concept-column",
        default="concept",
        help="Name of the column in the CSV containing concept names (default: 'concept')"
    )
    context_group.add_argument(
        "--context-column",
        help="Name of the column in the CSV containing context information"
    )
    
    # Search provider options
    search_group.add_argument(
        "--search-provider",
        choices=SEARCH_PROVIDERS,
        default="rapidapi",
        help="Search provider to use for content mining (default: rapidapi). 'tavily' requires TAVILY_API_KEY to be set in your environment."
    )
    search_group.add_argument(
        "--use-rapidapi",
        action="store_true",
        default=True, 
        help="Use RapidAPI search (same as --search-provider rapidapi)"
    )
    search_group.add_argument(
        "--min-score",
        type=float,
        default=1.5,
        help="Minimum educational score threshold for verification (default: 1.5)"
    )
    search_group.add_argument(
        "--skip-verification",
        action="store_true",
        default=False,
        help="Skip content verification step to speed up processing"
    )
    search_group.add_argument(
        "--no-skip-low-quality",
        action="store_true",
        default=False,
        help="Process all content even if low quality (slower, but more comprehensive)"
    )
    search_group.add_argument(
        "--content-threshold",
        type=float,
        default=1.3,
        help="Educational score threshold for content to process with LLM (default: 1.3)"
    )
    search_group.add_argument(
        "--skip-summarization",
        action="store_true",
        default=False,
        help="Skip LLM summarization step, keeping processed_content empty. Makes processing faster but results less refined."
    )
    
    # Performance options
    perf_group.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (1-50, default: 50)"
    )
    perf_group.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_CONCURRENT,
        help=f"Maximum concurrent requests (default: {DEFAULT_CONCURRENT}, based on CPU count)"
    )
    perf_group.add_argument(
        "--process-batch-size", 
        type=int,
        default=PROCESS_BATCH_SIZE,
        help=f"Number of contents to process at once (default: {PROCESS_BATCH_SIZE}, based on CPU count)"
    )
    perf_group.add_argument(
        "--max-workers",
        type=int,
        default=CPU_COUNT,
        help=f"Maximum number of worker processes for CPU-bound tasks (default: {CPU_COUNT}, based on CPU count)"
    )
    perf_group.add_argument(
        "--max-threads",
        type=int,
        default=min(4, CPU_COUNT),
        help=f"Maximum threads for OpenMP and other thread-based libraries (default: min(4, {CPU_COUNT}))"
    )
    perf_group.add_argument(
        "--safe-mode",
        action="store_true",
        default=True,
        help="Enable safe mode with conservative thread settings to prevent resource exhaustion (default: enabled)"
    )
    perf_group.add_argument(
        "--no-parallel-extraction",
        action="store_true",
        default=False,
        help="Disable parallel content extraction (not recommended, less efficient)"
    )
    perf_group.add_argument(
        "--performance-mode",
        action="store_true",
        default=False,
        help="Enable performance-optimized mode with faster processing (may reduce quality slightly)"
    )
    perf_group.add_argument(
        "--high-performance",
        action="store_true",
        default=False,
        help="Enable maximum performance mode (uses more resources but much faster)"
    )
    perf_group.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="How often to save results when using --continue-from (default: after every batch)"
    )
    
    # LLM and content processing options
    llm_group.add_argument(
        "--provider",
        help="LLM provider for content verification",
        default="gemini"
    )
    llm_group.add_argument(
        "--system-prompt",
        help="Custom system prompt for the LLM that processes content. If not provided, a default extractive summarization prompt will be used."
    )
    llm_group.add_argument(
        "--cache-results",
        action="store_true",
        default=True,
        help="Cache processing results to avoid duplicate LLM calls (default: enabled)"
    )
    llm_group.add_argument(
        "--no-batch-llm",
        action="store_true",
        default=False,
        help="Disable batch LLM processing (default: batch processing enabled)"
    )
    
    # Checkpointing options
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        help="Directory to save checkpoint files. If specified, checkpoints will be saved periodically."
    )
    checkpoint_group.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_INTERVAL,
        help=f"Number of term batches to process before saving a checkpoint (default: {CHECKPOINT_INTERVAL})"
    )
    
    # Logging options
    logging_group.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bar"
    )
    logging_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="ERROR",
        help="Set the logging level (default: ERROR to suppress most messages)"
    )
    logging_group.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress all output except critical errors (equivalent to --log-level CRITICAL)"
    )
    logging_group.add_argument(
        "--verbose-logging",
        action="store_true",
        default=False,
        help="Enable detailed logging of intermediate steps"
    )
    logging_group.add_argument(
        "--log-file",
        help="Path to save logs to a file in addition to console output"
    )
    logging_group.add_argument(
        "--minimal-output",
        action="store_true",
        default=False,
        help="Minimize console output to reduce overhead (only show essential information)"
    )
    
    return parser

def main():
    """Main CLI entry point."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate context CSV arguments
    if args.context_csv_file and not args.context_column:
        print("ERROR: When using --context-csv, you must also specify --context-column")
        sys.exit(1)
    
    # Handle search provider arguments compatibility
    if args.use_rapidapi and args.search_provider != "rapidapi":
        if not args.minimal_output:
            print("Note: --use-rapidapi is specified but --search-provider is not 'rapidapi'. Using specified search provider.")
        args.use_rapidapi = False
    
    try:
        # Pass all the work to the runner module
        asyncio.run(run_mining_pipeline(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 