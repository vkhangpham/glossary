"""
Command-line interface for web content mining using Firecrawl SDK.
Simplified CLI that only supports the modern Firecrawl approach.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

# Import runner functionality
from generate_glossary.utils.web_miner_runner import run_web_mining

# Load environment variables
load_dotenv('.env')

def setup_argument_parser():
    """Create and configure the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Mine web content for academic concepts using Firecrawl SDK (4x faster than traditional approaches)."
    )
    
    # Input/Output Options
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to file containing terms (one per line)"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for results JSON file"
    )
    
    # Processing Options
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=25,
        help="Number of concepts to process in each batch (default: 25)"
    )
    
    # Checkpoint Options
    parser.add_argument(
        "-c", "--checkpoint-dir",
        help="Directory for saving checkpoints"
    )
    
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume from the latest checkpoint"
    )
    
    # Logging Options
    parser.add_argument(
        "-l", "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output (equivalent to --log-level ERROR)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (equivalent to --log-level DEBUG)"
    )
    
    return parser

def main():
    """Main entry point for the CLI."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Adjust log level based on quiet/verbose flags
    if args.quiet:
        args.log_level = "ERROR"
    elif args.verbose:
        args.log_level = "DEBUG"
    
    # Check for Firecrawl API key
    if not os.getenv("FIRECRAWL_API_KEY"):
        print("\n⚠️  FIRECRAWL_API_KEY not set in environment")
        print("\nTo use this tool:")
        print("1. Get your API key from https://www.firecrawl.dev")
        print("2. Set it: export FIRECRAWL_API_KEY='fc-your-key'")
        print("3. Or add to .env file: FIRECRAWL_API_KEY=fc-your-key")
        print("\nFirecrawl pricing: $83/month for 10,000 concepts (Standard plan)")
        return 1
    
    try:
        # Run the mining pipeline
        results = run_web_mining(
            input_file=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            log_level=args.log_level
        )
        
        # Check for errors
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return 1
        
        # Success
        stats = results.get("statistics", {})
        if not args.quiet:
            print(f"\n✅ Successfully processed {stats.get('successful', 0)}/{stats.get('total', 0)} concepts")
            print(f"Results saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())