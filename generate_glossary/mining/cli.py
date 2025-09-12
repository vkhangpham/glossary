#!/usr/bin/env python3
"""
Modern CLI for web content mining using Firecrawl v2.0.

This module provides the `uv run mine-web` command that integrates with the
simplified mining API to extract academic concepts from web sources.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Any

from generate_glossary.mining.mining import mine_concepts
from generate_glossary.utils.logger import get_logger

logger = get_logger("mining.cli")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup comprehensive argument parser for the mining CLI."""
    parser = argparse.ArgumentParser(
        description="Mine web content for academic concepts using Firecrawl v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run mine-web terms.txt --output results.json
  uv run mine-web concepts.txt -o output.json --hybrid-mode
  uv run mine-web terms.txt -o results.json --max-age 86400000
  uv run mine-web terms.txt -o results.json --no-batch --no-summary --actions '[{"type": "click", "selector": ".load-more"}]'

Firecrawl v2.0 Features:
  • Batch scraping for 500% performance improvement
  • Smart crawling with natural language prompts
  • Enhanced caching with configurable maxAge
  • Summary format for optimized content extraction
  • Actions support for dynamic content interaction
        """
    )
    
    # Positional argument
    parser.add_argument(
        "terms_file",
        help="Path to file containing terms to mine (one per line)"
    )
    
    # Required arguments
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for results JSON file"
    )
    
    # Core processing arguments
    parser.add_argument(
        "-c", "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent operations (default: 5)"
    )
    
    # Firecrawl v2.0 feature arguments
    parser.add_argument(
        "--max-age",
        type=int,
        default=172800000,  # 2 days in milliseconds
        help="Cache duration in milliseconds (default: 172800000 = 2 days)"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_false",
        dest="use_summary",
        default=True,
        help="Disable summary format (batch scraping enabled by default)"
    )
    
    parser.add_argument(
        "--no-batch",
        action="store_false",
        dest="use_batch",
        default=True,
        help="Disable batch scraping (enabled by default for 500%% performance improvement)"
    )
    
    parser.add_argument(
        "--actions",
        type=str,
        help="JSON string for dynamic content actions (e.g., '[{\"type\": \"click\", \"selector\": \".load-more\"}]')"
    )
    
    parser.add_argument(
        "--hybrid-mode",
        action="store_true",
        help="Use both batch + smart extraction for best results"
    )
    
    # Logging and output arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output (sets log level to ERROR)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Detailed output (sets log level to DEBUG)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        default=True,
        help="Hide progress bars (enabled by default)"
    )
    
    return parser


def load_terms_from_file(file_path: str) -> List[str]:
    """Load and validate terms from input file."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Terms file not found: {file_path}")
        
        with path.open('r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]
        
        if not terms:
            raise ValueError(f"No valid terms found in {file_path}")
        
        logger.info(f"Loaded {len(terms)} terms from {file_path}")
        return terms
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Terms file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading terms file {file_path}: {e}")


def validate_firecrawl_api_key() -> bool:
    """Validate Firecrawl API key is available."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("\n⚠️  FIRECRAWL_API_KEY not set in environment")
        print("\nTo use this tool:")
        print("1. Get your API key from https://www.firecrawl.dev")
        print("2. Set it: export FIRECRAWL_API_KEY='fc-your-key'")
        print("3. Or add to .env file: FIRECRAWL_API_KEY=fc-your-key")
        print("\nFirecrawl v2.0 pricing: Optimized for academic research")
        return False
    return True


def configure_logging(args) -> None:
    """Configure logging based on CLI arguments."""
    if args.quiet:
        level_name = "ERROR"
    elif args.verbose:
        level_name = "DEBUG"
    else:
        level_name = args.log_level
    
    # Map level name to logging level
    level = getattr(logging, level_name, logging.INFO)
    
    # Set logger level and handler levels
    logger.setLevel(level)
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)
    
    # Set environment variable for downstream modules
    os.environ['LOGLEVEL'] = level_name
    
    logger.info(f"Log level set to {level_name}")


def main() -> int:
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args)
    
    # Validate API key
    if not validate_firecrawl_api_key():
        return 1
    
    try:
        # Load terms
        terms = load_terms_from_file(args.terms_file)
        
        
        # Parse actions if provided
        actions = None
        if args.actions:
            try:
                actions = json.loads(args.actions)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in --actions: {e}")
                return 1
        
        # Log configuration
        logger.info(f"Mining {len(terms)} concepts with Firecrawl v2.0")
        logger.info(f"Batch scraping: {'enabled' if args.use_batch else 'disabled'}")
        logger.info(f"Summary format: {'enabled' if args.use_summary else 'disabled'}")
        logger.info(f"Cache max age: {args.max_age}ms")
        
        # Run mining with simple status indication
        if not args.quiet:
            print("Processing concepts...")
        
        results = mine_concepts(
            concepts=terms,
            output_path=args.output,
            max_concurrent=args.max_concurrent,
            max_age=args.max_age,
            use_summary=args.use_summary,
            use_batch_scrape=args.use_batch,
            actions=actions,
            use_hybrid=args.hybrid_mode
        )
        
        # Report results
        if "error" in results:
            logger.error(f"Mining failed: {results['error']}")
            return 1
        
        stats = results.get("statistics", {})
        if not args.quiet:
            print(f"\n✅ Successfully processed {stats.get('successful', 0)}/{stats.get('total_concepts', 0)} concepts")
            print(f"📊 Total resources found: {stats.get('total_resources', 0)}")
            features = stats.get('features_used', {})
            features_str = ', '.join(f"{k}={v}" for k, v in features.items()) if isinstance(features, dict) else ', '.join(features)
            print(f"⚡ Firecrawl v2.0 features used: {features_str}")
            print(f"💾 Results saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())