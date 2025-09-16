#!/usr/bin/env python3
"""
Modern CLI for web content mining using Firecrawl v2.2.0.

This module provides the `uv run mine-web` command that integrates with the
simplified mining API to extract academic concepts from web sources.

New v2.2.0 Features:
- PDF parsing with page limits for better performance control
- Queue status monitoring for intelligent throttling
- 15x faster Map endpoint for URL discovery
- Enhanced webhook support with signatures
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
    parser = argparse.ArgumentParser(
        description="Mine web content for academic concepts using Firecrawl v2.2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run mine-web terms.txt --output results.json
  uv run mine-web concepts.txt -o output.json --hybrid-mode
  uv run mine-web terms.txt -o results.json --max-age 86400000
  uv run mine-web terms.txt -o results.json --max-pages 5 --queue-status
  uv run mine-web terms.txt -o results.json --use-map-endpoint --webhook-url https://example.com/webhook
  uv run mine-web terms.txt -o results.json --no-batch --no-summary --actions '[{"type": "click", "selector": ".load-more"}]'

Firecrawl v2.2.0 Features:
  • Batch scraping for 500% performance improvement
  • Smart crawling with natural language prompts
  • Enhanced caching with configurable maxAge
  • Summary format for optimized content extraction
  • Actions support for dynamic content interaction
  • Queue status monitoring for intelligent throttling
  • PDF parsing with page limits for better performance
  • 15x faster Map endpoint for URL discovery
  • Enhanced webhook support with signatures
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

    # Firecrawl v2.2.0 feature arguments
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum pages to process for PDF parsing (improves performance)"
    )

    parser.add_argument(
        "--queue-status",
        action="store_true",
        help="Enable queue status monitoring for intelligent throttling"
    )

    parser.add_argument(
        "--use-map-endpoint",
        dest="use_fast_map",
        action="store_true",
        help="Enable the 15x faster Map endpoint for URL discovery"
    )

    parser.add_argument(
        "--no-map-endpoint",
        action="store_false",
        dest="use_fast_map",
        default=True,
        help="Disable the 15x faster Map endpoint for URL discovery"
    )

    parser.add_argument(
        "--webhook-url",
        type=str,
        help="Webhook URL for receiving processing notifications"
    )

    parser.add_argument(
        "--webhook-events",
        type=str,
        help="Comma-separated webhook events to subscribe to (default: started,page,completed,failed)"
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
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("\n⚠️  FIRECRAWL_API_KEY not set in environment")
        print("\nTo use this tool:")
        print("1. Get your API key from https://www.firecrawl.dev")
        print("2. Set it: export FIRECRAWL_API_KEY='fc-your-key'")
        print("3. Or add to .env file: FIRECRAWL_API_KEY=fc-your-key")
        print("\nFirecrawl v2.2.0 pricing: Optimized for academic research")
        return False
    return True


def validate_v220_parameters(args) -> None:
    """Validate v2.2.0 specific parameters and combinations."""
    # Validate max_pages is positive
    if args.max_pages is not None and args.max_pages <= 0:
        raise ValueError("--max-pages must be a positive integer")

    # Validate queue monitoring requires API key (warning only since early gate already checked)
    if args.queue_status and not os.getenv("FIRECRAWL_API_KEY"):
        logger.warning("--queue-status requested but FIRECRAWL_API_KEY validation already handled upstream")

    # Validate webhook URL format if provided
    if args.webhook_url:
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
            r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if not url_pattern.match(args.webhook_url):
            raise ValueError("--webhook-url must be a valid HTTP/HTTPS URL")

    logger.debug("v2.2.0 parameter validation passed")


def configure_logging(args) -> None:
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
        
        # Validate v2.2.0 parameters
        validate_v220_parameters(args)

        # Log configuration
        logger.info(f"Mining {len(terms)} concepts with Firecrawl v2.2.0")
        logger.info(f"Batch scraping: {'enabled' if args.use_batch else 'disabled'}")
        logger.info(f"Summary format: {'enabled' if args.use_summary else 'disabled'}")
        logger.info(f"Cache max age: {args.max_age}ms")

        # Log v2.2.0 specific configuration
        if args.max_pages:
            logger.info(f"PDF page limit: {args.max_pages} pages")
        logger.info(f"Queue monitoring: {'enabled' if args.queue_status else 'disabled'}")
        logger.info(f"Fast Map endpoint: {'enabled' if args.use_fast_map else 'disabled'}")
        if args.webhook_url:
            logger.info(f"Webhook URL: {args.webhook_url}")
        
        # Run mining with simple status indication
        if not args.quiet:
            print("Processing concepts...")
        
        # Prepare webhook config if provided
        webhook_config = None
        if args.webhook_url:
            from generate_glossary.mining.webhooks import WebhookConfig
            try:
                # Parse webhook events if provided
                webhook_events = None
                if args.webhook_events:
                    webhook_events = [event.strip() for event in args.webhook_events.split(',') if event.strip()]

                if webhook_events:
                    webhook_config = WebhookConfig(url=args.webhook_url, events=webhook_events)
                else:
                    webhook_config = WebhookConfig(url=args.webhook_url)

            except Exception as webhook_error:
                logger.warning(f"Failed to create webhook config: {webhook_error}. Continuing without webhooks.")
                webhook_config = None

        results = mine_concepts(
            concepts=terms,
            output_path=args.output,
            max_concurrent=args.max_concurrent,
            max_age=args.max_age,
            use_summary=args.use_summary,
            use_batch_scrape=args.use_batch,
            actions=actions,
            use_hybrid=args.hybrid_mode,
            max_pages=args.max_pages,
            enable_queue_monitoring=args.queue_status,
            use_fast_map=args.use_fast_map,
            webhook_config=webhook_config
        )
        
        # Report results
        if "error" in results:
            logger.error(f"Mining failed: {results['error']}")
            return 1
        
        stats = results.get("statistics", {})
        if not args.quiet:
            print(f"\n✅ Successfully processed {stats.get('successful', 0)}/{stats.get('total_concepts', 0)} concepts")
            print(f"📊 Total resources found: {stats.get('total_resources', 0)}")

            # Show v2.2.0 feature usage - check both results and statistics
            v220_features = results.get('v2_2_0_features_used')
            if v220_features and isinstance(v220_features, dict):
                features_str = ', '.join(f"{k}={v}" for k, v in v220_features.items())
                print(f"⚡ Firecrawl v2.2.0 features used: {features_str}")
            else:
                # Fallback to statistics if available
                features = stats.get('features_used', {})
                if isinstance(features, dict) and features:
                    features_str = ', '.join(f"{k}={v}" for k, v in features.items())
                    print(f"⚡ Features used: {features_str}")
                elif isinstance(features, (list, tuple)) and features:
                    features_str = ', '.join(str(f) for f in features)
                    print(f"⚡ Features used: {features_str}")

            # Show v2.2.0 specific metrics - with proper guards
            v220_metrics = stats.get('v220_metrics', {})
            if isinstance(v220_metrics, dict) and v220_metrics:
                if v220_metrics.get('pdf_pages_processed'):
                    print(f"📄 PDF pages processed: {v220_metrics['pdf_pages_processed']}")
                if v220_metrics.get('queue_status_checks'):
                    print(f"🔄 Queue status checks: {v220_metrics['queue_status_checks']}")
                if v220_metrics.get('map_endpoint_used'):
                    print(f"🚀 Fast Map endpoint: 15x performance boost enabled")
                if v220_metrics.get('webhook_events'):
                    print(f"🔗 Webhook events sent: {v220_metrics['webhook_events']}")

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