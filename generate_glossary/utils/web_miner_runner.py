"""
Web content mining runner using Firecrawl SDK.
Simplified implementation that only uses the modern Firecrawl approach.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from generate_glossary.utils.firecrawl_web_miner import (
    mine_concepts_with_firecrawl,
    initialize_firecrawl,
    ConceptDefinition,
    WebResource
)

# Setup logger
logger = logging.getLogger(__name__)

def configure_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging with the specified level."""
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    numeric_level = levels.get(log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Quiet down noisy loggers
    for module in ["aiohttp", "urllib3", "asyncio", "firecrawl"]:
        logging.getLogger(module).setLevel(logging.ERROR)
    
    return logging.getLogger(__name__)

def read_terms(filepath: str) -> List[str]:
    """Read terms from file, one term per line."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load checkpoint data from file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoint(checkpoint_file: str, data: Dict[str, Any]):
    """Save checkpoint data to file."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def run_web_mining(
    input_file: str,
    output_path: str,
    batch_size: int = 25,
    checkpoint_dir: Optional[str] = None,
    resume: bool = False,
    log_level: str = "INFO",
    **kwargs
) -> Dict[str, Any]:
    """
    Run web content mining using Firecrawl SDK.
    
    Args:
        input_file: Path to file containing terms (one per line)
        output_path: Path to save results
        batch_size: Number of concepts to process in each batch
        checkpoint_dir: Directory for checkpoint files
        resume: Whether to resume from checkpoint
        log_level: Logging level
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Dictionary with results and statistics
    """
    # Configure logging
    logger = configure_logging(log_level)
    
    # Check Firecrawl API key
    if not os.getenv("FIRECRAWL_API_KEY"):
        logger.error("FIRECRAWL_API_KEY not set in environment")
        logger.error("Get your API key from https://www.firecrawl.dev")
        logger.error("Set it with: export FIRECRAWL_API_KEY='fc-your-key'")
        return {
            "error": "Firecrawl API key not configured",
            "results": {},
            "statistics": {"total": 0, "successful": 0, "failed": 0}
        }
    
    # Initialize Firecrawl
    app = initialize_firecrawl()
    if not app:
        logger.error("Failed to initialize Firecrawl client")
        return {
            "error": "Firecrawl initialization failed",
            "results": {},
            "statistics": {"total": 0, "successful": 0, "failed": 0}
        }
    
    # Read terms
    logger.info(f"Reading terms from {input_file}")
    terms = read_terms(input_file)
    logger.info(f"Found {len(terms)} terms to process")
    
    # Setup checkpoint if needed
    checkpoint_file = None
    processed_terms = set()
    results = {}
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{int(time.time())}.json")
        
        if resume:
            # Find latest checkpoint
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")])
            if checkpoints:
                latest = os.path.join(checkpoint_dir, checkpoints[-1])
                logger.info(f"Resuming from checkpoint: {latest}")
                checkpoint_data = load_checkpoint(latest)
                processed_terms = set(checkpoint_data.get("processed", []))
                results = checkpoint_data.get("results", {})
                logger.info(f"Resumed with {len(processed_terms)} already processed terms")
    
    # Filter out already processed terms
    remaining_terms = [t for t in terms if t not in processed_terms]
    if not remaining_terms:
        logger.info("All terms already processed")
        return {
            "results": results,
            "statistics": {
                "total": len(terms),
                "successful": len([r for r in results.values() if r.get("summary")]),
                "failed": len([r for r in results.values() if "error" in r])
            }
        }
    
    logger.info(f"Processing {len(remaining_terms)} remaining terms")
    
    # Process terms
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(remaining_terms), batch_size):
        batch = remaining_terms[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(remaining_terms) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} terms)")
        
        # Mine concepts
        batch_results = mine_concepts_with_firecrawl(batch)
        
        # Update results
        if "results" in batch_results:
            results.update(batch_results["results"])
            processed_terms.update(batch)
        
        # Save checkpoint
        if checkpoint_file:
            checkpoint_data = {
                "processed": list(processed_terms),
                "results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            logger.debug(f"Checkpoint saved: {len(processed_terms)} terms processed")
        
        # Progress update
        total_processed = len(processed_terms)
        success_count = sum(1 for r in results.values() if r.get("summary"))
        logger.info(f"Progress: {total_processed}/{len(terms)} terms, {success_count} successful")
    
    # Calculate final statistics
    elapsed = time.time() - start_time
    stats = {
        "total": len(terms),
        "processed": len(processed_terms),
        "successful": sum(1 for r in results.values() if r.get("summary")),
        "failed": sum(1 for r in results.values() if "error" in r),
        "total_resources": sum(len(r.get("resources", [])) for r in results.values()),
        "processing_time": elapsed,
        "avg_time_per_concept": elapsed / len(remaining_terms) if remaining_terms else 0
    }
    
    # Prepare output
    output_data = {
        "results": results,
        "statistics": stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    logger.info("="*60)
    logger.info("Web Mining Complete")
    logger.info("="*60)
    logger.info(f"Total terms: {stats['total']}")
    logger.info(f"Successful: {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total resources: {stats['total_resources']}")
    logger.info(f"Processing time: {stats['processing_time']:.1f}s")
    logger.info(f"Average time per concept: {stats['avg_time_per_concept']:.2f}s")
    
    return output_data

# Compatibility wrapper for CLI
def main(args):
    """Main entry point for CLI."""
    return run_web_mining(
        input_file=args.input,
        output_path=args.output,
        batch_size=getattr(args, 'batch_size', 25),
        checkpoint_dir=getattr(args, 'checkpoint_dir', None),
        resume=getattr(args, 'resume', False),
        log_level=getattr(args, 'log_level', 'INFO')
    )