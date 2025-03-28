#!/usr/bin/env python3
"""
Script to fix verification errors in lv2_resources.json file.
Uses the existing verification logic from verification_utils.py,
falling back to CPU if CUDA is not available.
"""

import os
import json
import argparse
import asyncio
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import verification utilities
from generate_glossary.utils.verification_utils import (
    verify_content_async,
    is_wikipedia_url,
    get_domain_trust_score
)
from generate_glossary.utils.web_miner import get_domain_priority

async def fix_verification_errors(input_file, output_file=None, min_score=2.6, update_all=False, dry_run=False):
    """
    Fix verification errors by applying the same verification logic
    from verification_utils.py.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the fixed JSON file (defaults to overwriting input)
        min_score: Minimum score threshold for verification
        update_all: Whether to update all entries or just those with errors
        dry_run: If True, just print stats without modifying the file
    """
    if output_file is None:
        output_file = input_file
    
    logger.info(f"Loading data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count for statistics
    total_terms = len(data)
    total_entries = sum(len(entries) for entries in data.values())
    error_entries = 0
    updated_entries = 0
    error_msg = "CUDA out of memory"
    
    # Process each term and its entries
    for term, entries in data.items():
        for _, entry in enumerate(entries):
            # Check if entry has error or if we should update all
            has_error = (not entry.get("is_verified", False) and 
                         error_msg in entry.get("verification_reason", ""))
            
            if has_error:
                error_entries += 1
            
            if has_error or update_all:
                url = entry.get("url", "")
                processed_content = entry.get("processed_content", "")
                
                # Get initial domain priority score for final score calculation
                initial_score = get_domain_priority(url)
                
                try:
                    # Use the actual verification function from verification_utils.py
                    # This will automatically use CPU if CUDA is not available
                    logger.info(f"Verifying content for '{term}' from {url}")
                    is_verified, reason, edu_score = await verify_content_async(
                        url=url,
                        content=processed_content,
                        min_score=min_score
                    )
                    
                    # Final score calculation (matching web_miner.py)
                    final_score = (initial_score * 0.4) + (edu_score / 5.0 * 0.6)
                    
                    # Update entry with new values
                    entry["score"] = float(final_score)
                    entry["is_verified"] = bool(is_verified)
                    entry["verification_reason"] = reason
                    
                    updated_entries += 1
                    logger.info(f"Updated verification for '{term}' entry from {url}")
                    
                except Exception as e:
                    logger.error(f"Error verifying content for '{term}' from {url}: {e}")
                    # If there was an error, only update if it had an error before
                    if has_error:
                        # Use domain-based scoring as fallback
                        if is_wikipedia_url(url):
                            boost = 1.0
                            base_score = 3.5  # Reasonable baseline for Wikipedia
                            boost_reason = "Wikipedia content"
                        else:
                            try:
                                domain_trust = get_domain_trust_score(url)
                                if domain_trust > 0.5:
                                    boost = (domain_trust - 0.5) * 2
                                    domain = urlparse(url).netloc
                                    boost_reason = f"trusted domain ({domain})"
                                else:
                                    boost = 0.0
                                    boost_reason = ""
                            except Exception:
                                boost = 0.0
                                boost_reason = ""
                                
                            # Base score for fallback cases
                            base_score = 2.5
                        
                        # Apply the boost
                        boosted_score = min(5.0, base_score + boost)
                        
                        # Determine verification result
                        is_verified = boosted_score >= min_score
                        
                        # Final score calculation
                        final_score = (initial_score * 0.4) + (base_score / 5.0 * 0.6)
                        
                        # Create verification reason
                        if boost > 0:
                            reason = f"Fallback verification with educational score {base_score:.2f}/5.0 + {boost:.2f} boost for {boost_reason} = {boosted_score:.2f}/5.0"
                        else:
                            reason = f"Fallback verification with educational score {base_score:.2f}/5.0"
                        
                        # Update entry with new values
                        entry["score"] = float(final_score)
                        entry["is_verified"] = bool(is_verified)
                        entry["verification_reason"] = reason
                        
                        updated_entries += 1
                        logger.info(f"Applied fallback verification for '{term}' from {url}")
    
    logger.info(f"Total terms: {total_terms}")
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Entries with verification errors: {error_entries}")
    logger.info(f"Entries updated: {updated_entries}")
    
    # Save updated data
    if not dry_run and updated_entries > 0:
        logger.info(f"Saving updated data to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Done!")
    elif dry_run:
        logger.info("Dry run complete. No changes were saved.")
    else:
        logger.info("No entries needed updating.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix verification errors in lv2_resources.json")
    parser.add_argument("--input", "-i", default="data/lv2/lv2_resources.json", 
                        help="Path to input JSON file (default: data/lv2/lv2_resources.json)")
    parser.add_argument("--output", "-o", 
                        help="Path to output JSON file (default: same as input)")
    parser.add_argument("--min-score", type=float, default=2.6,
                        help="Minimum score threshold for verification (default: 2.6)")
    parser.add_argument("--update-all", action="store_true",
                        help="Update all entries, not just those with errors")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run mode: only print stats, don't modify the file")
    
    args = parser.parse_args()
    
    asyncio.run(fix_verification_errors(
        input_file=args.input,
        output_file=args.output,
        min_score=args.min_score,
        update_all=args.update_all,
        dry_run=args.dry_run
    )) 