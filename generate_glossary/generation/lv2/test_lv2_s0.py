import os
import sys
import asyncio
import json
from pathlib import Path

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generate_glossary.generation.lv2.lv2_s0_get_research_areas import (
    read_level1_terms, process_level1_term, ensure_dirs_exist,
    logger, Config
)

# Test configuration
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
TEST_TERMS_COUNT = 5  # Number of level 1 terms to test

async def test_research_area_extraction():
    """Test research area extraction on a small subset of level 1 terms"""
    try:
        # Create test output directory
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        
        # Ensure required directories exist
        ensure_dirs_exist()
        
        # Read level 1 terms
        level1_terms = read_level1_terms(Config.LV1_INPUT_FILE)
        logger.info(f"Read {len(level1_terms)} level 1 terms")
        
        # Take a small subset for testing
        test_terms = level1_terms[:TEST_TERMS_COUNT]
        logger.info(f"Selected {len(test_terms)} terms for testing: {test_terms}")
        
        # Process each term
        results = []
        for term in test_terms:
            result = await process_level1_term(term)
            results.append(result)
            
            logger.info(f"Processed '{term}': found {len(result['research_areas'])} research areas")
            if result['research_areas']:
                logger.info(f"Sample research areas: {result['research_areas'][:5]}")
        
        # Save test results
        test_results = {
            "test_terms": test_terms,
            "results": results
        }
        
        test_output_file = os.path.join(TEST_OUTPUT_DIR, "research_area_test_results.json")
        with open(test_output_file, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results saved to {test_output_file}")
        
        # Print summary
        total_research_areas = sum(len(result['research_areas']) for result in results)
        logger.info(f"Summary: extracted {total_research_areas} research areas from {len(test_terms)} level 1 terms")
        logger.info(f"Average: {total_research_areas / len(test_terms):.1f} research areas per term")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

def main():
    """Main test function"""
    asyncio.run(test_research_area_extraction())

if __name__ == "__main__":
    main() 