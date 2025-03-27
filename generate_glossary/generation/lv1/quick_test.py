import os
import sys
import json
import asyncio
from pathlib import Path

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.generation.lv1.lv1_s0_get_dept_names import (
    web_search_bulk,
    process_level0_term
)

# Setup logging
logger = setup_logger("lv1.quick_test")

async def main():
    """Run a quick test of the level 0 term extraction with a single term"""
    try:
        # Use a single term for quick testing
        test_term = "business"
        logger.info(f"Testing level 0 term extraction with term: {test_term}")
        
        # Test web search API directly
        logger.info("Testing web search API...")
        query = f"site:.edu college of {test_term} list of departments"
        search_results = web_search_bulk([query])
        
        if search_results and search_results.get("data"):
            result_count = len(search_results["data"][0].get("results", []))
            logger.info(f"Web search returned {result_count} results for '{query}'")
            
            # Save sample results
            os.makedirs("data/lv1/test", exist_ok=True)
            with open("data/lv1/test/search_results_sample.json", "w") as f:
                # Only save a few results to keep file size small
                search_sample = {
                    "query": query,
                    "results": search_results["data"][0].get("results", [])[:5]  # Just first 5 results
                }
                json.dump(search_sample, f, indent=2)
                
            logger.info(f"Saved sample search results to data/lv1/test/search_results_sample.json")
        else:
            logger.error("Web search returned no results or failed. Check API keys and connectivity.")
            return
        
        # Test full term processing
        logger.info("Testing full term processing pipeline...")
        result = await process_level0_term(test_term)
        
        logger.info(f"Extracted {result['count']} departments for '{test_term}'")
        
        # Save extracted departments
        with open(f"data/lv1/test/{test_term}_departments.txt", "w") as f:
            for dept in result["departments"]:
                f.write(f"{dept}\n")
        
        # Save detailed results with quality scores
        with open(f"data/lv1/test/{test_term}_detailed.json", "w") as f:
            # Create a more readable format of the results
            detailed_results = {
                "level0_term": test_term,
                "department_count": result["count"],
                "departments": [
                    {
                        "name": dept,
                        "quality_score": result["quality_scores"].get(dept.lower(), 0),
                        "sources": result["url_sources"].get(dept.lower(), [])
                    }
                    for dept in result["departments"]
                ]
            }
            json.dump(detailed_results, f, indent=2)
            
        logger.info(f"Test completed successfully. Results saved to data/lv1/test/{test_term}_*")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 