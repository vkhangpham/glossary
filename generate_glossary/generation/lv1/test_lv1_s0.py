import os
import sys
import json
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
import random
import re
import time
import certifi

# Fix import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.generation.lv1.lv1_s0_get_dept_names import (
    process_level0_term, 
    read_level0_terms,
    consolidate_department_lists,
    filter_department_lists,
    extract_lists_from_html,
    score_department_list,
    QUALITY_THRESHOLD,
    USE_LLM_VALIDATION,
    LLM_VALIDATION_THRESHOLD,
    DEPT_KEYWORD_THRESHOLD
)

from dotenv import load_dotenv
import aiohttp
from aiohttp import TCPConnector
import ssl

load_dotenv('.env')

# Setup logging
logger = setup_logger("lv1.test.s0")

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# Constants
TEST_TERMS = ["business", "engineering", "humanities"]  # Reduced subset for faster testing
OUTPUT_DIR = os.path.join(BASE_DIR, "data/lv1/test")
COMPARISON_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_department_names.txt.old")
MAX_CONCURRENT_TERMS = 3  # Maximum number of terms to process in parallel

def copy_old_excel_results():
    """Copy the current Excel-based results for comparison"""
    try:
        # Check if old Excel-based results exist
        excel_results_path = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_department_names.txt")
        if os.path.exists(excel_results_path):
            comparison_path = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s0_department_names.txt.old")
            # Copy if not already copied
            if not os.path.exists(comparison_path):
                with open(excel_results_path, "r", encoding="utf-8") as src:
                    content = src.read()
                with open(comparison_path, "w", encoding="utf-8") as dst:
                    dst.write(content)
                logger.info(f"Copied original Excel-based results to {comparison_path}")
    except Exception as e:
        logger.error(f"Error copying Excel results: {str(e)}")

def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "sample_html"),
        os.path.join(BASE_DIR, "data/lv1/cache")
    ]
    
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"COMPARISON_FILE: {COMPARISON_FILE}")
    logger.info(f"CACHE_DIR: {os.path.join(BASE_DIR, 'data/lv1/cache')}")
    
    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

async def process_terms_concurrently(terms: List[str]) -> List[Dict[str, Any]]:
    """Process multiple level 0 terms concurrently with shared session"""
    results = []
    
    # Create a shared session with optimal settings
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.options |= ssl.OP_LEGACY_SERVER_CONNECT  # Allow legacy renegotiation
    
    connector = TCPConnector(ssl=ssl_context, limit=10, limit_per_host=2)
    cookie_jar = aiohttp.CookieJar(unsafe=True)
    
    # Create a throttling semaphore to limit concurrent term processing
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TERMS)
    
    async def process_term_with_throttling(term: str) -> Dict[str, Any]:
        """Process a single term with throttling"""
        async with semaphore:
            return await process_level0_term(term)
    
    # Process in small batches for better resource management
    async with aiohttp.ClientSession(connector=connector, cookie_jar=cookie_jar) as session:
        tasks = [process_term_with_throttling(term) for term in terms]
        results = await asyncio.gather(*tasks)
    
    return results

async def test_extraction_metrics():
    """Test extraction metrics with a small subset of level 0 terms"""
    try:
        start_time = time.time()
        
        # Create output directory
        ensure_dirs_exist()
        
        # Copy old Excel-based results for comparison
        copy_old_excel_results()
        
        # Process test terms concurrently
        logger.info(f"Testing extraction with level 0 terms: {TEST_TERMS}")
        results = await process_terms_concurrently(TEST_TERMS)
        
        for i, term in enumerate(TEST_TERMS):
            # Save individual term results
            logger.info(f"Extracted {results[i]['count']} departments for '{term}'")
            term_output = os.path.join(OUTPUT_DIR, f"{term}_departments.txt")
            with open(term_output, "w", encoding="utf-8") as f:
                for dept in results[i]["departments"]:
                    f.write(f"{dept}\n")
        
        # Collect all departments
        all_departments = []
        department_sources = {}  # Level0 term sources
        department_url_sources = {}  # URL sources
        department_quality_scores = {}  # Quality scores
        url_lists = {}  # Raw lists from each URL
        
        for result in results:
            level0_term = result["level0_term"]
            departments = result["departments"]
            url_sources = result.get("url_sources", {})
            quality_scores = result.get("quality_scores", {})
            
            # Track URL lists
            if "url_lists" in result:
                for url, lists in result["url_lists"].items():
                    if url not in url_lists:
                        url_lists[url] = []
                    url_lists[url].extend(lists)
            
            for dept in departments:
                department_key = dept.lower()
                # Track level0 term sources
                if department_key not in department_sources:
                    department_sources[department_key] = []
                department_sources[department_key].append(level0_term)
                
                # Track URL sources
                if department_key not in department_url_sources:
                    department_url_sources[department_key] = []
                if department_key in url_sources:
                    department_url_sources[department_key].extend(url_sources[department_key])
                    
                # Track quality scores
                if department_key in quality_scores:
                    if department_key not in department_quality_scores:
                        department_quality_scores[department_key] = quality_scores[department_key]
                    else:
                        department_quality_scores[department_key] = max(
                            department_quality_scores[department_key],
                            quality_scores[department_key]
                        )
                
            all_departments.extend(departments)
            
        # Sort departments by quality score for better analysis
        sorted_departments = sorted(
            [(dept, department_quality_scores.get(dept.lower(), 0)) for dept in all_departments],
            key=lambda x: x[1],
            reverse=True
        )
        sorted_unique_departments = []
        seen = set()
        for dept, _ in sorted_departments:
            dept_lower = dept.lower()
            if dept_lower not in seen:
                seen.add(dept_lower)
                sorted_unique_departments.append(dept)
        
        # Use quality-sorted departments for the primary output
        unique_departments = sorted_unique_departments
        
        # Save combined results with quality information
        combined_output = os.path.join(OUTPUT_DIR, "combined_departments.txt")
        with open(combined_output, "w", encoding="utf-8") as f:
            for dept in unique_departments:
                quality = department_quality_scores.get(dept.lower(), 0)
                sources = department_sources.get(dept.lower(), [])
                f.write(f"{dept} (Quality: {quality:.2f}, Sources: {', '.join(sources)})\n")
        
        # Save URL lists (simplified)
        url_lists_file = os.path.join(OUTPUT_DIR, "url_lists.json")
        with open(url_lists_file, "w", encoding="utf-8") as f:
            # Use a simplified version to keep file size manageable
            simplified_lists = {}
            for url, lists in url_lists.items():
                if len(simplified_lists) < 10:  # Limit to 10 URLs for report brevity
                    simplified_lists[url] = [list[:5] for list in lists[:2]]
            json.dump(simplified_lists, f, indent=2)
        
        # Save metadata
        metadata = {
            "execution_time": f"{time.time() - start_time:.2f} seconds",
            "total_departments": len(unique_departments),
            "level0_terms": len(TEST_TERMS),
            "department_counts_by_level0": {result["level0_term"]: result["count"] for result in results},
            "test_terms": TEST_TERMS,
            "total_urls": len(url_lists),
            "quality_threshold": QUALITY_THRESHOLD,
            "llm_validation_threshold": LLM_VALIDATION_THRESHOLD,
            "dept_keyword_threshold": DEPT_KEYWORD_THRESHOLD,
            "use_llm_validation": USE_LLM_VALIDATION,
            "max_concurrent_terms": MAX_CONCURRENT_TERMS
        }
        
        metadata_file = os.path.join(OUTPUT_DIR, "test_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully extracted {len(unique_departments)} unique departments")
        logger.info(f"Test results saved to {OUTPUT_DIR}")
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
        # Compare with Excel-based approach if possible
        if os.path.exists(COMPARISON_FILE):
            with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
                excel_departments = [line.strip() for line in f if line.strip()]
            
            excel_dept_set = {dept.lower() for dept in excel_departments}
            new_dept_set = {dept.lower() for dept in unique_departments}
            
            common = excel_dept_set.intersection(new_dept_set)
            only_in_excel = excel_dept_set - new_dept_set
            only_in_new = new_dept_set - excel_dept_set
            
            comparison = {
                "total_excel_departments": len(excel_departments),
                "total_new_departments": len(unique_departments),
                "common_departments": len(common),
                "only_in_excel_count": len(only_in_excel),
                "only_in_new_count": len(only_in_new),
                "overlap_percentage": round(len(common) / max(1, len(excel_dept_set)) * 100, 2),
                "example_only_in_excel": list(only_in_excel)[:5],
                "example_only_in_new": list(only_in_new)[:5]
            }
            
            comparison_file = os.path.join(OUTPUT_DIR, "comparison.json")
            with open(comparison_file, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)
            
            logger.info(f"Comparison with Excel-based approach saved to {comparison_file}")
            logger.info(f"Overlap percentage: {comparison['overlap_percentage']}%")
            
    except Exception as e:
        logger.error(f"Error testing extraction: {str(e)}", exc_info=True)
        raise

async def test_list_extraction_metrics():
    """Test the HTML list extraction heuristics with a sample HTML file"""
    try:
        # Ensure directories exist (already called in test_extraction_metrics)
        sample_html_dir = os.path.join(OUTPUT_DIR, "sample_html")
        
        # Check for existing cache files to test extraction
        cache_dir = os.path.join(BASE_DIR, "data/lv1/cache")
        if os.path.exists(cache_dir):
            # Get a sample of cache files (limit to 3 for faster execution)
            cache_files = list(Path(cache_dir).glob("*.txt"))[:3]
            
            if cache_files:
                logger.info(f"Testing list extraction with {len(cache_files)} sample HTML files")
                
                extraction_stats = {
                    "files_processed": len(cache_files),
                    "lists_extracted": 0,
                    "filtered_lists": 0,
                    "quality_scores": [],
                    "sample_extractions": []
                }
                
                for i, cache_file in enumerate(cache_files):
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            html_content = f.read()
                        
                        # Extract lists
                        extracted_lists = extract_lists_from_html(html_content)
                        extraction_stats["lists_extracted"] += len(extracted_lists)
                        
                        # Score each list without LLM validation
                        for extracted_list in extracted_lists:
                            # Clean items similar to filter_department_lists function
                            cleaned_items = []
                            for item in extracted_list:
                                # Remove common prefixes like "Department of", "School of"
                                item = re.sub(r'^(Department|School|College|Faculty|Division|Program|Institute) of ', '', item, flags=re.IGNORECASE)
                                # Remove trailing numbers, parenthetical info
                                item = re.sub(r'\s*\(\d+\).*$', '', item)
                                item = re.sub(r'\s*\d+\s*$', '', item)
                                # Remove URLs
                                item = re.sub(r'http\S+', '', item)
                                # Clean whitespace
                                item = ' '.join(item.split())
                                
                                if item and len(item) > 2 and len(item) < 100:
                                    cleaned_items.append(item)
                            
                            if len(cleaned_items) >= 3:
                                # Score the cleaned list
                                for test_term in TEST_TERMS:
                                    quality_score = score_department_list(cleaned_items, test_term)
                                    extraction_stats["quality_scores"].append(quality_score)
                        
                        # Count list types
                        # This is just an estimation since we can't directly track the source in our extraction function
                        for j, list_items in enumerate(extracted_lists[:2]):  # Only process first 2 lists per file
                            if j < 2:  # Further limit to max 2 lists per file
                                list_file = os.path.join(sample_html_dir, f"file{i+1}_list{j+1}.txt")
                                with open(list_file, "w", encoding="utf-8") as f:
                                    f.write(f"Source: {cache_file.name}\n")
                                    f.write(f"Items: {len(list_items)}\n")
                                    f.write("Content:\n")
                                    for item in list_items[:5]:  # Only show first 5 items
                                        f.write(f"- {item}\n")
                                
                                # Add example to stats
                                if len(extraction_stats["sample_extractions"]) < 2:
                                    # Try to score this list 
                                    score = score_department_list(list_items, TEST_TERMS[0])
                                    extraction_stats["sample_extractions"].append({
                                        "source": cache_file.name,
                                        "items": len(list_items),
                                        "quality_score": score,
                                        "sample_items": list_items[:3]
                                    })
                        
                        # Test filtering (only with first test term for speed)
                        filtered_lists = [
                            list_items for list_items in extracted_lists 
                            if score_department_list(list_items, TEST_TERMS[0]) >= QUALITY_THRESHOLD
                        ]
                        extraction_stats["filtered_lists"] += len(filtered_lists)
                            
                    except Exception as e:
                        logger.error(f"Error processing HTML file {cache_file}: {str(e)}")
                
                # Calculate quality score statistics
                if extraction_stats["quality_scores"]:
                    scores = extraction_stats["quality_scores"]
                    extraction_stats["quality_stats"] = {
                        "min": min(scores),
                        "max": max(scores),
                        "avg": sum(scores) / len(scores),
                        "median": sorted(scores)[len(scores) // 2],
                        "above_threshold": sum(1 for s in scores if s >= QUALITY_THRESHOLD) / len(scores)
                    }
                
                # Save extraction statistics
                stats_file = os.path.join(OUTPUT_DIR, "extraction_stats.json")
                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(extraction_stats, f, indent=2)
                
                logger.info(f"List extraction statistics saved to {stats_file}")
                logger.info(f"Extracted {extraction_stats['lists_extracted']} lists from {extraction_stats['files_processed']} files")
                logger.info(f"After filtering: {extraction_stats['filtered_lists']} lists")
            else:
                logger.warning("No cache files found for testing list extraction")
        else:
            logger.warning(f"Cache directory {cache_dir} not found")
            
    except Exception as e:
        logger.error(f"Error testing list extraction: {str(e)}", exc_info=True)
        raise

async def main():
    """Main test execution function"""
    try:
        total_start_time = time.time()
        logger.info("Starting tests for level 1 department extraction")
        
        # Test extraction metrics
        await test_extraction_metrics()
        
        # Test list extraction metrics
        await test_list_extraction_metrics()
        
        logger.info(f"Tests completed successfully in {time.time() - total_start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 