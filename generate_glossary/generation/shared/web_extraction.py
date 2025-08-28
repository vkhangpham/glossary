"""
Shared web extraction functionality for levels 1-3.

This module provides the generic s0 (web extraction) logic that can be 
configured for different levels through the level_config module.
"""

import os
import json
import time
import random
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter

from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import ensure_directories
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists
from .level_config import get_level_config


# Processing constants
MAX_SEARCH_RESULTS = 50
MAX_CONCURRENT_REQUESTS = 5
BATCH_SIZE = 100
MAX_SEARCH_QUERIES = 100

# LLM configuration
NUM_LLM_ATTEMPTS = 3
DEFAULT_LLM_MODEL_TYPES = ["default", "mini", "nano"]
DEFAULT_MIN_SCORE_FOR_LLM = 0.3


def load_input_terms(input_file: str) -> List[str]:
    """Load terms from input file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def build_search_queries(terms: List[str], patterns: List[str]) -> List[str]:
    """Build search queries from terms and patterns."""
    queries = []
    for term in terms:
        for pattern in patterns:
            queries.append(pattern.format(term=term))
    return queries


def create_llm_validation_prompt(level: int, term: str) -> str:
    """Create level-specific LLM validation prompt."""
    config = get_level_config(level)
    
    if level == 1:
        return f"""You are an expert in academic institution organization and department structures.

Your task is to analyze a provided list and extract ONLY the departments that are EXPLICITLY and DIRECTLY under the umbrella of {term}.

IMPORTANT: You must be EXTREMELY STRICT about this. Generic academic subjects should NOT be included unless they are EXPLICITLY stated to be part of {term} in particular.

Instructions:
1. Return a JSON array of valid department names from the list: ["department1", "department2", ...]  
2. Include ONLY departments that EXPLICITLY belong to {term}
3. Exclude ALL of the following:
   - Website menu items, navigation sections, or non-relevant content
   - Generic academic departments that could exist in many colleges
   - Staff directories, contact information, or administrative content
   - News, events, or other non-departmental content

Return only the JSON array, no additional text."""

    elif level == 2:
        return f"""You are an expert in academic research organization and specialization areas.

Your task is to analyze a provided list and extract ONLY the research areas, groups, or labs that are EXPLICITLY related to {term}.

Instructions:
1. Return a JSON array of valid research areas: ["area1", "area2", ...]
2. Include ONLY research areas that are relevant to {term}  
3. Focus on established research specializations, not generic terms
4. Exclude website navigation, administrative content, or irrelevant material

Return only the JSON array, no additional text."""

    elif level == 3:
        return f"""You are an expert in academic conferences and research topics.

Your task is to analyze a provided list and extract ONLY the conference topics, themes, or tracks that are relevant to {term}.

Instructions:
1. Return a JSON array of valid conference topics: ["topic1", "topic2", ...]
2. Include ONLY topics that would be relevant for conferences in {term}
3. Focus on specific research themes and specialized topics
4. Exclude general conference information, navigation, or administrative content

Return only the JSON array, no additional text."""

    else:
        raise ValueError(f"Unknown level: {level}")


async def validate_content_with_llm(content_list: List[str], term: str, level: int) -> List[str]:
    """Validate extracted content using LLM."""
    from generate_glossary.utils.llm_simple import infer_text, get_random_llm_config
    
    if not content_list:
        return []
    
    prompt = create_llm_validation_prompt(level, term)
    content_text = "\n".join(content_list)
    full_prompt = f"{prompt}\n\nList to analyze:\n{content_text}"
    
    all_validated = []
    
    for attempt in range(NUM_LLM_ATTEMPTS):
        try:
            provider, model = get_random_llm_config(level)
            
            response = infer_text(
                provider=provider,
                prompt=full_prompt,
                system_prompt="You are a helpful assistant that extracts structured information.",
                model=model
            )
            
            # Try to parse JSON response
            if response and response.strip():
                try:
                    # Clean up response - remove markdown formatting
                    clean_response = response.strip()
                    if clean_response.startswith('```'):
                        clean_response = clean_response.split('\n', 1)[1]
                    if clean_response.endswith('```'):
                        clean_response = clean_response.rsplit('\n', 1)[0]
                    
                    validated_items = json.loads(clean_response)
                    if isinstance(validated_items, list):
                        all_validated.extend([item.strip() for item in validated_items if item.strip()])
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract items from text
                    lines = response.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith(('```', '#', '*', '-')):
                            # Remove bullet points, numbers, quotes
                            clean_line = line.lstrip('123456789.-*"\'').strip()
                            if clean_line:
                                all_validated.append(clean_line)
                                
        except Exception as e:
            logger = setup_logger(f"lv{level}.s0")
            logger.warning(f"LLM validation attempt {attempt + 1} failed: {str(e)}")
            continue
    
    # Return items that appeared in multiple attempts (agreement-based filtering)
    if all_validated:
        item_counts = Counter(all_validated)
        config = get_level_config(level)
        return [item for item, count in item_counts.items() 
                if count >= min(config.agreement_threshold, len(all_validated))]
    
    return []


async def process_search_results(search_results: List[Dict], terms: List[str], level: int) -> Dict[str, List[str]]:
    """Process search results and extract relevant content."""
    logger = setup_logger(f"lv{level}.s0")
    config = get_level_config(level)
    
    # HTML fetch configuration
    html_config = HTMLFetchConfig(
        max_concurrent=MAX_CONCURRENT_REQUESTS,
        timeout=30,
        max_content_length=500000
    )
    
    # List extraction configuration  
    extraction_config = ListExtractionConfig(
        min_items=3,
        max_items=200,
        min_score=0.1
    )
    
    # Filter configuration
    filter_config = FilterConfig(
        min_score=DEFAULT_MIN_SCORE_FOR_LLM,
        keywords=config.quality_keywords
    )
    
    term_results = {term: [] for term in terms}
    
    for result in search_results:
        try:
            url = result.get('url', '')
            if not url:
                continue
                
            # Fetch webpage content
            webpage_content = await fetch_webpage(url, html_config)
            if not webpage_content:
                continue
            
            # Extract lists from HTML
            extracted_lists = extract_lists_from_html(webpage_content, extraction_config)
            
            # Score and filter lists
            scored_lists = []
            for list_data in extracted_lists:
                score = score_list(list_data['items'], config.quality_keywords)
                if score >= filter_config.min_score:
                    scored_lists.append({
                        **list_data,
                        'score': score,
                        'url': url
                    })
            
            # Associate with relevant terms
            for term in terms:
                term_lower = term.lower()
                for list_data in scored_lists:
                    # Simple relevance check - if term appears in URL or content
                    if (term_lower in url.lower() or 
                        any(term_lower in item.lower() for item in list_data['items'][:5])):
                        term_results[term].extend(list_data['items'])
                        
        except Exception as e:
            logger.warning(f"Error processing search result {result.get('url', 'unknown')}: {str(e)}")
            continue
    
    # Validate results with LLM
    validated_results = {}
    for term, items in term_results.items():
        if items:
            # Remove duplicates and clean items
            unique_items = list(set([item.strip() for item in items if item.strip()]))
            
            # LLM validation
            validated_items = await validate_content_with_llm(unique_items, term, level)
            validated_results[term] = validated_items
        else:
            validated_results[term] = []
    
    return validated_results


def save_results(results: Dict[str, List[str]], output_file: str, metadata_file: str, level: int):
    """Save extraction results to files."""
    logger = setup_logger(f"lv{level}.s0")
    
    # Prepare output data
    all_items = []
    for term, items in results.items():
        for item in items:
            all_items.append(f"{term} - {item}")
    
    # Shuffle for variety
    random.shuffle(all_items)
    
    # Save main output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_items:
            f.write(item + '\n')
    
    # Save metadata
    metadata = {
        'level': level,
        'total_terms_processed': len(results),
        'total_items_extracted': len(all_items),
        'items_per_term': {term: len(items) for term, items in results.items()},
        'processing_timestamp': time.time(),
        'config_used': {
            'batch_size': BATCH_SIZE,
            'max_search_results': MAX_SEARCH_RESULTS,
            'llm_attempts': NUM_LLM_ATTEMPTS
        }
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(all_items)} items to {output_file}")
    logger.info(f"Processed {len(results)} terms with average {len(all_items)/len(results):.1f} items per term")


async def extract_web_content(
    input_file: str,
    level: int,
    output_file: str,
    metadata_file: str
) -> Dict[str, Any]:
    """
    Generic web content extraction for any level.
    
    Args:
        input_file: Path to file containing input terms
        level: Generation level (1, 2, or 3)
        output_file: Path to save extracted content
        metadata_file: Path to save processing metadata
        
    Returns:
        Dictionary containing processing results and metadata
    """
    logger = setup_logger(f"lv{level}.s0")
    config = get_level_config(level)
    
    # Ensure directories exist
    ensure_directories(level)
    
    logger.info(f"Starting Level {level} web extraction: {config.processing_description}")
    
    # Load input terms
    input_terms = load_input_terms(input_file)
    logger.info(f"Loaded {len(input_terms)} input terms")
    
    # Build search queries
    search_queries = build_search_queries(input_terms, config.search_patterns)
    logger.info(f"Generated {len(search_queries)} search queries")
    
    # Limit queries to prevent overwhelming the search API
    if len(search_queries) > MAX_SEARCH_QUERIES:
        search_queries = random.sample(search_queries, MAX_SEARCH_QUERIES)
        logger.info(f"Limited to {MAX_SEARCH_QUERIES} queries")
    
    # Execute bulk web search
    search_config = WebSearchConfig(
        limit=MAX_SEARCH_RESULTS,
        per_query_limit=10
    )
    
    try:
        search_results = web_search_bulk(search_queries, search_config)
        logger.info(f"Got {len(search_results)} search results")
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        search_results = []
    
    # Process search results
    if search_results:
        extracted_results = await process_search_results(search_results, input_terms, level)
        logger.info(f"Extracted content for {len(extracted_results)} terms")
    else:
        logger.warning("No search results to process")
        extracted_results = {term: [] for term in input_terms}
    
    # Save results
    save_results(extracted_results, output_file, metadata_file, level)
    
    # Return processing metadata
    return {
        'level': level,
        'input_terms_count': len(input_terms),
        'search_queries_count': len(search_queries),
        'search_results_count': len(search_results),
        'extracted_terms_count': sum(len(items) for items in extracted_results.values()),
        'processing_description': config.processing_description
    }


# Alias for backward compatibility with runner imports
extract_web_content_simple = extract_web_content