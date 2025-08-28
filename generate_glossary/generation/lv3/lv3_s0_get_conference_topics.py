import os
import sys
import random
import json
import time
import asyncio
import aiohttp
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter
import ssl
import certifi
import datetime  # For checkpoint timestamps

# Fix import path for utils - Adjust based on the new file location (lv3)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.config import get_level_config, get_processing_config, ensure_directories
# No longer need Provider import - using direct provider strings

# Import shared web search utilities
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html, score_list
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists, init_llm

# Load environment variables and setup logging
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))
logger = setup_logger("lv3.s0") # Changed logger name
random.seed(42)

# Constants
MAX_SEARCH_RESULTS = 5
MAX_CONCURRENT_REQUESTS = 25
BATCH_SIZE = 10
MAX_SEARCH_QUERIES = 100

# Conference/Journal-related keywords for enhanced filtering
CONFERENCE_KEYWORDS = [
    "topics", "tracks", "sessions", "call for papers", "cfp", "submission", "author",
    "paper", "conference", "journal", "workshop", "symposium", "proceedings",
    "publication", "scope", "aims", "theme", "special issue", "area", "field",
    "contribution", "presentation", "research", "study", "domain", "technical session"
]

# Anti-keywords indicating non-conference/topic text
NON_CONFERENCE_KEYWORDS = [
    "login", "sign in", "register", "registration", "venue", "travel", "accommodation",
    "sponsor", "exhibit", "committee", "organization", "steering", "program committee",
    "contact", "about", "home", "sitemap", "search", "privacy", "terms", "copyright",
    "accessibility", "careers", "jobs", "staff", "directory", "email", "address",
    "location", "directions", "map", "visit", "tour", "events", "news", "calendar",
    "library", "bookstore", "student", "alumni", "giving", "donate", "support",
    "request", "form", "learn more", "read more", "view", "download", "upload",
    "submit", "send", "share", "follow", "like", "tweet", "post", "comment",
    "subscribe", "newsletter", "blog", "click here", "link", "back", "next",
    "previous", "continue", "past conferences", "archive", "important dates",
    "deadlines", "keynote", "speaker", "invited talk", "plenary", "tutorial",
    "social event", "banquet", "award", "prize", "fee", "payment", "policy"
]

# Conference/Journal pattern matches for regex matching (less critical but can help)
JOURNAL_PATTERNS = [
    r"(?:topics|tracks|areas) (?:of|include|cover):?",
    r"(?:call for|submit) papers?",
    r"track [\d\w]+:",
    r"session [\d\w]+:",
    r"scope and topics",
    r"themes and sub-themes",
    r"submission guidelines",
    r"instructions for authors"
]

# Search queries - Updated for journals
SEARCH_QUERIES = [
    "\"{journal}\" (topics | tracks | call for papers | cfp | scope | themes)",
    "site:.org \"{journal}\" (topics | tracks | areas)", # Many journals use .org
    "site:.com \"{journal}\" (topics | tracks | areas)", # Some use .com
    "conference \"{journal}\" (topics | tracks)",
    "journal \"{journal}\" (scope | topics)",
]

# LLM system prompt template for journal topic validation
JOURNAL_TOPIC_VALIDATION_SYSTEM_PROMPT_TEMPLATE = """You are a highly meticulous academic assistant specializing in identifying ONLY the most specific and directly relevant topics, tracks, or themes for academic conferences and journals.

Task: Analyze the provided list and extract ONLY items that are DIRECTLY and UNQUESTIONABLY topics, tracks, themes, or specific subject areas covered by the journal named **{journal}**.

Input: A list of potential topics/tracks/keywords/other text found on a webpage related to the journal.
Output: A Python-style list `[...]` containing ONLY the items that are DEFINITIVELY and DIRECTLY relevant topics or tracks for **{journal}**. Return ONLY the verified items, preserving their original phrasing.

EXTREMELY STRICT Exclusion Criteria - DO NOT INCLUDE:
1. **ANYTHING not clearly a specific topic, track, or theme** for this specific journal ({journal}). If there is ANY doubt, EXCLUDE it.
2. **General academic fields** unless they are explicitly listed as a track or topic for {journal}. (e.g., Exclude "Computer Science" unless it's listed as "Track: Computer Science").
3. **Organizational or administrative details**: EXCLUDE committee names, submission guidelines (unless they list topics), deadlines, venue information, registration details, sponsor names, keynote speakers, etc.
4. **Website navigation/meta elements**: EXCLUDE ALL menu items, headers, footers, "Home", "About", "Contact Us", "Past Conferences", "Archives", etc.
5. **Vague or overly broad terms**: EXCLUDE terms that could apply to many journals without specific context.
6. **Instructions or calls to action**: EXCLUDE "Submit Your Paper", "Register Now", "View Program", etc.
7. **Names of people or organizations**: EXCLUDE author names, committee member names, affiliations, etc.
8. **Keywords that are too generic**: EXCLUDE single words like "Research", "Paper", "Analysis" unless part of a specific listed topic.

Guidelines:
- Be EXTREMELY SELECTIVE. When in doubt, EXCLUDE.
- ONLY include items that represent a specific area of focus, research theme, or session track for **{journal}**.
- Preserve the original phrasing and capitalization as much as possible.
- Return an empty list `[]` if NO items meet these strict criteria.
- Output ONLY the Python-style list, nothing else.

Example (Conference: International Conference on Machine Learning - ICML):
Input List: ["Deep Learning", "Reinforcement Learning", "Submission Deadline: May 15", "Register Now", "Theory", "Computer Vision Applications", "Natural Language Processing", "Organizing Committee", "Poster Session II", "About ICML"]
Output: ["Deep Learning", "Reinforcement Learning", "Theory", "Computer Vision Applications", "Natural Language Processing"]

Example (Journal: Journal of Fluid Mechanics):
Input List: ["Fluid Dynamics", "Turbulence", "Contact Editor", "Submit Manuscript", "Aims and Scope", "Computational Fluid Dynamics", "Aerodynamics", "Past Issues", "Editorial Board", "Geophysical Fluid Dynamics"]
Output: ["Fluid Dynamics", "Turbulence", "Computational Fluid Dynamics", "Aerodynamics", "Geophysical Fluid Dynamics"]

I will ONLY include items that are DEFINITIVELY and DIRECTLY relevant topics or tracks for **{journal}**. Any item with even slight uncertainty will be excluded.

Analyze the following list with EXTREME STRICTNESS:"""

DEFAULT_MIN_SCORE_FOR_LLM = 0.55 # Slightly lower threshold might be okay for topics
DEFAULT_LLM_MODEL_TYPES = ["pro", "default", "mini"] # Default model types for attempts

# Checkpoint system configuration
CHECKPOINT_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")), "data/lv3/checkpoint")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "lv3_s0_checkpoint.json")

# Use centralized configuration
LEVEL = 3
level_config = get_level_config(LEVEL)
processing_config = get_processing_config(LEVEL)

# Ensure directories exist
ensure_directories(LEVEL)

def read_journals(input_path: str) -> List[str]:
    """Read journal names (level 2 terms) from input file"""
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            terms = [line.strip() for line in file.readlines() if line.strip()]
        logger.info(f"Successfully read {len(terms)} journals (level 2 terms)")
        return terms
    except Exception as e:
        logger.error(f"Failed to read journals: {str(e)}", exc_info=True)
        raise


def clean_raw_topic(item: str) -> str:
    """Clean a raw topic item extracted from journal webpage"""
    item = item.strip()
    # Remove common prefixes like "Topic:", "Track:", "Session:"
    item = re.sub(r'^(Topic|Track|Session|Area)[\s:]*', '', item, flags=re.IGNORECASE)
    # Remove numbering/bullets like "1.", "a)", "*"
    item = re.sub(r'^[\d\w][\.\)]\s*', '', item)
    item = re.sub(r'^[•\-\*\−\–\—]\s*', '', item)
    # Remove trailing numbers, parenthetical info (like citations)
    item = re.sub(r'\s*\(\d+(?:,\s*\d+)*\)$', '', item) # Matches (1), (1, 2)
    item = re.sub(r'\s*\[\d+(?:,\s*\d+)*\]$', '', item) # Matches [1], [1, 2]
    item = re.sub(r'\s*\d+\s*$', '', item)
    # Remove URLs
    item = re.sub(r'http\S+', '', item)
    # Clean whitespace
    item = ' '.join(item.split())
    # Remove trailing punctuation (commas, semicolons)
    item = re.sub(r'[;,]+$', '', item).strip()
    return item


def score_raw_topic_list(items: List[str], metadata: Dict[str, Any], context_term: str) -> float:
    """Score a raw topic list based on various heuristics"""
    # Adjust weights for raw topics
    weights = {
        "keyword": 0.40,      # Increased: Presence of topic keywords is crucial
        "structure": 0.05,    # Decreased: HTML structure less important than content
        "pattern": 0.20,      # Moderate: Look for patterns like "Track:", "Topic:"
        "non_term": 0.25,     # Increased: Penalty for non-topic terms (registration, venue etc.)
        "consistency": 0.05,  # Decreased: Formatting consistency less critical
        "size": 0.0,          # Removed: List size less important
        "html_type": 0.05     # Decreased: HTML element type less critical
    }

    # Ensure weights sum to 1 (or close enough)
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        weights = {k: v / total_weight for k, v in weights.items()}

    # Use the common scoring function from list_extractor, passing the adjusted weights and keywords
    try:
        score = score_list(
            items,
            metadata,
            context_term,
            weights=weights,
            keywords=CONFERENCE_KEYWORDS,
            anti_keywords=NON_CONFERENCE_KEYWORDS,
            patterns=JOURNAL_PATTERNS
        )
        return score
    except Exception as e:
        logger.error(f"Error scoring list: {str(e)}", exc_info=True)
        return 0.0


async def run_multiple_llm_extractions(
    all_extracted_lists_raw: List[Dict[str, Any]],
    level2_term: str, # Changed from level1_term
    filter_config: FilterConfig,
    num_attempts: int = processing_config.llm_attempts,
    agreement_threshold: int = processing_config.concept_agreement_threshold,
    logger: Optional[Any] = None,
    model_types: List[str] = DEFAULT_LLM_MODEL_TYPES
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[List[str]]]:
    """
    Run multiple LLM extractions and select conference topics that appear in multiple responses.
    Each attempt uses a randomly selected provider (Gemini/OpenAI) and model type.

    Args:
        all_extracted_lists_raw: Raw extracted lists to process
        level2_term: The level 2 term (conference/journal name) being processed
        filter_config: Configuration for filtering
        num_attempts: Number of LLM extraction attempts
        agreement_threshold: Minimum number of appearances required for a topic
        logger: Optional logger
        model_types: List of model types to use for each attempt

    Returns:
        Tuple containing:
        - final_lists: Combined lists with items that meet the agreement threshold
        - llm_candidates: The candidates sent to the LLM (from the first run)
        - llm_results: The consolidated results from multiple LLM runs
    """
    if not all_extracted_lists_raw:
        return [], [], []

    logger.info(f"Running multiple LLM extractions ({num_attempts}) for '{level2_term}'")

    # Score all lists
    scored_lists = []
    for idx, list_dict in enumerate(all_extracted_lists_raw):
        if not isinstance(list_dict, dict) or 'items' not in list_dict or not list_dict['items']:
            continue
        if 'metadata' not in list_dict:
            list_dict['metadata'] = {}

        # Use the conference topic scoring function
        score = score_raw_topic_list(list_dict['items'], list_dict.get('metadata', {}), level2_term)

        scored_lists.append({
            'index': idx,
            'list_dict': list_dict,
            'score': score
        })

    scored_lists.sort(key=lambda x: x['score'], reverse=True)

    # Filter lists based on min_score_for_llm
    min_score = filter_config.min_score_for_llm if filter_config.min_score_for_llm is not None else DEFAULT_MIN_SCORE_FOR_LLM
    filtered_lists = [sl for sl in scored_lists if sl['score'] >= min_score]
    logger.info(f"Found {len(filtered_lists)} lists that passed score threshold ({min_score})")

    # Ensure we have enough lists for the LLM
    MIN_LISTS_FOR_LLM = 10 # Keep this minimum target
    if len(filtered_lists) < MIN_LISTS_FOR_LLM and scored_lists:
        lists_to_add = min(MIN_LISTS_FOR_LLM - len(filtered_lists), len(scored_lists) - len(filtered_lists))
        if lists_to_add > 0:
            remaining_lists = [sl for sl in scored_lists if sl not in filtered_lists]
            remaining_lists.sort(key=lambda x: x['score'], reverse=True)
            top_remaining = remaining_lists[:lists_to_add]
            filtered_lists.extend(top_remaining)
            logger.info(f"Added {len(top_remaining)} additional lists to reach minimum of {MIN_LISTS_FOR_LLM} lists for LLM")
            filtered_lists.sort(key=lambda x: x['score'], reverse=True)

    filtered_raw_lists = [sl['list_dict'] for sl in filtered_lists]

    if not filtered_raw_lists:
        logger.warning(f"No lists available for LLM processing after filtering for '{level2_term}'")
        return [], [], []

    logger.info(f"Running LLM extraction on {len(filtered_raw_lists)} lists for '{level2_term}'")

    all_results = []
    all_candidates = []
    all_raw_results = []

    if len(model_types) < num_attempts:
        model_types = model_types * (num_attempts // len(model_types) + 1)
    current_model_types = model_types[:num_attempts]
    available_providers = ["gemini", "openai"]

    for attempt in range(num_attempts):
        current_provider = random.choice(available_providers)
        current_model_type = random.choice(model_types) # Randomly choose model type

        # Create filter config for this attempt
        current_filter_config = FilterConfig(
            scoring_fn=score_raw_topic_list, # Use conference scoring
            clean_item_fn=clean_raw_topic,    # Use conference cleaning
            provider=current_provider,
            use_llm_validation=True,
            binary_llm_decision=False, # Assuming extraction, not binary decision
             # Use the new conference prompt template
            binary_system_prompt=JOURNAL_TOPIC_VALIDATION_SYSTEM_PROMPT_TEMPLATE.format(journal=level2_term),
            min_score_for_llm=0.0, # Already filtered
            model_type=current_model_type
        )

        logger.info(f"Attempt {attempt+1}/{num_attempts} using RANDOM provider: {current_provider}, model: {current_model_type}")

        try:
            # Pass the pre-filtered lists to the LLM filtering/extraction function
            final_lists, llm_candidates, llm_results = await filter_lists(
                filtered_raw_lists, level2_term, current_filter_config, logger
            )

            all_results.append(final_lists)
            if attempt == 0:
                all_candidates = llm_candidates
            all_raw_results.extend(llm_results) # Accumulate raw results from all attempts

            logger.info(f"Attempt {attempt+1} found {len(final_lists)} verified lists/topics")

        except Exception as e:
            logger.error(f"Error in extraction attempt {attempt+1}: {str(e)}")

    # Consolidate results based on agreement threshold
    all_items = []
    for final_lists_attempt in all_results:
        for lst in final_lists_attempt:
            all_items.extend(lst)

    item_counts = {}
    for item in all_items:
        # Normalize items for counting (lowercase, strip)
        item_norm = item.lower().strip()
        if item_norm not in item_counts:
            # Store original casing with first occurrence
            item_counts[item_norm] = {'count': 0, 'original': item}
        item_counts[item_norm]['count'] += 1

    # Filter items by agreement threshold, keeping original casing
    agreed_items_dict = {
        norm: data['original']
        for norm, data in item_counts.items()
        if data['count'] >= agreement_threshold
    }
    agreed_items = list(agreed_items_dict.values()) # Get the original casings

    logger.info(f"Found {len(agreed_items)} conference topics/tracks that meet the agreement threshold ({agreement_threshold})")

    # Format results - single list containing all agreed items
    final_consolidated_list = [agreed_items] if agreed_items else []

    # Ensure llm_results aligns with candidates for metadata saving
    # We return all raw results, let the caller handle pairing if needed for detailed metadata
    return final_consolidated_list, all_candidates, all_raw_results


async def process_level2_term(level2_term: str, # Renamed from level1_term
                              provider: Optional[str] = None,
                              session: Optional[Any] = None,
                              general_semaphore: Optional[asyncio.Semaphore] = None,
                              browser_semaphore: Optional[asyncio.Semaphore] = None,
                              min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                              model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                              num_llm_attempts: int = processing_config.llm_attempts,
                              agreement_threshold: int = processing_config.concept_agreement_threshold,
                              prefetched_search_results: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Process a single level2 term (conference/journal) to extract topics/tracks"""
    logger.info(f"Processing level 2 term: {level2_term} (LLM Min Score: {min_score_for_llm}, Models: {model_types}, Attempts: {num_llm_attempts}, Agree: {agreement_threshold})")

    term_details = {
        "level2_term": level2_term, # Changed key
        "all_urls": [],
        "raw_extracted_lists": [],
        "llm_io_pairs": [],
        "final_consolidated_topics": [], # Changed key
        "error": None
    }
    url_to_raw_lists = {}

    try:
        search_config = WebSearchConfig(
            base_dir=str(level_config.data_dir.parent.parent),
            raw_search_dir=level_config.data_dir / "raw_search" # Use Lv3 path
        )
        html_config = HTMLFetchConfig(
            cache_dir=level_config.data_dir / "cache" # Use Lv3 path
        )
        list_config = ListExtractionConfig(
            keywords=CONFERENCE_KEYWORDS, # Use conference keywords
            anti_keywords=NON_CONFERENCE_KEYWORDS, # Use conference anti-keywords
            patterns=JOURNAL_PATTERNS # Use conference patterns
        )
        filter_config = FilterConfig(
            scoring_fn=score_raw_topic_list, # Use conference scoring
            clean_item_fn=clean_raw_topic,    # Use conference cleaning
            provider=provider,
            use_llm_validation=True,
            binary_llm_decision=False, # Assuming extraction
            binary_system_prompt=JOURNAL_TOPIC_VALIDATION_SYSTEM_PROMPT_TEMPLATE.format(journal=level2_term), # Use conference prompt
            min_score_for_llm=min_score_for_llm,
            # model_type set in run_multiple_llm_extractions
        )

        search_results = prefetched_search_results
        if search_results is None:
            # Use updated SEARCH_QUERIES
            queries = [query.format(journal=level2_term) for query in SEARCH_QUERIES]
            logger.info(f"Searching with {len(queries)} queries for '{level2_term}'")
            search_results = web_search_bulk(queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)

        if not search_results or not search_results.get("data"):
            logger.warning(f"No search results for '{level2_term}'")
            term_details["error"] = "No search results"
            return {
                "level2_term": level2_term, # Key changed
                "conference_topics": [], # Key changed
                "count": 0,
                "verified": False,
                "num_urls": 0,
                "num_lists": 0,
                "term_details": term_details,
                "url_to_raw_lists": {}
            }

        # Extract URLs
        all_urls = []
        for query_index, query_data in enumerate(search_results.get("data", [])):
            query_urls = [r.get("url") for r in query_data.get("results", []) if r.get("url")]
            logger.debug(f"Query {query_index+1} returned {len(query_urls)} URLs")
            all_urls.extend(query_urls)

        seen_urls = set()
        urls = [url for url in all_urls if not (url in seen_urls or seen_urls.add(url))]
        term_details["all_urls"] = urls

        if not urls:
            logger.warning(f"No URLs found in search results for '{level2_term}'")
            term_details["error"] = "No URLs found in search results"
            return {
                "level2_term": level2_term, # Key changed
                "conference_topics": [], # Key changed
                "count": 0,
                "verified": False,
                "num_urls": 0,
                "num_lists": 0,
                "term_details": term_details,
                "url_to_raw_lists": {}
            }

        logger.info(f"Found {len(urls)} unique URLs for '{level2_term}'")

        semaphore_to_use = general_semaphore or asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        all_extracted_lists_raw = []

        # Fetch and extract logic (remains largely the same, just uses updated configs)
        async def fetch_and_extract(session_to_use):
            nonlocal all_extracted_lists_raw, url_to_raw_lists # Allow modification
            fetch_tasks = [
                fetch_webpage(url, session_to_use, semaphore_to_use, browser_semaphore, html_config, level2_term, logger)
                for url in urls[:MAX_SEARCH_RESULTS] # Limit number of URLs processed
            ]
            html_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            for i, (url, result) in enumerate(zip(urls[:MAX_SEARCH_RESULTS], html_contents)):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching content from {url}: {str(result)}")
                    continue
                html_content = result
                if not html_content: continue

                extracted_lists = extract_lists_from_html(html_content, list_config) # Uses conference config
                if extracted_lists:
                    # Add URL to metadata if not already present
                    for list_data in extracted_lists:
                         if 'metadata' not in list_data: list_data['metadata'] = {}
                         list_data['metadata']['url'] = url # Ensure URL is tracked

                    all_extracted_lists_raw.extend(extracted_lists)
                    url_lists = []
                    for list_data in extracted_lists:
                        if isinstance(list_data, dict) and "items" in list_data:
                            items = list_data["items"]
                            if isinstance(items, list):
                                clean_items = [str(item) for item in items if item]
                                if clean_items: url_lists.append(clean_items)
                        elif isinstance(list_data, list):
                             clean_items = [str(item) for item in list_data if item]
                             if clean_items: url_lists.append(clean_items)
                    if url_lists:
                        url_to_raw_lists[url] = url_lists
                    logger.debug(f"Extracted {len(extracted_lists)} lists from {url}")

        if session is None:
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context(cafile=certifi.where()),
                limit=MAX_CONCURRENT_REQUESTS, limit_per_host=2, force_close=True
            )
            async with aiohttp.ClientSession(
                connector=connector, timeout=aiohttp.ClientTimeout(total=60), raise_for_status=False
            ) as new_session:
                await fetch_and_extract(new_session)
        else:
            await fetch_and_extract(session)

        term_details["raw_extracted_lists"] = all_extracted_lists_raw

        if not all_extracted_lists_raw:
            logger.warning(f"No lists extracted for '{level2_term}'")
            term_details["error"] = "No lists extracted from fetched HTML"
            return {
                "level2_term": level2_term, # Key changed
                "conference_topics": [], # Key changed
                "count": 0,
                "verified": False,
                "num_urls": len(urls),
                "num_lists": 0,
                "term_details": term_details,
                "url_to_raw_lists": url_to_raw_lists
            }

        logger.info(f"Extracted a total of {len(all_extracted_lists_raw)} raw lists for '{level2_term}'. Starting filtering...")

        # Run multiple LLM extractions
        final_llm_output_lists, llm_candidates, llm_results = await run_multiple_llm_extractions(
            all_extracted_lists_raw, level2_term, filter_config,
            num_attempts=num_llm_attempts,
            agreement_threshold=agreement_threshold,
            logger=logger,
            model_types=model_types
        )

        # Store LLM I/O pairs in term_details
        term_details["llm_io_pairs"] = []
        if llm_candidates and llm_results:
             num_pairs = min(len(llm_candidates), len(llm_results)) # Pair inputs with outputs
             if len(llm_candidates) != len(llm_results):
                  logger.warning(f"Mismatch between LLM candidates ({len(llm_candidates)}) and results ({len(llm_results)}) for {level2_term}. Pairing up to {num_pairs}.")
             for i in range(num_pairs):
                  candidate_input = llm_candidates[i].get('items', []) if isinstance(llm_candidates[i], dict) else llm_candidates[i]
                  llm_output = llm_results[i] # Assuming llm_results is a list of lists
                  term_details["llm_io_pairs"].append({
                       "input_list_to_llm": candidate_input,
                       "output_list_from_llm": llm_output,
                       # Add source URL from candidate if available
                       "source_url": llm_candidates[i].get("metadata", {}).get("url", "unknown") if isinstance(llm_candidates[i], dict) else "unknown"
                  })
        elif llm_results and not llm_candidates:
            # If only results are available (e.g., from multiple runs consolidation)
            for result_list in llm_results:
                 term_details["llm_io_pairs"].append({
                      "input_list_to_llm": "N/A (Consolidated)", # Indicate input wasn't directly paired
                      "output_list_from_llm": result_list,
                      "source_url": "N/A (Consolidated)"
                 })

        if not final_llm_output_lists:
            logger.warning(f"No lists passed LLM extraction/validation for '{level2_term}'")
            term_details["error"] = "No lists passed LLM extraction/validation"
            return {
                "level2_term": level2_term, # Key changed
                "conference_topics": [], # Key changed
                "count": 0,
                "verified": False,
                "num_urls": len(urls),
                "num_lists": len(all_extracted_lists_raw),
                "term_details": term_details,
                "url_to_raw_lists": url_to_raw_lists
            }

        logger.info(f"After LLM extraction/agreement, {len(final_llm_output_lists[0]) if final_llm_output_lists else 0} topics remain for '{level2_term}'")

        # Consolidate topics (using function designed for lists of lists)
        # Since run_multiple_llm_extractions returns a list containing one list of agreed items:
        consolidated_topics = consolidate_lists(
            final_llm_output_lists, # Pass the list containing the single list of agreed items
            level2_term,
            min_frequency=1,
            min_list_appearances=1,
            similarity_threshold=0.75 # Slightly higher threshold for topics
        )
        term_details["final_consolidated_topics"] = consolidated_topics # Key changed

        logger.info(f"Consolidated to {len(consolidated_topics)} unique conference topics/tracks for '{level2_term}'")

        # Return results
        return {
            "level2_term": level2_term, # Key changed
            "conference_topics": consolidated_topics, # Key changed
            "count": len(consolidated_topics),
            "verified": len(consolidated_topics) > 0,
            "num_urls": len(urls),
            "num_lists": len(all_extracted_lists_raw),
            "term_details": term_details,
            "url_to_raw_lists": url_to_raw_lists
        }

    except Exception as e:
        logger.error(f"Error processing term '{level2_term}': {str(e)}", exc_info=True)
        term_details["error"] = f"Unhandled exception: {str(e)}"
        return {
            "level2_term": level2_term, # Key changed
            "conference_topics": [], # Key changed
            "count": 0,
            "verified": False,
            "num_urls": len(term_details.get('all_urls', [])),
            "num_lists": len(term_details.get('raw_extracted_lists', [])),
            "error": str(e),
            "term_details": term_details,
            "url_to_raw_lists": url_to_raw_lists
        }


def perform_bulk_web_search(level2_terms: List[str], search_config: WebSearchConfig) -> Dict[str, Dict[str, Any]]:
    """Perform a bulk web search for multiple level 2 terms (conferences/journals)"""
    if not level2_terms:
        return {}

    queries_per_term = SEARCH_QUERIES # Use the updated conference queries
    num_queries_per_term = len(queries_per_term)
    max_terms_per_search = max(1, MAX_SEARCH_QUERIES // num_queries_per_term) # Ensure at least 1

    logger.info(f"Performing bulk web search for {len(level2_terms)} terms (max {max_terms_per_search} per batch, {num_queries_per_term} queries per term)")

    all_queries = []
    term_to_query_indices = {}
    for term in level2_terms:
        start_idx = len(all_queries)
        # Use .format(level2_term=term) for conference queries
        current_term_queries = [query.format(journal=term) for query in queries_per_term]
        all_queries.extend(current_term_queries)
        term_to_query_indices[term] = (start_idx, len(all_queries))

    search_results = web_search_bulk(all_queries, search_config, logger=logger, limit=MAX_SEARCH_RESULTS)

    if not search_results or not search_results.get("data"):
        logger.warning(f"No search results found for any of the {len(level2_terms)} terms")
        return {}

    term_to_results = {}
    for term, (start_idx, end_idx) in term_to_query_indices.items():
        term_query_results = search_results.get("data", [])[start_idx:end_idx]
        # Only add if we have non-empty results for this term
        if any(qr.get("results") for qr in term_query_results):
             term_to_results[term] = {"data": term_query_results}
        else:
             logger.debug(f"No results found for term '{term}' in bulk search response.")


    logger.info(f"Got search results for {len(term_to_results)} out of {len(level2_terms)} terms")
    return term_to_results


async def process_level2_terms_batch(batch: List[str], # Renamed from level1
                                   provider: Optional[str] = None,
                                   session: Optional[Any] = None,
                                   general_semaphore: Optional[asyncio.Semaphore] = None,
                                   browser_semaphore: Optional[asyncio.Semaphore] = None,
                                   min_score_for_llm: Optional[float] = DEFAULT_MIN_SCORE_FOR_LLM,
                                   model_types: List[str] = DEFAULT_LLM_MODEL_TYPES,
                                   num_llm_attempts: int = processing_config.llm_attempts,
                                   agreement_threshold: int = processing_config.concept_agreement_threshold
                                   ) -> List[Dict[str, Any]]:
    """Process a batch of level 2 terms with optimized bulk web searching"""
    if not batch:
        return []

    logger.info(f"Processing batch of {len(batch)} level 2 terms")

    search_config = WebSearchConfig(
        base_dir=str(level_config.data_dir.parent.parent),
        raw_search_dir=level_config.data_dir / "raw_search" # Use Lv3 path
    )

    num_queries_per_term = len(SEARCH_QUERIES)
    max_terms_per_search = max(1, MAX_SEARCH_QUERIES // num_queries_per_term)

    search_results_by_term = {}
    for i in range(0, len(batch), max_terms_per_search):
        search_batch = batch[i:i + max_terms_per_search]
        logger.info(f"Performing bulk web search for {len(search_batch)} level 2 terms (batch {i//max_terms_per_search + 1})")
        batch_results = perform_bulk_web_search(search_batch, search_config) # Uses updated function
        search_results_by_term.update(batch_results)

    tasks = []
    for term in batch:
        prefetched_results = search_results_by_term.get(term)
        task = process_level2_term( # Use updated function
            term,
            provider,
            session,
            general_semaphore,
            browser_semaphore,
            min_score_for_llm,
            model_types,
            num_llm_attempts,
            agreement_threshold,
            prefetched_results
        )
        tasks.append(task)

    return await asyncio.gather(*tasks)


def ensure_dirs_exist():
    """Ensure all required Lv3 directories exist"""
    dirs_to_create = [
        level_config.data_dir / "cache", # Lv3
        level_config.data_dir / "raw_search", # Lv3
        level_config.data_dir / "detailed_meta", # Lv3
        os.path.dirname(level_config.get_step_output_file(0)), # Lv3
        os.path.dirname(level_config.get_step_metadata_file(0)), # Lv3
        CHECKPOINT_DIR  # Add checkpoint directory
    ]

    logger.info(f"BASE_DIR: {str(level_config.data_dir.parent.parent)}")
    logger.info(f"LV2_INPUT_FILE: {str(get_level_config(2).get_final_file())}") # Input is Lv2
    logger.info(f"OUTPUT_FILE (Topics): {level_config.get_step_output_file(0)}") # Output is Lv3
    logger.info(f"META_FILE: {level_config.get_step_metadata_file(0)}") # Output is Lv3
    logger.info(f"CACHE_DIR: {level_config.data_dir / "cache"}") # Output is Lv3
    logger.info(f"RAW_SEARCH_DIR: {level_config.data_dir / "raw_search"}") # Output is Lv3
    logger.info(f"DETAILED_META_DIR: {level_config.data_dir / "detailed_meta"}") # Output is Lv3
    logger.info(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")

    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

def save_checkpoint(processed_terms: Set[str], all_results: List[Dict[str, Any]], input_file_path: str):
    """Save a checkpoint of processed terms and results"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_data = {
        "timestamp": timestamp,
        "processed_terms": list(processed_terms),
        "results": all_results,
        "input_file_path": input_file_path
    }
    
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f)
        logger.info(f"✅ Saved checkpoint with {len(processed_terms)} processed terms")
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}", exc_info=True)

def load_checkpoint() -> Tuple[Set[str], List[Dict[str, Any]], Optional[str]]:
    """Load previous checkpoint if it exists"""
    if not os.path.exists(CHECKPOINT_FILE):
        logger.info("No checkpoint file found, starting fresh")
        return set(), [], None
    
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        processed_terms = set(checkpoint_data.get("processed_terms", []))
        results = checkpoint_data.get("results", [])
        input_file_path = checkpoint_data.get("input_file_path")
        
        logger.info(f"✅ Loaded checkpoint from {checkpoint_data.get('timestamp', 'unknown')} with {len(processed_terms)} processed terms")
        return processed_terms, results, input_file_path
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}", exc_info=True)
        return set(), [], None

async def main_async():
    """Async main execution function for Lv3 conference topic extraction"""
    try:
        parser = argparse.ArgumentParser(description="Extract conference/journal topics for level 2 terms.")
        parser.add_argument("--provider", help="LLM provider (e.g., gemini, openai)")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for processing terms (default: {BATCH_SIZE})")
        parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS, help=f"Max concurrent term processing requests (default: {MAX_CONCURRENT_REQUESTS})")
        # Updated input file help text and default
        parser.add_argument("--input-file", default=str(get_level_config(2).get_final_file()), help=f"Path to the input file containing level 2 terms (conferences/journals) (default: {str(get_level_config(2).get_final_file())})")
        parser.add_argument("--append", action='store_true', help="Append results to existing output files instead of overwriting.")
        parser.add_argument("--llm-attempts", type=int, default=processing_config.llm_attempts, help=f"Number of LLM extraction attempts per term (default: {processing_config.llm_attempts})")
        parser.add_argument("--agreement-threshold", type=int, default=processing_config.concept_agreement_threshold, help=f"Minimum appearances threshold for topics (default: {processing_config.concept_agreement_threshold})")
        parser.add_argument("--min-score-for-llm", type=float, default=DEFAULT_MIN_SCORE_FOR_LLM, help=f"Minimum heuristic score to send a list to LLM (default: {DEFAULT_MIN_SCORE_FOR_LLM})")
        parser.add_argument("--llm-model-types", type=str, default=",".join(DEFAULT_LLM_MODEL_TYPES), help=f"Comma-separated LLM model types for attempts (e.g., default,pro,mini) (default: {','.join(DEFAULT_LLM_MODEL_TYPES)})")
        parser.add_argument("--resume", action='store_true', help="Resume from last checkpoint")
        parser.add_argument("--skip-checkpoint", action='store_true', help="Skip checkpoint saving (useful for testing)")

        args = parser.parse_args()

        provider = args.provider
        batch_size = args.batch_size
        max_concurrent = args.max_concurrent
        input_file_path = args.input_file # Now Lv2 input
        append_mode = args.append
        num_llm_attempts = args.llm_attempts
        agreement_threshold = args.agreement_threshold
        min_score_for_llm = args.min_score_for_llm
        llm_model_types = [mt.strip() for mt in args.llm_model_types.split(',') if mt.strip()]
        resume_from_checkpoint = args.resume
        skip_checkpoint = args.skip_checkpoint

        # Ensure directories exist before anything else
        ensure_dirs_exist()

        # Check for and load checkpoint if resuming
        processed_terms_set = set()
        all_results = []
        if resume_from_checkpoint:
            processed_terms_set, checkpoint_results, checkpoint_input_file = load_checkpoint()
            if processed_terms_set and checkpoint_results:
                all_results = checkpoint_results
                
                # If the input file from the checkpoint doesn't match the provided one, warn user
                if checkpoint_input_file and checkpoint_input_file != input_file_path:
                    logger.warning(f"⚠️ Checkpoint was created with input file '{checkpoint_input_file}' but current input file is '{input_file_path}'")
                    logger.warning("Continuing with current input file but skipping already processed terms")

        # Log settings
        logger.info(f"Starting conference topic extraction using level 2 terms from '{input_file_path}'")
        if provider: logger.info(f"Using provider: {provider}")
        logger.info(f"Batch size: {batch_size}, Max concurrent: {max_concurrent}")
        logger.info(f"LLM attempts: {num_llm_attempts}, Agreement threshold: {agreement_threshold}")
        logger.info(f"LLM min score threshold: {min_score_for_llm}, LLM model types: {llm_model_types}")
        if append_mode: logger.info("Append mode enabled.")
        if resume_from_checkpoint: logger.info(f"Resuming from checkpoint with {len(processed_terms_set)} already processed terms")
        if skip_checkpoint: logger.info("Checkpoint saving is disabled")
        logger.info(f"Using optimized web search with dynamically calculated max terms per batch")

        # Read level 2 terms
        level2_terms = read_journals(input_file_path)
        logger.info(f"Read {len(level2_terms)} level 2 terms from input file")
        
        # Filter out already processed terms if resuming
        if resume_from_checkpoint and processed_terms_set:
            remaining_terms = [term for term in level2_terms if term not in processed_terms_set]
            logger.info(f"Filtered out {len(level2_terms) - len(remaining_terms)} already processed terms")
            level2_terms = remaining_terms
        
        logger.info(f"Processing {len(level2_terms)} level 2 terms")

        output_file = level_config.get_step_output_file(0) # Lv3 topics file
        meta_file = level_config.get_step_metadata_file(0)     # Lv3 meta file
        current_provider = provider or "gemini"

        logger.info(f"Using provider: {current_provider} with {num_llm_attempts} LLM attempts (models: {llm_model_types}), agreement threshold {agreement_threshold}, and score threshold {min_score_for_llm}")

        # aiohttp setup (remains the same)
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, CookieJar
        from generate_glossary.utils.web_search.html_fetch import MAX_CONCURRENT_BROWSERS

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = TCPConnector(
            ssl=ssl_context, limit=max_concurrent, limit_per_host=3,  # Increased limit_per_host
            force_close=True, enable_cleanup_closed=True
        )
        cookie_jar = CookieJar(unsafe=True)
        start_time = time.time()

        general_semaphore = asyncio.Semaphore(max_concurrent)
        browser_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BROWSERS)
        logger.info(f"Using general fetch semaphore limit: {max_concurrent}")
        logger.info(f"Using headless browser semaphore limit: {MAX_CONCURRENT_BROWSERS}")

        timeout = ClientTimeout(total=3600, connect=30)
        async with ClientSession(
            connector=connector, cookie_jar=cookie_jar, timeout=timeout, raise_for_status=False
        ) as session:
            for i in range(0, len(level2_terms), batch_size):
                batch = level2_terms[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(level2_terms) + batch_size - 1)//batch_size
                logger.info(f"Processing batch {batch_num}/{total_batches}")

                try:
                    # Use updated batch processor
                    batch_results = await process_level2_terms_batch(
                        batch,
                        current_provider,
                        session,
                        general_semaphore,
                        browser_semaphore,
                        min_score_for_llm,
                        llm_model_types,
                        num_llm_attempts,
                        agreement_threshold
                    )
                    
                    # Add processed terms to the set and results to the list
                    processed_terms_set.update(batch)
                    all_results.extend(batch_results)

                    # Log progress
                    elapsed = time.time() - start_time
                    terms_processed = min(i + batch_size, len(level2_terms))
                    if resume_from_checkpoint:
                        total_processed = len(processed_terms_set)
                        logger.info(f"Total processed terms (including from checkpoint): {total_processed}")
                    
                    terms_per_second = terms_processed / max(1, elapsed)
                    eta_seconds = (len(level2_terms) - terms_processed) / max(0.1, terms_per_second)
                    logger.info(f"Processed {terms_processed}/{len(level2_terms)} terms ({terms_processed/len(level2_terms)*100:.1f}%) in {elapsed:.1f}s ({terms_per_second:.2f} terms/s, ETA: {eta_seconds/60:.1f}m)")
                    
                    # Save checkpoint after each batch unless disabled
                    if not skip_checkpoint:
                        save_checkpoint(processed_terms_set, all_results, input_file_path)
                    
                except Exception as batch_error:
                    logger.error(f"❌ Error processing batch {batch_num}/{total_batches}: {batch_error}", exc_info=True)
                    # Save checkpoint on error to preserve progress
                    if not skip_checkpoint:
                        save_checkpoint(processed_terms_set, all_results, input_file_path)
                    # Continue with next batch rather than stopping completely
                    logger.info(f"Continuing with next batch despite error in batch {batch_num}")
                    continue

                # Small delay between batches
                await asyncio.sleep(2)

        # Aggregation and Saving Logic
        found_conference_topics = [] # Renamed
        verified_terms_count = 0
        processed_stats = {
            "total_urls_processed": 0,
            "total_raw_lists_extracted": 0, # Renamed from total_lists_found
            "verified_conference_topics_count": 0 # Renamed
        }
        level2_term_result_counts = {} # Renamed
        level2_to_conference_topics = {} # Renamed
        conference_topic_sources = {} # Renamed
        # Quality scores might be less meaningful for topics, but keep structure
        conference_topic_quality_scores = {} # Renamed

        all_term_details = [] # For detailed metadata (optional saving)
        all_raw_url_lists = {} # For raw lists (optional saving)

        for result in all_results:
            level2_term = result["level2_term"] # Key changed

            # Store details for later saving (optional)
            # if "term_details" in result: all_term_details.append(result["term_details"])
            # if "url_to_raw_lists" in result and result["url_to_raw_lists"]: all_raw_url_lists[level2_term] = result["url_to_raw_lists"]

            if result.get("error"):
                logger.error(f"Skipping aggregation for term '{level2_term}' due to processing error: {result['error']}")
                continue

            topics = result["conference_topics"] # Key changed
            # quality_scores = result.get("quality_scores", {}) # May remove quality score focus later
            verified = result.get("verified", False)
            num_urls = result.get("num_urls", 0)
            num_lists = result.get("num_lists", 0) # This is the count of raw lists extracted

            processed_stats["total_urls_processed"] += num_urls
            processed_stats["total_raw_lists_extracted"] += num_lists

            if verified:
                verified_terms_count += 1
                topic_count = len(topics)
                processed_stats["verified_conference_topics_count"] += topic_count

                if level2_term not in level2_to_conference_topics:
                    level2_to_conference_topics[level2_term] = []
                level2_to_conference_topics[level2_term].extend(topics)
                found_conference_topics.extend(topics)
                level2_term_result_counts[level2_term] = topic_count

                for topic in topics:
                    if topic not in conference_topic_sources:
                        conference_topic_sources[topic] = []
                    conference_topic_sources[topic].append(level2_term)
                    # Store dummy quality score for now
                    conference_topic_quality_scores[topic] = 1.0

        logger.info(f"Consolidated results from {len(all_results)} processed level 2 terms.")
        logger.info(f"Found {verified_terms_count} verified terms with topics.")

        # Optional saving of detailed metadata and raw lists would go here if enabled

        # Resource cleanup
        logger.info("⚠️ Resource management: Explicitly closing any remaining open resources")
        try:
            import gc
            logger.debug("Clearing memory and running garbage collection...")
            # all_term_details.clear() # If saving disabled
            # all_raw_url_lists.clear() # If saving disabled
            gc.collect()
            logger.debug("Memory cleanup complete")
        except Exception as cleanup_error:
            logger.warning(f"Non-critical error during cleanup: {cleanup_error}")

        # --- Final Output and Metadata Saving ---

        existing_unique_topics = set()
        if append_mode and os.path.exists(output_file):
            logger.info(f"Loading existing topics from {output_file} for append mode.")
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_unique_topics = {line.strip() for line in f if line.strip()}
                logger.info(f"Loaded {len(existing_unique_topics)} existing unique topics.")
            except Exception as e:
                logger.warning(f"Could not read existing output file {output_file}: {e}. Starting fresh.", exc_info=True)
                existing_unique_topics = set()

        # Add newly found unique topics (lowercase for comparison, store original case if possible)
        final_unique_topics_set = set(existing_unique_topics)
        newly_added_count = 0
        # Use a dictionary to preserve original casing while checking uniqueness in lowercase
        final_topics_dict = {topic.lower(): topic for topic in existing_unique_topics}

        for topic in found_conference_topics:
            topic_lower = topic.lower()
            if topic_lower not in final_topics_dict:
                final_topics_dict[topic_lower] = topic # Store original casing
                newly_added_count += 1

        logger.info(f"Added {newly_added_count} new unique conference topics.")

        # Save output file (unique topics)
        final_unique_topics_list = sorted(list(final_topics_dict.values())) # Sort by original casing
        random.shuffle(final_unique_topics_list) # Randomize order

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for topic in final_unique_topics_list:
                    f.write(f"{topic}\n") # Save with original casing
            logger.info(f"Saved {len(final_unique_topics_list)} total unique conference topics to {output_file}")
        except Exception as e:
            logger.error(f"Failed to write output file {output_file}: {e}", exc_info=True)

        # Save metadata file
        metadata = {}
        if append_mode and os.path.exists(meta_file):
            logger.info(f"Loading existing metadata from {meta_file} for append mode.")
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info("Existing metadata loaded successfully.")
            except Exception as e:
                 logger.warning(f"Could not read or parse existing metadata file {meta_file}: {e}. Creating new metadata.", exc_info=True)
                 metadata = {}

        # Update metadata fields (using updated names)
        # Handle potential nested structure from previous runs
        meta_content = metadata.get("metadata", metadata) # Access nested dict if exists, else use root

        meta_content["execution_time"] = f"{time.time() - start_time:.2f} seconds"
        meta_content["total_unique_conference_topics"] = len(final_unique_topics_list) # Renamed
        meta_content["total_level2_terms_processed"] = meta_content.get("total_level2_terms_processed", 0) + len(level2_terms) # Renamed
        meta_content["verified_level2_terms_count"] = meta_content.get("verified_level2_terms_count", 0) + verified_terms_count # Renamed

        existing_result_counts = meta_content.get("level2_term_result_counts", {})
        existing_result_counts.update(level2_term_result_counts)
        meta_content["level2_term_result_counts"] = existing_result_counts

        existing_mapping = meta_content.get("level2_to_conference_topics_mapping", {})
        for term, topics in level2_to_conference_topics.items():
            if term not in existing_mapping: existing_mapping[term] = []
            existing_topics_set = set(existing_mapping[term])
            for topic in topics:
                 if topic not in existing_topics_set:
                      existing_mapping[term].append(topic)
                      existing_topics_set.add(topic)
        meta_content["level2_to_conference_topics_mapping"] = existing_mapping # Renamed

        existing_sources = meta_content.get("conference_topic_level2_sources", {}) # Renamed
        for topic, sources in conference_topic_sources.items():
             if topic not in existing_sources: existing_sources[topic] = []
             existing_source_set = set(existing_sources[topic])
             for source in sources:
                  if source not in existing_source_set:
                       existing_sources[topic].append(source)
                       existing_source_set.add(source)
        meta_content["conference_topic_level2_sources"] = existing_sources # Renamed

        # Quality scores may not be relevant, store placeholder
        existing_quality_scores = meta_content.get("conference_topic_quality_scores", {})
        existing_quality_scores.update(conference_topic_quality_scores) # Add/overwrite
        meta_content["conference_topic_quality_scores"] = existing_quality_scores # Renamed

        meta_content["provider"] = current_provider
        meta_content["max_concurrent"] = max_concurrent
        meta_content["batch_size"] = batch_size
        meta_content["num_llm_attempts"] = num_llm_attempts
        meta_content["agreement_threshold"] = agreement_threshold
        meta_content["min_score_for_llm"] = min_score_for_llm
        meta_content["llm_model_types_used"] = llm_model_types[:num_llm_attempts]

        meta_content["total_urls_processed"] = meta_content.get("total_urls_processed", 0) + processed_stats["total_urls_processed"]
        meta_content["total_raw_lists_extracted"] = meta_content.get("total_raw_lists_extracted", 0) + processed_stats["total_raw_lists_extracted"] # Renamed

        total_terms_ever_processed = meta_content["total_level2_terms_processed"]
        total_verified_ever = meta_content["verified_level2_terms_count"]
        total_verified_topics_ever = sum(len(topics) for topics in meta_content.get("level2_to_conference_topics_mapping", {}).values()) # Renamed

        meta_content["processing_stats"] = {
             "avg_urls_per_term": meta_content["total_urls_processed"] / max(1, total_terms_ever_processed),
             "avg_raw_lists_per_term": meta_content["total_raw_lists_extracted"] / max(1, total_terms_ever_processed),
             "avg_final_topics_per_verified_term": total_verified_topics_ever / max(1, total_verified_ever) # Renamed
         }

        # Save the updated structure
        final_meta_structure = {
            "level2_to_conference_topics_mapping": { # Renamed
                 term: topics for term, topics in meta_content["level2_to_conference_topics_mapping"].items()
            },
            "conference_topic_level2_sources": { # Renamed
                 topic: sources for topic, sources in meta_content["conference_topic_level2_sources"].items()
            },
            "conference_topic_quality_scores": meta_content.get("conference_topic_quality_scores", {}), # Renamed
            "metadata": meta_content # Store the rest under the 'metadata' key
        }


        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(final_meta_structure, f, indent=2)
            logger.info(f"Final aggregated metadata saved to {meta_file}")
        except Exception as e:
            logger.error(f"Failed to write metadata file {meta_file}: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

    logger.info("Conference topic extraction completed")

def main():
    """Main execution function"""
    # Apply nest_asyncio if available to handle potential nested loops in dependencies
    try:
        import nest_asyncio
        nest_asyncio.apply()
        logger.info("Applied nest_asyncio patch.")
    except ImportError:
        logger.debug("nest_asyncio not found, skipping patch.")
    except RuntimeError as e:
         # Ignore if loop is already running (e.g., in Jupyter)
         if "cannot apply patch" not in str(e).lower():
              logger.warning(f"Could not apply nest_asyncio patch: {e}")

    asyncio.run(main_async())


if __name__ == "__main__":
    main() 