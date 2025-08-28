import asyncio
import json
import re
import time
import ast # Added for safe literal evaluation
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Set
from collections import Counter
import logging
import os
from datetime import datetime

# NLP and Graph Libraries
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # Import lemmatizer

# Ensure stopwords are downloaded (run once)
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

try:
    # Check/download WordNet data needed for lemmatizer
    wnl = WordNetLemmatizer()
    wnl.lemmatize('tests') # Try lemmatizing to trigger download check
except LookupError:
    import nltk
    nltk.download('wordnet', quiet=True)

from generate_glossary.utils.llm_simple import infer_structured, infer_text, get_random_llm_config

# Constants
QUALITY_THRESHOLD = 0.7  # Minimum quality score to consider a list (0-1)
LLM_VALIDATION_THRESHOLD = 0.6  # Minimum quality score to consider sending to LLM validation
PRE_FILTER_THRESHOLD = 0.55  # Threshold for pre-filtering based on heuristic score
USE_LLM_VALIDATION = True  # Whether to use LLM validation for lists
BINARY_LLM_DECISION = True  # Whether to use binary LLM decisions instead of scoring
LLM_VALIDATION_BATCH_SIZE = 5  # Number of lists to validate with LLM at once
RATE_LIMIT_DELAY = 1  # seconds between LLM calls
MAX_LISTS_FOR_LLM = 20 # Maximum number of lists to send for LLM validation/extraction
MIN_LISTS_FOR_LLM = 10 # Minimum number of lists to aim for sending to LLM, if available
JACCARD_THRESHOLD = 0.15 # Lowered threshold for connecting lists
FALLBACK_TOP_N = 5 # Number of lists to send in fallback
FALLBACK_MIN_SCORE = 0.4 # Minimum score for a list to be considered in fallback

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer() # Initialize lemmatizer once

class FilterConfig:
    """Configuration for list filtering"""
    def __init__(self,
                 quality_threshold: float = QUALITY_THRESHOLD,
                 llm_validation_threshold: float = LLM_VALIDATION_THRESHOLD,
                 pre_filter_threshold: float = PRE_FILTER_THRESHOLD,
                 use_llm_validation: bool = USE_LLM_VALIDATION,
                 binary_llm_decision: bool = BINARY_LLM_DECISION,
                 llm_validation_batch_size: int = LLM_VALIDATION_BATCH_SIZE,
                 provider: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 binary_system_prompt: Optional[str] = None,
                 scoring_fn: Optional[Callable] = None,
                 clean_item_fn: Optional[Callable] = None,
                 min_score_for_llm: Optional[float] = None, # Minimum heuristic score to send to LLM
                 model_type: Optional[str] = "default"): # LLM model type (e.g., default, pro)
        self.quality_threshold = quality_threshold
        self.llm_validation_threshold = llm_validation_threshold
        self.pre_filter_threshold = pre_filter_threshold
        self.use_llm_validation = use_llm_validation
        self.binary_llm_decision = binary_llm_decision
        self.llm_validation_batch_size = llm_validation_batch_size
        self.provider = provider or "gemini"
        self.system_prompt = system_prompt
        self.binary_system_prompt = binary_system_prompt
        self.scoring_fn = scoring_fn
        self.clean_item_fn = clean_item_fn
        self.min_score_for_llm = min_score_for_llm # Store the minimum score
        self.model_type = model_type # Store the model type


# init_llm function removed - using direct LLM calls
        provider=provider,
        model=model_name,
        temperature=0.3
    )


def deep_clean_list(items: List[str]) -> Set[str]:
    """Perform deeper cleaning: lowercasing, tokenizing, lemmatizing, removing stopwords."""
    cleaned_tokens = set()
    for item in items:
        try:
            # Lowercase and tokenize
            tokens = word_tokenize(item.lower())
            # Lemmatize, remove stopwords and non-alphanumeric tokens
            filtered_tokens = {
                LEMMATIZER.lemmatize(token) # Apply lemmatization
                for token in tokens 
                if token.isalnum() and token not in STOPWORDS # Removed length filter
            }
            cleaned_tokens.update(filtered_tokens)
        except Exception as e:
            logging.debug(f"Tokenization/cleaning error for item '{item}': {e}")
            continue
    # Remove empty strings that might result from lemmatization issues
    cleaned_tokens.discard('') 
    return cleaned_tokens


async def filter_lists(
    extracted_lists: List[Dict[str, Any]],
    context_term: str,
    config: FilterConfig,
    logger: Optional[logging.Logger] = None,
    save_debug: bool = False  # Set to False by default to avoid memory issues
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[List[str]]]:
    """
    Improved filtering with minimal processing but some smart heuristics
    before sending lists to the LLM.

    Args:
        extracted_lists: List of dictionaries with items and metadata
        context_term: The related term for context
        config: Filtering configuration
        logger: Optional logger
        save_debug: Whether to save intermediate results for debugging (default: False)

    Returns:
        Tuple containing:
        - final_lists: List of filtered lists (or verified sub-lists if using list extraction)
        - llm_candidates_actual: The subset of lists sent to the LLM.
        - llm_results_processed: The processed results from the LLM for the candidates.
    """
    # Directory for saving debug information
    debug_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                           "data", "debug", "filtering")
    
    # Only create directory if debug saving is enabled
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not extracted_lists:
        # Return empty structures matching the return type
        if logger:
            logger.info(f"⚠️ No lists to filter for '{context_term}'")
        return [], [], []
        
    if logger:
        logger.info(f"🔄 Starting improved filtering for '{context_term}'. Received {len(extracted_lists)} raw lists.")

    start_time = time.time()
    
    # Save raw extracted lists for analysis only if debug enabled
    if save_debug:
        raw_lists_file = os.path.join(debug_dir, f"{context_term}_raw_lists_{timestamp}.json")
        with open(raw_lists_file, 'w') as f:
            json.dump(extracted_lists, f, indent=2, default=str)
        if logger:
            logger.info(f"💾 Saved raw lists to {raw_lists_file}")
    
    # Stage 1: Basic Cleaning
    logger.debug(f"🧹 Stage 1: Cleaning {len(extracted_lists)} raw lists for '{context_term}'")
    cleaned_lists_raw = [] # Store potentially duplicated lists first
    
    for list_idx, list_data in enumerate(extracted_lists):
        original_items = list_data["items"] # Keep original for reference if needed
        metadata = list_data.get("metadata", {})
        source_url = metadata.get("url", "unknown_url")
        
        # Skip if no items initially
        if not original_items:
            continue
            
        # Clean items using the provided clean function or basic cleaning
        cleaned_items = []
        if config.clean_item_fn:
            cleaned_items = [config.clean_item_fn(item) for item in original_items]
            # Filter out empty items after cleaning
            cleaned_items = [item for item in cleaned_items if item]
        else:
            # Basic cleaning (apply if no specific function provided)
            for item in original_items:
                # Remove numbering, URLs, and excessive whitespace
                item = re.sub(r'\s*\(\d+\).*$', '', item)
                item = re.sub(r'\s*\d+\.\s*', '', item)
                item = re.sub(r'\s*\d+\s*$', '', item)
                item = re.sub(r'http\S+', '', item)
                # Remove trailing punctuation
                item = re.sub(r'[,:;]+$', '', item)
                item = ' '.join(item.split())
                if item:
                    cleaned_items.append(item)
        
        # Skip if too few items remain after cleaning
        if len(cleaned_items) < 3:
            if logger:
                 logger.debug(f"👎 List {list_idx} skipped: too few items after cleaning ({len(cleaned_items)} < 3)")
            continue
        
        # Pre-filtering: Identify likely research areas using heuristics
        is_likely_research_area_list = False
        
        # Heuristic 1: List items contain term-specific research keywords
        context_term_lower = context_term.lower()
        research_keywords = [
            f"{context_term_lower}",
            "research", "theory", "experimental", "applied", "computational",
            "quantum", "theoretical", "physics", "chemistry", "biology", "science",
            "engineering", "mathematics", "computer", "technology", "systems"
        ]
        
        term_keywords = []
        if context_term_lower == "physics":
            term_keywords = ["quantum", "particle", "nuclear", "astrophysics", "relativity", 
                            "cosmology", "condensed matter", "optics", "atomic"]
        elif context_term_lower == "biology":
            term_keywords = ["molecular", "cellular", "genomics", "ecology", "evolution", 
                            "microbiology", "genetics", "biochemistry", "neuroscience"]
        elif context_term_lower == "computer science":
            term_keywords = ["algorithm", "artificial intelligence", "machine learning", 
                            "data science", "systems", "networking", "security", "graphics"]
        
        research_keywords.extend(term_keywords)
        
        # Count keyword matches in the list
        keyword_matches = 0
        for item in cleaned_items:
            item_lower = item.lower()
            if any(keyword in item_lower for keyword in research_keywords):
                keyword_matches += 1
        
        keyword_ratio = keyword_matches / len(cleaned_items) if cleaned_items else 0
        
        # Heuristic 2: Consistent naming patterns (title case, similar lengths)
        title_case_count = sum(1 for item in cleaned_items if item.istitle() or item.isupper())
        title_case_ratio = title_case_count / len(cleaned_items) if cleaned_items else 0
        
        # Check consistency in item lengths
        item_lengths = [len(item) for item in cleaned_items]
        avg_length = sum(item_lengths) / len(item_lengths) if item_lengths else 0
        length_variance = sum((length - avg_length) ** 2 for length in item_lengths) / len(item_lengths) if item_lengths else 0
        length_consistency = min(1.0, 100 / (length_variance + 10))  # Normalize to 0-1 range
        
        # Heuristic 3: Navigation or UI elements (avoid these)
        navigation_indicators = ["home", "about", "contact", "login", "sign in", "register", "menu", "site map", "privacy"]
        nav_matches = sum(1 for item in cleaned_items if any(nav in item.lower() for nav in navigation_indicators))
        nav_ratio = nav_matches / len(cleaned_items) if cleaned_items else 0
        
        # Calculate a simple quality score
        quality_score = (keyword_ratio * 0.5) + (title_case_ratio * 0.2) + (length_consistency * 0.2) - (nav_ratio * 0.5)
        
        # Additional heuristic: Prefer moderate-sized lists (not too short, not too long)
        size_factor = 0.0
        if 4 <= len(cleaned_items) <= 20:
            size_factor = 0.2
        elif len(cleaned_items) > 20:
            size_factor = 0.1
        
        quality_score += size_factor
        
        # Store list with cleaned items and quality score
        cleaned_lists_raw.append({
            "items": cleaned_items, 
            "original_items": original_items, 
            "metadata": metadata,
            "source_url": source_url,
            "quality_score": quality_score
        })

    cleaning_time = time.time() - start_time
    if logger:
        logger.debug(f"⏱️ Cleaning completed in {cleaning_time:.2f}s")
    
    # Save cleaned lists for analysis only if debug enabled
    if save_debug:
        cleaned_lists_file = os.path.join(debug_dir, f"{context_term}_cleaned_lists_{timestamp}.json")
        with open(cleaned_lists_file, 'w') as f:
            json.dump(cleaned_lists_raw, f, indent=2, default=str)
        if logger:
            logger.info(f"💾 Saved cleaned lists to {cleaned_lists_file}")
    
    # Deduplicate based on cleaned items content (order-insensitive)
    logger.debug(f"🔍 Deduplicating {len(cleaned_lists_raw)} cleaned lists")
    unique_lists_content = set()
    deduplicated_lists = [] # This will hold the unique lists
    for list_entry in cleaned_lists_raw:
        # Create a unique representation (sorted tuple of items)
        list_tuple = tuple(sorted(list_entry["items"]))
        if list_tuple not in unique_lists_content:
            unique_lists_content.add(list_tuple)
            deduplicated_lists.append(list_entry) # Add the first occurrence

    dedup_time = time.time() - start_time - cleaning_time
    if logger:
        logger.debug(f"⏱️ Deduplication completed in {dedup_time:.2f}s")
        logger.info(f"📋 After cleaning and deduplication: {len(deduplicated_lists)} unique lists remain")
    
    # Save deduplicated lists for analysis only if debug enabled
    if save_debug:
        dedup_lists_file = os.path.join(debug_dir, f"{context_term}_dedup_lists_{timestamp}.json")
        with open(dedup_lists_file, 'w') as f:
            json.dump(deduplicated_lists, f, indent=2, default=str)
        if logger:
            logger.info(f"💾 Saved deduplicated lists to {dedup_lists_file}")
    
    if not deduplicated_lists:
        if logger:
            logger.warning(f"⚠️ No lists remain after cleaning and deduplication for '{context_term}'")
        return [], [], []

    # Select candidates for LLM processing based on quality score
    # Sort by quality score in descending order
    deduplicated_lists.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    
    # Define a reasonable maximum for LLM processing
    max_llm_lists = 50  # Cap at 50 lists to avoid overloading the LLM
    
    # Select the top-scoring lists first, up to the maximum
    llm_candidates_actual = deduplicated_lists[:max_llm_lists]
    
    if len(deduplicated_lists) > max_llm_lists:
        if logger:
            logger.warning(f"⚠️ Limiting LLM processing to top {max_llm_lists} lists out of {len(deduplicated_lists)} total")

    # Save LLM candidate lists for analysis only if debug enabled
    if save_debug:
        llm_candidate_lists_file = os.path.join(debug_dir, f"{context_term}_llm_candidates_{timestamp}.json")
        with open(llm_candidate_lists_file, 'w') as f:
            json.dump(llm_candidates_actual, f, indent=2, default=str)
        if logger:
            logger.info(f"💾 Saved LLM candidate lists to {llm_candidate_lists_file}")

    # Stage 3: LLM Processing
    llm_start_time = time.time()
    final_lists = []
    llm_results_processed = []
    
    if not llm_candidates_actual:
        logger.warning(f"⚠️ No candidates remain after minimal filtering for '{context_term}'")
        return [], [], []

    if llm_candidates_actual:
        if config.binary_llm_decision:
            # Use Binary LLM Validation
            logger.info(f"🧠 Sending {len(llm_candidates_actual)} lists to LLM (model: {config.model_type}) for extraction")
            llm_extraction_start = time.time()
            llm_results = await validate_and_extract_lists_with_llm(
                [cand["items"] for cand in llm_candidates_actual],
                context_term,
                config,
                logger
            )
            llm_extraction_time = time.time() - llm_extraction_start
            logger.debug(f"⏱️ LLM extraction completed in {llm_extraction_time:.2f}s")
            
            final_lists = llm_results # The result itself is the list of verified lists
            llm_results_processed = llm_results # Store the direct output for metadata
            
            if logger:
                verified_count = len(final_lists)
                total_items = sum(len(lst) for lst in final_lists)
                logger.info(f"📋 LLM extraction yielded {verified_count} verified lists with {total_items} total items")
                if verified_count > 0 and len(final_lists[0]) > 0:
                    logger.debug(f"📋 Sample from first verified list: {final_lists[0][:5]}")
        else:
            # Use Non-Binary LLM Validation (List Extraction/Verification)
            logger.info(f"🧠 Sending {len(llm_candidates_actual)} lists to LLM (model: {config.model_type}) for extraction")
            llm_extraction_start = time.time()
            llm_results = await validate_and_extract_lists_with_llm(
                [cand["items"] for cand in llm_candidates_actual],
                context_term,
                config,
                logger
            )
            llm_extraction_time = time.time() - llm_extraction_start
            logger.debug(f"⏱️ LLM extraction completed in {llm_extraction_time:.2f}s")
            
            final_lists = llm_results # The result itself is the list of verified lists
            llm_results_processed = llm_results # Store the direct output for metadata
            
            if logger:
                verified_count = len(final_lists)
                total_items = sum(len(lst) for lst in final_lists)
                logger.info(f"📋 LLM extraction yielded {verified_count} verified lists with {total_items} total items")
                if verified_count > 0 and len(final_lists[0]) > 0:
                    logger.debug(f"📋 Sample from first verified list: {final_lists[0][:5]}")
    else:
        if logger:
            logger.warning(f"⚠️ No suitable lists found to send for LLM validation for '{context_term}'")
        # Ensure return types match even when no LLM call happens
        final_lists = []
        llm_results_processed = []
    
    # Save final LLM results for analysis only if debug enabled
    if save_debug:
        llm_results_file = os.path.join(debug_dir, f"{context_term}_llm_results_{timestamp}.json")
        with open(llm_results_file, 'w') as f:
            # Create a paired structure to include both input and output
            llm_result_pairs = []
            for idx, result_list in enumerate(llm_results_processed):
                if idx < len(llm_candidates_actual):
                    llm_result_pairs.append({
                        "input_list": llm_candidates_actual[idx].get("items", []),
                        "output_list": result_list,
                        "source_url": llm_candidates_actual[idx].get("source_url", "unknown"),
                        "metadata": llm_candidates_actual[idx].get("metadata", {}),
                        "quality_score": llm_candidates_actual[idx].get("quality_score", 0)
                    })
            json.dump(llm_result_pairs, f, indent=2, default=str)
        if logger:
            logger.info(f"💾 Saved LLM results to {llm_results_file}")
        
    llm_time = time.time() - llm_start_time
    total_time = time.time() - start_time
        
    if logger:
        final_item_count = sum(len(lst) for lst in final_lists)
        logger.info(f"✅ Filtering complete for '{context_term}' in {total_time:.2f}s: {len(final_lists)} final lists, {final_item_count} total items")
        
        # Log timing breakdown
        logger.debug(f"⏱️ Timing breakdown: cleaning {cleaning_time:.2f}s, deduplication {dedup_time:.2f}s, " +
                     f"LLM processing {llm_time:.2f}s")
    
    # Ensure the final return matches the expected Tuple structure
    return final_lists, llm_candidates_actual, llm_results_processed


async def validate_lists_with_llm_binary(
    lists: List[List[str]], 
    context_term: str, 
    config: FilterConfig,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, bool]]:
    """
    Validate lists using LLM with a binary yes/no decision.
    
    Args:
        lists: List of lists to validate
        context_term: The related term for context
        config: Filtering configuration
        logger: Optional logger
        
    Returns:
        List of dictionaries with binary validation results
    """
    if not lists or not config.binary_system_prompt:
        return []
    
    llm = init_llm(config.provider, config.model_type) # Pass model_type
    results = []
    
    num_batches = (len(lists) + config.llm_validation_batch_size - 1) // config.llm_validation_batch_size
    
    for i in range(num_batches):
        batch_lists = lists[i * config.llm_validation_batch_size : (i + 1) * config.llm_validation_batch_size]
        prompts = []
        for list_to_validate in batch_lists:
            # Format list for prompt (e.g., Python list string)
            list_str = json.dumps(list_to_validate, indent=None) # Compact JSON list
            prompt = f"{config.binary_system_prompt.format(level0_term=context_term)}\n\nList:\n{list_str}"
            prompts.append(prompt)
        
        try:
            if logger:
                 logger.info(f"Sending batch {i+1}/{num_batches} ({len(batch_lists)} lists) to LLM ({config.provider}, model: {config.model_type}) for binary validation...")
                 
            responses = await llm.batch_prompt(prompts)
            
            for response_text in responses:
                # Process response - expect simple 'yes' or 'no' (case-insensitive)
                decision = response_text.strip().lower()
                is_valid = (decision == 'yes')
                results.append({"is_valid": is_valid, "raw_response": response_text})
                
        except Exception as e:
            if logger:
                 logger.error(f"LLM binary validation batch failed: {str(e)}")
            # Add error result for each item in the failed batch
            results.extend([{"is_valid": False, "error": str(e)} for _ in batch_lists])
            
        # Add delay between batches to avoid rate limiting
        if i < num_batches - 1:
            await asyncio.sleep(RATE_LIMIT_DELAY)
            
    return results


async def validate_lists_with_llm(
    lists: List[List[str]], 
    context_term: str, 
    config: FilterConfig,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Validate lists using LLM to determine if they are relevant and assign quality scores
    
    Args:
        lists: List of lists to validate
        context_term: The related term for context
        config: Filtering configuration
        logger: Optional logger
        
    Returns:
        List of dictionaries with validation results including quality scores
    """
    if not lists or not config.use_llm_validation:
        return []
    
    llm = init_llm(config.provider, config.model_type) # Pass model_type
    validation_results = []
    
    # Use a default system prompt if none provided
    system_prompt = config.system_prompt or f"""You are an expert in analyzing and categorizing lists of information.

Your task is to evaluate whether the provided list contains items that are the research areas, or a list of departments, of The College of {context_term.upper()}.

Consider:
1. Whether the items represent specific things related to {context_term}. Only include items that are under the umbrella of {context_term}.
2. Whether they use consistent naming patterns and terminology
3. Whether they appear to be a cohesive list rather than navigation elements or other website components

Non-relevant items might include website navigation links, general categories, or unrelated content."""
    
    # Process in batches to avoid exceeding context limits
    for batch_start in range(0, len(lists), config.llm_validation_batch_size):
        batch = lists[batch_start:batch_start + config.llm_validation_batch_size]
        
        # Construct prompt for LLM
        prompt = f"""I need to validate whether the following lists contain items related to {context_term}.
For each list, rate the quality on a scale of 0-1 where:
- 0 means the list contains no relevant items
- 1 means the list contains all relevant items

Please respond with JSON in this format:
```json
[
  {{
    "list_id": 0,
    "quality_score": 0.8,
    "is_valid_list": true,
    "explanation": "This list contains valid items because..."
  }},
  ...
]
```

Here are the lists to validate:

"""
        
        for i, item_list in enumerate(batch):
            sample_items = item_list[:10]  # Take first 10 items as sample
            prompt += f"List {i}:\n" + "\n".join([f"- {item}" for item in sample_items])
            if len(item_list) > 10:
                prompt += f"\n... (and {len(item_list) - 10} more items)"
            prompt += "\n\n"
        
        # Call LLM
        try:
        response = infer_text(
            provider=provider or "openai",
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT
        )
            
            # Parse JSON response
            try:
                json_str = response.text
                # Extract JSON if it's wrapped in markdown code blocks
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                results = json.loads(json_str)
                
                # Add validation results
                for result in results:
                    list_id = result.get("list_id", 0)
                    if list_id < len(batch):
                        validation_results.append({
                            "list": batch[list_id],
                            "quality_score": result.get("quality_score", 0),
                            "is_valid_list": result.get("is_valid_list", False),
                            "explanation": result.get("explanation", "")
                        })
            except json.JSONDecodeError as e:
                if logger:
                    logger.error(f"Error parsing LLM response: {str(e)}")
                    logger.error(f"Raw response: {response.text}")
                
                # Try to extract quality scores from text if JSON parsing fails
                for i, item_list in enumerate(batch):
                    list_marker = f"List {i}"
                    score_pattern = r"quality[_\s]score:?\s*([01](?:\.\d+)?)"
                    
                    # Look for score in text
                    if list_marker in response.text:
                        text_after_list = response.text.split(list_marker)[1].split("List")[0]
                        score_match = re.search(score_pattern, text_after_list, re.IGNORECASE)
                        if score_match:
                            score = float(score_match.group(1))
                            is_valid = score >= 0.5  # Assume valid if score >= 0.5
                            validation_results.append({
                                "list": batch[i],
                                "quality_score": score,
                                "is_valid_list": is_valid,
                                "explanation": "Extracted from non-JSON response"
                            })
        except Exception as e:
            if logger:
                logger.error(f"Error calling LLM for list validation: {str(e)}")
        
        # Add delay between batches
        await asyncio.sleep(RATE_LIMIT_DELAY)
    
    return validation_results


async def validate_and_extract_lists_with_llm(
    lists: List[List[str]],
    context_term: str,
    config: FilterConfig,
    logger: Optional[logging.Logger] = None
) -> List[List[str]]:
    """
    Validate lists using an LLM designed to extract valid items, returning verified sub-lists.
    Handles potential markdown and JSON formatting in responses.
    """
    if not lists:
        return []

    if not config.binary_system_prompt: # Reusing binary_system_prompt as the extraction prompt
        raise ValueError("System prompt (binary_system_prompt) is required for LLM list extraction.")
        
    llm = init_llm(config.provider, config.model_type) # Pass model_type
    verified_lists = [] # Store the lists of verified items returned by the LLM
    
    num_batches = (len(lists) + config.llm_validation_batch_size - 1) // config.llm_validation_batch_size

    for i in range(num_batches):
        batch_lists = lists[i * config.llm_validation_batch_size : (i + 1) * config.llm_validation_batch_size]
        
        if logger:
            logger.info(f"Sending batch {i+1}/{num_batches} ({len(batch_lists)} lists) to LLM ({config.provider}, model: {config.model_type}) for list extraction...")
        
        # Process each list individually instead of using batch_prompt
        responses = []
        for list_to_validate in batch_lists:
            # Format list for prompt (e.g., Python list string)
            # Use json.dumps for robust list representation
            list_str = json.dumps(list_to_validate, indent=None) 
            # Use the binary_system_prompt which is now designed for extraction
            prompt = f"{config.binary_system_prompt.format(level0_term=context_term, level1_term=context_term, level2_term=context_term)}\n\nInput List:\n{list_str}"
            
            try:
                # Call LLM for each list individually
        response = infer_text(
            provider=provider or "openai",
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT
        )
                responses.append(response.text)
                # Small delay between requests to avoid rate limits
                await asyncio.sleep(0.1)
            except Exception as e:
                if logger:
                    logger.error(f"Error calling LLM for list extraction: {str(e)}")
                responses.append("")  # Add empty response to maintain index alignment
            
        for response_text in responses:
            extracted_list = []
            try:
                # 1. Clean potential markdown code fences (```python ... ```, ```json ... ```, ``` ... ```)
                cleaned_response = response_text.strip()
                # Remove leading/trailing ``` optionally followed by language name and newline
                cleaned_response = re.sub(r'^```[a-z]*\n?', '', cleaned_response)
                # Remove trailing newline and ```
                cleaned_response = re.sub(r'\n?```$', '', cleaned_response)
                # Remove leading/trailing single backticks if present
                cleaned_response = cleaned_response.strip('`')
                cleaned_response = cleaned_response.strip()
                
                # 2. Attempt to parse as Python list literal or JSON list
                if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                    try:
                        # Try ast.literal_eval first (safer for Python list syntax)
                        parsed_list = ast.literal_eval(cleaned_response)
                        if isinstance(parsed_list, list):
                            # Ensure all items are strings
                            extracted_list = [str(item) for item in parsed_list if isinstance(item, (str, int, float))]
                        else:
                            if logger:
                                logger.warning(f"LLM response parsed but is not a list: {cleaned_response[:100]}...")
                    except (ValueError, SyntaxError, TypeError):
                        # If ast fails, try json.loads (for JSON list syntax)
                        try:
                            parsed_list = json.loads(cleaned_response)
                            if isinstance(parsed_list, list):
                                extracted_list = [str(item) for item in parsed_list if isinstance(item, (str, int, float))]
                            else:
                                 if logger:
                                     logger.warning(f"LLM response parsed (JSON) but is not a list: {cleaned_response[:100]}...")
                        except json.JSONDecodeError as json_err:
                            if logger:
                                logger.warning(f"Could not parse LLM response as Python or JSON list: '{cleaned_response[:100]}...'. Error: {json_err}")
                else:
                    if logger:
                         logger.warning(f"LLM response does not appear to be a list: '{cleaned_response[:100]}...'")
                         
            except Exception as parse_error:
                if logger:
                    logger.error(f"Error processing LLM response: '{response_text[:100]}...'. Error: {parse_error}")
            
            # Only add if the extracted list is not empty
            if extracted_list:
                verified_lists.append(extracted_list)
            # else: (implicit) if parsing fails or list is empty, it's skipped
            
        if i < num_batches - 1:
            await asyncio.sleep(RATE_LIMIT_DELAY)

    return verified_lists # Return the lists of verified items


def consolidate_lists(all_lists: List[List[str]], 
                    context_term: Optional[str] = None,
                    min_frequency: int = 1,
                    min_list_appearances: int = 1,
                    similarity_threshold: float = 0.7) -> List[str]:
    """
    Find common items across multiple lists
    Uses frequency analysis and intersection-union approach
    
    Args:
        all_lists: All extracted lists of items
        context_term: Optional related term for context (prioritize items containing this)
        min_frequency: Minimum frequency of an item to be included
        min_list_appearances: Minimum number of lists an item must appear in
        similarity_threshold: Threshold for considering items similar with word overlap
    
    Returns:
        Consolidated list of unique items
    """
    if not all_lists:
        return []
    
    # Normalize all items for case-insensitive comparison
    normalized_lists = [[item.lower() for item in lst] for lst in all_lists]
    
    # Create a frequency counter for all items
    all_items = [item for lst in normalized_lists for item in lst]
    item_counts = Counter(all_items)
    
    # Find items that appear with sufficient frequency
    consolidated_items = []
    seen = set()
    
    for item, count in sorted(item_counts.items(), key=lambda x: x[1], reverse=True):
        # Skip items we've already seen in normalized form
        if item in seen:
            continue
            
        # Count how many different lists this item appears in
        list_appearances = sum(1 for lst in normalized_lists if item in lst)
        
        if count >= min_frequency or list_appearances >= min_list_appearances:
            # Check for similar items to avoid duplicates
            is_similar = False
            for existing in list(consolidated_items):  # Use a copy of the list for iteration
                # Check for substantial string similarity or substring relationship
                existing_lower = existing.lower()
                
                # String containment check
                if item in existing_lower or existing_lower in item:
                    is_similar = True
                    # Keep the better version (prefer longer more detailed versions)
                    if len(item) > len(existing) * 1.2:  # New item is significantly longer
                        consolidated_items.remove(existing)
                        consolidated_items.append(item)
                        seen.add(existing_lower)
                    break
                
                # Word overlap check
                item_words = set(item.split())
                existing_words = set(existing_lower.split())
                if len(item_words) > 0 and len(existing_words) > 0:
                    overlap = len(item_words & existing_words) / max(len(item_words), len(existing_words))
                    if overlap >= similarity_threshold:
                        is_similar = True
                        # Keep the one with higher count or the longer one if counts are equal
                        existing_count = item_counts.get(existing_lower, 0)
                        if count > existing_count or (count == existing_count and len(item) > len(existing)):
                            consolidated_items.remove(existing)
                            consolidated_items.append(item)
                            seen.add(existing_lower)
                        break
            
            # Add the item if not similar to any existing item
            if not is_similar:
                consolidated_items.append(item)
                seen.add(item)
    
    # If context_term is provided, prioritize items containing it
    if context_term:
        context_term_lower = context_term.lower()
        # Sort so that items containing the context_term appear first
        consolidated_items.sort(key=lambda x: (0 if context_term_lower in x.lower() else 1, -item_counts.get(x.lower(), 0)))
    else:
        # Otherwise sort by frequency
        consolidated_items.sort(key=lambda x: -item_counts.get(x.lower(), 0))
    
    # Capitalize for consistency
    return [item.capitalize() for item in consolidated_items] 