import asyncio
import json
import re
import time
import ast # Added for safe literal evaluation
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Set
from collections import Counter
import logging

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

from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM

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
                 clean_item_fn: Optional[Callable] = None):
        self.quality_threshold = quality_threshold
        self.llm_validation_threshold = llm_validation_threshold
        self.pre_filter_threshold = pre_filter_threshold
        self.use_llm_validation = use_llm_validation
        self.binary_llm_decision = binary_llm_decision
        self.llm_validation_batch_size = llm_validation_batch_size
        self.provider = provider or Provider.GEMINI
        self.system_prompt = system_prompt
        self.binary_system_prompt = binary_system_prompt
        self.scoring_fn = scoring_fn
        self.clean_item_fn = clean_item_fn


def init_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Initialize LLM with specified provider
    
    Args:
        provider: Optional provider name
        
    Returns:
        LLM instance
    """
    if not provider:
        provider = Provider.GEMINI  # Default to Gemini
        
    return LLMFactory.create_llm(
        provider=provider,
        model=GEMINI_MODELS["default"] if provider == Provider.GEMINI else OPENAI_MODELS["default"],
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
    logger: Optional[logging.Logger] = None
) -> Tuple[List[List[str]], List[Dict[str, Any]], List[List[str]]]:
    """
    Filter lists using community detection based on shared cleaned content.

    Args:
        extracted_lists: List of dictionaries with items and metadata
        context_term: The related term for context
        config: Filtering configuration
        logger: Optional logger

    Returns:
        Tuple containing:
        - final_lists: List of filtered lists (or verified sub-lists if using list extraction)
        - llm_candidates_actual: The subset of lists sent to the LLM.
        - llm_results_processed: The processed results from the LLM for the candidates.
    """
    if not extracted_lists:
        # Return empty structures matching the new return type
        return [], [], []
        
    if logger:
        logger.info(f"Starting filtering for {context_term}. Received {len(extracted_lists)} raw lists.")

    # Stage 1: Clean, Score, and Pre-filter
    scored_cleaned_lists_raw = [] # Store potentially duplicated lists first
    
    for list_data in extracted_lists:
        original_items = list_data["items"] # Keep original for reference if needed
        metadata = list_data.get("metadata", {})
        
        # Skip if no items initially
        if not original_items:
            continue
            
        # Clean items first using the provided clean function
        cleaned_items = []
        if config.clean_item_fn:
            cleaned_items = [config.clean_item_fn(item) for item in original_items]
            # Filter out empty items after cleaning
            cleaned_items = [item for item in cleaned_items if item]
        else:
            # Basic cleaning (apply if no specific function provided)
            for item in original_items:
                item = re.sub(r'\s*\(\d+\).*$', '', item)
                item = re.sub(r'\s*\d+\s*$', '', item)
                item = re.sub(r'http\S+', '', item)
                item = ' '.join(item.split())
                if item:
                    cleaned_items.append(item)
        
        # Skip if too few items remain after cleaning
        if len(cleaned_items) < 3:
            if logger:
                 logger.debug(f"Skipping list due to < 3 items after cleaning: {original_items}")
            continue
            
        # Now, score the *cleaned* list using the provided scoring function or fallback
        quality_score = 0.0
        if config.scoring_fn:
            # Pass cleaned items to the scoring function
            quality_score = config.scoring_fn(cleaned_items, metadata, context_term)
        else:
            # Fallback scoring
            keyword_ratio = metadata.get("keyword_ratio", 0)
            pattern_ratio = metadata.get("pattern_ratio", 0)
            non_term_ratio = metadata.get("non_term_ratio", 1)
            nav_score = metadata.get("structure_analysis", {}).get("nav_score", 1)
            quality_score = (keyword_ratio * 0.4 + pattern_ratio * 0.3 + (1 - non_term_ratio) * 0.2 + (1 - nav_score) * 0.1)
            if logger:
                 # Change level to DEBUG as it's not a critical error
                 logger.debug("Using fallback scoring logic. Provide config.scoring_fn for better results.")
        
        # Store list with quality score and cleaned items temporarily
        scored_cleaned_lists_raw.append({
            "items": cleaned_items, 
            "original_items": original_items, 
            "metadata": metadata,
            "quality_score": quality_score
        })

    # Deduplicate based on cleaned items content (order-insensitive)
    unique_lists_content = set()
    scored_cleaned_lists = [] # This will hold the unique lists
    for list_entry in scored_cleaned_lists_raw:
        # Create a unique representation (sorted tuple of items)
        list_tuple = tuple(sorted(list_entry["items"]))
        if list_tuple not in unique_lists_content:
            unique_lists_content.add(list_tuple)
            scored_cleaned_lists.append(list_entry) # Add the first occurrence

    if logger:
        logger.debug(f"Scored and cleaned {len(scored_cleaned_lists_raw)} lists.")
        logger.info(f"Reduced to {len(scored_cleaned_lists)} unique lists after deduplication.")
    
    if not scored_cleaned_lists:
        return [], [], []

    # --- Community Detection Stage --- 

    # 1. Deep Cleaning for community analysis
    deep_cleaned_map = {} # index -> set of deeply cleaned tokens
    for i, list_entry in enumerate(scored_cleaned_lists):
        # Use the already cleaned items for deeper cleaning
        deep_cleaned_map[i] = deep_clean_list(list_entry["items"])

    # 2. Community Finding (Graph based on Jaccard Similarity of deep-cleaned tokens)
    G = nx.Graph()
    list_indices = list(range(len(scored_cleaned_lists)))
    G.add_nodes_from(list_indices)

    edges_added = 0
    for i in range(len(list_indices)):
        for j in range(i + 1, len(list_indices)):
            idx1 = list_indices[i]
            idx2 = list_indices[j]
            set1 = deep_cleaned_map[idx1]
            set2 = deep_cleaned_map[idx2]
            
            # Calculate Jaccard Similarity if sets are non-empty
            if set1 and set2:
                intersection_size = len(set1.intersection(set2))
                union_size = len(set1.union(set2))
                if union_size > 0:
                    jaccard_sim = intersection_size / union_size
                    # Add edge if similarity exceeds threshold
                    if jaccard_sim >= JACCARD_THRESHOLD:
                        G.add_edge(idx1, idx2)
                        edges_added += 1
                
    if logger:
        # Use the edges_added count for logging
        logger.debug(f"Built graph with {G.number_of_nodes()} nodes and {edges_added} edges based on Jaccard similarity >= {JACCARD_THRESHOLD}.")

    # Find communities (connected components)
    raw_components = list(nx.connected_components(G))
    # Filter for components of size > 1
    connected_components = [list(c) for c in raw_components if len(c) > 1]
    num_raw_components = len(raw_components)
    num_communities = len(connected_components)
    num_lists_in_communities = sum(len(c) for c in connected_components)
    
    if logger:
        logger.debug(f"Found {num_raw_components} raw connected components.")
        logger.info(f"Found {num_communities} communities (size > 1) involving {num_lists_in_communities} lists.")

    llm_candidates_actual = []
    if not connected_components:
        logger.warning("No list communities (size > 1) found. Falling back to top-N scoring on all unique lists.")
        # --- Refined Fallback Logic --- 
        # Sort all unique lists by their heuristic score
        scored_cleaned_lists.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)
        # Filter by minimum score threshold
        fallback_candidates_potential = [
            l for l in scored_cleaned_lists if l.get('quality_score', 0.0) >= FALLBACK_MIN_SCORE
        ]
        # Select the top N lists from the filtered candidates (or fewer)
        num_to_send = min(len(fallback_candidates_potential), FALLBACK_TOP_N)
        llm_candidates_actual = fallback_candidates_potential[:num_to_send]
        logger.info(f"[Fallback] Selected top {len(llm_candidates_actual)} lists (score >= {FALLBACK_MIN_SCORE}) to send to LLM.")
        # --- End Refined Fallback Logic --- 
    else:
        # --- Community Selection Logic --- 
        # 3. Community Scoring
        community_scores = []
        for component_indices in connected_components:
            component_lists = [scored_cleaned_lists[idx] for idx in component_indices]
            if not component_lists: continue
            avg_score = sum(l['quality_score'] for l in component_lists) / len(component_lists)
            community_scores.append({
                'score': avg_score, 
                'lists': component_lists, 
                'indices': component_indices # Keep track of original indices for logging
            })

        # 4. Community Selection
        community_scores.sort(key=lambda x: x['score'], reverse=True)
        
        if logger:
            # Log details of top few communities for debugging
            top_communities_log = []
            for i, comm in enumerate(community_scores[:5]): # Log top 5
                top_communities_log.append(f"  #{i+1}: Score={comm['score']:.3f}, Size={len(comm['lists'])}, Indices={comm['indices']}")
            logger.debug("Top 5 communities (by avg score):\n" + "\n".join(top_communities_log))

        selected_lists_set = set() # Track selected list items to avoid duplicates
        
        # First pass: Ensure minimum
        for community in community_scores:
            if len(llm_candidates_actual) >= MIN_LISTS_FOR_LLM:
                break
            for list_entry in community['lists']:
                 # Check if list content already selected to prevent adding same list from lower ranked community
                 list_tuple = tuple(sorted(list_entry["items"]))
                 if list_tuple not in selected_lists_set:
                      if len(llm_candidates_actual) < MAX_LISTS_FOR_LLM: # Still respect overall max
                           llm_candidates_actual.append(list_entry)
                           selected_lists_set.add(list_tuple)
                      else:
                           break # Stop adding if max reached
            if len(llm_candidates_actual) >= MAX_LISTS_FOR_LLM: break

        # Second pass: Add more up to max from remaining top communities
        if len(llm_candidates_actual) < MAX_LISTS_FOR_LLM:
            for community in community_scores:
                if len(llm_candidates_actual) >= MAX_LISTS_FOR_LLM:
                    break
                for list_entry in community['lists']:
                    list_tuple = tuple(sorted(list_entry["items"]))
                    if list_tuple not in selected_lists_set:
                        if len(llm_candidates_actual) < MAX_LISTS_FOR_LLM:
                            llm_candidates_actual.append(list_entry)
                            selected_lists_set.add(list_tuple)
                        else:
                            break # Stop adding if max reached
                if len(llm_candidates_actual) >= MAX_LISTS_FOR_LLM: break

        # Ensure final list count respects MAX (should be handled by checks within loops)
        llm_candidates_actual = llm_candidates_actual[:MAX_LISTS_FOR_LLM]
        
        if logger:
             logger.info(f"Selected {len(llm_candidates_actual)} lists from top communities for LLM (target {MIN_LISTS_FOR_LLM}-{MAX_LISTS_FOR_LLM}).")
        # --- End Community Selection Logic --- 

    # Stage 3: LLM Processing (Common for both community and fallback selection)
    final_lists = []
    llm_results_processed = []
    
    if not llm_candidates_actual:
        logger.info("No candidates selected for LLM processing after filtering/community selection.")
        return [], [], []

    # Perform LLM validation/extraction on the selected subset
    if config.binary_llm_decision: 
        llm_decisions = await validate_lists_with_llm_binary(
            [l["items"] for l in llm_candidates_actual], context_term, config, logger
        )
        llm_results_processed = llm_decisions
        for i, decision in enumerate(llm_decisions):
            if i < len(llm_candidates_actual) and decision.get("is_valid_list", False):
                final_lists.append(llm_candidates_actual[i]["items"])
        logger.info(f"LLM binary validation resulted in {len(final_lists)} verified lists.")
    else:
        llm_extracted_sublists = await validate_and_extract_lists_with_llm(
             [l["items"] for l in llm_candidates_actual], context_term, config, logger
        )
        llm_results_processed = llm_extracted_sublists
        for sublist in llm_extracted_sublists:
             if sublist: 
                 final_lists.append(sublist)
        logger.info(f"LLM list extraction returned {len(final_lists)} non-empty verified sub-lists.")

    return final_lists, llm_candidates_actual, llm_results_processed


async def validate_lists_with_llm_binary(
    lists: List[List[str]], 
    context_term: str, 
    config: FilterConfig,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, bool]]:
    """
    Validate lists using LLM with binary (yes/no) decisions
    
    Args:
        lists: List of lists to validate
        context_term: The related term for context
        config: Filtering configuration
        logger: Optional logger
        
    Returns:
        List of dictionaries with binary validation results
    """
    if not lists or not config.use_llm_validation:
        return []
    
    llm = init_llm(config.provider)
    validation_results = []
    
    # Use a default binary system prompt if none provided
    binary_system_prompt = config.binary_system_prompt or f"""You are an expert in classifying and organizing lists.

Your task is to evaluate whether a provided list contains valid items related to {context_term}.

You must return a clear YES or NO decision for each list.
- Answer YES if the list primarily contains items related to {context_term}
- Answer NO if the list contains menu items, website sections, non-relevant content, etc.

THIS IS CRITICAL: Your decision must be binary (YES/NO) with no middle ground or uncertainty."""
    
    # Process in batches to avoid exceeding context limits
    for batch_start in range(0, len(lists), config.llm_validation_batch_size):
        batch = lists[batch_start:batch_start + config.llm_validation_batch_size]
        
        # Construct prompt for LLM
        prompt = f"""I need to determine whether these lists contain valid items related to {context_term}.

For each list, provide a binary (YES/NO) decision on whether it contains relevant items.

Please respond with JSON in this format:
```json
[
  {{
    "list_id": 0,
    "is_valid_list": true,
    "explanation": "These are clearly relevant because..."
  }},
  {{
    "list_id": 1,
    "is_valid_list": false,
    "explanation": "These appear to be navigation menu items because..."
  }},
  ...
]
```

Here are the lists to validate:

"""
        
        for i, item_list in enumerate(batch):
            sample_items = item_list[:15]  # Take more items to give LLM better context
            prompt += f"List {i}:\n" + "\n".join([f"- {item}" for item in sample_items])
            if len(item_list) > 15:
                prompt += f"\n... (and {len(item_list) - 15} more items)"
            prompt += "\n\n"
        
        # Add context about what makes a valid list
        prompt += f"""
Remember, valid lists for {context_term} will:
1. Contain items that are specifically related to {context_term}
2. Use consistent naming patterns
3. Have appropriate terminology for the field

Invalid lists might contain:
1. Website navigation items ("Home", "Contact", "About")
2. Administrative categories
3. Random website content
4. Items unrelated to {context_term}

For each list, provide a clear YES or NO decision in the is_valid_list field.
"""
        
        # Call LLM
        try:
            response = llm.infer(prompt=prompt, system_prompt=binary_system_prompt)
            
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
                            "list_id": list_id + batch_start,  # Adjust list_id to be relative to all lists
                            "items": batch[list_id],
                            "is_valid_list": result.get("is_valid_list", False),
                            "explanation": result.get("explanation", "")
                        })
                        
                        # Log the decision for debugging
                        if logger:
                            decision = "YES" if result.get("is_valid_list", False) else "NO"
                            logger.debug(f"LLM decision for list {list_id + batch_start}: {decision}")
            except json.JSONDecodeError as e:
                if logger:
                    logger.error(f"Error parsing LLM response: {str(e)}")
                    logger.error(f"Raw response: {response.text}")
                
                # Try to extract decisions from text if JSON parsing fails
                for i, item_list in enumerate(batch):
                    list_marker = f"List {i}"
                    
                    # Look for decision in text
                    text_after_list = response.text.split(list_marker)[-1].lower()
                    if "yes" in text_after_list.split("list")[0]:
                        validation_results.append({
                            "list_id": i + batch_start,
                            "items": batch[i],
                            "is_valid_list": True,
                            "explanation": "Extracted from non-JSON response"
                        })
                    elif "no" in text_after_list.split("list")[0]:
                        validation_results.append({
                            "list_id": i + batch_start,
                            "items": batch[i],
                            "is_valid_list": False,
                            "explanation": "Extracted from non-JSON response"
                        })
        except Exception as e:
            if logger:
                logger.error(f"Error calling LLM for list validation: {str(e)}")
        
        # Add delay between batches
        await asyncio.sleep(RATE_LIMIT_DELAY)
    
    return validation_results


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
    
    llm = init_llm(config.provider)
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
            response = llm.infer(prompt=prompt, system_prompt=system_prompt)
            
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
    Validate lists using LLM and extract verified sub-lists based on the prompt.

    Args:
        lists: List of lists to validate.
        context_term: The related term for context.
        config: Filtering configuration (expects binary_system_prompt to be set for list extraction).
        logger: Optional logger.

    Returns:
        List containing the verified sub-lists extracted by the LLM for each input list.
        If an input list results in no verified items, an empty list is returned for that position.
    """
    if not lists or not config.use_llm_validation or config.binary_llm_decision:
        # This function should only be called when list extraction is intended
        if logger:
            logger.warning("validate_and_extract_lists_with_llm called inappropriately. Check FilterConfig.")
        return [[] for _ in lists] # Return empty lists if called incorrectly

    llm = init_llm(config.provider)
    all_extracted_sublists = []

    # Ensure the prompt for list extraction is available
    list_extraction_prompt = config.binary_system_prompt # Reusing binary_system_prompt
    if not list_extraction_prompt:
        if logger:
            logger.error("LLM list extraction requires a system prompt (binary_system_prompt in FilterConfig).")
        return [[] for _ in lists]

    # Process in batches
    for batch_start in range(0, len(lists), config.llm_validation_batch_size):
        batch = lists[batch_start:batch_start + config.llm_validation_batch_size]
        batch_results = [[] for _ in batch] # Initialize results for this batch

        # Construct prompt for LLM - expecting a list output for each input list
        prompt = f"Analyze the following lists. For each list, return a Python-style list containing ONLY the items that are directly relevant research areas or courses for The Department of {context_term}. Return an empty list `[]` if no items are relevant.\n\n"

        list_inputs_formatted = []
        for i, item_list in enumerate(batch):
            # Limit item count sent to LLM if necessary, but maybe more context is better?
            sample_items = item_list # Send all items for better context
            list_str = f"Input List {i}:\n{json.dumps(sample_items, indent=2)}"
            list_inputs_formatted.append(list_str)

        prompt += "\n\n".join(list_inputs_formatted)
        prompt += f"\n\nRespond ONLY with the verified lists, one per line, in Python list format. Example:\nOutput for List 0: ['Verified Item 1', 'Verified Item 2']\nOutput for List 1: []\n..."

        try:
            response = llm.infer(prompt=prompt, system_prompt=list_extraction_prompt)
            response_text = response.text.strip()

            if logger:
                 logger.debug(f"LLM response for list extraction (Batch starting {batch_start}):\n{response_text}")

            # Attempt to parse the response - expecting one line per list
            lines = response_text.splitlines()
            parsed_count = 0
            for i, line in enumerate(lines):
                if i >= len(batch): continue # Shouldn't happen if LLM follows instructions

                line = line.strip()
                # Try to find the list structure (e.g., "Output for List 0: [...]")
                match = re.match(r".*Output for List \d+:\s*(.*)", line, re.IGNORECASE)
                list_str = match.group(1).strip() if match else line

                try:
                    # Use ast.literal_eval for safer evaluation than eval()
                    extracted_list = ast.literal_eval(list_str)
                    if isinstance(extracted_list, list):
                         # Basic validation: ensure items are strings
                         validated_list = [str(item).strip() for item in extracted_list if isinstance(item, (str, int, float)) and str(item).strip()]
                         batch_results[i] = validated_list
                         parsed_count += 1
                         if logger:
                              logger.debug(f"  Successfully parsed list for batch index {i}: {validated_list}")
                    else:
                         if logger:
                              logger.warning(f"  Could not parse line {i+1} as list (wrong type: {type(extracted_list)}): {line}")
                except (ValueError, SyntaxError, TypeError) as e:
                    if logger:
                         logger.warning(f"  Could not parse line {i+1} as list (Error: {e}): {line}")

            if parsed_count != len(batch) and logger:
                 logger.warning(f"LLM list extraction: Parsed {parsed_count} lists, but expected {len(batch)} for this batch.")
            
            all_extracted_sublists.extend(batch_results)

        except Exception as e:
            if logger:
                logger.error(f"Error calling LLM for list extraction (Batch starting {batch_start}): {str(e)}", exc_info=True)
            # Add empty lists for the failed batch
            all_extracted_sublists.extend([[] for _ in batch])

        # Add delay between batches
        if batch_start + config.llm_validation_batch_size < len(lists):
             await asyncio.sleep(RATE_LIMIT_DELAY)

    # Ensure the number of results matches the number of input lists
    if len(all_extracted_sublists) != len(lists):
         if logger:
              logger.error(f"LLM list extraction mismatch: Expected {len(lists)} results, got {len(all_extracted_sublists)}. Padding with empty lists.")
         # Pad with empty lists if necessary
         all_extracted_sublists.extend([[] for _ in range(len(lists) - len(all_extracted_sublists))])

    return all_extracted_sublists


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