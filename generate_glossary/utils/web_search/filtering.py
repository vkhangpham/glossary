import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from collections import Counter
import logging

from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS, BaseLLM

# Constants
QUALITY_THRESHOLD = 0.7  # Minimum quality score to consider a list (0-1)
LLM_VALIDATION_THRESHOLD = 0.6  # Minimum quality score to consider sending to LLM validation
PRE_FILTER_THRESHOLD = 0.55  # Threshold for pre-filtering based on heuristic score
USE_LLM_VALIDATION = True  # Whether to use LLM validation for lists
BINARY_LLM_DECISION = True  # Whether to use binary LLM decisions instead of scoring
LLM_VALIDATION_BATCH_SIZE = 5  # Number of lists to validate with LLM at once
RATE_LIMIT_DELAY = 1  # seconds between LLM calls


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
        model=GEMINI_MODELS["default"] if provider == Provider.GEMINI else OPENAI_MODELS["mini"],
        temperature=0.3
    )


async def filter_lists(
    extracted_lists: List[Dict[str, Any]], 
    context_term: str,
    config: FilterConfig,
    logger: Optional[logging.Logger] = None
) -> List[List[str]]:
    """
    Filter extracted lists based on quality score and optionally LLM validation
    
    Args:
        extracted_lists: List of dictionaries with items and metadata
        context_term: The related term for context
        config: Filtering configuration
        logger: Optional logger
        
    Returns:
        List of filtered lists
    """
    if not extracted_lists:
        return []
    
    # Stage 1: Apply pre-filtering to reduce lists that need LLM validation
    pre_filtered_lists = []
    
    for list_data in extracted_lists:
        items = list_data["items"]
        metadata = list_data.get("metadata", {})
        
        # Skip if no items
        if not items:
            continue
            
        # Score the list using the provided scoring function or fallback
        quality_score = 0.0
        if config.scoring_fn:
            quality_score = config.scoring_fn(items, metadata, context_term)
        else:
            # Fallback scoring just based on metadata
            keyword_ratio = metadata.get("keyword_ratio", 0)
            pattern_ratio = metadata.get("pattern_ratio", 0)
            non_term_ratio = metadata.get("non_term_ratio", 1)
            nav_score = metadata.get("structure_analysis", {}).get("nav_score", 1)
            quality_score = (keyword_ratio * 0.4 + pattern_ratio * 0.3 + (1 - non_term_ratio) * 0.2 + (1 - nav_score) * 0.1)
        
        # Clean items using the provided clean function
        if config.clean_item_fn:
            cleaned_items = [config.clean_item_fn(item) for item in items]
            # Filter out empty items
            cleaned_items = [item for item in cleaned_items if item]
        else:
            # Basic cleaning
            cleaned_items = []
            for item in items:
                # Remove trailing numbers, parenthetical info
                item = re.sub(r'\s*\(\d+\).*$', '', item)
                item = re.sub(r'\s*\d+\s*$', '', item)
                # Remove URLs
                item = re.sub(r'http\S+', '', item)
                # Clean whitespace
                item = ' '.join(item.split())
                
                if item:
                    cleaned_items.append(item)
        
        # Skip if too few items remain after cleaning
        if len(cleaned_items) < 3:
            continue
        
        # Store list with quality score and cleaned items
        pre_filtered_lists.append({
            "items": cleaned_items,
            "original_items": items,
            "metadata": metadata,
            "quality_score": quality_score
        })
    
    if logger:
        logger.debug(f"Pre-filtered to {len(pre_filtered_lists)} lists (from {len(extracted_lists)} original lists)")
    
    # Check if we're using an enhanced mode where all lists should be validated by LLM
    force_llm_validation = config.use_llm_validation and config.binary_llm_decision
    
    # Stage 2: Apply filtering based on configuration
    if force_llm_validation:
        # When forcing LLM validation for all lists, we'll use pre-filtering 
        # only to eliminate very low quality lists
        candidate_lists = [l for l in pre_filtered_lists if l["quality_score"] >= config.pre_filter_threshold]
        
        if logger:
            logger.debug(f"Sending {len(candidate_lists)} lists for LLM validation (forced validation mode)")
        
        # Get decisions from LLM for all candidate lists
        validated_lists = []
        
        # Process in batches to avoid overloading the LLM API
        for i in range(0, len(candidate_lists), config.llm_validation_batch_size):
            batch = candidate_lists[i:i+config.llm_validation_batch_size]
            
            llm_decisions = await validate_lists_with_llm_binary(
                [l["items"] for l in batch],
                context_term,
                config,
                logger
            )
            
            # Add only lists that are positively verified by LLM
            for j, decision in enumerate(llm_decisions):
                if j < len(batch):
                    is_validated = decision.get("is_valid_list", False)
                    if is_validated:
                        validated_lists.append(batch[j])
            
            # Add a small delay between batches
            if i + config.llm_validation_batch_size < len(candidate_lists):
                await asyncio.sleep(RATE_LIMIT_DELAY)
        
        if logger:
            logger.debug(f"LLM validated {len(validated_lists)} lists out of {len(candidate_lists)} candidates")
        
        return [l["items"] for l in validated_lists]
    else:
        # Use the original three-stage filtering approach
        # Sort by quality score (highest first)
        pre_filtered_lists.sort(key=lambda x: x["quality_score"], reverse=True)
        
        high_quality_lists = [l for l in pre_filtered_lists if l["quality_score"] >= config.quality_threshold]
        medium_quality_lists = [l for l in pre_filtered_lists if config.pre_filter_threshold <= l["quality_score"] < config.quality_threshold]
        
        if logger:
            logger.debug(f"Found {len(high_quality_lists)} high-quality lists (score >= {config.quality_threshold})")
            logger.debug(f"Found {len(medium_quality_lists)} medium-quality lists for potential LLM validation")
        
        # If we have no lists to potentially validate, just return high quality lists
        if not medium_quality_lists:
            return [l["items"] for l in high_quality_lists]
        
        # If LLM validation is disabled, use a higher threshold for the medium quality lists
        if not config.use_llm_validation:
            additional_lists = [l for l in medium_quality_lists if l["quality_score"] >= config.pre_filter_threshold + 0.1]
            if logger:
                logger.debug(f"Adding {len(additional_lists)} medium-quality lists without LLM validation")
            return [l["items"] for l in high_quality_lists + additional_lists]
        
        # Stage 3: LLM validation for medium quality lists
        # To reduce API costs, limit the number of lists for LLM validation
        max_llm_validations = min(len(medium_quality_lists), config.llm_validation_batch_size)
        llm_candidate_lists = medium_quality_lists[:max_llm_validations]
        
        if logger:
            logger.debug(f"Validating {len(llm_candidate_lists)} medium-quality lists with LLM")
        
        # Get decisions from LLM
        validated_lists = []
        
        if config.binary_llm_decision:
            llm_decisions = await validate_lists_with_llm_binary(
                [l["items"] for l in llm_candidate_lists],
                context_term,
                config,
                logger
            )
            
            # Add validated lists that meet the decision threshold
            for i, decision in enumerate(llm_decisions):
                if i < len(llm_candidate_lists):
                    is_validated = decision.get("is_valid_list", False)
                    if is_validated:
                        validated_lists.append(llm_candidate_lists[i])
            
            if logger:
                logger.debug(f"LLM validated {len(validated_lists)} additional lists")
        else:
            # Use scoring approach
            llm_results = await validate_lists_with_llm(
                [l["items"] for l in llm_candidate_lists],
                context_term,
                config,
                logger
            )
            
            # Update scores based on LLM validation
            for i, result in enumerate(llm_results):
                if i < len(llm_candidate_lists):
                    llm_candidate_lists[i]["quality_score"] = result.get("quality_score", llm_candidate_lists[i]["quality_score"])
                    if llm_candidate_lists[i]["quality_score"] >= config.quality_threshold:
                        validated_lists.append(llm_candidate_lists[i])
        
        # Combine high quality lists with validated medium quality lists
        final_lists = high_quality_lists + validated_lists
        
        # Sort by quality score (highest first)
        final_lists.sort(key=lambda x: x["quality_score"], reverse=True)
        
        if logger:
            logger.debug(f"Final filtered to {len(final_lists)} high-quality lists")
        
        # Return the items from final lists
        return [l["items"] for l in final_lists]


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